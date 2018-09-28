import time
import os
import threading
import subprocess
import requests
from threads import Threads
import ast


class Task(object):
	'''
	task description.
	'''
	def __init__(self, job, role, id, status, logger):
		'''
		initialize a task
		'''
		(self.job_name, self.job_dir, self.task_template, self.num_ps, self.num_worker) = job
		self.role = role
		self.id = id
		self.name = self.job_name + '-' + self.role + '-' + str(self.id)
		self.status = status
		self.logger = logger
		self.volume_mounted = False

	def set_resource_spec(self, cpu, mem, bw=0, gpu=0):
		'''
		cpu and mem is must
		'''
		self.cpu = cpu
		self.mem = mem
		self.bw = bw
		self.gpu = gpu

	def set_container(self, image, script, work_volume, data_volume):
		'''
		container description
		work_vomule = (work_dir, host_workdir_prefix, name='k8s-mxnet-work-volume')
		data_volume = (data_dir, host_data_dir, name='k8s-mxnet-data-volume')
		'''
		self.image = image
		self.script = script
		(work_dir, host_workdir_prefix, name) = work_volume
		postfix = self.name + '/'
		host_work_dir = host_workdir_prefix + postfix
		self.work_volume = (work_dir, host_work_dir, name)
		self.data_volume = data_volume

	def set_data_source(self, hdfs_url='', local_mounted=True):
		'''data specification, if data not in local host, fetch from HDFS'''
		self.hdfs_url = hdfs_url # a list of datasets
		self.local_mounted = local_mounted

	def set_train(self, prog, batch_size=1, kv_store='sync', num_examples=0, num_epochs=100):
		self.prog = prog
		self.batch_size = batch_size
		self.kv_store = kv_store
		self.num_examples = num_examples
		self.num_epochs = num_epochs
		self.epoch_size = num_examples/batch_size

	def set_mxnet_env(self, kv_store_big_array_bound=1000000, ps_verbose=0):
		'''set env MXNET_KVSTORE_BIGARRAY_BOUND, PS_VERBOSE'''
		self.kv_store_big_array_bound = kv_store_big_array_bound
		self.ps_verbose = 0

	def set_placement(self, node):
		'''
		the node ip for task placement
		'''
		self.node = node
	
	def _mount_volume(self):
		'''
		directories on hosts mounted to containers
		'''
		if self.volume_mounted:
			return

		# create work dir
		(work_dir, host_work_dir, name) = self.work_volume
		cmd = 'ssh ' + self.node + ' "sudo rm -rf ' + host_work_dir + '; mkdir -p ' + host_work_dir + '"'
		os.system(cmd)

		self.volume_mounted = True

		# if it is worker task or data not mounted from local host, then read data from HDFS
		if self.role == "ps":
			return
		if self.local_mounted:
			return
		if self.hdfs_url is None or self.hdfs_url == '':
			raise ValueError('HDFS data URL is not specified')

		(data_dir, host_data_dir, name) = self.data_volume
		pool = Threads()
		for data in self.hdfs_url:
			fn = data.split("/")[-1]
			local_file = host_data_dir + fn
			# force copy even exist: some file may be broken due to interruption
			cmd = 'ssh ' + self.node + ' "/usr/local/hadoop/bin/hadoop fs -copyToLocal -f ' + data + ' ' + local_file + '"'
			os.system(cmd)
			thread = threading.Thread(target=(lambda cmd=cmd: os.system(cmd)), args=())
			pool.add(thread)
		pool.start()
		pool.wait()

	def _create(self):
		'''create task definition, i.e., yaml file'''
		variables = dict()
		variables['JOB_NAME'] = self.job_name
		variables['NUM_PS'] = str(self.num_ps)
		variables['NUM_WORKER'] = str(self.num_worker)

		variables['TASK_NAME'] = self.name
		variables['TASK_ID'] = str(self.id)
		variables['TASK_ROLE'] = self.role
		variables['TASK_NODE'] = self.node
		variables['CPU'] = str(self.cpu)
		variables['MEM'] = str(self.mem) + "Gi"
		variables['GPU'] = str(self.gpu)

		variables['IMAGE'] = self.image
		variables['SCRIPT'] = self.script
		variables['PROG'] = self.prog

		(work_dir, host_work_dir, name) = self.work_volume
		variables['WORK_DIR'] = work_dir
		variables['HOST_WORK_DIR'] = host_work_dir
		variables['WORK_VOLUME'] = name

		(data_dir, host_data_dir, name) = self.data_volume
		variables['DATA_DIR'] = data_dir
		variables['HOST_DATA_DIR'] = host_data_dir
		variables['DATA_VOLUME'] = name

		variables['KV_STORE'] = self.kv_store
		variables['BATCH_SIZE'] = str(self.batch_size)
		variables['MXNET_KVSTORE_BIGARRAY_BOUND'] = str(self.kv_store_big_array_bound)
		variables['PS_VERBOSE'] = str(self.ps_verbose)
		
		# generate yaml file
		jinja = self.job_dir + self.name + '.jinja'
		os.system("cp " + self.task_template + " " + jinja)

		temp_file = jinja + '.temp'
		for key, value in variables.items():
			os.system('sed -e "s@\$' + key + '@' + value + '@g" "' + jinja + '"' + ' > ' + temp_file)
			os.system('rm ' + jinja)
			os.system('mv ' + temp_file + ' ' + jinja)
		self.yaml = self.job_dir + self.name + '.yaml'
		os.system("python " + self.job_dir+"render-template.py " + jinja + " > " + self.yaml)

	def start(self):
		'''start the task in k8s'''
		self.logger.info("starting task " + self.name + "...")
		self._mount_volume()
		self._create()
		subprocess.check_output("kubectl create -f " + self.yaml, shell=True) # start pods in k8s

	def delete(self, all):
		'''delete the task'''
		pool = Threads()
		thread = threading.Thread(target=(os.system('kubectl delete -f ' + self.yaml)), args=())
		pool.add(thread)

		if all:
			(work_dir, host_work_dir, name) = self.work_volume
			cmd = 'timeout 10 ssh ' + self.node + ' "sudo rm -r ' + host_work_dir + '"'
			thread = threading.Thread(target=(lambda cmd=cmd: os.system(cmd)), args=())
			pool.add(thread)

		pool.start()
		pool.wait()

	def scale_in(self):
		self.logger.info("scale in task " + self.name + "...")
		(work_dir, host_work_dir, name) = self.work_volume
		local_file = host_work_dir + "SCALING.txt"
		if self.role == "ps":
			cmd = "ssh " + self.node + " 'echo DEC_SERVER > " + local_file + "'"
		else:
			cmd = "ssh " + self.node + " 'echo DEC_WORKER > " + local_file + "'"
		try:
			subprocess.check_output(cmd, shell=True)
		except Exception as e:
			self.logger.error(self.name + str(e))
		# wait until finished
		cmd = "ssh " + self.node + " 'cat " + local_file + "'"
		output = subprocess.check_output(cmd, shell=True)
		# the other side is opening and writing the file, try again
		while (output != "FINISH"):
			output = subprocess.check_output(cmd, shell=True)
			self.logger.info(self.name + " " + output)
			time.sleep(1)
			if output is not None:
				output.replace('\n','')

		self.delete(True)

	def scale_out(self):
		self.logger.info("scale out task " + self.name + "...")
		self._mount_volume()
		(work_dir, host_work_dir, name) = self.work_volume
		local_file = host_work_dir + "SCALING.txt"
		if self.role == "ps":
			cmd = "ssh " + self.node + " 'echo INC_SERVER > " + local_file + "'"
		else:
			cmd = "ssh " + self.node + " 'echo INC_WORKER > " + local_file + "'"
		subprocess.check_output(cmd, shell=True)
		self.start()

	def get_progress(self):
		'''get the task progress from each worker'''
		if self.role == "ps":
			return
		(work_dir, host_work_dir, name) = self.work_volume
		local_file = host_work_dir + 'progress.txt'
		cmd = "ssh " + self.node + " 'cat " + local_file + "'"
		output = subprocess.check_output(cmd, shell=True)
		# the other side is opening and writing the file, try again
		counter = 0
		while (output == '' or output == None):
			output = subprocess.check_output(cmd, shell=True)
			time.sleep(0.001 * (10 ** counter))
			counter = counter + 1
			if counter > 2:
				break
		if output is not None and output != '':  # should not be empty, even no progress, there should be initialization values written in files.
			stat_dict = ast.literal_eval(output.replace('\n', ''))
			if "progress" in stat_dict:
				progress = stat_dict["progress"] # (epoch, batch)
				return progress
			else:
				self.logger.error("Task:: " + "progress output does not have progress or val-loss value")
		else:
			self.logger.info("Task:: " + "the progress output is empty.")
		return (0,0)

	def get_speed(self):
		'''get the worker training speed'''
		if self.role == "ps":
			return
		(work_dir, host_work_dir, name) = self.work_volume
		local_file = host_work_dir + 'speed.txt'
		cmd = "ssh " + self.node + " 'cat " + local_file + "'"
		output = subprocess.check_output(cmd, shell=True)
		# the other side is opening and writing the file, try again
		try:
			counter = 0
			while(output == '' or output == None):
				output = subprocess.check_output(cmd, shell=True)
				time.sleep(0.001*(10**counter))
				counter = counter + 1
				if counter > 2:
					self.logger.error(self.name + " read training speed timeout.")
					break
			stb_speed = float('%.3f'%(float(output.replace('\n', '').split(' ')[1])))
			return stb_speed
		except:
			self.logger.error("failed to read speed.txt for task " + self.name)
		return 0

	def _get_pod(self):
		'''
		get the names of the pods belonging to the task
		NAME                                    READY     STATUS    RESTARTS   AGE
		1-measurement-imagenet-ps-0-mzv2z       1/1       Running   0          1m
		'''
		cmd = 'kubectl get pods --selector=' + 'task_name=' + self.name + ' --namespace=default | grep ' + self.name
		output = subprocess.check_output(cmd, shell=True)
		try:
			pod = output.replace('\n','').split(' ')[0]
			return pod
		except Exception as e:
			self.logger.error("Get pod name error! " + str(e))
		
	def get_metrics(self):
		'''get the metrics of the pod'''
		# heapster               192.168.192.16    <none>        80/TCP              5d
		cmd = "kubectl get services --namespace=kube-system | grep heapster |awk '{print $2}'"
		heapster_cluster_ip = subprocess.check_output(cmd, shell=True).replace('\n','')
		if heapster_cluster_ip == '':
			heapster_cluster_ip = '192.168.192.16'
		
		'''
		{
		  "metrics": [
		   {
		    "timestamp": "2017-08-14T08:10:00Z",
		    "value": 0
		   }
		  ],
		  "latestTimestamp": "2017-08-14T08:10:00Z"
		 }
		'''
		pod = self._get_pod()
		self.logger.debug("The pod name of " + self.name + ": " + str(pod))
		keys = ['cpu/usage_rate', 'memory/usage', 'network/tx_rate', 'network/rx_rate']	# cpu: milli core, mem: bytes, net: bytes/second
		metrics = {}
		for key in keys:
			url = 'http://' + heapster_cluster_ip + '/api/v1/model/namespaces/default/pods/' + pod + '/metrics/' + key
			try:
				output = requests.get(url, verify=False).json()
				value = int(output['metrics'][-1]['value'])	# get latest value, maybe empty since heapster update metrics per minute
			except:
				value = 0
			metrics[key] = value
		self.logger.debug(self.name + ": " + str(metrics))
		return metrics

