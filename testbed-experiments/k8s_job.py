import time
import datetime
import os
import sys
import threading
import subprocess
from multiprocessing.pool import ThreadPool
from threads import Threads
from k8s_task import Task

class K8SJob(object):
	'''job description.
	Parameters
	----------
	id: int
	num_ps: int
	num_workers: int
	other parameters: string or list of strings
	work to be done on Tuesday 8/8/2017: 
		(1) modify template file, worker and server mount different dirs
		(2) modify template file, set work_dir and export it as an env
		(3) add support for gpu and get_progress() if necessary
	'''
	def __init__(self, id, type, model_name, workload_id, dir_prefix, logger):
		'''initialize a job
		job type: eg., measurement-imagenet, i.e., category-dataset
		'''
		self.id = id
		self.type = type
		self.model_name = model_name
		self.workload_id = workload_id
		self.name = str(id) + '-' + type + '-' + model_name
		
		now = time.time()
		self.timestamp = str(datetime.datetime.fromtimestamp(now).strftime('%Y-%m-%d-%H:%M:%S'))
		self.dir = dir_prefix + self.name + '-' + self.timestamp + '/'

		self.logger = logger
		
		self.num_ps = None
		self.ps_cpu = None
		self.ps_mem = None
		self.ps_bw = None
		
		self.num_workers = None
		self.worker_cpu = None
		self.worker_mem = None
		self.ps_bw = None
		
		self.ps_placement = []  #current placement
		self.worker_placement = []

		self.speed_list = []
		self.progress_list = None # [(epoch, batch)]
		self.ps_metrics = []
		self.worker_metrics = []
		self.ps_pods = []
		self.worker_pods = []

		# for experiment
		self.arrival_slot = None
		self.arrival_time = None
		self.start_slot = None
		self.start_time = None
		self.end_slot = None
		self.end_time = None
		self.status = 'initialized'
		self.progress = 0
		self.training_speeds = dict()  #(num_ps, num_workers): speed
		self.val_losses = dict() # epoch : validation_loss
		self.num_epochs = 0
		self.epoch_size = 0

		self.ps_tasks = dict()
		self.worker_tasks = dict() # id:obj
		self.ps_task_id = 0
		self.worker_task_id = 0
		self.scale_ps_placement = None # the state to be achieved
		self.scale_worker_placement = None

		self.running_time = 0.0
		self.tic = None
		self.training = True
		self.dom_share = 0

	def info(self):
		return "Job id: " + str(self.id) + " type: " + str(self.type) + " arrival_slot: " + str(self.arrival_slot) \
						 + " progress: " + str(self.progress) + " total epochs: " + str(self.num_epochs)

	def set_ps_resources(self, num_ps, ps_cpu, ps_mem, ps_bw=0, ps_gpu=0):
		'''resource requirements of parameter servers'''
		self.num_ps = num_ps
		self.ps_cpu = ps_cpu
		self.ps_mem = ps_mem
		self.ps_bw = ps_bw
		self.ps_gpu = ps_gpu
	
	def set_worker_resources(self, num_workers, worker_cpu, worker_mem, worker_bw=0, worker_gpu=0):
		'''resource requirements of workers'''
		self.num_workers = num_workers
		self.worker_cpu = worker_cpu
		self.worker_mem = worker_mem
		self.worker_bw = worker_bw
		self.worker_gpu = worker_gpu
	
	def set_ps_placement(self, ps_placement):
		'''the placement of parameter servers'''
		if isinstance(ps_placement, list):
			if len(ps_placement) == self.num_ps:
				self.ps_placement = ps_placement
			else:
				raise RuntimeError('ps_placement length' + str(len(ps_placement)) + ' is not consistent with num_ps ' + str(len(self.num_ps)))
		else:
			raise TypeError('ps_placement is not a list')

	def set_worker_placement(self, worker_placement):
		'''the placement of workers'''
		if isinstance(worker_placement, list):
			if len(worker_placement) == self.num_workers:
				self.worker_placement = worker_placement
			else:
				raise RuntimeError('worker_placement length' + str(len(worker_placement)) + ' is not consistent with num_workers ' + str(len(self.num_workers)))
		else:
			raise TypeError('worker_placement is not a list')

	def set_container(self, image, script, work_dir, host_workdir_prefix, work_volume='k8s-mxnet-work-volume'):
		'''container description'''
		# self.image = image
		self.image = "yhpeng/k8s-mxnet-gpu-experiment-scaling"
		self.script = script
		self.work_dir = work_dir
		self.host_workdir_prefix = host_workdir_prefix
		self.work_volume = work_volume
		
	def set_data(self, hdfs_data, data_dir, host_data_dir, data_mounted=True, data_volume='k8s-mxnet-data-volume'):
		'''data specification, if data not in local host, fetch from HDFS'''
		self.hdfs_data = hdfs_data # dataset list including training data and validation data
		self.data_dir = data_dir
		self.host_data_dir = host_data_dir
		self.data_mounted = data_mounted
		self.data_volume = data_volume
	
	def set_train(self, prog, batch_size, kv_store, scale_bs=False, num_examples=0, num_epochs=100):
		self.prog = prog
		self.tot_batch_size = batch_size
		self.kv_store = kv_store
		self.scale_bs = scale_bs
		self.num_examples = num_examples
		self.num_epochs = num_epochs # for unknown num_epochs, will update it in progressor with estimation
		self.epoch_size = num_examples/batch_size

	def set_mxnet(self, kv_store_big_array_bound=1000000, ps_verbose=0):
		'''set env MXNET_KVSTORE_BIGARRAY_BOUND'''
		self.kv_store_big_array_bound = kv_store_big_array_bound
		self.ps_verbose = ps_verbose

	def get_progress(self):
		progresses = dict()
		def run(task, progresses):
			progresses[task.id] = task.get_progress()
		pool = ThreadPool(len(self.worker_tasks))
		for (id, task) in self.worker_tasks.items():
			pool.apply_async(run, (task, progresses,))
		pool.close()
		pool.join()
		return progresses.values()

	def get_speed(self):
		speeds = dict()
		def run(task, speeds):
			speeds[task.id] = task.get_speed()
		pool = ThreadPool(len(self.worker_tasks))
		for (id, task) in self.worker_tasks.items():
			pool.apply_async(run, (task,speeds,))
		pool.close()
		pool.join()
		self.logger.debug(speeds.values())
		return speeds.values()

	def get_metrics(self):
		ps_metrics = dict()
		worker_metrics = dict()

		def run(task, metrics):
			metrics[task.id] = task.get_metrics()
		try:
			pool = ThreadPool(len(self.ps_tasks) + len(self.worker_tasks))
			for (id, task) in self.ps_tasks.items():
				#pool.apply_async(run, (task, ps_metrics,))
				run(task, ps_metrics)
			for (id, task) in self.worker_tasks.items():
				# pool.apply_async(run, (task, worker_metrics,))
				run(task, worker_metrics)
		except Exception as e:
			self.logger.error(self.name + ": " + str(e))
		pool.close()
		pool.join()
		self.logger.debug(self.name + ": " + str(ps_metrics)+str(worker_metrics))
		return (ps_metrics.values(), worker_metrics.values())

	def _create_task(self, role):
		if role == "ps":
			job_meta = (self.name, self.dir, self.task_template, self.num_ps, self.num_workers)
			self.ps_task_id += 1
			task = Task(job_meta, "ps", self.ps_task_id, "Initializing", self.logger)
			task.set_resource_spec(self.ps_cpu, self.ps_mem, self.ps_bw, self.ps_gpu)
			work_vomule = (self.work_dir, self.host_workdir_prefix, self.work_volume)
			data_volume = (self.data_dir, self.host_data_dir, self.data_volume)
			task.set_container(self.image, self.script, work_vomule, data_volume)
			task.set_data_source(self.hdfs_data, True)
			task.set_train(self.prog, self.tot_batch_size, self.kv_store, self.num_examples, self.num_epochs)
			task.set_mxnet_env(self.kv_store_big_array_bound, self.ps_verbose)
			self.ps_tasks[self.ps_task_id] = task
		else:
			job_meta = (self.name, self.dir, self.task_template, self.num_ps, self.num_workers)
			self.worker_task_id += 1
			task = Task(job_meta, "worker", self.worker_task_id, "Initializing", self.logger)
			task.set_resource_spec(self.worker_cpu, self.worker_mem, self.worker_bw, self.worker_gpu)
			work_vomule = (self.work_dir, self.host_workdir_prefix, self.work_volume)
			data_volume = (self.data_dir, self.host_data_dir, self.data_volume)
			task.set_container(self.image, self.script, work_vomule, data_volume)
			task.set_data_source(self.hdfs_data, True)
			task.set_train(self.prog, self.tot_batch_size, self.kv_store, self.num_examples, self.num_epochs)
			task.set_mxnet_env(self.kv_store_big_array_bound, self.ps_verbose)
			self.worker_tasks[self.worker_task_id] = task
		return task

	def start(self):
		'''start the job in k8s'''
		if self.num_workers == 0 and self.num_ps == 0:
			return

		self.logger.info("starting job " + self.name + "...")
		os.system('mkdir -p ' + self.dir)  # job working dir
		self.task_template = self.dir + "k8s-mxnet-task-template.jinja"
		os.system("cp ../templates/k8s-mxnet-task-template.jinja " + self.task_template)
		os.system("cp ../templates/render-template.py " + self.dir)
		for i in range(self.num_ps):
			task = self._create_task("ps")
			task.set_placement(self.ps_placement[i])

		for i in range(self.num_workers):
			task = self._create_task("worker")
			task.set_placement(self.worker_placement[i])

		def run(task):
			task.start()
		pool = ThreadPool(len(self.ps_tasks)+ len(self.worker_tasks))
		for (id, ps_task) in self.ps_tasks.items():
			pool.apply_async(run, (ps_task,))
		for (id, worker_task) in self.worker_tasks.items():
			pool.apply_async(run, (worker_task,))
		pool.close()
		pool.join()
		self.tic = time.time()

	def delete(self, del_all=True):
		'''delete the job.
		Parameters
		----------
		del_all: whether to delete all, including histories.
		'''
		toc = time.time()
		if self.tic > 0:
			self.running_time += (toc - self.tic)
		if self.progress >= self.num_epochs:
			os.system("mkdir -p job_time/")
			f = open('job_time/job_'+str(self.id), 'w')
			f.write(self.name + ": " + str(self.running_time))
			f.close()
		def run(task, del_all):
			task.delete(del_all)
		num = len(self.ps_tasks)+ len(self.worker_tasks)
		if num <= 0:
			return
		pool = ThreadPool(num)
		for (id, ps_task) in self.ps_tasks.items():
			pool.apply_async(run, (ps_task, del_all,))
		for (id, worker_task) in self.worker_tasks.items():
			pool.apply_async(run, (worker_task, del_all,))
		self.worker_tasks = dict()
		self.ps_tasks = dict()
		pool.close()
		pool.join()

		# delete job working dir
		if del_all:
			subprocess.check_output("rm -rf " + self.dir, shell=True)

	def scale_in(self):
		'''dynamically scale the job while keeping job running in k8s, to be implemented'''
		self.logger.info("scale job " + self.name + "...")
		if self.scale_ps_placement is None or self.scale_worker_placement is None:
			raise RuntimeError("Job:: " + "did not set scale_ps_placement or scale_worker_placement")
		assert len(self.scale_ps_placement)==self.num_ps
		assert len(self.scale_worker_placement)==self.num_workers

		self.logger.info("previous placement: " + str(self.ps_placement) + str(self.worker_placement))
		self.logger.info("next placement: " + str(self.scale_ps_placement) + str(self.scale_worker_placement))

		ps_intersect = list(set(self.ps_placement) & set(self.scale_ps_placement))
		worker_intersect = list(set(self.worker_placement) & set(self.scale_worker_placement))
		if len(ps_intersect) == 0 or len(worker_intersect) == 0:
			# delete all tasks and start all tasks
			self.logger.info("Restart all tasks...")
			self.delete(True)
			self.ps_tasks = dict()
			self.worker_tasks = dict()
		else:
			self.logger.info("Scaling in ...")
			# first scale in
			scale_ps_placement_cp = list(self.scale_ps_placement)
			for (id, task) in self.ps_tasks.copy().items():
				if task.node not in scale_ps_placement_cp:
					self.logger.info("scale in ps task node " + task.node)
					task.scale_in() # one by one
					del self.ps_tasks[id]
				else:
					scale_ps_placement_cp.remove(task.node)

			scale_worker_placement_cp = list(self.scale_worker_placement)
			for (id, task) in self.worker_tasks.copy().items():
				if task.node not in scale_worker_placement_cp:
					self.logger.info("scale in worker task node " + task.node)
					task.scale_in() # one by one
					del self.worker_tasks[id]
				else:
					scale_worker_placement_cp.remove(task.node)

		self.ps_placement = self.scale_ps_placement
		self.worker_placement = self.scale_worker_placement
		self.scale_ps_placement = None
		self.scale_worker_placement = None

	def scale_out(self):
		if len(self.ps_tasks) == 0 and len(self.worker_tasks) == 0:
			self.start()
		else:
			self.logger.info("Scaling out ...")
			# scale out
			ps_placement_cp = list(self.ps_placement)
			for (id, task) in self.ps_tasks.copy().items():
				if task.node not in ps_placement_cp:
					# create a new task
					task = self._create_task("ps")
					task.scale_out()
				else:
					ps_placement_cp.remove(task.node)

			worker_placement_cp = list(self.worker_placement)
			for (id, task) in self.worker_tasks.copy().items():
				if task.node not in worker_placement_cp:
					# create a new task
					task = self._create_task("worker")
					task.scale_out()
				else:
					worker_placement_cp.remove(task.node)



		
		
		
		
		
		
		
		