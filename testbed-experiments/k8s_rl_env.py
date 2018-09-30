import Queue
import time
import numpy as np
import parameters as pm
from cluster import Cluster
import log
import os
from scheduler_base import Scheduler
import jobrepo
from k8s_job import K8SJob


class K8S_RL_Env(Scheduler):
	def __init__(self, name, trace, logger, training_mode=True):
		Scheduler.__init__(self, name, trace, logger)

		self.cwd = os.getcwd() + '/'
		self.job_id = 0

		self.epsilon = 0.0
		self.training_mode = training_mode
		self.sched_seq = []
		self.job_prog_in_ts = dict()
		self.window_jobs = None
		self.jobstats = dict()
		for stats_name in ["arrival", "ts_completed", "tot_completed", "duration", "uncompleted", "running", "total", "backlog", "cpu_util", "gpu_util"]:
			self.jobstats[stats_name] = []
		if pm.PS_WORKER and pm.BUNDLE_ACTION:
			self.action_freq = [0 for _ in range(3)]
		self._init_k8s()
		# prepare for the first timeslot
		self._prepare()
		self.tic = time.time()

	def _init_k8s(self):
		self.logger.info("clear all existing jobs...")
		os.system("kubectl delete jobs --all")

	def _prepare(self):
		# admit new jobs
		self.logger.info("Timeslot " + str(self.curr_ts) + " ...")
		self.logger.info("Time has passed: " + "%.3f"%(time.time()-self.tic))
		num_arrv_jobs = 0
		k8s_jobs = []
		if self.curr_ts in self.trace:
			for job in self.trace[self.curr_ts]:
				job.reset()
				model = job.model
				for i in range(len(jobrepo.job_repos)):
					_type, _model = jobrepo.job_repos[i]
					if model == _model:
						self.job_id += 1
						k8s_job = K8SJob(self.job_id, _type, _model, i, self.cwd, self.logger)
						jobrepo.set_config(k8s_job)
						k8s_job.num_epochs = job.num_epochs  # here specify total number of trained epochs
						k8s_job.arrival_time = time.time()
						k8s_job.arrival_slot = self.curr_ts
						k8s_job.num_worker = 0
						k8s_job.num_ps = 0
						k8s_job.worker_placement = []
						k8s_job.ps_placement = []
						cpu, gpu = job.resr_worker  # do not care memory and bandwidth
						k8s_job.worker_cpu = cpu
						k8s_job.worker_gpu = gpu/4
						cpu, gpu = job.resr_ps
						k8s_job.ps_cpu = cpu
						k8s_job.ps_gpu = gpu/4
						k8s_jobs.append(k8s_job)
						self.uncompleted_jobs.add(k8s_job) # map job to k8s job
						if not self.training_mode:
							k8s_job.training = False
						num_arrv_jobs += 1
						self.logger.debug(job.info())
						break
			assert len(k8s_jobs) == len(self.trace[self.curr_ts])
		self.logger.info("# of job arrival: " + str(num_arrv_jobs))
		self.jobstats["arrival"].append(num_arrv_jobs)
		self.jobstats["total"].append(len(self.completed_jobs)+len(self.uncompleted_jobs))
		self.jobstats["backlog"].append(max(len(self.uncompleted_jobs)-pm.SCHED_WINDOW_SIZE,0))

		# reset
		self._sched_states() # get scheduling states in this ts

		# killing all running jobs
		for k8s_job in self.running_jobs:
			k8s_job.delete(True) # True or False?
		self.running_jobs.clear()
		# load balancing placement
		self.node_used_resr_queue = Queue.PriorityQueue()
		for i in range(pm.CLUSTER_NUM_NODES):
			self.node_used_resr_queue.put((0, i))
		self.cluster.clear()

		for k8s_job in self.uncompleted_jobs:
			if pm.ASSIGN_BUNDLE and pm.PS_WORKER:  # assign each job a bundle of ps and worker first to avoid job starvation
				_, node = self.node_used_resr_queue.get()
				resr_worker = np.array([k8s_job.worker_cpu, k8s_job.worker_gpu*4])
				resr_ps = np.array([k8s_job.ps_cpu, k8s_job.ps_gpu*4])
				resr_reqs = resr_worker + resr_ps
				succ, node_used_resrs = self.cluster.alloc(resr_reqs, node)
				if succ:
					k8s_job.num_ps = 1
					k8s_job.ps_placement= [pm.CLUSTER_NODES[node]]
					k8s_job.num_workers = 1
					k8s_job.worker_placement= [pm.CLUSTER_NODES[node]]
					k8s_job.dom_share = np.max(1.0 * (resr_worker + resr_ps) / self.cluster.CLUSTER_RESR_CAPS)
					self.running_jobs.add(k8s_job)
				else:
					k8s_job.num_ps = 0
					k8s_job.ps_placement = []
					k8s_job.num_workers = 0
					k8s_job.worker_placement = []
					k8s_job.dom_share = 0
				self.node_used_resr_queue.put((np.sum(node_used_resrs), node))  # always put back to avoid blocking in step()
			else:
				k8s_job.num_workers = 0
				k8s_job.worker_placement = []
				if pm.PS_WORKER:
					k8s_job.num_ps = 0
					k8s_job.ps_placement = []
				k8s_job.dom_share = 0

		if pm.VARYING_SKIP_NUM_WORKERS:
			self.skip_num_workers = np.random.randint(1, pm.MAX_NUM_WORKERS)
		else:
			self.skip_num_workers = 8 #np.random.randint(0,pm.MAX_NUM_WORKERS)
		if pm.VARYING_PS_WORKER_RATIO:
			self.ps_worker_ratio = np.random.randint(3,6)
		else:
			self.ps_worker_ratio = 5

	def _move(self):
		self._progress()
		# next timeslot
		self.curr_ts += 1
		if self.curr_ts == pm.MAX_TS_LEN:
			self.end = True
			self.logger.info("The end timeslot!")
			self.logger.info("Results: " + str(self.get_results()))
			self.logger.info("Stats: " + str(self.get_jobstats()))
			for k8s_job in self.uncompleted_jobs:
				self.logger.info("Uncompleted job "+ str(k8s_job.name) + " tot_epoch: "+str(k8s_job.num_epochs) + " prog: " + str(k8s_job.progress) + " workers: " + str(k8s_job.num_workers))
			exit()
		self._prepare()

	# step forward by one action
	def step(self, output):
		# mask and adjust probability
		mask = np.ones(pm.ACTION_DIM)
		for i in range(len(self.window_jobs)):
			if self.window_jobs[i] is None: # what if job workers are already maximum
				if pm.PS_WORKER:
					if pm.BUNDLE_ACTION:  # worker, ps, bundle
						mask[3 * i] = 0.0
						mask[3 * i + 1] = 0.0
						mask[3 * i + 2] = 0.0
					else:
						mask[2*i] = 0.0
						mask[2 * i + 1] = 0.0
				else:
					mask[i] = 0.0
			else:
				if pm.PS_WORKER:
					worker_full = False
					ps_full = False
					if self.window_jobs[i].num_workers >= pm.MAX_NUM_WORKERS:
						worker_full = True
					if self.window_jobs[i].num_ps >= pm.MAX_NUM_WORKERS:
						ps_full = True
					if worker_full:
						if pm.BUNDLE_ACTION:
							mask[3 * i] = 0.0
						else:
							mask[2*i] = 0.0
					if ps_full:
						if pm.BUNDLE_ACTION:
							mask[3*i+1] = 0.0
						else:
							mask[2 * i + 1] = 0.0
					if (worker_full or ps_full) and pm.BUNDLE_ACTION:
						mask[3*i+2] = 0.0

		masked_output = np.reshape(output[0]*mask, (1,len(mask)))
		sum_prob = np.sum(masked_output)
		action_vec = np.zeros(len(mask))
		move_on = True
		valid_state = False
		if ((not pm.PS_WORKER) and sum(mask[:len(self.window_jobs)]) == 0) \
				or (pm.PS_WORKER and (not pm.BUNDLE_ACTION) and sum(mask[:2*len(self.window_jobs)]) == 0) \
				or (pm.PS_WORKER and pm.BUNDLE_ACTION and sum(mask[:3*len(self.window_jobs)]) == 0):
			self.logger.debug("All jobs are None, move on and do not save it as a sample")
			self._move()
		elif sum_prob <= 0:
			self.logger.info("All actions are masked or some action with probability 1 is masked!!!")
			if pm.EXPERIMENT_NAME is None:
				self.logger.info("Output: " + str(output)) # Output: [[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. 0.  1.  0.]], WHY?
				self.logger.info("Mask: " + str(mask))
				self.logger.info("Window_jobs: " + str(self.window_jobs))
				num_worker_ps_str = ""
				for job in self.window_jobs:
					if job:
						num_worker_ps_str += str(job.name) + ": " + str(job.num_ps) + " " + str(job.num_workers) + ","
				self.logger.info("Job: " + num_worker_ps_str)
			self._move()
		else:
			masked_output = masked_output/sum_prob
			if self.training_mode:
				# select action
				if np.random.rand() > pm.MASK_PROB: # only valid for training mode
					masked_output = np.reshape(output[0], (1,len(mask)))
				action_cumsum = np.cumsum(masked_output)
				action = (action_cumsum > np.random.randint(1, pm.RAND_RANGE) / float(pm.RAND_RANGE)).argmax()

				if pm.EPSILON_GREEDY:
					if np.random.rand() < self.epsilon:
						val_actions = []
						for i in range(len(masked_output[0])):
							if masked_output[0][i] > pm.MIN_ACTION_PROB_FOR_SKIP:
								val_actions.append(i)
						action = val_actions[np.random.randint(0, len(val_actions))]

				if pm.INJECT_SAMPLES:
					if (not pm.REAL_SPEED_TRACE) and (not pm.PS_WORKER):
						allMaxResr = True
						for job in self.window_jobs:
							if job:
								if job.num_workers > self.skip_num_workers:
									continue
								else:
									allMaxResr = False
									break
						if allMaxResr and masked_output[0][len(action_vec)-1] > pm.MIN_ACTION_PROB_FOR_SKIP and np.random.rand() <= pm.SAMPLE_INJECTION_PROB:  # choose to skip if prob larger than a small num, else NaN
							action = len(action_vec) - 1
							self.logger.debug("Got 1.")
					elif pm.REAL_SPEED_TRACE and pm.PS_WORKER:
						# shuffle = np.random.choice(len(self.window_jobs), len(self.window_jobs), replace=False)  # shuffle is a must, otherwise NN selects only the first several actions!!!
						if pm.JOB_RESR_BALANCE and pm.BUNDLE_ACTION:
							max_num_ps_worker = 0
							min_num_ps_worker = 10**10
							index_min_job = -1
							for i in range(len(self.window_jobs)):
								job = self.window_jobs[i]
								if job:
									num_ps_worker = job.num_ps + job.num_workers
									if num_ps_worker > max_num_ps_worker:
										max_num_ps_worker = num_ps_worker
									if num_ps_worker < min_num_ps_worker:
										min_num_ps_worker = num_ps_worker
										index_min_job = i
							if max_num_ps_worker and min_num_ps_worker and index_min_job != -1 and max_num_ps_worker/min_num_ps_worker > np.random.randint(3,6):
								if masked_output[0][3*index_min_job+2] > pm.MIN_ACTION_PROB_FOR_SKIP and masked_output[0][3*index_min_job] > pm.MIN_ACTION_PROB_FOR_SKIP:
									if np.random.rand() < 0.5:
										action = 3*index_min_job+2
									else:
										action = 3*index_min_job

						shuffle = [_ for _ in range(len(self.window_jobs))]
						for i in shuffle:
							job = self.window_jobs[i]
							if job:
								if pm.BUNDLE_ACTION:
									# if one of three actions: ps/worker/bundle has low probability, enforce to select it
									if min(self.action_freq) > 0 and min(self.action_freq)*1.0/sum(self.action_freq) < 0.02:
										index = np.argmin(self.action_freq)
										if mask[3*i+index] > 0 and masked_output[0][3*i+index] > pm.MIN_ACTION_PROB_FOR_SKIP:
											action = 3 * i + index
											self.logger.debug("Got 0: " + str(index))
											break
									if (job.num_workers == 0 or job.num_ps == 0) and mask[3*i+2] > 0 and masked_output[0][3*i+2] > pm.MIN_ACTION_PROB_FOR_SKIP:
										action = 3*i+2
										self.logger.debug("Got 1")
										break
									elif job.num_ps >= job.num_workers and mask[3*i] > 0 and masked_output[0][3*i] > pm.MIN_ACTION_PROB_FOR_SKIP and np.random.rand() < 0.5:
										# increase this job's worker
										action = 3*i
										self.logger.debug("Got 2.")
										break
									elif job.num_workers >= job.num_ps*self.ps_worker_ratio and np.random.rand() < 0.5:
										if mask[3*i+2] > 0 and masked_output[0][3*i+2] > pm.MIN_ACTION_PROB_FOR_SKIP and mask[3*i+1] > 0 and masked_output[0][3*i+1] > pm.MIN_ACTION_PROB_FOR_SKIP:
											if np.random.rand() < 0.5:
												# increase this job's bundle
												action = 3*i+2
												self.logger.debug("Got 3.")
											else:
												# incrase ps
												action = 3*i+1
												self.logger.debug("Got 4.")
										break
								else:
									if job.num_workers == 0 and mask[2*i] > 0 and masked_output[0][2*i] > pm.MIN_ACTION_PROB_FOR_SKIP and np.random.rand() < 0.01:
										action = 2 * i
										self.logger.debug("Got 1.")
										break
									elif job.num_ps == 0 and mask[2*i+1] > 0 and masked_output[0][2*i+1] > pm.MIN_ACTION_PROB_FOR_SKIP and np.random.rand() < 0.01:
										action = 2 * i + 1
										self.logger.debug("Got 2.")
										break
									elif job.num_ps >= job.num_workers*self.ps_worker_ratio and mask[2*i] > 0 and masked_output[0][2*i] > pm.MIN_ACTION_PROB_FOR_SKIP and np.random.rand() < 0.5:
										# increase this job's worker
										action = 2*i
										self.logger.debug("Got 3.")
										break
									elif job.num_workers >= job.num_ps*self.ps_worker_ratio and mask[2*i+1] > 0 and masked_output[0][2*i+1] > pm.MIN_ACTION_PROB_FOR_SKIP and np.random.rand() < 0.5:
										# increase this job's ps
										action = 2*i+1
										self.logger.debug("Got 4.")
										break
			else:
				if pm.SELECT_ACTION_MAX_PROB: # only available for validation
					action = np.argmax(masked_output)  # output is [[...]] # always select the action with max probability
				else:
					action_cumsum = np.cumsum(masked_output)
					action = (action_cumsum > np.random.randint(1, pm.RAND_RANGE) / float(pm.RAND_RANGE)).argmax()

			action_vec[action] = 1
			# check whether skip this timeslot
			if pm.SKIP_TS and action == len(action_vec) - 1:
				self._move()
				# filter out the first action that causes 0 reward??? NO
				# if sum([job.num_workers+job.num_ps for job in self.uncompleted_jobs]) > 0:
				valid_state = True
				self.sched_seq.append(None)
				self.logger.debug("Skip action is selected!")
				self.logger.debug("Output: " + str(output))
				self.logger.debug("Masked output: " + str(masked_output))
			else:
				# count action freq
				if pm.PS_WORKER and pm.BUNDLE_ACTION:
					self.action_freq[action % 3] += 1

				# allocate resource
				if pm.PS_WORKER:
					if pm.BUNDLE_ACTION:
						job = self.window_jobs[action/3]
					else:
						job = self.window_jobs[action/2]
				else:
					job = self.window_jobs[action]
				if job is None:
					self._move()
					self.logger.debug("The selected action is None!")
				else:
					_, node = self.node_used_resr_queue.get()
					# get resource requirement of the selected action
					resr_worker = np.array([job.worker_cpu, job.worker_gpu*4])
					resr_ps = np.array([job.ps_cpu, job.ps_gpu*4])
					if pm.PS_WORKER:
						if pm.BUNDLE_ACTION:
							if action%3 == 0:
								resr_reqs = resr_worker
							elif action%3 == 1:
								resr_reqs = resr_ps
							else:
								resr_reqs = resr_worker + resr_ps
						else:
							if action%2 == 0:  # worker
								resr_reqs = resr_worker
							else:
								resr_reqs = resr_ps
					else:
						resr_reqs = resr_worker
					succ, node_used_resrs = self.cluster.alloc(resr_reqs, node)
					if succ:
						move_on = False
						# change job tasks and placement
						if pm.PS_WORKER:
							if pm.BUNDLE_ACTION:
								if action%3 == 0:  # worker
									job.num_workers += 1
									job.worker_placement.append(pm.CLUSTER_NODES[node])
								elif action%3 == 1: # ps
									job.num_ps += 1
									job.ps_placement.append(pm.CLUSTER_NODES[node])
								else:  # bundle
									job.num_ps += 1
									job.ps_placement.append(pm.CLUSTER_NODES[node])
									job.num_workers += 1
									job.worker_placement.append(pm.CLUSTER_NODES[node])
							else:
								if action %2 == 0:  # worker
									job.num_workers += 1
									job.worker_placement.append(pm.CLUSTER_NODES[node])
								else: # ps
									job.num_ps += 1
									job.ps_placement.append(pm.CLUSTER_NODES[node])
						else:
							job.num_workers += 1
							job.worker_placement.append(pm.CLUSTER_NODES[node])

						job.dom_share = np.max(1.0 * (job.num_workers * resr_worker + job.num_ps * resr_ps) / self.cluster.CLUSTER_RESR_CAPS)
						self.node_used_resr_queue.put((np.sum(node_used_resrs), node))
						self.running_jobs.add(job)
						valid_state = True
						self.sched_seq.append(job)
					else:
						self._move()
						self.logger.debug("No enough resources!")
		if move_on:
			reward = self.rewards[-1] * move_on
		else:
			reward = 0
		return masked_output, action_vec, reward, move_on, valid_state    # invalid state, action and output when move on except for skip ts


	def get_jobstats(self):
		self.jobstats["duration"] = [(job.end_time - job.arrival_time) for job in self.completed_jobs]
		for name, value in self.jobstats.items():
			self.logger.debug(name + ": length " + str(len(value)) + " " + str(value))
		return self.jobstats

	def _sched_states(self):
		self.states = []
		for job in self.running_jobs:
			self.states.append((job.id, job.workload_id, job.num_workers, job.num_ps))

	def get_job_reward(self):
		job_reward = []
		for job in self.sched_seq:
			if job is None: # skip
				if len(self.job_prog_in_ts) > 0:
					job_reward.append(self.rewards[-1]/len(self.job_prog_in_ts))
				else:
					job_reward.append(0)
			else:
				job_reward.append(self.job_prog_in_ts[job])
		self.sched_seq = []
		self.job_prog_in_ts.clear()

		self.logger.info("Action Frequency: " + str(self.action_freq))
		return job_reward


	def get_sched_states(self):
		return self.states


	def _progress(self):
		# start all jobs with allocated ps/worker
		self.logger.info("# of jobs to run: " + str(len(self.running_jobs)))
		for job in self.running_jobs:
			job.start()
			if job.progress == 0:
				job.start_time = time.time()
				job.start_slot = self.curr_ts

		rewards = dict()
		num_ts_completed = 0
		counter = 0
		while True:
			# sleep 1 minutes
			time.sleep(60)

			# read progress
			for job in self.running_jobs.copy():
				try:
					speed_list = job.get_speed()
					if sum(speed_list) == 0:
						self.logger.info("speed is 0, continue")
						continue
				except Exception as e:
					self.logger.info("get speed error!" + str(e))
					continue
				try:
					progress_list = job.get_progress()
				except Exception as e:
					self.logger.info("get progress error!" + str(e))
					continue
				try:
					(ps_metrics, worker_metrics) = job.get_metrics()
				except Exception as e:
					self.logger.info("get metrics error!" + str(e))
					continue

				# compute cpu usage
				ps_cpu_usage_list = []
				for metrics in ps_metrics:
					ps_cpu_usage_list.append(metrics['cpu/usage_rate'] / 1000.0)
				worker_cpu_usage_list = []
				for metrics in worker_metrics:
					worker_cpu_usage_list.append(metrics['cpu/usage_rate'] / 1000.0)

				progress = 0
				for epoch, batch in progress_list:
					progress += epoch
					progress += 1.0 * batch / job.epoch_size
				rewards[job.id] = progress/job.num_epochs
				self.job_prog_in_ts[job] = rewards[job.id]
				job.progress += progress

				self.logger.info("job name: " + job.name + " model name: " + job.model_name + ", kv_store: " + job.kv_store + \
				            ", batch_size: " + str(job.tot_batch_size) + \
				            ", num_ps: " + str(job.num_ps) + ", num_worker: " + str(job.num_worker) + \
				            ", progress_list: " + str(progress_list) + \
				            ", total progress: " + str(job.progress) + \
				            ", total num_epochs: " + str(job.num_epochs) + \
				            ", speed_list: " + str(speed_list) + ", sum_speed (samples/second): " + str(sum(speed_list)) + \
				            ", sum_speed(batches/second): " + str(sum(speed_list) / int(job.tot_batch_size)) + \
				            ", ps cpu usage: " + str(ps_cpu_usage_list) + \
				            ", worker cpu usage: " + str(worker_cpu_usage_list)
				            )
				if job.progress >= job.num_epochs:
					self.logger.info("ob " + str(job.name) + " finished!")
					job.end_time = time.time()
					job.end_slot = self.curr_ts
					job.delete(True)
					self.running_jobs.remove(job)
					self.uncompleted_jobs.remove(job)
					self.completed_jobs.add(job)
					num_ts_completed += 1

			counter += 1
			if counter == pm.TS_DURATION/60:
				break
		reward = sum(rewards.values())
		self.rewards.append(reward)

		self.jobstats["running"].append(len(self.running_jobs))
		self.jobstats["tot_completed"].append(len(self.completed_jobs))
		self.jobstats["uncompleted"].append(len(self.uncompleted_jobs))
		self.jobstats["ts_completed"].append(num_ts_completed)
		cpu_util, gpu_util = self.cluster.get_cluster_util()
		self.jobstats["cpu_util"].append(cpu_util)
		self.jobstats["gpu_util"].append(gpu_util)


	def observe(self):
		'''
		existing resource share of each job: 0-1
		job type 0-8
		job normalized progress 0-1
		num of backlogs: percentage of total number of jobs in the trace
		'''
		# cluster_state = self.cluster.get_cluster_state()
		# for test, first use dominant resource share of each job as input state
		q = Queue.PriorityQueue()
		for job in self.uncompleted_jobs:
			if pm.PS_WORKER:
				if job.num_workers >= pm.MAX_NUM_WORKERS and job.num_ps >= pm.MAX_NUM_WORKERS: # and, not or
					continue
			else:
				if job.num_workers >= pm.MAX_NUM_WORKERS:  # not schedule it any more
					continue
			if pm.JOB_SORT_PRIORITY == "Resource":
				q.put((job.dom_share, job.arrival_slot, job))
			elif pm.JOB_SORT_PRIORITY == "Arrival":
				q.put((job.arrival_slot, job.arrival_slot, job))
			elif pm.JOB_SORT_PRIORITY == "Progress":
				q.put((1-job.progress/job.num_epochs, job.arrival_slot, job))

		if pm.ZERO_PADDING:
			state = np.zeros(shape=pm.STATE_DIM)  # zero padding instead of -1
		else:
			state = -1*np.ones(shape=pm.STATE_DIM)
		self.window_jobs = [None for _ in range(pm.SCHED_WINDOW_SIZE)]

		shuffle = np.array([i for i in range(pm.SCHED_WINDOW_SIZE)]) # default keep order
		if pm.JOB_ORDER_SHUFFLE:
			shuffle = np.random.choice(pm.SCHED_WINDOW_SIZE, pm.SCHED_WINDOW_SIZE, replace=False)

		# resource share / job arrival / progress
		for order in shuffle:
			if not q.empty():
				_, _, job = q.get()
				j = 0
				for (input,enable) in pm.INPUTS_GATE: # INPUTS_GATE=[("TYPE",True), ("STAY",False), ("PROGRESS",False), ("DOM_RESR",False), ("WORKERS",True)]
					if enable:
						if input == "TYPE":
							if not pm.INPUT_RESCALE:
								if not pm.TYPE_BINARY:
									state[j][order] = job.workload_id
								else:
									bin_str = "{0:b}".format(job.workload_id).zfill(4)
									for bin_ch in bin_str:
										state[j][order] = int(bin_ch)
										j += 1
									j -= 1
							else:
								state[j][order] = float(job.workload_id)/8
						elif input == "STAY":
							if not pm.INPUT_RESCALE:
								state[j][order] = self.curr_ts - job.arrival_slot
							else:
								state[j][order] = float(self.curr_ts - job.arrival_slot) / 100
						elif input == "PROGRESS":
							state[j][order] = 1 - job.progress/job.num_epochs
						elif input == "DOM_RESR":
							state[j][order] = job.dom_share
						elif input == "WORKERS":
							if not pm.INPUT_RESCALE:
								state[j][order] = job.num_workers
							else:
								state[j][order] = float(job.num_workers)/pm.MAX_NUM_WORKERS
						elif input == "PS":
							if not pm.INPUT_RESCALE:
								state[j][order] = job.num_ps
							else:
								state[j][order] = float(job.num_ps) / pm.MAX_NUM_WORKERS
						else:
							raise RuntimeError
						j += 1
				self.window_jobs[order] = job

		# backlog = float(max(len(self.uncompleted_jobs) - pm.SCHED_WINDOW_SIZE, 0))/len(pm.TOT_NUM_JOBS)
		self.logger.debug("ts: " + str(self.curr_ts) \
						  + " backlog: " + str(max(len(self.uncompleted_jobs) - pm.SCHED_WINDOW_SIZE, 0)) \
						  + " completed jobs: " + str(len(self.completed_jobs)) \
						  + " uncompleted jobs: " + str(len(self.uncompleted_jobs)))
		return state


	# for SL labels
	def _state(self, label_job_id, role="worker"): # whether this action selection leads to worker increment or ps increment
		# cluster_state = self.cluster.get_cluster_state()
		input = self.observe()  #  NN input
		label = np.zeros(pm.ACTION_DIM)
		for i in range(pm.SCHED_WINDOW_SIZE):
			job = self.window_jobs[i]
			if job and job.id == label_job_id:
				if pm.PS_WORKER:
					if pm.BUNDLE_ACTION:
						if role == "worker":
							label[i * 3] = 1
						elif role == "ps":
							label[i * 3 + 1] = 1
						elif role == "bundle":
							label[i * 3 + 2] = 1
					else:
						if role == "worker":
							label[i * 2] = 1
						elif role == "ps":
							label[i * 2 + 1] = 1
				else:
					label[i] = 1
		self.data.append((input,label))


def test():
	import log, trace
	logger = log.getLogger(name="agent_" + str(id), level="INFO")
	job_trace = trace.Trace(logger).get_trace()
	env = K8S_RL_Env("K8S_RL", job_trace, logger)
	while not env.end:
		data = env.step()
		for item in data:
			print item
		print "-----------------------------"
		raw_input("Next? ")

	print env.get_results()


if __name__ == '__main__':
	test()





