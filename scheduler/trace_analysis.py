import trace
import log
import Queue
import numpy
import parameters
import matplotlib.pyplot as plt


def main():
	logger = log.getLogger(name="Trace_analysis",level="INFO")
	job_trace = trace.Trace(logger).get_trace()
	jobs = Queue.PriorityQueue()
	types = []
	for value in job_trace.values():
		for job in value:
			if job.type not in types:
				jobs.put((job.type, job))
				types.append(job.type)

	test_types = [1]
	figure = plt.figure(1,figsize=(15,7.5))
	fig_index = 1
	while not jobs.empty():
		_, job = jobs.get()
		if job.type in test_types:
			vary_workers_sp =[]
			job.num_ps = 3
			job.curr_ps_placement = [1,2,3]
			for i in range(16):
				job.progress = 0
				job.num_workers = i+1
				# job.num_ps = i+1
				job.curr_worker_placement = [j+1 for j in range(i+1)]
				# job.curr_ps_placement = [j + 1 for j in range(i + 1)]
				speed = job.step()
				vary_workers_sp.append(speed)

			linear_sp = [vary_workers_sp[0]*(i+1) for i in range(len(vary_workers_sp))]
			plt.subplot(2,4,fig_index)
			fig_index += 1
			plt.plot([i+1 for i in range(len(vary_workers_sp))], vary_workers_sp, "go-", label="type " +str(job.type))
			plt.plot([i + 1 for i in range(len(linear_sp))], linear_sp, "r--", label="type " + str(job.type) + " ideal")

			legend = plt.legend(loc='best', shadow=False)
			frame = legend.get_frame()
			frame.set_facecolor('1')
			plt.xlabel("# of workers")
			plt.ylabel('Epochs')
	plt.suptitle("How workers(bundles) affects training speed of different jobs", fontsize=20)
	figure.savefig('job_analysis.pdf')
	plt.show()












if __name__ == "__main__":
	main()