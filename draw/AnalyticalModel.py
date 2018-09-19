import numpy as np
import ast
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import sys
import Queue
import copy
plt.style.use(["seaborn-bright", "double-figure.mplstyle"])
class LoadBalanceUnit:
    def __init__(self, ma_id, ma_capcity):
        self.machine_id = ma_id
        self.machine_load = sum(ma_capcity)

    def __cmp__(self, other):
        return cmp(self.machine_load, other.machine_load)


if __name__ == '__main__':
	# res net 50 speed
	fix_worker_dict_resnet = [[0, 0, 0, 0, 0, 0, 0, 0], [0, 20.828, 24.813, 29.122, 28.98, 29.345, 29.44, 29.144], [0, 38.5, 53.79, 54.486000000000004, 57.006, 58.101, 57.91, 57.823], [0, 50.306, 70.229, 80.189, 84.587, 86.26, 85.803, 86.25999999999999], [0, 62.339999999999996, 93.651, 98.35900000000001, 111.858, 115.01, 113.326, 112.48400000000001], [0, 71.42699999999999, 107.153, 115.659, 139.852, 142.161, 137.728, 139.067], [0, 75.93299999999999, 119.185, 130.779, 163.21699999999998, 166.61399999999998, 164.05, 169.475], [0, 81.54599999999999, 133.624, 147.062, 0, 194.40300000000002, 191.57799999999997, 194.489]]
	fix_paraserver_dict_resnet = [[0, 0, 0, 0, 0, 0, 0, 0], [0, 20.828, 38.5, 50.306, 62.339999999999996, 71.42699999999999, 75.93299999999999, 81.54599999999999], [0, 24.813, 53.79, 70.229, 93.651, 107.153, 119.185, 133.624], [0, 29.122, 54.486000000000004, 80.189, 98.35900000000001, 115.659, 130.779, 147.062], [0, 28.98, 57.006, 84.587, 111.858, 139.852, 163.21699999999998, 0], [0, 29.345, 58.101, 86.26, 115.01, 142.161, 166.61399999999998, 194.40300000000002], [0, 29.44, 57.91, 85.803, 113.326, 137.728, 164.05, 191.57799999999997], [0, 29.144, 57.823, 86.25999999999999, 112.48400000000001, 139.067, 169.475, 194.489]]

	# vgg-16 speed
	fix_worker_dict_vgg = [[0, 0, 0, 0, 0, 0, 0, 0], [0, 6.739, 13.932, 17.97, 19.592, 21.717, 23.029, 21.82], [0, 14.049, 26.149, 30.447, 30.338, 38.669, 40.192, 42.22], [0, 20.647, 31.384999999999998, 35.757999999999996, 46.024, 43.554, 57.766, 60.447], [0, 20.799, 29.866, 45.985, 46.129, 65.88900000000001, 73.71799999999999, 78.537], [0, 29.657000000000004, 44.315, 53.998000000000005, 69.776, 73.51100000000001, 78.238, 94.369], [0, 32.06700000000001, 49.889, 67.241, 79.12, 90.60499999999999, 89.489, 109.02100000000002], [0, 31.134, 57.748000000000005, 70.243, 0, 93.42099999999999, 102.65700000000001, 0]]
	fix_paraserver_dict_vgg = [[0, 0, 0, 0, 0, 0, 0, 0], [0, 6.739, 14.049, 20.647, 20.799, 29.657000000000004, 32.06700000000001, 31.134], [0, 13.932, 26.149, 31.384999999999998, 29.866, 44.315, 49.889, 57.748000000000005], [0, 17.97, 30.447, 35.757999999999996, 45.985, 53.998000000000005, 67.241, 70.243], [0, 19.592, 30.338, 46.024, 46.129, 69.776, 79.12, 0], [0, 21.717, 38.669, 43.554, 65.88900000000001, 73.51100000000001, 90.60499999999999, 93.42099999999999], [0, 23.029, 40.192, 57.766, 73.71799999999999, 78.238, 89.489, 102.65700000000001], [0, 21.82, 42.22, 60.447, 78.537, 94.369, 109.02100000000002, 0]]

	models = ["resnet-50", "vgg-16", "resnext-110", "inception-bn", "seq2seq", "cnn-text-classification", "dssm", "wlm"]
	local_computation_time = [0.449, 0.535, 0.226, 0.815, 0.075, 0.585, 0.567, 0.154]  # unit: s
	model_size = [102.2, 553.4, 6.92, 42.1, 36.5, 24, 6, 19.2]
	batch_num = [115, 115, 390, 120, 780, 193, 349, 165]
	inter_bandwidth = [91.875, 233.0, 59.5, 145.875, 120.125, 60.75, 92.125, 10.375]  # unit: MB server bandwidth 4352 MB/s
	intra_bandwidth = [306.5, 427.75, 63.0, 1082.125, 181.125, 159.625, 65.625, 22.875]  # unit: MB server bandwidth 1510MB/s
	resource_demand_worker = [[2, 4], [2, 4], [2, 4], [2, 4], [4, 0], [2, 4], [4, 0], [1, 4]]  # cpu, gpu, 1 cpu = 1 slots, 1 gpu = 4 slots
	resource_demand_paraserver = [[3, 0], [4, 0], [3, 0], [3, 0], [1, 0], [3, 0], [1, 0], [1, 0]]
	num_machines = 6
	analyze_fix_worker_resnet = [[0 for col in range(0, 8)] for row in range(0, 8)]
	analyze_fix_paraserver_resnet = [[0 for col in range(0, 8)] for row in range(0, 8)]
	analyze_fix_worker_vgg = [[0 for col in range(0, 8)] for row in range(0, 8)]
	analyze_fix_paraserver_vgg = [[0 for col in range(0, 8)] for row in range(0, 8)]
	for index in range(0, 2):
		print "model: ", models[index]
		for wk in range(1, 8):
			for ps in range(1, 8):
				print "ps {0} wk {1}".format(ps, wk)
				# decide placement on machine
				curr_allocation = [[0, 0] for ma in range(0, num_machines)]
				machines = [[0, 0] for ma in range(0, num_machines)]
				for i in range(0, wk): # place worker first
					machine_que = Queue.PriorityQueue()
					for j in range(0, num_machines):
						if machines[j][0] + resource_demand_worker[index][0] <= 8 and machines[j][1] + resource_demand_worker[index][1] <= 8:  # only consider one type of resource
							occupiedresource = copy.deepcopy(machines[j])
							machine_que.put(LoadBalanceUnit(j, occupiedresource))
							del occupiedresource
					if not machine_que.empty():
						tmp = machine_que.get()
						ma_id = tmp.machine_id
						# print "wk ma id {0} occupied resource {1}".format(ma_id, machines[ma_id])
						curr_allocation[ma_id][0] += 1
						machines[ma_id][0] += resource_demand_worker[index][0]
						machines[ma_id][1] += resource_demand_worker[index][1]
					else:
						print "no machine available for wk"
						# logging.debug("job idx {0} machine id {1}".format(jobidx, ma_id))
				for i in range(0, ps): # place para server later
					machine_que = Queue.PriorityQueue()
					for j in range(0, num_machines):
						if machines[j][0] + resource_demand_paraserver[index][0] <= 8 and machines[j][1] + resource_demand_paraserver[index][1] <= 8:
							occupiedresource = copy.deepcopy(machines[j])
							machine_que.put(LoadBalanceUnit(j, occupiedresource))
							del occupiedresource
					if not machine_que.empty():
						tmp = machine_que.get()
						ma_id = tmp.machine_id
						# print "ps ma id {0} occupied resource {1}".format(ma_id, machines[ma_id])
						curr_allocation[ma_id][1] += 1
						machines[ma_id][0] += resource_demand_paraserver[index][0]
						machines[ma_id][1] += resource_demand_paraserver[index][1]
					else:
						print "no machine available for ps"
						# logging.debug("job idx {0} machine id {1}".format(jobidx, ma_id))
				print curr_allocation
				transmission_time = 0
				bandwidth = [0 for row in range(0, num_machines)] # inter bandwidth of each transmission path
				for ma in range(0, num_machines):
					num = curr_allocation[ma][0] * (ps - curr_allocation[ma][1]) + curr_allocation[ma][1] * (wk - curr_allocation[ma][0]) # path starting from worker + path starting from ps
					if num != 0:
						bandwidth[ma] = 1510 / float(num)

				for ma in range(0, num_machines):

					if curr_allocation[ma][0] != 0:
						assert bandwidth[ma] > 0
						max_trans_time = 0
						intra_bandwidth = 4352 / float(curr_allocation[ma][0])
						for ma_ps in range(0, num_machines):
							if ma_ps == ma:
								continue
							if curr_allocation[ma_ps][1] != 0:
								assert bandwidth[ma_ps] > 0
								# print ma, ma_ps, ps, curr_allocation[ma][1], curr_allocation[ma_ps][1]
								trans_model = model_size[index] / float(ps - curr_allocation[ma][1])
								min_bdw = min(bandwidth[ma], bandwidth[ma_ps])
								max_trans_time = max(max_trans_time, trans_model / float(min_bdw))
						# max over intra, inter data transmission time
						max_trans_time = max(max_trans_time, model_size[index] * curr_allocation[ma][1] / ps / float(intra_bandwidth))
						transmission_time = max(transmission_time, max_trans_time)
				# for ma in range(0, num_machines):
				# 	if curr_allocation[ma][0] != 0:
				# 		intra_bandwidth = 4352 / float(curr_allocation[ma][0])
				# 		if curr_allocation[ma][1] == 0:
				# 			inter_bandwidth = 1510 / float(curr_allocation[ma][0])
				# 		else:
				# 			inter_bandwidth = 1510 / float(wk)
				# 		tmp_transmission_time = max(model_size[index] * (1 - float(curr_allocation[ma][1]) / ps) / float(inter_bandwidth), model_size[index] * curr_allocation[ma][1] / ps / float(intra_bandwidth))  # unit sec
				# 		transmission_time = max(transmission_time, tmp_transmission_time)
				speed = wk / float(local_computation_time[index] + 2 * transmission_time) * 32 # number of samples trained in 1s
				if index == 0:
					analyze_fix_worker_resnet[wk][ps] = speed
					analyze_fix_paraserver_resnet[ps][wk] = speed
				elif index == 1:
					analyze_fix_worker_vgg[wk][ps] = speed
					analyze_fix_paraserver_vgg[ps][wk] = speed
		# for i in range(0, len(analyze_fix_worker)):
		# 	print analyze_fix_worker[i]
		# for i in range(0, len(analyze_fix_worker)):
		# 	print analyze_fix_paraserver[i] # 1 mini batch = 32 samples

	colors = ["red", "blue", "green", "black", "orange", "pink"]
	markers = ["o", "s", "^", "*", "v", "h"]
	plot_num = [1, 3, 5]
	for i in range(0, len(plot_num)):
		index = [1, 2, 3, 4, 5, 6, 7]
		rects1 = plt.plot(index, np.array(fix_worker_dict_resnet[plot_num[i]][1:])/32.0, color=colors[i], label="worker={0}".format(plot_num[i]), marker=markers[i])
	plt.xlabel('# of PSs')
	plt.ylabel('Speed (steps/s)')
	#plt.legend(loc='upper right', bbox_to_anchor=(0.84, 1.06), fancybox=False, shadow=False, ncol=2)
	plt.legend(loc=(0,0.7), frameon=False, fontsize=20, ncol=2)
	plt.ylim((0, 6))
	ax = plt.gca()
	print ax
	ax.yaxis.set_major_locator(mtick.MaxNLocator(5))
	# ax.xaxis.set_major_locator(mtick.MaxNLocator(6))
	plt.tight_layout()
	plt.savefig('fig_speed_fix_wk.pdf', format='pdf', dpi=1000)
	plt.show()
	plt.clf()
	# sys.exit(0)

	for i in range(0, len(plot_num)):
		index = [1, 2, 3, 4, 5, 6, 7]
		rects1 = plt.plot(index, np.array(fix_paraserver_dict_resnet[plot_num[i]][1:])/32.0, color=colors[i], label="PS={0}".format(plot_num[i]), marker=markers[i])
	plt.xlabel('# of workers')
	plt.ylabel('Speed (steps/s)')
	plt.legend(loc='best', frameon=False, fontsize=24)
	# plt.legend(loc='upper right', bbox_to_anchor=(0.82, 1.03), fancybox=False, shadow=False, ncol=2)
	# plt.ylim((50, 200))frameon=False
	# set interval of y axis label
	ax = plt.gca()
	plt.ylim((0, 6))
	# print ax
	ax.yaxis.set_major_locator(mtick.MaxNLocator(4))
	# ax.xaxis.set_major_locator(mtick.MaxNLocator(6))
	plt.tight_layout()
	plt.savefig('fig_speed_fix_ps.pdf', format='pdf', dpi=1000)
	plt.show()
	# sys.exit(0)