import sys
import math
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
import numpy as np


plt.style.use(["seaborn-bright", "double-figure.mplstyle"])
# plt.rcParams["font.family"] = "Times New Roman"


def process_results(file):
	f = open(file, 'r')
	jcts = []
	makespans = []
	rewards = []
	steps = []
	for line in f:
		if line == '\n':
			break
		segs = line.replace("\n", '').split(" ")
		steps.append(int(segs[1].replace(":","")))
		jcts.append(float(segs[2]))
		# makespans.append(float(segs[3]))
		# rewards.append(float(segs[4]))
	# smoothing jct by averaging results
	temp_jcts = []
	for i in range(len(jcts)):
		index = max(0,i-4)
		temp_jcts.append(sum(jcts[index:i+1])/len(jcts[index:i+1]))

	return (steps, temp_jcts, makespans, rewards)


'''
curve fitting
'''
def draw(data1, data2, data3):
	style = ['go-', 'bD-', 'g--', 'c^-', 'mx-', 'ks-']
	fig, ax = plt.subplots()
	#ax.plot(data1[0], data1[1], style[0],  label=data1[2])
	ax.plot(data2[0], data2[1], style[1], label=data2[2], markevery=4)
	ax.plot(data3[0], data3[1], style[2], label=data3[2])

	for i in range(len(data2[0])): # failed to set markers
		if data2[0][i] == 2000 or data2[0][i] == 1000:
			ax.plot([data2[0][i]],[data2[1][i]], 'rD')

	# #ax.annotate('New types of jobs', xy=(1000, 3.8), xytext=(1300, 4.6),fontsize=20,
	# 			arrowprops=dict(arrowstyle=matplotlib.patches.ArrowStyle("Fancy", head_length=0.3, head_width=0.3, tail_width=0.2), connectionstyle="arc3", facecolor='black'),
	# 			)
	# #ax.annotate('                    ', xy=(2100, 3.0), xytext=(1300, 4.6),fontsize=20,
	# 			arrowprops=dict(arrowstyle=matplotlib.patches.ArrowStyle("Fancy", head_length=0.3, head_width=0.3, tail_width=0.2), connectionstyle="arc3", facecolor='black'),
	# 			)
	legend = ax.legend(loc='best', shadow=False)
	frame = legend.get_frame()
	frame.set_facecolor('1')

	ax.set_xlabel('Step')
	ax.set_ylabel('Avg. Job Completion Time')
	ax.set_ylim(bottom=0)
	# plt.locator_params(axis='y', nticks=4, tight=True)
	ax.xaxis.set_major_locator(mtick.MaxNLocator(6))
	ax.yaxis.set_major_locator(mtick.MaxNLocator(4))
	ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

	plt.gcf().subplots_adjust(bottom=0.2, left=0.2)
	plt.savefig('training_process_varying_job_types.pdf', format='pdf', dpi=1000)
	plt.show()


def draw_jct():
	# train_steps, train_jcts, _, _ = process_results("./data/rl_train_jct_changing_workload.txt")
	valid_steps, valid_jcts, _, _ = process_results("./data/rl_validation_changing_workload.txt")

	# aim (True, ('2.8825+-0.216790028984', '25.34+-0.564269439187', '2.40165523761+-0.0481009935628'))
	aim_steps = valid_steps
	aim_jcts = [5.7 for _ in range(len(valid_steps))]

	drf_steps = valid_steps
	drf_jcts = [9.731 for _ in range(len(valid_steps))]

	print valid_steps

	draw((drf_steps, drf_jcts, ""),(valid_steps, valid_jcts, r'$DL^2$'), (aim_steps, aim_jcts, "Ideal"))


draw_jct()
