import sys
import math
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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
		makespans.append(float(segs[3]))
		rewards.append(float(segs[4]))

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
	style = ['g-', 'b-', 'r--', 'c^-', 'mx-', 'ks-']
	fig, ax = plt.subplots()
	ax.plot(np.array(data1[0])*300, np.array(data1[1])*300, style[0],  label=data1[2])
	ax.plot(np.array(data2[0])*300, np.array(data2[1])*300, style[1],   label=data2[2], markevery=4)
	ax.plot(np.array(data3[0])*300, np.array(data3[1])*300, style[2], label=data3[2])

	legend = ax.legend(loc='best', shadow=False)
	frame = legend.get_frame()
	frame.set_facecolor('1')

	ax.set_xlabel('Time (s)')
	ax.set_ylabel('Avg. Job Completion Time (s)')
	# plt.locator_params(axis='y', nticks=4, tight=True)
	ax.xaxis.set_major_locator(mtick.MaxNLocator(5))
	ax.yaxis.set_major_locator(mtick.MaxNLocator(5))
	ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

	plt.gcf().subplots_adjust(bottom=0.2, left=0.2)
	plt.savefig('training_process.pdf', format='pdf', dpi=1000)
	plt.show()


def draw_jct():
	sl_steps, sl_jcts, _, _ = process_results("./data/sim_testbed_offline_sl.txt")
	rl_steps, rl_jcts, _, _ = process_results("./data/sim_testbed_online_rl.txt")

	sl_steps = sl_steps[:sl_steps.index(200)+1]
	sl_jcts = sl_jcts[:len(sl_steps)]

	# DRF 4.031
	drf_steps = sl_steps + rl_steps
	drf_jcts = [1.79 for _ in range(len(drf_steps))]

	rl_steps = [sl_steps[-1]+_ for _ in rl_steps]
	draw((sl_steps, sl_jcts, "SL"),(rl_steps, rl_jcts, "DRL"), (drf_steps, drf_jcts, "DRF"))


draw_jct()
