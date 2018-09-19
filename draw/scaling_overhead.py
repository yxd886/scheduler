
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
import sys

plt.style.use(["seaborn-bright", "double-figure.mplstyle"])


def draw(xlabel, xticks, values, std_devs, file):
	fig, ax = plt.subplots()

	index = [0.75 + i for i in range(len(values))]
	index = np.array(index)
	print index[-1] + 0.75
	ax.set_xlim([0, index[-1] + 0.75])
	bar_width = 0.5
	opacity = 0.8

	patterns = ["/", "\\", "-", '|', '.', '*']

	colors = ['pink', 'green', 'red', 'cyan', 'yellow', 'lightgray']
	print index
	print values
	print std_devs

	for i in range(len(values)):
		rects1 = plt.bar(index[i], height=values[i], yerr=std_devs[i], width=bar_width,
						 alpha=opacity,
						 color=colors[i],
		                 ecolor='black',
		                 capsize=10,
						 align='center',
						 hatch=patterns[i])


	ax.set_xticks(np.arange(len(values)) + bar_width / 2)
	# ax.set_xticklabels(xticks, rotation=30)
	if max(values) < 4 and max(values)>1:
		ax.set_ylim([0,4])
	ax.yaxis.set_major_locator(mtick.MaxNLocator(4))
	#ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
	plt.xlabel(xlabel)
	plt.ylabel('Time(ms)')
	plt.xticks(index, xticks)

	plt.gcf().subplots_adjust(bottom=0.28, left=0.2)
	plt.show()

	fig.savefig(file)



xticks = []

mean_overhead = [0.00636886,  0.00256794,  0.0043196,  0.00290034,  0.00321858, 0.00241583,  0.00268425,  0.00344782,  0.00258504,  0.0035235]
devs = [0.00686049,  0.00277294,  0.00448122,  0.00299107,  0.00436244, 0.0027795 ,  0.00347478,  0.00431656,  0.00308403,  0.00296039]

overheads = []
std_devs = []
num_ps = [2,4,6,8,10]
for i in num_ps:
	xticks.append(str(i))
	overheads.append(sum(mean_overhead[0:i])*1000)
	std_devs.append(np.mean(devs[0:i])*1000)

draw("# of PSs added", xticks, overheads, std_devs, "overhead_varying_increased_ps.pdf")





#
# Scaling Experiment
#
# 5 worker 1 server, INC_SERVER (multiple machines):
# overhead for worker 9 : [0.00922548, 0.00807262, 0.0125743, 0.00870764, 0.0119425, 0.00797322, 0.00962866, 0.0120467, 0.00875266, 0.00787945];
# overhead for worker 11: [0.00096635, 0.000935602, 0.00562032, 0.000961836, 0.000977192, 0.00095329, 0.000989218, 0.00110443, 0.00100524, 0.00631449];
# overhead for worker 13: [0.0186433, 0.000947081, 0.000952409, 0.00283837, 0.000965379, 0.000952374, 0.000993241, 0.0009613, 0.00100882, 0.00105353];
# overhead for worker 15: [0.00150513, 0.0010434, 0.00101399, 0.00099455, 0.00106164, 0.00111907, 0.00110727, 0.00203557, 0.00105212, 0.0012793];
# overhead for worker 17: [0.00150405, 0.00184099, 0.00143697, 0.000999287, 0.00114621, 0.00108122, 0.000702843, 0.00109111, 0.00110638, 0.00109073];
#
# mean_step = [ 0.00636886,  0.00256794,  0.0043196,  0.00290034,  0.00321858, 0.00241583,  0.00268425,  0.00344782,  0.00258504,  0.0035235 ];
# std_step = [ 0.00686049,  0.00277294,  0.00448122,  0.00299107,  0.00436244, 0.0027795 ,  0.00347478,  0.00431656,  0.00308403,  0.00296039];