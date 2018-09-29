import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick

#print(mpl.__version__)
mpl.rcParams['hatch.linewidth'] = 0.5


def draw(fifo_values, fifo_devs, srtf_values, srtf_devs, figure_name):
	'''
	average completion time compared with other schedulers
	'''

	plt.style.use(["seaborn-bright", "double-figure.mplstyle"])

	labels = ('SL', 'SL+DRL')
	xticks = ('FIFO' ,'SRTF')    # ticks of bars, change here

	fig, ax = plt.subplots()

	bar_width = 0.4
	opacity = 0.8

	start_index = 0.5
	index = [start_index + bar_width*i for i in range(len(fifo_values))]
	index = np.array(index)

	# ax.set_ylim([0, 140])
	# xticks = np.arange(0, 120, 20)
	# ax.set_xticks(xticks)

	patterns = ["/", "\\", "-", 'o', '.', '*']

	colors = ['p', 'g', 'r', 'orange', 'r', 'b', 'm', 'y', 'steelblue', 'gold', 'lightcoral', 'chocolate']
	colors = ['#e1e4ff', 'green','#e1e4ff' ,'1']
	#colors = [str(1-i) for i in np.linspace(0,1,len(times))]

	for i in range(len(fifo_values)):
		rects1 = plt.bar(index[i], fifo_values[i], bar_width, yerr=fifo_devs[i],
		                 alpha=opacity,
		                 color=colors[i%len(colors)],  # (0.9,0.9,0.9),
		                 align='center',
						 ecolor='black',
						 capsize=10,
		                 hatch=patterns[i%len(patterns)])
		# for p in rects1.patches:
		# 	ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=24)

	index = [1.7 + bar_width * i for i in range(len(srtf_values))]
	index = np.array(index)
	for i in range(len(srtf_values)):
		rects2 = plt.bar(index[i]+0.3, srtf_values[i], bar_width, yerr=srtf_devs[i],
		                 alpha=opacity,
		                 color=colors[i%len(colors)],  # (0.9,0.9,0.9),
		                 align='center',
						 ecolor='black',
						 capsize=10,
		                 label=labels[i],
						hatch=patterns[i%len(patterns)])
		# for p in rects2.patches:
		# 	ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=24)

	ax.yaxis.set_major_locator(mtick.MaxNLocator(4))
	ax.set_xlim([0, index[i]+0.8])
	ax.set_ylim([0., 25])
	#plt.ylabel('Norm. Avg. JCT')
	plt.xticks([0.7,2.2], xticks, fontsize=32, weight='medium')
	legend = ax.legend(loc='best', shadow=False)
	for label in legend.get_texts():
		label.set_fontsize(22)
	plt.ylabel('Avg. Job Completion Time')
	plt.gcf().subplots_adjust(bottom=0.15, left=0.2)

	frame = legend.get_frame()
	frame.set_facecolor('1')
	plt.gcf().subplots_adjust()
	plt.savefig(figure_name, format='pdf', dpi=1000)
	plt.show()




fifo_values = (14.219, 7.118)
fifo_devs = (1.220, 0.386)
srtf_values = (13.609, 8.557)
srtf_devs = (1.342, 0.247)
draw(fifo_values, fifo_devs, srtf_values, srtf_devs, "fifo_srtf_sl_rl.pdf")


#FIFO: 4.315+-0.168
#SRTF: 4.158+-0.092
# ('SRTF', ('7.55733333333+-0.247701881751', '63.68+-0.64', '0.943517751775+-0.00921247471239'))
# ('FIFO', ('7.11866666667+-0.386740510192', '60.74+-1.11642285896', '1.00578585414+-0.0168249917616'))