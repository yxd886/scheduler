import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick

#print(mpl.__version__)
mpl.rcParams['hatch.linewidth'] = 0.5


def draw(jct_times, makespan_times, figure_name):
	'''
	average completion time compared with other schedulers
	'''

	plt.style.use(["ggplot", "double-figure.mplstyle"])

	labels = ('DL2', 'DRF', 'Optimus')
	xticks = ('JCT' ,'Makespan')    # ticks of bars, change here

	# normalize Optimus to 1
	Optimus_time = jct_times[0]
	for i in range(len(jct_times)):
		jct_times[i] /= float(Optimus_time)

	Optimus_time = makespan_times[0]
	for i in range(len(makespan_times)):
		makespan_times[i] /= float(Optimus_time)

	fig, ax = plt.subplots()

	bar_width = 0.4
	opacity = 0.8

	start_index = 0.5
	index = [start_index + bar_width*i for i in range(len(jct_times))]
	index = np.array(index)

	# ax.set_ylim([0, 140])
	# xticks = np.arange(0, 120, 20)
	# ax.set_xticks(xticks)

	patterns = ["/", "\\", "-", 'o', '.', '*']

	colors = ['b', 'g', 'r', 'orange', 'r', 'b', 'm', 'y', 'steelblue', 'gold', 'lightcoral', 'chocolate']
	colors = ['r', '#e1e4ff' ,'1']
	#colors = [str(1-i) for i in np.linspace(0,1,len(times))]

	for i in range(len(jct_times)):
		rects1 = plt.bar(index[i], jct_times[i], bar_width,
		                 alpha=opacity,
		                 color=colors[i%len(colors)],  # (0.9,0.9,0.9),
		                 align='center',
		                 hatch=patterns[i%len(patterns)])
		for p in rects1.patches:
			ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=24)

	index = [1.7 + bar_width * i for i in range(len(makespan_times))]
	index = np.array(index)
	for i in range(len(makespan_times)):
		rects2 = plt.bar(index[i]+0.3, makespan_times[i], bar_width,
		                 alpha=opacity,
		                 color=colors[i%len(colors)],  # (0.9,0.9,0.9),
		                 align='center',
		                 label=labels[i],
						hatch=patterns[i%len(patterns)])
		for p in rects2.patches:
			ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=24)

	ax.yaxis.set_major_locator(mtick.MaxNLocator(5))
	ax.set_xlim([0, index[i]+0.8])
	# ax.set_ylim([0.75, max(jct_times) + 0.5])
	#plt.ylabel('Norm. Avg. JCT')
	plt.xticks([0.9,2.4], xticks, fontsize=32, weight='medium')
	legend = ax.legend(loc=(0.46,0.55), shadow=False)
	for label in legend.get_texts():
		label.set_fontsize(22)

	frame = legend.get_frame()
	frame.set_facecolor('1')
	plt.gcf().subplots_adjust()
	plt.show()
	fig.savefig(figure_name)
