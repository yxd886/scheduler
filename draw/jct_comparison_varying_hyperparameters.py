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

	colors = ['#1f77a4', 'lightgray', 'pink', 'green', 'red', 'cyan', 'yellow', 'lightgray']
	print index
	print values
	print std_devs

	for i in range(len(values)):
		rects1 = plt.bar(index[i], height=values[i], yerr=std_devs[i], width=bar_width,
						 alpha=opacity,
						 color=colors[0],
		                 ecolor='black',
		                 capsize=10,
						 align='center')
						 # hatch=patterns[i])

	ax.set_xticks(np.arange(len(values)) + bar_width / 2)
	# ax.set_xticklabels(xticks, rotation=30)
	if max(values) < 4 and max(values)>1:
		ax.set_ylim([0,4])
	ax.yaxis.set_major_locator(mtick.MaxNLocator(4))

	#plt.xlabel(xlabel)
	plt.ylabel('Avg. Job Completion Time')
	plt.xticks(index, xticks)

	# plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
	plt.tight_layout()
	plt.show()

	fig.savefig(file)


def draw_vary_cluster_num():
	xticks = ('1', '2', '3', '4', '5')
	jcts = [6.408, 6.442, 6.819, 6.394, 6.921]
	std_devs = [0.804, 0.515, 0.559, 0.740, 0.655]
	draw("# of clusters", xticks, jcts, std_devs, "jct_vary_cluster_num.pdf")


def draw_vary_hidden_layers():
	xticks = ('1', '2', '3', '4', '5')
	jcts = [6.579, 5.724, 6.815, 7.652, 8.529]
	std_devs = [0.785, 0.844, 0.329, 0.354, 0.368]
	draw("# of layers",xticks, jcts, std_devs, "jct_vary_hide_layer_num.pdf")


def draw_sl_loss_funcs():
	xticks = ('MS', 'CE', 'AD')
	jcts = [6.867, 5.906, 6.010]
	std_devs = [0.815, 0.461, 0.649]
	draw("SL loss", xticks, jcts, std_devs, "jct_vary_SL_loss_fun.pdf")


def draw_vary_sched_window():
	xticks = ('10', '20', '30', '40', '50')
	jcts = [7.709, 6.958, 6.501, 5.873, 5.816]
	std_devs = [0.500, 0.606, 0.327, 0.734, 1.046]
	draw("Sched. window", xticks, jcts, std_devs, "jct_vary_window_size.pdf")


def draw_vary_num_neurons():
	xticks = ('32', '64', '96', '128', '256')
	jcts = [6.302, 6.77, 6.158, 5.943, 6.461]
	std_devs = [0.366, 0.421, 0.598, 0.695, 0.828]
	draw("# of neurons", xticks, jcts, std_devs, "jct_vary_neuron_num.pdf")

def draw_fifo_srtf():
	xticks = ('FIFO', 'FIFO-DL2', 'SRTF', 'SRTF-DL2')
	jcts = [3.396, 2.952, 2.953, 3.431]
	std_devs = [0.166, 0.111, 0.134, 0.212]
	draw("# of neurons", xticks, jcts, std_devs, "jct_vary_neuron_num.pdf")

def draw_vary_reward():
	xticks = ('R1', 'R2', 'R3')
	jcts = (7.077, 6.577, 9.256)
	std_devs = [1.268, 0.432, 0.152]
	draw('Reward functions', xticks, jcts, std_devs, 'jct_vary_reward_funcs.pdf')




if __name__ == "__main__":
	draw_vary_sched_window()
	draw_vary_num_neurons()
	draw_vary_hidden_layers()
	draw_sl_loss_funcs()
	draw_vary_cluster_num()
	draw_vary_reward()
	sys.exit(0)
	# if len(sys.argv) != 2:
	# 	print "please input job arrival distribution"
	# 	exit(1)
	# main(sys.argv[1])