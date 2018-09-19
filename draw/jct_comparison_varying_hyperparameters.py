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

	#plt.xlabel(xlabel)
	plt.ylabel('JCT')
	plt.xticks(index, xticks)

	plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
	plt.show()

	fig.savefig(file)


def draw_vary_cluster_num():
	xticks = ('1', '2', '3', '4', '8')
	jcts = [3.079, 2.939, 3.203, 3.147, 3.103]
	std_devs = [0.197, 0.143, 0.199, 0.185, 0.089]
	draw("# of clusters", xticks, jcts, std_devs, "jct_vary_cluster_num.pdf")


def draw_vary_hidden_layers():
	xticks = ('1', '2', '3', '4', '5')
	jcts = [3.029, 3.232, 3.260, 3.572, 3.671]
	std_devs = [0.181, 0.228, 0.241, 0.249, 0.273]
	draw("# of layers",xticks, jcts, std_devs, "jct_vary_hide_layer_num.pdf")


def draw_sl_loss_funcs():
	xticks = ('MS', 'CE', 'AD')
	jcts = [3.141, 2.984, 3.221]
	std_devs = [0.131, 0.191, 0.227]
	draw("SL loss", xticks, jcts, std_devs, "jct_vary_SL_loss_fun.pdf")


def draw_vary_sched_window():
	xticks = ('10', '15', '20', '25', '30')
	jcts = [2.967, 2.976, 2.995, 2.989, 3.111]
	std_devs = [0.163, 0.143, 0.217, 0.152, 0.118]
	draw("Sched. window", xticks, jcts, std_devs, "jct_vary_window_size.pdf")


def draw_vary_num_neurons():
	xticks = ('32', '64', '96', '128', '256')
	jcts = [3.025, 2.952, 2.953, 3.227, 3.406]
	std_devs = [0.199, 0.111, 0.134, 0.173, 0.241]
	draw("# of neurons", xticks, jcts, std_devs, "jct_vary_neuron_num.pdf")

def draw_fifo_srtf():
	xticks = ('FIFO', 'FIFO-DL2', 'SRTF', 'SRTF-DL2')
	jcts = [3.396, 2.952, 2.953, 3.431]
	std_devs = [0.166, 0.111, 0.134, 0.212]
	draw("# of neurons", xticks, jcts, std_devs, "jct_vary_neuron_num.pdf")

def draw_vary_reward():
	xticks = ('R1', 'R2', 'R3')
	jcts = (3.0265, 3.0328, 3.798)
	std_devs = [0.145, 0.137, 0.044]
	draw('Reward functions', xticks, jcts, std_devs, 'jct_vary_reward_funcs.pdf')


# ('Norm_Progress', ('3.0265+-0.145519090003', '25.79+-0.622012861603', '2.3625514075+-0.0522627663532'))
# ('Num_Uncompleted_Jobs', ('3.03283333333+-0.13742887049', '25.83+-0.508035431835', '2.35508356943+-0.0463825536455'))
# ('Job_Progress', ('3.79833333333+-4.4408920985e-16', '27.8+-0.0', '2.20753234049+-4.4408920985e-16'))
#
# # def main(dist):
# 	if dist == "Uniform":
# 		draw_uniform()
# 	elif dist == "Poisson":
# 		draw_possion()
# 	elif dist == "Google_Trace":
# 		draw_GGtrace()
# 	else:
# 		print "ERROR: Wrong dist"



if __name__ == "__main__":
	# draw_vary_sched_window()
	# draw_vary_num_neurons()
	# draw_vary_hidden_layers()
	# draw_sl_loss_funcs()
	# draw_vary_cluster_num()
	draw_vary_reward()
	sys.exit(0)
	# if len(sys.argv) != 2:
	# 	print "please input job arrival distribution"
	# 	exit(1)
	# main(sys.argv[1])