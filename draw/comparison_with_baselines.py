import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import sys

plt.style.use(["seaborn-bright", "double-figure.mplstyle"])


def draw_jct(xticks, values, std_devs, file):
	fig, ax = plt.subplots()

	index = [0.75 + i for i in range(len(values))]
	index = np.array(index)
	print index[-1] + 0.75
	ax.set_xlim([0, index[-1] + 0.75])
	bar_width = 0.5
	opacity = 0.8

	patterns = ["/", "\\", "-", '|', '.', '*']

	colors = ['#1f77a4', 'pink', 'green', 'red', 'cyan', 'yellow', 'k']
	print index
	print values
	print std_devs

	for i in range(len(values)):
		rects1 = plt.bar(index[i], height=values[i], yerr=std_devs[i], width=bar_width,
						 alpha=opacity,
						 color=colors[0],
		                 ecolor='black',
		                 capsize=10,
						 align='center',)
						# hatch=patterns[i])

	ax.set_xticks(np.arange(len(values)) + bar_width / 2)
	ax.set_xticklabels(xticks, rotation=20)
	ax.yaxis.set_major_locator(mtick.MaxNLocator(5))
	# ax.yaxis.set_label_coords(0, 1)

	# plt.xlabel('Models')
	plt.ylabel('Avg. Job Completion Time')
	plt.xticks(index, xticks)

	# plt.gcf().subplots_adjust(bottom=0.28, left=0.2)
	plt.tight_layout()
	plt.show()

	fig.savefig(file)

def draw_makespan(xticks, values, std_devs, file):
	fig, ax = plt.subplots()

	index = [0.75 + i for i in range(len(values))]
	index = np.array(index)
	print index[-1] + 0.75
	ax.set_xlim([0, index[-1] + 0.75])
	bar_width = 0.5
	opacity = 0.8

	patterns = ["/", "\\", "-", '|', '.', '*']

	colors = ['#1f77a4', 'pink', 'green', 'red', 'cyan', 'yellow', 'k']
	print index
	print values
	print std_devs

	for i in range(len(values)):
		rects1 = plt.bar(index[i], height=values[i], yerr=std_devs[i], width=bar_width,
						 alpha=opacity,
						 color=colors[0],
		                 ecolor='black',
		                 capsize=10,
						 align='center',)
						# hatch=patterns[i])

	ax.set_xticks(np.arange(len(values)) + bar_width / 2)
	ax.set_xticklabels(xticks, rotation=20)
	ax.yaxis.set_major_locator(mtick.MaxNLocator(5))

	# plt.xlabel('Models')
	plt.ylabel('Makespan')
	plt.xticks(index, xticks)

	# plt.gcf().subplots_adjust(bottom=0.28, left=0.2)
	plt.tight_layout()
	plt.show()

	fig.savefig(file)


xticks = ('DL$^2$', 'DRF', 'Tetris', 'Optimus', 'OfflineRL')

# # uniform distribution
# def draw_uniform():
# 	jcts = [2.984, 4.031, 6.055, 3.433]
# 	std_devs = [0.191, 0.388, 0.632, 0.662]
# 	draw_jct(xticks, jcts, std_devs, "jct_comparison_uniform_workload.pdf")
#
# 	makespans = [25.490, 29.59, 32.22, 26.6]
# 	std_devs = [0.481, 0.933, 1.467, 1.034]
# 	draw_makespan(xticks, makespans, std_devs, "makespan_comparison_uniform_workload.pdf")
#
# # poisson distribution
# def draw_possion():
# 	jcts = [2.611, 3.459, 4.649, 2.701]
# 	std_devs = [0.145, 0.276, 0.651, 0.352]
# 	draw_jct(xticks, jcts, std_devs, "jct_comparison_possion_workload.pdf")
#
# # google trace distribution
# def draw_GGtrace():
# 	jcts = [2.726, 3.307, 5.229, 2.847]
# 	std_devs = [0.156, 0.349, 0.737, 0.368]
# 	draw_jct(xticks, jcts, std_devs, "jct_comparison_GGtrace_workload.pdf")
#

# ali trace distribution
def draw_Alitrace():
	jcts = [5.724, 10.246, 8.955, 7.318, 9.418]
	std_devs = [0.844, 0.914, 0.226, 0.529, 1.194]
	draw_jct(xticks, jcts, std_devs, "jct_comparison_Alitrace_workload.pdf")




#
# def main(dist):
# 	if dist == "Uniform":
# 		draw_uniform()
# 	elif dist == "Poisson":
# 		draw_possion()
# 	elif dist == "Google_Trace":
# 		draw_GGtrace()
# 	else:
# 		print "ERROR: Wrong dist"



if __name__ == "__main__":
	# draw_uniform()
	# draw_possion()
	# draw_GGtrace()
	draw_Alitrace()
	# sys.exit(0)
	# if len(sys.argv) != 2:
	# 	print "please input job arrival distribution"
	# 	exit(1)
	# main(sys.argv[1])
