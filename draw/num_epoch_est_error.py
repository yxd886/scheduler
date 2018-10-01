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
	ax.yaxis.set_major_locator(mtick.MaxNLocator(4))

	plt.xlabel(xlabel)
	plt.ylabel('Avg. Job Completion Time', fontsize=24)
	plt.xticks(index, xticks)

	# plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
	plt.tight_layout()
	plt.show()

	fig.savefig(file)

def draw_est_error():
	xticks = ('5', '10', '15', '20', '25')
	jcts = [6.024, 5.948, 6.851, 6.987, 7.217]
	std_devs = [0.810, 0.853, 0.355, 1.100, 0.595]
	draw("Error (%)", xticks, jcts, std_devs, "jct_vary_epoch_est_error.pdf")


if __name__ == "__main__":
	draw_est_error()