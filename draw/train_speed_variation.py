import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
import sys

plt.style.use(["seaborn-bright", "double-figure.mplstyle"])



def draw(xlabel, xticks, values, std_devs, optimus_values, optimus_stddevs, file):
	fig, ax = plt.subplots()

	colors = ['#1f77a4', 'lightgray', 'pink', 'green', 'red', 'cyan', 'yellow', 'lightgray']
	x = [int(i) for i in xticks]
	ax.plot(x, values, 'gD-', label = r'$DL^2$')
	ax.plot(x, optimus_values, 'b^-', label = "Optimus")


	# ax.set_xticklabels(xticks, rotation=30)
	ax.xaxis.set_major_locator(mtick.MaxNLocator(5))
	ax.yaxis.set_major_locator(mtick.MaxNLocator(5))
	plt.legend(loc=(0.01,0.62), shadow=False)

	plt.xlabel(xlabel)
	plt.ylabel('Avg. Job Completion Time', fontsize=24)
	plt.ylim(5,10)

	# plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
	plt.tight_layout()
	plt.show()

	fig.savefig(file, dpi=1000)


def draw_est_error():
	xticks = ('5', '10', '15', '20', '25')
	jcts = [6.207, 6.669, 6.648, 6.806, 7.119]
	std_devs = [0.520, 0.289, 0.746, 0.665, 0.508]
	optimus_jcts = [7.318, 7.856, 8.310, 8.988, 9.352]
	optimus_devs = [0.529, 0.472, 0.508, 0.205, 0.691]
	draw("Variation (%)", xticks, jcts, std_devs, optimus_jcts, optimus_devs, "jct_vary_train_speed_error.pdf")

if __name__ == "__main__":
	draw_est_error()