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

	colors = ['pink', 'green', 'red', 'cyan', 'yellow', 'k']
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
	ax.set_xticklabels(xticks, rotation=30)
	ax.yaxis.set_major_locator(mtick.MaxNLocator(5))

	# plt.xlabel('Models')
	plt.ylabel('JCT')
	plt.xticks(index, xticks)

	plt.gcf().subplots_adjust(bottom=0.28, left=0.2)
	# plt.show()

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

	colors = ['pink', 'green', 'red', 'cyan', 'yellow', 'k']
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
	ax.set_xticklabels(xticks, rotation=30)
	ax.yaxis.set_major_locator(mtick.MaxNLocator(5))

	# plt.xlabel('Models')
	plt.ylabel('Makespan')
	plt.xticks(index, xticks)

	plt.gcf().subplots_adjust(bottom=0.28, left=0.2)
	# plt.show()

	fig.savefig(file)


xticks = ('DL$^2$', 'DRF', 'Tetris', 'Optimus')

# uniform distribution
def draw_uniform():
	jcts = [2.984, 4.031, 6.055, 3.433]
	std_devs = [0.191, 0.388, 0.632, 0.662]
	draw_jct(xticks, jcts, std_devs, "jct_comparison_uniform_workload.pdf")

	makespans = [25.490, 29.59, 32.22, 26.6]
	std_devs = [0.481, 0.933, 1.467, 1.034]
	draw_makespan(xticks, makespans, std_devs, "makespan_comparison_uniform_workload.pdf")


# makespans = ()


# poisson distribution
def draw_possion():
	jcts = [2.611, 3.459, 4.649, 2.701]
	std_devs = [0.145, 0.276, 0.651, 0.352]
	draw_jct(xticks, jcts, std_devs, "jct_comparison_possion_workload.pdf")

# google trace distribution
def draw_GGtrace():
	jcts = [2.726, 3.307, 5.229, 2.847]
	std_devs = [0.156, 0.349, 0.737, 0.368]
	draw_jct(xticks, jcts, std_devs, "jct_comparison_GGtrace_workload.pdf")


def main(dist):
	if dist == "Uniform":
		draw_uniform()
	elif dist == "Poisson":
		draw_possion()
	elif dist == "Google_Trace":
		draw_GGtrace()
	else:
		print "ERROR: Wrong dist"



if __name__ == "__main__":
	draw_uniform()
	draw_possion()
	draw_GGtrace()
	sys.exit(0)
	if len(sys.argv) != 2:
		print "please input job arrival distribution"
		exit(1)
	main(sys.argv[1])


# data
# DL2
# Uniform result: JCT $2.984\pm0.191$, Makespan $25.490\pm0.481$
# Poisson: JCT $2.611\pm0.145$, Makespan $29.120\pm0.602$
# Google Trace: JCT $2.726\pm0.156$, Makespan $27.060\pm0.291$
#
# % DRF
# %('Google_Trace', ('3.30666666667+-0.349039157308', '29.72+-0.696850055607', '2.03761989039+-0.0462061516815'))
# %('Poisson', ('3.45883333333+-0.275684048224', '32.86+-0.836899038116', '1.83923882898+-0.045382150314'))
# %('Uniform', ('4.0315+-0.388209887389', '29.59+-0.933220231242', '2.05257507591+-0.0613740818988'))
#
# % Tetris
# %('Google_Trace', ('5.22933333333+-0.737231005559', '31.9+-1.24579292019', '1.90592730855+-0.0661674425285'))
# %('Poisson', ('4.64883333333+-0.651216831102', '33.55+-1.01906820184', '1.80766693256+-0.0570652924748'))
# %('Uniform', ('6.05516666667+-0.632052410803', '32.22+-1.46751490623', '1.89089177794+-0.0866798078063'))
#
# % Optimus
# % 5% speed estimation error
# %('Google_Trace', ('2.8465+-0.367675468315', '28.24+-0.586856030045', '2.13695714507+-0.0443141748774'))
# %('Poisson', ('2.70016666666+-0.352295212381', '31.24+-0.743236167042', '1.93489408513+-0.0478121104149'))
# %('Uniform', ('3.43316666667+-0.662187641248', '26.6+-1.03440804328', '2.27401382991+-0.0849248517586'))

