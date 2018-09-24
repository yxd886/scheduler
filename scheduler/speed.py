import numpy as np
import ast
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator


def fit():
	# fit a speed function for each model
	speed_funcs = dict()
	records = []
	with open("../trace/testbed/ps-worker-linearity.txt", "r") as f:
		for line in f:
			records.append(ast.literal_eval(line.replace('\n','')))
	speed_maps = dict()
	for record in records:
		model, sync_mode, tot_batch_size, num_ps, num_worker, speeds, ps_cpu_usages, worker_cpu_usages = record
		if model not in speed_maps:
			speed_maps[model] = []
		speed_maps[model].append((num_ps, num_worker, sum(speeds)))
	for model in speed_maps.keys():
		x = []; y = []; z = []
		for _num_ps, _num_worker, _speed in speed_maps[model]:
			x.append(_num_ps)
			y.append(_num_worker)
			z.append(_speed)
		interp = scipy.interpolate.Rbf(np.array(x), np.array(y), np.array(z), function='linear')
		speed_funcs[model] = interp
	return speed_funcs

speed_funcs = fit()


# draw the 3D figure to have some basic understanding of training speed and resources
def draw(model):
	assert model in speed_funcs.keys()

	fig = plt.figure()
	# ax = fig.gca(projection='3d')
	ax = Axes3D(fig)
	ax.invert_xaxis()

	xi = yi = np.linspace(0, 16, 50)
	xi, yi = np.meshgrid(xi, yi)
	zi = speed_funcs[model](xi, yi)
	ax.plot_surface(xi, yi, zi, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	ax.set_xlabel('ps', fontsize=16)
	ax.set_ylabel('workers', fontsize=16)
	ax.set_zlabel('speed (steps/s)', fontsize=16)
	for tl in ax.get_xticklabels():
		tl.set_fontsize(12)
	for tl in ax.get_yticklabels():
		tl.set_fontsize(12)
	for tl in ax.get_zticklabels():
		tl.set_fontsize(12)
	ax.set_title(model, fontsize=20)
	plt.show()
	# plt.savefig(model + '-speed.pdf', format='pdf', dpi=1000)



if __name__ == '__main__':
	for model in speed_funcs.keys():
		draw(model)
