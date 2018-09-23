import numpy as np
import ast
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator

'''
analyze trace in testbed
'''

linear_scalability_maps = dict()
ratio_maps =  dict()
model_names = {"resnet-50":"ResNet", "vgg-16":"VGG", "resnext-110":"ResNeXt", "inception-bn":"Inception", "seq2seq":"Seq2Seq", "cnn-text-classification":"CTC", "dssm":"DSSM", "wlm":"WLM"}

def ratio_fit():
	# fit a speed function for each model
	speed_funcs = dict()
	records = []
	with open("../trace/testbed/ps-worker-ratio.txt", "r") as f:
		for line in f:
			records.append(ast.literal_eval(line.replace('\n','')))
	speed_maps = dict()
	for record in records:
		model, sync_mode, tot_batch_size, num_ps, num_worker, speeds, ps_cpu_usages, worker_cpu_usages = record
		if model not in speed_maps:
			speed_maps[model] = []
		speed_maps[model].append((num_ps, num_worker, sum(speeds)))
	print speed_maps['resnet-50']
	for model in speed_maps.keys():
		x = []; y = []; z = []
		for _num_ps, _num_worker, _speed in speed_maps[model]:
			print model, _num_ps, _speed
			if model not in ratio_maps:
				ratio_maps[model] = []
			ratio_maps[model].append(_speed)

			x.append(_num_ps)
			y.append(_num_worker)
			z.append(_speed)
		interp = scipy.interpolate.Rbf(np.array(x), np.array(y), np.array(z), function='linear')
		speed_funcs[model] = interp
	return speed_funcs


def linear_fit():
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
	print speed_maps['resnet-50']
	for model in speed_maps.keys():
		x = []; y = []; z = []
		for _num_ps, _num_worker, _speed in speed_maps[model]:
			if _num_ps == _num_worker and _num_ps%2==0:
				print model, _num_ps, _speed
				if model not in linear_scalability_maps:
					linear_scalability_maps[model] = []
				linear_scalability_maps[model].append(_speed)

			x.append(_num_ps)
			y.append(_num_worker)
			z.append(_speed)
		interp = scipy.interpolate.Rbf(np.array(x), np.array(y), np.array(z), function='linear')
		speed_funcs[model] = interp
	return speed_funcs

speed_funcs = linear_fit()
ratio_fit()


def get_speed(self, model, num_ps, num_worker):
	assert model in self.speed_funcs.keys() and num_ps <= pm.MAX_NUM_WORKERS+1 and num_worker <= pm.MAX_NUM_WORKERS+1
	return self.speed_funcs[model](num_ps, num_worker)


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




def draw_linearity():
	# ps:worker=2:2, 4:4, ..., 10:10, 12:12
	# draw a figure showing the scalability of 3 models
	plt.style.use(["seaborn-bright", "double-figure.mplstyle"])
	fig, ax = plt.subplots()
	plt.xlabel('# of workers')
	plt.ylabel('Scalability')

	styles = ["bD-", "r*-", "g>-", "y--", "kD-", "c-", "m-", "b*-"]
	count = 0
	for i in range(len(linear_scalability_maps)):
		dnn = linear_scalability_maps.keys()[i]
		if dnn != "resnet-50" and dnn != "vgg-16" and dnn != "seq2seq":
			continue
		for j in range(len(linear_scalability_maps[dnn])):
			if linear_scalability_maps[dnn][j] == 0:
				linear_scalability_maps[dnn][j] = (linear_scalability_maps[dnn][j-1] + linear_scalability_maps[dnn][j+1])/2
		x = np.array([2*_ for _ in range(1,len(linear_scalability_maps[dnn])+1)])
		y = 2*np.array(linear_scalability_maps[dnn])/(linear_scalability_maps[dnn][0])
		plt.plot(x, y, styles[count%len(styles)], label=model_names[dnn])
		count += 1
	legend = ax.legend(loc=(0,0.48), shadow=False)
	frame = legend.get_frame()
	frame.set_facecolor('1')
	plt.xlim(2,13)
	plt.ylim(2,13)
	ax.xaxis.set_major_locator(MaxNLocator(4))
	ax.yaxis.set_major_locator(MaxNLocator(4))
	# plt.title("PS:Worker=1:1")
	plt.tight_layout()

	plt.savefig('training_scalability.pdf',format='pdf', dpi=1000)
	plt.show()


def draw_ratio():
	# ps:worker = 2:10, 3:9, 4:8, 6:6, 8:4, 9:3, 10:2
	# draw a bar figure for 3 models under different ps/worker ratios
	plt.style.use(["seaborn-bright", "single-figure.mplstyle"])
	fig, ax = plt.subplots(figsize=(8,4))
	plt.xlabel('# of workers')
	plt.ylabel('Norm. Speed')


	styles = ["b-", "r*-", "c^-", "y--", "kD-", "g-", "m-", "b*-"]
	for i in range(len(ratio_maps)):
		dnn = ratio_maps.keys()[i]
		# if dnn != "resnet-50" and dnn != "vgg-16" and dnn != "seq2seq":
		# 	continue
		x = np.array([_ for _ in range(1,len(ratio_maps[dnn])+1)])
		y = np.array(ratio_maps[dnn])/max((ratio_maps[dnn]))
		plt.plot(x, y, styles[i%len(styles)], label=dnn)
	legend = ax.legend(loc='best', shadow=False)
	frame = legend.get_frame()
	frame.set_facecolor('1')
	ax.set_xticklabels(['','(2,10)','(3,9)','(4,8)','(6,6)','(8,4)','(9,3)','(10,2)'])
	# plt.xlim(1,7)
	# plt.ylim(1,7)
	# plt.title("PS:Worker=1:1")
	plt.tight_layout()
	plt.savefig('ps_worker_ratio.pdf',format='pdf', dpi=1000)
	plt.show()


if __name__ == '__main__':
	# draw_linearity()
	draw_ratio()
	exit()
	for model in speed_funcs.keys():
		draw(model)
