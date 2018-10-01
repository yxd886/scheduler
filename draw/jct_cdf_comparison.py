import os
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import matplotlib.transforms
import matplotlib
import time
import datetime
import numpy as np
import ast
import yaml

from matplotlib.ticker import MaxNLocator





def job_cdf_compare():
	plt.style.use(["seaborn-bright", "double-figure.mplstyle"])
	drf_jcts = []
	with open("./data/DRF_JCTs.txt", 'r') as f:
		for line in f:
			jcts = ast.literal_eval(line.replace('\n',''))
			drf_jcts += jcts

	dl2_jcts = []
	with open("./data/DL2_JCTs.txt", 'r') as f:
		for line in f:
			stats = ast.literal_eval(line.replace('\n',''))
			if stats['step'] == 400:
				jcts = stats['jcts']
				for jct in jcts:
					dl2_jcts += jct
				break

	print "average job duration: DRF ", sum(drf_jcts)/len(drf_jcts), " DL2 ", sum(dl2_jcts)/len(dl2_jcts)
	fig, ax = plt.subplots()
	# bins = [_ for _ in range(0, max(drf_jcts+dl2_jcts), max(drf_jcts+dl2_jcts)/1000)]
	counts, bin_edges = np.histogram(drf_jcts, bins=1000)
	cdf = np.cumsum(counts, dtype=float)
	cdf /= cdf[-1]
	cdf = np.append(np.array([0.0]), cdf)
	ax.plot(bin_edges*5, cdf*100, 'b-')
	# bins = [_ for _ in range(0, max(drf_jcts+dl2_jcts), max(drf_jcts+dl2_jcts)/1000)]
	counts, bin_edges = np.histogram(dl2_jcts, bins=1000)
	cdf = np.cumsum(counts, dtype=float)
	cdf /= cdf[-1]
	cdf = np.append(np.array([0.0]), cdf)
	ax.plot(bin_edges*5, cdf * 100, 'g-')
	ax.xaxis.set_major_locator(MaxNLocator(40))
	plt.xlabel("Job Completion Time (min)")
	plt.ylabel("CDF (%)")
	plt.ylim(0,100)
	plt.xlim(left=1)

	plt.xscale('log')
	plt.gcf().subplots_adjust(bottom=0.2, left=0.18)
	plt.tight_layout()
	plt.show()
	fig.savefig("jct_cdf_comparison.pdf")

job_cdf_compare()


