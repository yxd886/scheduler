import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
import sys

plt.style.use(["seaborn-bright", "double-figure.mplstyle"])



def draw():
	fig, ax = plt.subplots()
	plt.ylabel('Avg. Job Completion Time')
	plt.tight_layout()
	plt.show()
	fig.savefig("padding.pdf", dpi=1000)

if __name__ == "__main__":
	draw()