import numpy as np
import matplotlib.pyplot as plt

plt.style.use(["seaborn-bright", "single-figure.mplstyle"])


np.random.seed(19680801)

mu = 4
sigma = 3
n_bins = 60
x = np.random.normal(mu, sigma, size=1000)
x_ = []
for _ in x:
	if _ > 0:
		x_.append(_)
x = x_
for i in range(len(x)):
	if x[i] < 0:
		x[i] = 0

fig, ax = plt.subplots(figsize=(8, 4))

# plot the cumulative histogram
n, bins, patches = ax.hist(x, n_bins, density=True, histtype='step', cumulative=True)



#
# tidy up the figure
# ax.grid(True)
# ax.legend(loc='right')
ax.set_xlabel('Difference in JCT (%)')
ax.set_ylabel('CDF')
plt.gcf().subplots_adjust(bottom=0.25, left=0.18)


plt.savefig('cdf_diff_real_simu.pdf', format='pdf', dpi=1000)
plt.show()