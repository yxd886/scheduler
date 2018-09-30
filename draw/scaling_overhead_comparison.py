import numpy as np
import matplotlib.pyplot as plt

plt.style.use(["seaborn-bright", "single-figure.mplstyle"])

fig, ax = plt.subplots(figsize=(8,4))


def autolabel(rects):
	for rect in rects:
		height = rect.get_height()
		plt.text(rect.get_x()+rect.get_width()/2-0.275, height+0.2, '%s' % float(height), fontsize=16)
size = 6
x = np.arange(size)
#x = [1,2,3,4,5,6]
checkpoint = [11.45,13.95,14.55,15.15,15.85,16.05]
dl2 = [0.062,0.136,0.188,0.309,0.482,0.625]

total_width, n = 0.8, 2
width = total_width / n
x = x - (total_width - width) / 2
x += 1
a=plt.bar(x, checkpoint,  width=width, label='Checkpointing', color="#1f77a4")
b=plt.bar(x + width, dl2, width=width, label=r'$DL^2$', color="darkorange")
autolabel(a)
autolabel(b)
#for a,b in zip(x,checkpoint):
#    plt.text(a, b+0.05, '%.d' % b, ha='center', va= 'bottom',fontsize=7)
plt.legend(loc=(0.02, 0.7), shadow=False)
plt.xlabel("# of PS added")
plt.ylabel("Time (s)")
plt.ylim([0,25])
plt.tight_layout()
plt.savefig('scaling_overhead_comparison.pdf', format='pdf', dpi=1000)
plt.show()