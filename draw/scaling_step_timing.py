import numpy as np
import matplotlib.pyplot as plt



plt.style.use(["seaborn-bright", "single-figure.mplstyle"])

fig, ax = plt.subplots(figsize=(8,4))

x = ["Inception","ResNet-18","ResNet-50","ResNet-101","ResNeXt-101","ResNet-152","AlexNet","VGG-16"]
step_1 = [1.41,1.425,1.42,1.371,1.364,1.438,1.295,1.398]
step_2 = [0.781,0.297,8.01,0.969,0.622,0.916,0.262,0.299]
step_3 = [19.731,19.546,51.17,42.667,37.548,50.102,98.946,142.843]
step_4 = [12.532,4.071,77.168,16.48,46.477,23.755,115.497,2.041]

bar_width = 0.5
plt.bar(x, step_1, label='Step 1', width=bar_width)
plt.bar(x, step_2, bottom = step_1, label='Step 2', width=bar_width)
plt.bar(x, step_3, bottom = step_2, label='Step 3', color="#1f77a4", width=bar_width)
plt.bar(x, step_4, bottom = step_3, label='Step 4', color='darkorange', width=bar_width)
plt.legend(shadow=False)

plt.xticks(rotation=30)
# plt.xlabel("Model")
plt.ylabel("Time (ms)")
plt.tight_layout()
plt.savefig('scaling_step_timing.pdf', format='pdf', dpi=1000)
plt.show()