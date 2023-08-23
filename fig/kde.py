import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import arviz as az

az.style.use("arviz-grayscale")
from cycler import cycler
default_cycler = cycler(color=["#000000", "#6a6a6a", "#bebebe", "#2a2eec"])
plt.rc('axes', prop_cycle=default_cycler)
plt.rc('figure', dpi=300)

_, ax = plt.subplots(figsize=(12, 4))
bw = 0.4
np.random.seed(42)
y = np.random.normal(8, size=8)
x = np.linspace(y.min() - bw * 3, y.max() + bw * 3, 100)
kernels = np.transpose([stats.norm.pdf(x, i, bw) for i in y])
ax.plot(x, kernels, 'k--', alpha=0.5)
ax.plot(y, np.zeros(len(y)), 'C1o')
ax.plot(x, kernels.sum(1))

ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor('w')
plt.savefig('KDE_example.png')
