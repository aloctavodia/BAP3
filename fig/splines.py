import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from formulae import design_matrices


from cycler import cycler

default_cycler = cycler(color=["#000000", "#6a6a6a", "#bebebe", "#2a2eec"])
plt.rc("axes", prop_cycle=default_cycler)
plt.rc("figure", dpi=300)

def splines(knots, x_true=None, y_true=None):
    # Get x-y values from true function
    if x_true is None:
        x_min, x_max = 0, 6
        x_true = np.linspace(x_min, x_max, 500)
    else:
        x_min, x_max = min(x_true), max(x_true)  
    
    if y_true is None:
        y_true = np.sin(x_true)

    # Prepare figure
    _, axes = plt.subplots(1, 3, figsize=(12, 4),
                         constrained_layout=True,
                         sharex=True, sharey=True)
    axes = np.ravel(axes)

    # Plot the knots and true function
    for ax in axes:
        ax.vlines(knots, -1, 1, color='grey', ls=':')
        ax.plot(x_true, y_true, 'C2--', lw=4, alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])

    labels = ["Constant", "Linear", "Quadratic"]
    for order, ax, label in zip(range(0, 4), axes, labels):
        B = basis(x_true, order, knots)
        y_hat = ols(B, y_true)
        ax.plot(x_true, y_hat, 'k.')
        ax.set_title(label)

def basis(x_true, order, knots):
    """compute basis functions"""
    B = []
    for i in range(0, order+1):
        B.append(x_true**i)
    for k in knots:
        B.append(np.where(x_true < k, 0, (x_true - k) ** order))
    B = np.array(B).T
    return B


def ols(X, y):
    """Compute ordinary least squares in closed-form"""
    β = np.linalg.solve(X.T @ X, X.T @ y)
    return X @ β

splines([1.67, 4.17])
plt.savefig("piecewise.png")

####################################################################

np.random.seed(123)

cmap = mpl.colormaps['gray']

x = pd.DataFrame({"x":np.linspace(0., 1., 500)})
knots = [0.25, 0.5, 0.75]

B0 = design_matrices("bs(x, knots=knots, degree=0)", data=x)
B1 = design_matrices("bs(x, knots=knots, degree=1)", data=x)
B3 = design_matrices("bs(x, knots=knots, degree=3, intercept=True)", data=x)

_, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey='row')
for idx, (B, title) in enumerate(zip((B1, B3),
                                     ("Piecewise linear",
                                      "Cubic spline"))):
    B = np.array(B.common)[:,1:]
    colors = cmap(np.linspace(0, 0.85, B.shape[1]))
    # plot spline basis functions
    for i in range(B.shape[1]):
        axes[0, idx].plot(x, B[:,i],
                          color=colors[i], 
                          lw=2, ls="--")
    # we generate some positive random coefficients 
    # there is nothing wrong with negative values
    β = np.abs(np.random.normal(0, 1, size=B.shape[1]))
    # plot spline basis functions scaled by its β
    for i in range(B.shape[1]):
        axes[1, idx].plot(x, B[:,i]*β[i],
                          color=colors[i],
                          lw=2, ls="--")
    # plot the sum of the basis functions
    axes[1, idx].plot(x, np.dot(B, β), color='k', lw=3)
    # plot the knots
    axes[0, idx].plot(knots, np.zeros_like(knots), "ko")
    axes[1, idx].plot(knots, np.zeros_like(knots), "ko")
    axes[0, idx].set_title(title)

plt.savefig('splines_weighted.png', bbox_inches='tight')
