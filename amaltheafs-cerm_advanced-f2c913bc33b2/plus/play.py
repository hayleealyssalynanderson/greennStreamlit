from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt

from ScenarioGenerator import ScenarioGenerator

quantiles = [0,.01, .05, .5,1]

horizon = 10

alpha = .2
beta = 1.5
gamma = .05
R = 10
e = 5
p = 5
theta = .1

scenario = ScenarioGenerator(horizon, alpha, beta, gamma, R, e, p, theta)
scenario.compute()

mean = scenario.expectation[1:,-1]
cov = scenario.var_tilde[-1,1:,1:]

def fmt(x):
    s = 100 * 2*np.pi * np.sqrt(np.linalg.det(cov)) * x
    s = f"{s:.1f}"
    return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"


levels = quantiles / (2*np.pi * np.sqrt(np.linalg.det(cov)))

mean1, mean2 = mean

x = np.linspace(mean1-20, mean1+200, 1000)
y = np.linspace(mean2-20, mean2+200, 1000)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

rv = multivariate_normal(mean, cov)
fig = plt.figure(figsize=(16,12))
contour = plt.contourf(x, y, rv.pdf(pos), levels, colors=("orangered","orange","gold","lightyellow"))

#plt.clabel(contour, levels, inline=True, colors=("gold", "orange", "red"), fmt=fmt)
plt.title("iso-quantiles of climate risks at time "+str(horizon))
plt.xlabel("physical risk")
plt.ylabel("transition risk")
plt.show()