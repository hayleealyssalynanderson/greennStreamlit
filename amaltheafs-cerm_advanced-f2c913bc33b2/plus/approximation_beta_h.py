import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from sklearn.preprocessing import PolynomialFeatures
from regression import reg_new_poly, precision

with open("Work/data.csv") as file_name:
    next(file_name)
    array = np.loadtxt(file_name, delimiter=",")

array = np.delete(array, 0, 1)

beta, H, el, ul1, ul2 = [list(array[:,k]) for k in [1,3,5,6,7]]

betas = np.linspace(0,5,1000)
Hs = np.linspace(0,1,1000)
betas, Hs = np.meshgrid(betas, Hs)

betas_resized = np.resize(betas,1000000)
Hs_resized = np.resize(Hs, 1000000)

grid = np.array([betas_resized, Hs_resized]).T

poly = PolynomialFeatures(degree=2)

grid_new = poly.fit_transform(grid)

prediction = reg_new_poly.predict(grid_new)

prediction_el, prediction_ul1, prediction_ul2 = [prediction[:,i] for i in range(3)]

prediction_el = np.resize(prediction_el, (1000,1000))
prediction_ul1 = np.resize(prediction_ul1, (1000,1000))
prediction_ul2 = np.resize(prediction_ul2, (1000,1000))

#for expected loss

fig = plt.figure(figsize=(16,12))
ax = plt.axes(projection="3d")

surf = ax.plot_surface(betas, Hs, prediction_el, cmap=cm.coolwarm, linewidth=0, antialiased=False, label = "regression prediction")
surf._facecolors2d=surf._facecolor3d
surf._edgecolors2d=surf._edgecolor3d
ax.scatter(beta, H, el, label="data point")

ax.set_xlabel('beta')
ax.set_ylabel("H")
ax.set_zlabel("expected loss")
ax.set_title("prediction of expected loss (precision = "+str(precision)[:5]+")")
ax.legend()

plt.show()

#for unexpected loss at risk1

fig = plt.figure(figsize=(16,12))
ax = plt.axes(projection="3d")

surf = ax.plot_surface(betas, Hs, prediction_ul1, cmap=cm.coolwarm, linewidth=0, antialiased=False, label = "regression prediction")
surf._facecolors2d=surf._facecolor3d
surf._edgecolors2d=surf._edgecolor3d
ax.scatter(beta, H, ul1, label="data point")

ax.set_xlabel('beta')
ax.set_ylabel("H")
ax.set_zlabel("unexpected loss at risk 5%")
ax.set_title("prediction of unexpected loss at risk 5% (precision = "+str(precision)[:5]+")")
ax.legend()

plt.show()

#for unexpected loss at risk2

fig = plt.figure(figsize=(16,12))
ax = plt.axes(projection="3d")

surf = ax.plot_surface(betas, Hs, prediction_ul2, cmap=cm.coolwarm, linewidth=0, antialiased=False, label = "regression prediction")
surf._facecolors2d=surf._facecolor3d
surf._edgecolors2d=surf._edgecolor3d
ax.scatter(beta, H, ul2, label="data point")

ax.set_xlabel('beta')
ax.set_ylabel("H")
ax.set_zlabel("unexpected loss at risk 1%")
ax.set_title("prediction of unexpected loss at risk 1% (precision = "+str(precision)[:5]+")")
ax.legend()

plt.show()
