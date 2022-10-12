import numpy as np
from regression import reg_poly, reg_lasso
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

alpha = np.linspace(0, .03, 100)
beta = np.linspace(0,2.0,100)
gamma = np.linspace(0,.01,100)
H = np.linspace(.001,.5,100)
theta = np.linspace(.001,3,100)
coeffs = np.array([alpha, beta, gamma, H, theta]).T

fig = plt.figure(figsize=(16,12))
ax = plt.axes(projection="3d")

poly = PolynomialFeatures(degree=2)
array_poly = poly.fit_transform(coeffs)

prediction = reg_poly.predict(array_poly)

x_poly, y_poly, z_poly = [list(prediction[:,k]) for k in range(3)]

with open("Work/data.csv") as file_name:
    next(file_name)
    array = np.loadtxt(file_name, delimiter=",")

array = np.delete(array, 0, 1)

x_line, y_line, z_line = [list(array[:,5+k]) for k in range(3)]

ax.scatter(x_line, y_line, z_line, 'gray')

ax.plot(x_poly, y_poly, z_poly, 'gray')

ax.set_xlabel('expected loss')
ax.set_ylabel("5"+"%"+" unexpected loss")
ax.set_zlabel("1"+"%"+" unexpected loss")

ax.set_title("scatter plot of data points")

plt.show()