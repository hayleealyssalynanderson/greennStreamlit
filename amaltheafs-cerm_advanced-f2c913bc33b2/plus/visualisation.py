import numpy as np
import matplotlib.pyplot as plt
from regression import poly_prediction, lasso_prediction

fig = plt.figure(figsize=(16,12))
ax = plt.axes(projection="3d")

with open("Work/data.csv") as file_name:
    next(file_name)
    array = np.loadtxt(file_name, delimiter=",")

array = np.delete(array, 0, 1)

x_line, y_line, z_line = [list(array[:,5+k]) for k in range(3)]

ax.set_xlabel('expected loss')
ax.set_ylabel("5"+"%"+" unexpected loss")
ax.set_zlabel("1"+"%"+" unexpected loss")

ax.scatter(x_line, y_line, z_line, 'gray')

x_poly, y_poly, z_poly = [list(poly_prediction[:,k]) for k in range(3)]

x_lasso, y_lasso, z_lasso = [list(lasso_prediction[:,k]) for k in range(3)]

ax.scatter(x_lasso, y_lasso, z_lasso, color='green')

ax.scatter(x_poly, y_poly, z_poly, color='red')

ax.set_title("scatter plot of data points")

plt.show()