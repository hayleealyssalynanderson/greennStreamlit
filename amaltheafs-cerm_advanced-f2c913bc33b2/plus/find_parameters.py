import numpy as np
import matplotlib.pyplot as plt
import csv

from sklearn.linear_model import Lasso

gdp=[]

with open("Work\world-gdp-growth-rate.csv") as file_name:
    next(file_name)
    reader = csv.reader(file_name, delimiter=',')
    for row in reader:
        gdp.append(float(row[1]))

R = np.mean(gdp)
e = np.var(gdp)

print(R,e)
