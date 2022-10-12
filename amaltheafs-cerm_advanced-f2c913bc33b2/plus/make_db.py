import pandas as pd
import numpy as np

from timer import Timer
from draw_distribution import draw_distribution
from parameters import *

"""
This script adds rows to feed the data.csv file of [parameters, losses] data points
"""

#path of the portfolio

portfolio_path = 'Work/inputs/portfolio1000loansJules.dump'

#input parameters

N = 1000
risk1 = .05
risk2 = .01

n_points = 500

base_alpha = alpha
base_beta = beta
base_gamma = gamma
base_R = R
base_e = e
base_p = p
base_theta = theta

#base parameters to create parameters dataset

base = np.array([[base_alpha, base_beta, base_gamma, base_R, base_e, base_p, base_theta]])
base = np.resize(base, (n_points,5))

noise = np.random.normal(size = (n_points, 5))

test = base + base * noise
test = np.where(test>0, test, 1e-3)

#generation of parameters dictionary

keylist = ["alpha", "beta", "gamma", "R", "e", "p", "theta", "expected_loss", "unexpected_loss1", "unexpected_loss2"]
dic = {key: [] for key in keylist}

for k in range(7):
    dic[keylist[k]] = list(test[:,k])

#generation of new data points

for i in range(n_points):

    #initialization of timer for each iteration

    timer_i = Timer(text = "Simulation "+str(i+1)+" computed in ")
    timer_i.start()

    parameters = test[i]
    alpha, beta, gamma, R, e, p, theta = parameters[:7]
    
    simulation = draw_distribution(portfolio_path, horizon, alpha, beta, gamma, R, e, p, theta, N, risk1, risk2, disp="n")

    #end of timer for iteration

    timer_i.stop()

    #append new losses to dictionary

    for j in range(3):
        dic[keylist[5+j]].append(simulation[j])

#append new data points to database

df = pd.DataFrame.from_dict(dic)
df.to_csv('data.csv', mode='a', index_label = True, header=False)