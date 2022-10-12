from typing import Text
import numpy as np
import matplotlib.pyplot as plt

from LargeCERMEngine import LargeCERMEngine
from ScenarioGenerator import ScenarioGenerator
from Portfolio import load_from_file
from Ratings import TEST_8_RATINGS

portfolio_path = 'Work/inputs/portfolio1000loansJules.dump'

horizon = 10

alpha = .02
beta = 1.5
gamma = .005
H = .3
theta = .1

N = 100

risk1 = .05
risk2 = .01

scenario = ScenarioGenerator(horizon, alpha, beta, gamma, H, theta)
scenario.compute()

portfolio = load_from_file(portfolio_path)

engine = LargeCERMEngine(portfolio, TEST_8_RATINGS, scenario)
engine.compute(N)

ind1 = int(np.floor(N*(1-risk1)))
ind2 = int(np.floor(N*(1-risk2)))

losses = engine.loss_results

cumulative_losses = losses.copy()
for i in range(1,horizon):
    cumulative_losses[:,:,i] += cumulative_losses[:,:,i-1]

sorted_losses = np.sort(cumulativelosses, axis=0)
