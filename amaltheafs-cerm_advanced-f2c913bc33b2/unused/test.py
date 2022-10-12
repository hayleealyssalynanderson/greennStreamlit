import numpy as np
import matplotlib.pyplot as plt
from ScenarioGenerator import correlation_from_covariance

from CERMEngine import CERMEngine
from ScenarioGenerator import ScenarioGenerator
from Portfolio import load_from_file, show
from Ratings import TEST_8_RATINGS
from heatmap import heatmap, annotate_heatmap
from timer import Timer

horizon = 10

scenario = ScenarioGenerator(horizon, .02, 1.5, .005, .3, .1)

scenario.compute()

alpha = scenario.alpha
gamma = scenario.gamma
beta = scenario.beta
H = scenario.hhw
theta = scenario.theta

A = np.array([[0, 0, 0], [-gamma, 1, -alpha], [0, beta, 0]])
external = np.array([[1,0,0],[0,H**2,0],[0,0,theta**2]])

delta = (alpha**2*theta**2+H**2+gamma**2)/(2*alpha-alpha**2*beta*(1+alpha*beta))

Delta = np.array([[1, 0, 0],[0, (1+alpha*beta)*delta/beta, delta],[0, delta, theta**2+beta*(1+alpha*beta)*delta]])
print(correlation_from_covariance(Delta))