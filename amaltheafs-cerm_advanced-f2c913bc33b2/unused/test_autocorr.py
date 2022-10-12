import numpy as np
import scipy.linalg as lin
import matplotlib.pyplot as plt
from ScenarioGenerator import ScenarioGenerator

horizon = 20

scenario = ScenarioGenerator(2*horizon, .02, 1.5, .005, .3, .1)
scenario.compute()

alpha = scenario.alpha
gamma = scenario.gamma
beta = scenario.beta
H = scenario.hhw
theta = scenario.theta

A = np.array([[0, 0, 0], [-gamma, 1, -alpha], [0, beta, 0]])
external = np.array([[1,0,0],[0,H**2,0],[0,0,theta**2]])

#test pour transition à t=5, tau=1

#t=0
var=np.zeros((3,3))
for i in range(1,6):
    var=A@var@A.T+external
var5=var.copy()
var6=A@var@A.T+external

xi5=np.sqrt(np.diag(var5))
xi6=np.sqrt(np.diag(var6))

cov=A@var5

invxi5=np.reshape(1/xi5,(3,1))
invxi6=np.reshape(1/xi6,(3,1))

invsd=invxi6@invxi5.T

autocorr=invsd*cov
