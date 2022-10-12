import numpy as np
import scipy.linalg as lin
import matplotlib.pyplot as plt
from ScenarioGenerator import ScenarioGenerator

"""this script draws in 3D the auto- and cross- correlations of physical and transition risks
over a given horizon and given climate scenario for various delay values"""

#input parameters

horizon = 20
alpha = .02
beta = 1.5
gamma = .005
H = .3
theta = .1

#scenario generation

scenario = ScenarioGenerator(2*horizon, .02, 1.5, .005, .3, .1)
scenario.compute()

#generation of the incremental matrix

A = np.array([[0, 0, 0], [-scenario.gamma, 1, -scenario.alpha], [0, scenario.beta, 0]])

#initialization of times, delays, and autocorrelations

times=[]
taus=[]
autocorrelation_phy=[]
autocorrelation_tra=[]

#execution

for t in range(1,horizon):    
    for tau in range(1,horizon-t-1):
        times.append(t)
        taus.append(tau)
        autocorrelation = lin.inv(lin.sqrtm(scenario.var_at(t+tau)))@np.linalg.matrix_power(A,tau)@lin.sqrtm(scenario.var_at(t))
        autocorrelation_phy.append(autocorrelation[1,1])
        autocorrelation_tra.append(autocorrelation[2,2])


fig = plt.figure(figsize=plt.figaspect(0.5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.azim = -30
ax1.dist = 10
ax1.elev = 10
ax1.set_xlabel('t')
ax1.set_ylabel('tau')
ax1.set_zlabel('correlation')
ax1.scatter(times, taus, autocorrelation_phy, 'green', label = "physical auto-correlation between t+tau and t")
ax1.legend()

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.azim = -30
ax2.dist = 10
ax2.elev = 10
ax2.set_xlabel('t')
ax2.set_ylabel('tau')
ax2.set_zlabel('correlation')
ax2.scatter(times, taus, autocorrelation_tra, 'red', label = "transition auto-correlation between t+tau and t")
ax2.legend()

plt.show()