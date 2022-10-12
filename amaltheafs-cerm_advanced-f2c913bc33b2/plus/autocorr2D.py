import numpy as np
import scipy.linalg as lin
import matplotlib.pyplot as plt
from ScenarioGenerator import ScenarioGenerator
from MatrixAndThresholdGenerator import correlation_from_covariance
from parameters import *

"""this script draws in 2D the auto- and cross- correlations of physical and transition risks
over a given horizon and given climate scenario for various delay values"""


#scenario generation

scenario = ScenarioGenerator(2*horizon, alpha, beta, gamma, R, e, p, theta)
scenario.compute()

#generation of the incremental matrix

A = np.array([[0, 0, 0], [-scenario.gamma, 1, -scenario.alpha], [0, scenario.beta, 0]])

#initialization of autocorrelations

autocorrelation = np.zeros((horizon, horizon-1, scenario.nb_rf, scenario.nb_rf))
autocorrelation_phy=np.zeros((horizon-1,horizon-1))
autocorrelation_tra=np.zeros((horizon-1,horizon-1))
autocorrelation_phy_tra=np.zeros((horizon-1,horizon))
autocorrelation_tra_phy=np.zeros((horizon-1,horizon))

#initialization of times and delays for which is drawn the graph

times=range(1,horizon)
taus=range(1,horizon)

#execution

for t in times:

    #logging of variance matrix at time t

    var_t=scenario.var_at(t)
    corr=correlation_from_covariance(var_t)

    #logging of simultaneous cross-correlations, i.e. for delay tau=0

    autocorrelation_phy_tra[t-1,0] = corr[2,1]
    autocorrelation_tra_phy[t-1,0] = corr[1,2]

    #execution for each possible delay

    for tau in taus:

        #logging of variance matrix at time t+tau

        var_delay=scenario.var_at(t+tau)

        #logging of inverse [standard deviations (macro-correlations)] at times t and t+tau

        at_time=np.reshape(1/np.sqrt(np.diag(var_t)), (3,1))
        at_delay=np.reshape(1/np.sqrt(np.diag(var_delay)), (3,1))

        #following the formula from the paper

        invsd=(at_delay@at_time.T)
        autocorrelation = invsd*(np.linalg.matrix_power(A,tau)@var_t)

        #logging all auto- and cross-correlations

        autocorrelation_phy[t-1,tau-1] = autocorrelation[1,1]
        autocorrelation_tra[t-1,tau-1] = autocorrelation[2,2]
        autocorrelation_phy_tra[t-1,tau] = autocorrelation[2,1]
        autocorrelation_tra_phy[t-1,tau] = autocorrelation[1,2]

#plotting results

fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(16,12))
fig.suptitle('Evolution of autocorrelations as functions of delay')

ax1.plot(times, autocorrelation_phy, label=["tau = "+str(tau) for tau in taus])
ax1.set_title('physical auto-correlation')
ax1.legend(loc="upper left",prop={"size":7})
ax1.set_ylabel("correlation")

ax2.plot(times, autocorrelation_tra, label=["tau = "+str(tau) for tau in taus])
ax2.set_title('transition auto-correlation')

taus=[0]+list(taus)

ax3.plot(times, autocorrelation_tra_phy, label=["tau = "+str(tau) for tau in taus])
ax3.set_title('physical after transition correlation')
ax3.legend(loc="upper left", prop={"size":7})
ax3.set_ylabel("correlation")
ax3.set_xlabel("time (years)")

ax4.plot(times, autocorrelation_phy_tra, label=["tau = "+str(tau) for tau in taus])
ax4.set_title('transition after physical correlation')
ax4.set_xlabel("time (years)")

plt.show()