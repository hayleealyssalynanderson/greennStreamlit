import numpy as np
import matplotlib.pyplot as plt

from new_climate_model import NewClimateModel

"""draws the evolution of risk intensities over a given horizon and given climate coefficients"""

horizon = 10

alpha = .02
beta = 1.5
gamma = .005
R = .0369
e = .0326
p = .01
theta = .01

scenario = NewClimateModel(horizon, alpha, beta, gamma, R, e, p, theta)

scenario.compute()

macros=scenario.macro_correlation.T

gdp = scenario.gdp[0]

mu = np.resize(scenario.mu, horizon)
ssigma = scenario.ssigma

esp_gdp = mu + ssigma/2
med = mu
var_gdp = (np.exp(ssigma)-1)*np.exp(2*mu + ssigma)

#plt.figure(figsize=(16,12))
#plt.plot(range(1,horizon),macros[1:,:], label=["economic","physical","transition"])
#plt.title("Evolution of macro-correlations under calibrated scenario over a "+str(horizon)+"-year horizon")
#plt.ylabel("value (in economic risk units)")
#plt.xlabel("time (years)")
#plt.legend()
#plt.show()

plt.figure(figsize=(16,12))
plt.plot(gdp)
plt.title("Evolution of gdp under calibrated scenario over a "+str(horizon)+"-year horizon")
plt.ylabel("gdp")
plt.xlabel("time (years)")
plt.legend()
plt.show()