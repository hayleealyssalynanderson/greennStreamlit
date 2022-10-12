import matplotlib.pyplot as plt
from ScenarioGenerator import ScenarioGenerator
from parameters import *

"""This script draws the evolution of risk intensities over a given horizon and given climate coefficients, along with their limits"""

#scenario generation

scenario = ScenarioGenerator(horizon, alpha, beta, gamma, R, e, p, theta)
scenario.compute()

#logging of all macro-correlations evolutions

macros=scenario.macro_correlation.T

#plotting

plt.figure(figsize=(16,12))
plt.plot(range(1,horizon),macros[1:,:], label=["economic","physical","transition"])
plt.title("Evolution of macro-correlations under calibrated scenario over a "+str(horizon)+"-year horizon")
plt.ylabel("value (in economic risk units)")
plt.xlabel("time (years)")
plt.legend()
plt.show()