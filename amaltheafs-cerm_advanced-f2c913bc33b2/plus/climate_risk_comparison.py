import numpy as np
import matplotlib.pyplot as plt

from LargeCERMEngine import LargeCERMEngine
from ScenarioGenerator import ScenarioGenerator
from Portfolio import load_from_file
from Ratings import TEST_8_RATINGS
from timer import Timer

######################################
### COMPARISON W/ AND W/O CLIMATE RISK
######################################

horizon = 10
N = 300

filename = 'Work/inputs/portfolio1000loansJules.dump'

portfolio = load_from_file(filename)

scenario_no_cr = ScenarioGenerator(horizon, 0, 0, 0, .0001, .0001)
scenario_cr = ScenarioGenerator(horizon, .06, .5, .005, .22, .33)

scenario_no_cr.compute()
scenario_cr.compute()

engine_no_cr = LargeCERMEngine(portfolio, TEST_8_RATINGS, scenario_no_cr)
engine_cr = LargeCERMEngine(portfolio, TEST_8_RATINGS, scenario_cr)

engine_no_cr.compute(N)
engine_cr.compute(N)

portfolio_result_no_cr = engine_no_cr.portfolio_result
portfolio_result_cr = engine_cr.portfolio_result

losses_distribution_no_cr = portfolio_result_no_cr.losses()
losses_distribution_cr = portfolio_result_cr.losses()

risk = 0.05
ind = int(np.floor(N*(1-risk)))

el_no_cr = losses_distribution_no_cr.sum(dtype=np.int64)/N
ul_no_cr = np.sort(losses_distribution_no_cr, axis=0)[ind]

el_cr = losses_distribution_cr.sum(dtype=np.int64)/N
ul_cr = np.sort(losses_distribution_cr, axis=0)[ind]

plt.figure(figsize=(12, 8))
plt.title("Comparison of losses w and w/o climate change")
plt.xlabel("Loss")
plt.ylabel("Number of occurrences")
bins=np.histogram(np.hstack((losses_distribution_no_cr,losses_distribution_cr)), bins=max(100,N//50))[1]
plt.hist(losses_distribution_no_cr, bins, alpha=.5, label="w/o climate change")
plt.hist(losses_distribution_cr, bins, alpha=.5, label="w/ climate change")

plt.axvline(x=el_no_cr,color='cyan')
plt.text(el_no_cr+20,N/100,"Expected Loss without climate risk",rotation=90)
plt.axvline(x=ul_no_cr,color='darkblue')
plt.text(ul_no_cr+20,N/100,"Unexpected Loss at risk 5% without climate risk",rotation=90)

plt.axvline(x=el_cr,color='orange')
plt.text(el_cr+20,N/50,"Expected Loss with climate risk",rotation=90)
plt.axvline(x=ul_cr,color='red')
plt.text(ul_cr+20,N/50,"Unexpected Loss at risk 5% with climate risk",rotation=90)

plt.xlabel("Loss")
plt.ylabel("Number of occurences")
plt.title("Comparison of losses w/ and w/o climate risk for given portfolio with "+str(N)+" iterations")

plt.legend()

plt.show()

