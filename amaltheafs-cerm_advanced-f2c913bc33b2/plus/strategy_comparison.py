import numpy as np
import matplotlib.pyplot as plt

from CERMEngine import CERMEngine
from ScenarioGenerator import ScenarioGenerator
from Portfolio import load_from_file
from Ratings import TEST_8_RATINGS
from timer import Timer

######################################
### COMPARISON W/ AND W/O CLIMATE RISK
######################################

horizon = 10
N = 3000

filename = 'Work/inputs/portfolio1000loansJules.dump'

portfolio = load_from_file(filename)
nb_groups = len(portfolio.groups)
nb_ratings = 8

scenario = ScenarioGenerator(horizon, .02, 1.5, .005, .3, .1)
scenario.compute()

strategy1 = "regular"

strategy2 = np.zeros((nb_groups, nb_ratings, 3))
strategy2[:, :, 0] = np.ones((nb_groups, nb_ratings))
strategy2[:, :, 1] = .5*np.ones((nb_groups, nb_ratings))
strategy2[:, :, 2] = .5*np.ones((nb_groups, nb_ratings))
 
strategy3 = np.zeros((nb_groups, nb_ratings, 3))
strategy3[:, :, 0] = np.ones((nb_groups, nb_ratings))
strategy3[:, :, 1] = 0*np.ones((nb_groups, nb_ratings))
strategy3[:, :, 2] = 0*np.ones((nb_groups, nb_ratings))

engine1 = CERMEngine(portfolio, TEST_8_RATINGS, scenario, sensitivity_objective = strategy1)
engine2 = CERMEngine(portfolio, TEST_8_RATINGS, scenario, sensitivity_objective = strategy2)
engine3 = CERMEngine(portfolio, TEST_8_RATINGS, scenario, sensitivity_objective = strategy3)

engine1.compute(N)
engine2.compute(N)
engine3.compute(N)

portfolio_result1 = engine1.portfolio_result
portfolio_result2 = engine2.portfolio_result
portfolio_result3 = engine3.portfolio_result

losses_distribution1 = portfolio_result1.losses()
losses_distribution2 = portfolio_result2.losses()
losses_distribution3 = portfolio_result3.losses()

risk = 0.05
ind = int(np.floor(N*(1-risk)))

el1 = losses_distribution1.sum(dtype=np.int64)/N
ul1 = np.sort(losses_distribution1, axis=0)[ind]

el2 = losses_distribution2.sum(dtype=np.int64)/N
ul2 = np.sort(losses_distribution2, axis=0)[ind]

el3 = losses_distribution3.sum(dtype=np.int64)/N
ul3 = np.sort(losses_distribution3, axis=0)[ind]

plt.figure(figsize=(16, 12))
plt.title("Comparison of losses w and w/o climate change")
plt.xlabel("Loss")
plt.ylabel("Number of occurrences")
bins=np.histogram(np.hstack((losses_distribution1, losses_distribution2, losses_distribution3)), bins=max(50,N//50))[1]
plt.hist(losses_distribution1, bins, alpha=.5, label="no-effort strategy")
plt.hist(losses_distribution2, bins, alpha=.5, label="some effort")
plt.hist(losses_distribution3, bins, alpha=.5, label="limit-case effort")

plt.axvline(x=el1,color='cyan')
plt.text(el1+20,N/100,"Expected Loss without transition effort",rotation=90)
plt.axvline(x=ul1,color='darkblue')
plt.text(ul1+20,N/100,"Unexpected Loss at risk 5% without transition effort",rotation=90)

plt.axvline(x=el2,color='orange')
plt.text(el2+20,N/50,"Expected Loss with some transition effort",rotation=90)
plt.axvline(x=ul2,color='red')
plt.text(ul2+20,N/50,"Unexpected Loss at risk 5% with some transition effort",rotation=90)

plt.axvline(x=el3,color='grey')
plt.text(el3+20,N/50,"Expected Loss with limit-case transition effort",rotation=90)
plt.axvline(x=ul3,color='green')
plt.text(ul3+20,N/50,"Unexpected Loss at risk 5% with limit-case transition effort",rotation=90)

plt.xlabel("Loss")
plt.ylabel("Number of occurences")
plt.title("Comparison of losses w/ and w/o transition effort for portfolio of structure with "+str(N)+" iterations")

plt.legend()

plt.show()

