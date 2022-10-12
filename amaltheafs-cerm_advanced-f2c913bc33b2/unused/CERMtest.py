import numpy as np
import matplotlib.pyplot as plt

from CERMEngine import CERMEngine
from ScenarioGenerator import ScenarioGenerator
from Portfolio import load_from_file, show
from Ratings import TEST_8_RATINGS
from heatmap import heatmap, annotate_heatmap
from timer import Timer


horizon = 20
N = 10

filename = 'Work/inputs/portfolio1000loansJules.dump'

portfolio = load_from_file(filename)

scenario_cr = ScenarioGenerator(horizon, .02, 1.5, .005, .3, .4)

scenario_cr.compute()

engine_cr = CERMEngine(portfolio, TEST_8_RATINGS, scenario_cr)

engine_cr.compute(N)

portfolio_result_cr = engine_cr.portfolio_result

losses_distribution_cr = portfolio_result_cr.losses()

risk = 0.05
ind = int(np.floor(N*(1-risk)))

el_cr = losses_distribution_cr.sum(dtype=np.int64)/N
ul_cr = np.sort(losses_distribution_cr, axis=0)[ind]

plt.figure(figsize=(12, 8))
plt.title("Comparison of losses w and w/o climate change")
plt.xlabel("Loss")
plt.ylabel("Number of occurrences")
plt.hist(losses_distribution_cr, bins=50, alpha=.5, label="w/ climate change")

plt.axvline(x=el_cr,color='orange')
plt.text(el_cr+20,N/50,"Expected Loss with climate risk",rotation=90)
plt.axvline(x=ul_cr,color='red')
plt.text(ul_cr+20,N/50,"Unexpected Loss at risk 5% with climate risk",rotation=90)


plt.xlabel("Loss")
plt.ylabel("Number of occurences")
plt.title("Comparison of losses w/ and w/o climate risk for given portfolio with "+str(N)+" iterations")

plt.legend()

plt.show()



























