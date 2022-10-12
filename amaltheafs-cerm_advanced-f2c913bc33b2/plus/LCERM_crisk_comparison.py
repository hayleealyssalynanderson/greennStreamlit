from typing import Text
import numpy as np
import matplotlib.pyplot as plt

from LargeCERMEngine import LargeCERMEngine
from ScenarioGenerator import ScenarioGenerator
from Portfolio import load_from_file
from Ratings import TEST_8_RATINGS

"""This script returns, for a chosen portfolio:
-a comparison between loss distributions with and with climate risk, meaning one under a chosen climate scenario and one where climate risks are not considered
-a comparison between the evolutions of expected, 5%- and 1%-unexpected losses for the same cases"""

#chosen portfolio
portfolio_path = 'Work/inputs/portfolio1000loansJules.dump'

#chosen horizon of time
horizon = 10

#chosen climate coefficients
alpha = .02
beta = 1.5
gamma = .005
H = .3
theta = .1

#number of iterations
N = 20000

#chosen risk values
risk1 = .05
risk2 = .01

ind1 = int(np.floor(N*(1-risk1)))
ind2 = int(np.floor(N*(1-risk2)))

portfolio = load_from_file(portfolio_path)

#without climate risk

scenario_wo = ScenarioGenerator(horizon, 0, 0, 0, 1e-5, 1e-5)
scenario_wo.compute()

engine_wo = LargeCERMEngine(portfolio, TEST_8_RATINGS, scenario_wo)
engine_wo.compute(N)

losses_wo = engine_wo.loss_results

sorted_losses_wo = np.sort(losses_wo, axis=0)

el_wo, ul1_wo, ul2_wo = sorted_losses_wo.sum(axis=(0,1))/N, (sorted_losses_wo.sum(axis=1))[ind1], (sorted_losses_wo.sum(axis=1))[ind2]

draws_wo = np.sort(losses_wo.sum(axis=(1,2)))

for t in range(1,horizon):
        el_wo[t] += el_wo[t-1]
        ul1_wo[t] += ul1_wo[t-1]
        ul2_wo[t] += ul2_wo[t-1]

expected_loss_wo = el_wo[-1]
unexpected_loss1_wo = ul1_wo[-1]
unexpected_loss2_wo = ul2_wo[-1]

#with climate risk

scenario_w = ScenarioGenerator(horizon, alpha, beta, gamma, H, theta)
scenario_w.compute()

engine_w = LargeCERMEngine(portfolio, TEST_8_RATINGS, scenario_w)
engine_w.compute(N)

losses_w = engine_w.loss_results

sorted_losses_w = np.sort(losses_w, axis=0)

el_w, ul1_w, ul2_w = sorted_losses_w.sum(axis=(0,1))/N, (sorted_losses_w.sum(axis=1))[ind1], (sorted_losses_w.sum(axis=1))[ind2]

draws_w = np.sort(losses_w.sum(axis=(1,2)))

for t in range(1,horizon):
        el_w[t] += el_w[t-1]
        ul1_w[t] += ul1_w[t-1]
        ul2_w[t] += ul2_w[t-1]

expected_loss_w = el_w[-1]
unexpected_loss1_w = ul1_w[-1]
unexpected_loss2_w = ul2_w[-1]

#figures

plt.figure(figsize=(16, 12))

bins=np.histogram(np.hstack((draws_wo,draws_w)), bins=max(100,N//50))[1]
plt.hist(draws_wo, bins, alpha=.5, color='skyblue', label="w/o climate change")
plt.hist(draws_w, bins, alpha=.5, color = 'sandybrown', label="w/ climate change")

plt.axvline(x = expected_loss_wo, color='lightblue')
plt.text(expected_loss_wo+20, 20,"expected loss without climate risk",rotation=90)
plt.axvline(x = unexpected_loss1_wo, color='deepskyblue')
plt.text(unexpected_loss1_wo+20,20,"unexpected loss without climate risk at risk "+str(int(100*risk1))+"%",rotation=90)
plt.axvline(x = unexpected_loss2_wo, color='mediumblue')
plt.text(unexpected_loss2_wo+20,20,"unexpected loss without climate risk at risk "+str(int(100*risk2))+"%",rotation=90)

plt.axvline(x = expected_loss_w, color='yellow')
plt.text(expected_loss_w+20,20,"expected loss with climate risk",rotation=90)
plt.axvline(x = unexpected_loss1_w, color='orange')
plt.text(unexpected_loss1_w+20,20,"unexpected loss with climate risk at risk "+str(int(100*risk1))+"%",rotation=90)
plt.axvline(x = unexpected_loss2_w, color='red')
plt.text(unexpected_loss2_w+20,20,"unexpected loss with climate risk at risk "+str(int(100*risk2))+"%",rotation=90)

plt.xlabel("loss")
plt.ylabel("number of occurences")
plt.title("histogram of loss distribution for given portfolio with "+str(N)+" iterations")

plt.legend()
plt.show()

plt.figure(figsize=(16,12))

plt.plot(el_wo, color = 'lightblue', label="expected loss without climate risk")
plt.plot(ul1_wo, color = 'deepskyblue', label="unexpected loss without climate risk at risk "+str(int(100*risk1))+"%")
plt.plot(ul2_wo, color = 'mediumblue', label="unexpected loss without climate risk at risk "+str(int(100*risk2))+"%")

plt.plot(el_w, color = 'yellow', label="expected loss with climate risk")
plt.plot(ul1_w, color = 'orange', label="unexpected loss with climate risk at risk "+str(int(100*risk1))+"%")
plt.plot(ul2_w, color = 'red', label="unexpected loss with climate risk at risk "+str(int(100*risk2))+"%")

plt.xlabel("time")
plt.ylabel("loss")
plt.title("evolution of relevant losses for given portfolio with "+str(N)+" iterations")

plt.legend()
plt.show()