import numpy as np
import matplotlib.pyplot as plt

from CERMEngine import CERMEngine
from ScenarioGenerator import ScenarioGenerator
from Portfolio import load_from_file
from Ratings import TEST_8_RATINGS
from timer import Timer

horizon = 10
N = 100
risk=.05
ind=int(np.floor(N*(1-risk)))

filename = 'bitbucket/CERM/inputs/portfolio100loans.dump'

portfolio = load_from_file(filename)
scenario_cr = ScenarioGenerator(horizon, .02, 1.5, .005, .3, .1)
scenario_cr.compute()
engine_cr = CERMEngine(portfolio, TEST_8_RATINGS, scenario_cr)

nb_groups=engine_cr.matrixGenerator.nb_groups
nb_ratings=engine_cr.matrixGenerator.nb_ratings
nb_rf=scenario_cr.nb_rf
micro_correlation=engine_cr.matrixGenerator.micro_correlation

standard_micro_correlation=np.zeros((nb_groups, nb_ratings, nb_rf))
standard_micro_correlation[:,:,0]=np.ones((nb_groups, nb_ratings))

portfolio_losses=np.zeros((N, horizon, nb_rf))

for i in range(0, nb_rf):
    micro_correlation=standard_micro_correlation.copy()
    micro_correlation[:,:,i]=np.ones((nb_groups, nb_ratings))
    engine_cr.compute(N)
    portfolio_result = engine_cr.portfolio_result
    loss_evolution=np.zeros((horizon))
    #for the following loops, need to collect all evolutions 
    #of all iterations for each portfolio: each portfolioresult
    #should have a N*T array attribute of losses by iteration and time
    #that we conveniently name portfolio_loss_evolution here
    for loan in portfolio_result.loans():
        loss_evolution=loan.loss_evolution
        loss_evolution+=np.array(loss_evolution)
    portfolio_loss_evolution=np.zeros((N,horizon))
    #portfolio_loss_evolution=...
    portfolio_losses[:,:,i]=portfolio_loss_evolution


#economic portfolio
eco=portfolio_losses[:,:0].copy()
el_eco=eco.sum(axis=0)/N
ul_eco=np.sort(eco)[ind]

#physical portfolio
phy=portfolio_losses[:,:1]
el_phy=phy.sum(axis=0)/N
ul_phy=np.sort(phy)[ind]

#transition portfolio
tra=portfolio_losses[:,:2]
el_tra=tra.sum(axis=0)/N
ul_tra=np.sort(tra)[ind]

#the following

plt.figure(figsize=(16,12))

fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(16,12))
ax1.plot(ul_eco,label=".05 risk - unexpected loss for economic portfolio",color="darkblue")
ax1.plot(el_eco,label="expected loss for economic portfolio",color="blue")
ax1.legend(loc="upper left")
ax1.set_title('economic exposure only')
ax1.set_xlabel('time')
ax1.set_ylabel('loss')

ax2.plot(ul_phy,label=".05 risk - unexpected loss for physical portfolio",color="darkgreen")
ax2.plot(el_phy,label="expected loss for physical portfolio",color="green")
ax2.legend(loc="upper left")
ax2.set_title('physical exposure only')
ax2.set_xlabel('time')
ax2.set_ylabel('loss')

ax3.plot(ul_tra,label=".05 risk - unexpected loss for transition portfolio",color="darkred")
ax3.plot(el_tra,label="expected loss for transition portfolio",color="red")
ax3.legend(loc="upper left")
ax3.set_title('transition exposure only')
ax3.set_xlabel('time')
ax3.set_ylabel('loss')

ax4.plot(ul_eco,label="5% risk - unexpected loss for economic portfolio",color="darkblue")
ax4.plot(el_eco,label="expected loss for economic portfolio",color="blue")
ax4.plot(ul_phy,label="5% risk - unexpected loss for physical portfolio",color="darkgreen")
ax4.plot(el_phy,label="expected loss for physical portfolio",color="green")
ax4.plot(ul_tra,label="5% risk - unexpected loss for transition portfolio",color="darkred")
ax4.plot(el_tra,label="expected loss for transition portfolio",color="red")
ax4.legend(loc="upper left")
ax4.set_title('altogether')
ax4.set_xlabel('time')
ax4.set_ylabel('loss')

plt.title("Comparison of climate metrics for different climate exposures")

plt.legend()
plt.savefig("Comparison of climate metrics for different climate exposures")
plt.show()