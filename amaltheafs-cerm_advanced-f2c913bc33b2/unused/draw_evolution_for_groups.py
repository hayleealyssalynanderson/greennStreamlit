from typing import Text
import numpy as np
import matplotlib.pyplot as plt

from LargeCERMEngine import LargeCERMEngine
from ScenarioGenerator import ScenarioGenerator
from Portfolio import load_from_file
from Ratings import TEST_8_RATINGS


def draw_evolution(portfolio_path, horizon, alpha, beta, gamma, H, theta, N, risk1, risk2, disp="y"):
    """draw_distribution computes the LCERM N times to return a histogram of loss distribution, as well as
    the expected loss, and the unexpected losses associated with risks risk1 and risk2"""
    
    scenario = ScenarioGenerator(horizon, alpha, beta, gamma, H, theta)
    scenario.compute()

    portfolio = load_from_file(portfolio_path)

    engine = LargeCERMEngine(portfolio, TEST_8_RATINGS, scenario)
    engine.compute(N)

    ind1 = int(np.floor(N*(1-risk1)))
    ind2 = int(np.floor(N*(1-risk2)))

    losses = engine.loss_results

    nb_groups = losses.shape[1]
    size = int(-np.floor(-np.sqrt(nb_groups)))

    sorted_losses = np.sort(losses, axis=0)

    el, ul1, ul2 = sorted_losses.sum(axis=0)/N, sorted_losses[ind1,:,:], sorted_losses[ind2,:,:]

    if disp=="y":
        plt.figure(figsize=(16, 12))
        for g in range(nb_groups):
            plt.subplot(size, size, g+1)
            plt.plot(el[g,:])
            plt.plot(ul1[g,:])
            plt.plot(ul2[g,:])

        plt.xlabel("loss")
        plt.ylabel("number of occurences")
        plt.title("histogram of loss distribution for given portfolio with "+str(N)+" iterations")
        plt.show()

        plt.legend()