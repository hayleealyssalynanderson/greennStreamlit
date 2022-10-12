from typing import Text
import numpy as np
import matplotlib.pyplot as plt

from LargeCERMEngine import LargeCERMEngine
from ScenarioGenerator import ScenarioGenerator
from Portfolio import load_from_file
from Ratings import TEST_8_RATINGS
from  matplotlib.colors import LinearSegmentedColormap

"""This script defines the draw_distribution function, used in simulator.py"""

def draw_distribution(portfolio_path, horizon, alpha, beta, gamma, R, e, p, theta, N, risk1, risk2, disp="y"):
    """
    Returns a histogram of loss distribution under given parameters, as well as the evolutions of
    the expected loss, and the unexpected losses associated with risks risk1 and risk2.

    Parameters
    ----------
        portfolio_path (string): 
            the path to the file from which to load the portfolio
        horizon (int): 
            the time horizon (in years) of the simulation
        alpha (float): 
            transition efficiency coefficient (reduced)
        beta (float): 
            transition effort reactivity coefficient
        gamma (float): 
            climate change intensity of the economic activity (idiosyncratic)
        R (float): 
            hypothetical climate-free average growth rate of log GDP
        e (float): 
            idiosyncratic economic risk
        p (float): 
            idiosyncratic physical risk
        theta (float): 
            independent transition coefficient
        N (int): 
            number of iterations
        risk1 (float): 
            risk to compute for first unexpected loss (between 0 and 1)
        risk2 (float): 
            risk to compute for second unexpected loss (between 0 and 1)
    """
    #generation of climate scenario

    scenario = ScenarioGenerator(horizon, alpha, beta, gamma, R, e, p, theta)
    scenario.compute()

    #loading of input portfolio

    portfolio = load_from_file(portfolio_path)

    #computation of all losses through LCERM

    engine = LargeCERMEngine(portfolio, TEST_8_RATINGS, scenario)
    engine.compute(N)

    #definition of risk indices matching risk1 and risk2
    
    ind1 = int(np.floor(N*(1-risk1)))
    ind2 = int(np.floor(N*(1-risk2)))

    #logging of all losses

    losses = engine.loss_results

    #logging of final physical and transition cumulative risks (??? should there really be a - (minus) ??? without it we have precisely contrary results to what we want to obtain)

    cumulative_growth_factors = -engine.cumulative_growth_factors[1:, :]

    #logging of final losses for plane distribution

    scenario_losses = losses.sum(axis=(1,2))
    
    #sorting of all losses

    sorted_losses = np.sort(losses, axis=0)

    #logging of non-cumulative expected loss, unexpected loss at risk risk1, unexpected loss at risk risk2 at each time

    el, ul1, ul2 = sorted_losses.sum(axis=(0,1))/N, (sorted_losses.sum(axis=1))[ind1], (sorted_losses.sum(axis=1))[ind2]

    #logging of all final losses or all iterations

    draws = np.sort(losses.sum(axis=(1,2)))

    #computation of cumulative expected loss, unexpected loss at risk risk1, unexpected loss at risk risk2 at each time

    for t in range(1,horizon):
        el[t] += el[t-1]
        ul1[t] += ul1[t-1]
        ul2[t] += ul2[t-1]

    #logging of final cumulative expected loss, unexpected loss at risk risk1, unexpected loss at risk risk2

    expected_loss = el[-1]
    unexpected_loss1 = ul1[-1]
    unexpected_loss2 = ul2[-1]

    #plots if wanted

    if disp=="y":

        #plotting of final loss distribution, along with expected loss and unexpected losses at risks risk1 and risk2

        plt.figure(figsize=(16, 12))
        plt.hist(draws, bins=max(200, N//100), alpha=.7, label="histogram of loss distribution for given portfolio")

        plt.axvline(x = expected_loss, color='pink')
        plt.text(expected_loss+20, N/100,"expected loss",rotation=90)
        plt.axvline(x = unexpected_loss1, color='orange')
        plt.text(unexpected_loss1+20, N/100,"unexpected loss at risk "+str(int(100*risk1))+"%",rotation=90)
        plt.axvline(x = unexpected_loss2, color='red')
        plt.text(unexpected_loss2+20, N/100,"unexpected loss at risk "+str(int(100*risk2))+"%",rotation=90)

        plt.xlabel("loss")
        plt.ylabel("number of occurences")
        plt.title("histogram of loss distribution for given portfolio with "+str(N)+" iterations")

        plt.legend()
        plt.show()

        #plotting of evolution of expected loss and unexpected losses at risks risk1 and risk2

        plt.figure(figsize=(16,12))
        plt.plot(el, label="expected loss")
        plt.plot(ul1, label="unexpected loss at risk "+str(int(100*risk1))+"%")
        plt.plot(ul2, label="unexpected loss at risk "+str(int(100*risk2))+"%")

        plt.xlabel("time")
        plt.ylabel("loss")
        plt.title("evolution of relevant losses for given portfolio with "+str(N)+" iterations")

        plt.legend()
        plt.show()

        #plotting of loss-scenario (physical and transition) distribution

        plt.figure(figsize=(16,12))

        c = ["green", "greenyellow", "yellow", "orange", "darkorange", "red", "darkred"]
        v = [0,.05,.1,.4,.6,.9,1.]
        l = list(zip(v,c))
        cmap=LinearSegmentedColormap.from_list('rg',l, N=256)

        plt.scatter(cumulative_growth_factors[0], cumulative_growth_factors[1], c=scenario_losses, cmap=cmap, label="data point")

        plt.xlabel("cumulative physical risk")
        plt.ylabel("cumulative transition risk")
        plt.title("scenario-loss distribution")
        plt.colorbar()

        plt.show()

    return expected_loss, unexpected_loss1, unexpected_loss2