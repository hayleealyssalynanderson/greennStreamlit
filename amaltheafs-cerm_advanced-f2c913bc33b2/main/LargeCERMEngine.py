import numpy as np

from MatrixAndThresholdGenerator import MatrixAndThresholdGenerator
from timer import Timer
from Ratings import TEST_8_RATINGS

"""This script defines the LCERMEngine class, in which is input a portfolio, a climatic scenario, and a ratings table
 to eventually assess climate charges and other insightful quantities when the portfolio is large enough"""

"""Final losses can be greater than the principal of the portfolio considering that all loans are amortized.
Indeed, amortization makes default a non-absorbing state: when a loan defaults ang goes back to rating i<K, 
we consider that the bank has taken out a loan with same EAD after the former loan had been defaulted on.
This is why the bank can eventually lose more than its portfolio principal. 
Amortization coefficient should be negligible in that regard."""

#loan_profile is the rating profile objective of the bank
#here, it is drawn from S&P 1996

loan_profile=np.array([4,21,60,72,34,20,2,0])/213

class LargeCERMEngine:
    """The CERM, when it is large enough, disregards individual idiosyncratic risks (the LCERM)"""

    def __init__(self, portfolio, ratings, scenarios, LGD=1, sensitivity_objective="regular"):
        """Initializing the LCERM with the portfolio, the ratings for which it is conducted, the climatic coefficients.
        The LGD is chosen constant for all loans for now but will be set to evolve.
        sensitivity_objective describes the transition objective of the bank. """

        self.scenarios = scenarios
        self.portfolio = portfolio
        self.ratings = ratings
        self.LGD = LGD

        #printing portfolio parameters

        print('Initializing CERM for {} loans in {} groups, {} ratings with horizon {}'
              .format(len(portfolio.loans()), len(portfolio.groups), len(ratings.list()), scenarios.horizon))

        #initialization of timer for generation of migration matrices and thresholds

        timer = Timer(text="Matrix and threshold generated")
        timer.start()

        #generation of migration quantities under given climate scenario, ratings, number of groups, loan profile
        
        """One should note that the quantities generated here depend on the climate coefficients only and do not
        need computation of the climate scenario (which is random)"""

        self.matrixGenerator = MatrixAndThresholdGenerator(ratings, portfolio.groups, scenarios, loan_profile, sensitivity_objective)
        self.matrixGenerator.compute()

        #timer ends

        timer.stop()

    def compute(self, nb_iteration):
        """"compute the LCERM nb_iteration times"""

        #initialization of timer for full computation of the LCERM

        timer = Timer(text="CERM Full compute computed for " + str(nb_iteration) + " iterations")
        timer.start()

        #logging of scenario, horizon, number of groups

        scenarios = self.scenarios
        horizon = scenarios.horizon
        nb_groups = len(self.portfolio.groups)

        #initialization of losses
        
        self.loss_results = np.zeros((nb_iteration, nb_groups, horizon))

        #initialization of final cumulative growth factors

        self.cumulative_growth_factors = np.zeros((scenarios.nb_rf, nb_iteration))

        #execution

        for iteration in range(nb_iteration):

            #initialization of timer for each iteration

            timer_it = Timer(text="CERM computation done for iteration " + str(iteration))
            timer_it.start()

            #generation of random climate scenario for each iteration

            scenarios.compute()

            #generation of cumulative growth factors, gdp calculation

            scenarios.gdp_compute(disp = "n")

            #initialization of loss array

            loss = np.zeros((nb_groups, horizon))

            #generation of time-run migration matrices (product of consecutive migration matrices) 
            #dependent on climate scenario (risks, migration matrices)

            self.matrixGenerator.conditional()
            product = self.matrixGenerator.product.copy()

            #logging of the EAD array of all exposures at default of the portfolio for every time, group, and initial rating, under given ratings table

            EAD = self.portfolio.EAD(horizon, TEST_8_RATINGS)

            #loss computation

            for t in range(horizon):
                for g in range(nb_groups):

                    #loss computation for each time and group: no random part because the portfolio is large enough

                    loss[g,t] = EAD[g,t]@product[g,t,:,-1]

            #logging of all losses

            self.loss_results[iteration] = loss.copy()

            #logging of final cumulative growth factors

            self.cumulative_growth_factors[:, iteration] = scenarios.cumulative_growth_factors[:, -1]

            #iteration timer stops

            timer_it.stop()

        #computation timer stops

        timer.stop()
