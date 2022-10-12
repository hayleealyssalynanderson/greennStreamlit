import matplotlib.pyplot as plt
import numpy as np

from MatrixAndThresholdGenerator import MatrixAndThresholdGenerator
from LoanResult import LoanResult
from PortfolioResult import PortfolioResult
from timer import Timer

"""This script defines the CERMEngine class, in which is input a portfolio, a climatic scenario, and a ratings table
 to eventually assess climate charges and other insightful quantities"""

#loan_profile is the rating profile objective of the bank: here, it is drawn from S&P 1996

loan_profile=np.array([4,21,60,72,34,20,2,0])/213

#definition of the CERMEngine class

class CERMEngine:

    def __init__(self, portfolio, ratings, scenarios, LGD=1, sensitivity_objective="regular"):
        self.scenarios = scenarios
        self.portfolio = portfolio
        self.ratings = ratings
        self.LGD = LGD

        #printing portfolio parameters

        print('Initializing CERM for {} loans in {} groups, {} ratings with horizon {}'.format(len(portfolio.loans()), len(portfolio.groups), len(ratings.list()), scenarios.horizon))

        #initialization of timer for generation of migration matrices and thresholds

        timer = Timer(text="Matrix and threshold generated")
        timer.start()

        #generation of migration matrices and thresholds under given climate scenario

        self.matrixGenerator = MatrixAndThresholdGenerator(ratings, portfolio.groups, scenarios, loan_profile, sensitivity_objective)
        self.matrixGenerator.compute()

        #end of timer

        timer.stop()

        self.portfolio_result = None

    def compute(self, nb_iteration):

        #initialization of timer for computation of CERM

        timer = Timer(text="CERM Full compute computed for " + str(nb_iteration) + " iterations")
        timer.start()
        scenarios = self.scenarios
        nb_ratings = len(self.ratings.list())
        rating_at_default = nb_ratings

        self.portfolio_result = PortfolioResult(self.portfolio, nb_iteration, scenarios.horizon)

        for iteration in range(nb_iteration):
            timer_it = Timer(text="CERM computation done for iteration " + str(iteration))
            timer_it.start()
            scenarios.compute()
            self.matrixGenerator.init()
            self.matrixGenerator.compute()
            for i in range(len(self.portfolio.loans())):
                loan = self.portfolio.loans()[i]
                group = self.portfolio.group_index(loan.group())

                if iteration == 0:
                    loan_results = LoanResult(loan)
                    loan_results.nb_iter = nb_iteration
                    self.portfolio_result.append(loan_results)
                else:
                    loan_results = self.portfolio_result.loans()[i]

                loan_results.rating_evolution = [self.ratings.rating_pos(loan.initialRating)]
                loan_results.loss_evolution = []
                el = 0
                mm_x_mm = np.eye(nb_ratings-1)
                loan_at_default = False
                loss = 0
                t_at_default = -1
                for t in range(scenarios.horizon):
                    # migration_matrix = self.matrixGenerator.migration_matrices[t, group]

                    last_loan_rating = loan_results.rating_evolution[-1]

                    new_rating = last_loan_rating
                    # Just add Loss if rating is not at default ???
                    if len(loan_results.rating_evolution) == t + 1:
                        # Si rating = "default" = last rating , compute loss
                        if last_loan_rating == rating_at_default:
                            # loan.loss.append(loan.EAD(self.time)*(1-loan.recovery_rate[-1]))
                            # loan_results.losses.append(loan.ead(t)*self.LGD)
                            # loan_results.losses.append(loan.principal*self.LGD)
                            loss = loan.ead(t)*self.LGD
                            loan_results.loss_evolution.append(loss)
                            t_at_default = t
                            loan_at_default = True

                    # Si pas "at default" compute new rating, and set loss at t = 0
                        else:
                            # probability_of_default = migration_matrix[last_loan_rating - 1, nb_ratings - 1]
                            # loan_results.probabilities_of_default.append(probability_of_default)

                            # b=self.parameters.collateral_factor_loadings[g,r-1]
                            # eps_tilde=np.random.normal()
                            # RR=norm.cdf(b.T@Z+np.sqrt(1-b.T@(self.parameters.correlation_matrix@b))*eps_tilde)[0]

                            new_rating = self.compute_rating_at_t(scenarios, t, group, nb_ratings, last_loan_rating)
                            loan_results.rating_evolution.append(new_rating)
                            loan_results.loss_evolution.append(0)

                    el_result = self.compute_expected_losses(mm_x_mm, loan, t, group, nb_ratings)
                    mm_x_mm = el_result[0]
                    el += el_result[1]

                    if loan_at_default:
                        break

                #self.portfolio_result.whole_losses[iteration,i,:]=np.array(loan_results.loss_evolution)

                loan_results.losses.append(loss)
                loan_results.t_at_default.append(t_at_default)
                loan_results.expected_losses.append(el)
            timer_it.stop()
        timer.stop()

    def compute_rating_at_t(self, scenarios, t, group, nb_ratings, last_loan_rating):
        a = self.matrixGenerator.factor_loadings[t, group, last_loan_rating - 1]
        eps = np.random.normal()
        x = a@scenarios.risks_at(t) + np.sqrt(1 - a@scenarios.rf_correlation_at(t)@a.T)*eps
        thresholds = self.matrixGenerator.thresholds[t, group, last_loan_rating - 1]

        rating_at_t = 0
        while rating_at_t <= nb_ratings - 1 and thresholds[rating_at_t] > x:
            rating_at_t += 1

        return rating_at_t

    def compute_expected_losses(self, mm_x_mm, loan, t, group, nb_ratings):
        mm = self.matrixGenerator.migration_matrices
        K = nb_ratings
        if t == 0:
            mm_min1 = mm[0, group]
        else:
            mm_min1 = mm[t-1, group]

        mm_t = mm[t, group]
        mm_x_mm = mm_x_mm * mm_min1[:K-1, :K-1]
        el = (mm_x_mm*mm_t[:K-1, K-1]).sum()*self.LGD*loan.ead(t)
        return [mm_x_mm, el]
