import numpy as np


class PortfolioResult:
    def __init__(self, source_portfolio, nb_iteration,horizon):
        self.source_portfolio = source_portfolio
        self.portfolio = []
        self.nb_iteration = nb_iteration
        self.whole_losses = np.zeros((nb_iteration, len(self.portfolio), horizon))

    def append(self, loan):
        self.portfolio.append(loan)

    def loans(self):
        return self.portfolio

    def last_ratings(self):
        last_ratings = []
        for loan in self.portfolio:
            last_ratings.append(loan.rating_evolution[-1])
        return last_ratings

    def losses(self):
        losses = [0] * self.nb_iteration
        for loan in self.portfolio:
            losses = np.add(losses, loan.losses)
        return losses

    def expected_losses(self):
        el = [0] * self.nb_iteration
        for loan in self.portfolio:
            el = np.add(el, loan.expected_losses)
        return el


