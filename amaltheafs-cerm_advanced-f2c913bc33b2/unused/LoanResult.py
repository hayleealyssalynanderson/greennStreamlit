from datetime import date
from Loan import Loan


# @title Generation of Loan class
class LoanResult(Loan):
    """to keep track of the evolution of the loan over one simulation"""

    def __init__(self, loan):
        self.loan = loan
        # Intermediary Result
        self.rating_evolution = []
        self.loss_evolution = []
        # Output
        # Loss at default, time at default and expected loss for each iteration
        self.t_at_default = []
        self.losses = []
        self.expected_losses = []

    def date_at_default(self, it):
        start_date = date.today()
        default_date = date(start_date.year + self.t_at_default[it], start_date.month, start_date.day)
        return default_date

