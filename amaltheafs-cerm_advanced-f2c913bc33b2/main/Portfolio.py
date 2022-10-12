import pickle
import numpy as np
from timer import Timer

"""This script defines the Portfolio class"""

class Portfolio:
    """
    A class to represent a portfolio, to input in the CERM.
    
    Attributes
    ----------
    name : str
        name given to the portfolio
    groups : str
        list of groups
    portfolio : Loan
        list of loans    
    """

    def __init__(self, portfolioName):
        """
        Constructs all the necessary attributes for the Portfolio object.
        
        Parameters
        ----------
            portfolioName : str
                name given to the portfolio 
        """
        self.name = portfolioName
        self.groups = []
        self.portfolio = []

    def add_loan(self, loan):
        """
        Adds a loan to the portfolio.
        
        Parameters
        ----------
            loan : Loan
                loan of Loan type to append to the portfolio in matching group
        """
        group = loan.group()
        if group not in self.groups:
            self.groups.append(group)
        self.portfolio.append(loan)

    def loans(self):
        """
        Returns all the loans in the portfolio.
        """
        return self.portfolio

    def group_index(self, group):
        """
        Returns the index in the list self.groups.

        Parameters
        ----------
            group : str
                name of the group
        """
        return self.groups.index(group)

    def grouped_portfolio(self):
        """
        Returns the list of loans, listed by group.
        """
        grouped_portfolio = []
        for g in self.groups:
            grouped_portfolio.append([])
        for loan in self.portfolio:
            grouped_portfolio[self.group_index(loan.group())].append(loan)
        return grouped_portfolio

    def EAD(self, horizon, ratings):
        """
        Returns the EAD array necessary to vectorize loss computation.

        Parameters
        ----------
            horizon : int
                time horizon
            ratings : Ratings
                ratings chosen in the study
        """
        grouped_portfolio = self.grouped_portfolio()
        nb_ratings = len(ratings.list())
        EAD = np.zeros((len(grouped_portfolio), horizon, nb_ratings))
        for loan in self.portfolio:
            EAD[self.group_index(loan.group()), :, ratings.rating_pos(loan.initialRating)] += loan.ead_evolution(horizon)
        return EAD

def dump_to_file(portfolio, fileName):
    """
    Transforms a portfolio dump file into Portfolio type.
    
    Parameters
    ----------
    
        filename : str
            path of the file
    """
    with open(fileName, 'wb') as outp:
        pickle.dump(portfolio, outp, pickle.HIGHEST_PROTOCOL)


def load_from_file(fileName):
    """
    Loads portfolio file with input portfolio path.
    
    Parameters
    ----------
        fileName: str
            portfolio path
    """
    t = Timer(text="Portfolio loaded")
    t.start()
    with open(fileName, 'rb') as inp:
        portfolio = pickle.load(inp)
    t.stop()
    return portfolio

def show(Portfolio):
    """
    Returns the portfolio in a dictionary way to facilitate interpretation.
    
    Parameters
    ----------
        Portfolio: Portfolio
            portfolio of type Portfolio
    """
    group_names = Portfolio.groups
    dic = dict.fromkeys(group_names)
    for key in list(dic.keys()):
        dic[key] = np.zeros((0,4))
    for loan in Portfolio.portfolio:
        info = np.array([loan.principal, loan.maturity, loan.interest_rate, loan.initialRating])
        dic[loan._Loan__group] = np.vstack((dic[loan._Loan__group],info))
    return dic
