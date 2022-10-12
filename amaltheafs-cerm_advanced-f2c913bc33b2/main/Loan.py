from datetime import date


# @title Generation of Loan class
class Loan:
    """
    A class to represent a loan, of which type is every loan in the portfolio input in the CERM.
    
    Attributes
    ----------
    id: uuid
        id of loan
    type: str
        type of loan
    group: str
        name of group of loan
    borrower: str
        name of borrower
    maturity: int
        maturity of loan in years
    principal: float
        principal of loan
    interest_rate: float
        interest rate of loan
    initialRating: str
        initial rating of loan status
    creation_date: date
        date of creation of the loan
    """

    def __init__(self, id, type, group, borrower, maturity, principal, interest_rate, initialRating, creation_date: date = date.today()):
        """
        Constructs all the necessary attributes for the Loan object.
        
        Parameters
        ----------
            id: uuid
                id of loan
            type: str
                type of loan
            group: str
                name of group of loan
            borrower: str
                name of borrower
            maturity: int
                maturity of loan in years
            principal: float
                principal of loan
            interest_rate: float
                interest rate of loan
            initialRating: str
                initial rating of loan status
            creation_date: date
                date of creation of the loan
        """
        self.id = id
        self.type = type
        self.__group = group
        self.borrower = borrower
        self.initialRating = initialRating
        self.loan_type = creation_date
        self.maturity = maturity
        self.maturity_date = creation_date.replace(year=creation_date.year + maturity)
        self.principal = principal
        self.interest_rate = interest_rate
        self.creation_date = creation_date

    def ead(self, t):
        """
        Returns the EAD of the loan at time t.
        
        Parameters
        ----------
        t: int
            time (date) in years
        """
        ir = self.interest_rate
        m = self.maturity
        p = self.principal
        if t <= m:
            if ir==0:
                return p
            else:
                return p * ((1 + ir) ** m - (1 + ir) ** t) / ((1 + ir) ** m - 1)
        else:
            return 0

    def ead_evolution(self, end_time):
        """
        Returns the list of all EAD values taken by the loan.
        
        Parameters
        ----------
        end_time: int
            time (date) in years until which to calculate EAD
        """
        return [self.ead(t) for t in range(end_time)]

    def group(self):
        """
        Returns the group in which the loan is.
        """
        return self.__group

