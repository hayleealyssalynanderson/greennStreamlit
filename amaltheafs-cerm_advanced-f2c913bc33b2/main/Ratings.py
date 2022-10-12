import numpy as np

"""This script defines the Ratings class, used to assess the ratings table used by the bank"""

REG_MATRIX_8 = .01 * np.array([
    [90.81, 8.33, .68, .06, .12, 0, 0, 0],
    [.7, 90.65, 7.79, .64, .06, .14, .02, 0],
    [.09, 2.27, 91.05, 5.52, .74, .26, .01, .06],
    [.02, .33, 5.95, 86.93, 5.3, 1.17, .12, .18],
    [.03, .14, .67, 7.73, 80.53, 8.84, 1.00, 1.06],
    [0, .11, .24, .43, 6.48, 83.46, 4.07, 5.2],
    [.22, 0, .22, 1.3, 2.38, 11.24, 64.86, 19.79],
    [0, 0, 0, 0, 0, 0, 0, 100]
])

RATINGS_8 = [
    "AAA",
    "AA",
    "A",
    "BBB",
    "BB",
    "B+",
    "B",
    "D"  # Default
]


class Ratings:
    """
    A class to represent the ratings of a bank.
    
    ...
    Attributes
    ----------
        ratingsName: str
            name of the ratings table
        ___ratings: array
            ratings table (migration matrix)
    """

    def __init__(self, ratingsName, ratings=RATINGS_8):
        """
        Constructs all the necessary attributes for the Ratings object.
        
        Parameters
        ----------
            ratingsName: str
                name of the ratings table
            ___ratings: list
                list of ratings
        """
        self.name = ratingsName
        self.__ratings = ratings

    def rating_pos(self, rating):
        """
        Returns the index corresponding to input rating in ratings table.
        
        Parameters
        ----------
            rating: str
                rating (letter)
                
        Returns
        -------
            self.__ratings.index(rating): int
                index of input rating in ratings table
        """
        return self.__ratings.index(rating)

    def list(self):
        """
        Lists all possible ratings.
        
        Returns
        -------
            self.__ratings: list
                list of ratings
        """
        return self.__ratings

    def reg_mat(self):
        """
        Returns migration matrix.
        
        Returns
        -------
            [ratings_table]: array
                ratings table (migration matrix)"""
        return REG_MATRIX_8


TEST_8_RATINGS = Ratings("8 ratings with its reg matrix")

