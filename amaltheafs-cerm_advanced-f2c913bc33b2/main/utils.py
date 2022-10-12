import numpy as np
from scipy.stats import norm

"""This script lists some functions that are useful for more important other scripts"""

def correlation(PD):
    """
    The correlation function as given in the original paper
    
    Parameters
    ----------
        PD: float (between 0 and 1)
            probability of default
    """

    return .12*(1-np.exp(-50*PD))/(1-np.exp(-50))+.24*(1-(1-np.exp(-50*PD))/(1-np.exp(-50)))

def correlation_from_covariance(covariance):
    """Gives the correlation matrix associated to an input covariance matrix
    
    Parameters
    ----------
        covariance: SPD matrix
            covariance matrix
    """
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    _correlation = covariance / outer_v
    _correlation[covariance == 0] = 0
    return _correlation

def migmat_to_thresh(M):
    """Transforms a migration matrix into migration thresholds
    
    Parameters
    ----------
        M: array
            array of migration probabilities
    """
    L = len(M)
    z = np.empty((L, L))
    for i in range(L):
        for j in range(L):
            z[i, j] = norm.ppf(min(sum(M[i, j:L]), 1))
    return z

def thresh_to_migmat(Z):
    """
    Transforms matrices of migration thresholds into migration matrices
    Optimized when Z is of shape (horizon, nb_groups, matrix.shape = (nb_ratings, nb_ratings))
    
    Parameters
    ----------
        Z: array
            array of migration thresholds
    """

    #to optimize calculations, the formula (28) is linearized here

    C = norm.cdf(Z)
    K = C.shape[-1]
    A = np.zeros((K,K))
    B = np.zeros((K,K))

    #definition of transformation matrix

    for i in range(1,K):
        A[i,i] = 1
        A[i,i-1] = -1

    A_big = np.resize(A, C.shape)

    #definition of affine term

    for i in range(K):
        B[i,0] = 1

    B_big = np.resize(B, C.shape)

    return C@A_big + B_big