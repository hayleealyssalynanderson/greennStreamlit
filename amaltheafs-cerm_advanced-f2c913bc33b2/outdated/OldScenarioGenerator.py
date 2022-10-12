import numpy as np
from timer import Timer
from utils import correlation_from_covariance

"""This script defines the ScenarioGenerator class, which computes the evolution of a climate trajectory given climate coefficients"""

class OldScenarioGenerator:
    """
    A class to represent a climate scenario, to input in the CERM.
    
    Attributes
    ----------
    horizon: int
        horizon in years
    rf_list: list (str)
        names of risk factors
    nb_rf: int
        number of risk factors
    alpha: float
        transition efficiency
    beta: float
        transition effort reactivity to climate change
    gamma: float
        climate change intensity of the economic activity (carbon intensity)
    hhw: float
        HotHouseWorld coefficient
    theta: float
        independent transition effort
    macro_correlation: array (float)
        macro-correlations
    rf_correlation: array (float)
        array of risk factors correlation matrices
    vars: array (float)
        array of risk factors covariance matrices
    """

    def __init__(self, horizon, alpha, beta, gamma, H, theta):
        """
        Constructs all the necessary attributes for the ScenarioGenerator object.
        
        Parameters
        ----------
            horizon: int
                horizon in years
            alpha: float
                transition efficiency
            beta: float
                transition effort reactivity to climate change
            gamma: float
                climate change intensity of the economic activity (carbon intensity)
            hhw: float
                HotHouseWorld coefficient
            theta: float
                independent transition effort
        """
        self.horizon = horizon
        self.rf_list = ["economic", "physical", "transition"]
        self.nb_rf = len(self.rf_list)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.hhw = H
        self.theta = theta

        self.risk = None
        self.macro_correlation = None
        self.rf_correlation = None

    def compute(self):
        """
        Computes all calculations for a random climate trajectory
        """

        #initialization of timer for computation of scenario

        timer = Timer(text="Scenarios generated")
        timer.start()

        #definition of incremental matrix

        design = np.array([[0, 0, 0], [-self.gamma, 1, -self.alpha], [0, self.beta, 0]])

        #definition of variance of idiosyncratic risk

        external = np.array([[1, 0, 0], [0, self.hhw ** 2, 0], [0, 0, self.theta ** 2]])

        #initialization of risks

        z_risk = np.zeros((self.nb_rf, 1))
        var = np.zeros((self.nb_rf, self.nb_rf))

        #initialization of risk evolution

        self.risk = np.zeros((self.nb_rf,  self.horizon))

        #initialization of covariance evolution

        self.vars = np.zeros((self.nb_rf, self.nb_rf, self.horizon))

        #initialization of macro-correlations evolutions

        self.macro_correlation = np.zeros((self.nb_rf,  self.horizon))

        #initialization of correlation matrix evolution

        self.rf_correlation = np.zeros((self.horizon, self.nb_rf, self.nb_rf))

        # first occurrence of climate risks, as model starts at time 0
        #eps_h, eps_theta = np.random.normal(size=2)
        #z_risk = design @ z_risk + np.array([[np.random.normal(), self.hhw * eps_h, self.theta * eps_theta]]).T
        #var = design @ var @ design.T + external
        #self.macro_correlation[:, [0]] = (np.array([np.sqrt(np.diag(var))])).T
        #self.risk[:, [0]] = z_risk / self.macro_correlation[:, [0]]
        #self.risk[:, [0]][self.macro_correlation[:, [0]] == 0] = z_risk[self.macro_correlation[:, [0]] == 0]

        #execution

        for t in range(1, self.horizon):

            #update of risks

            eps_e, eps_h, eps_theta = np.random.normal(size=3)
            z_risk = design @ z_risk + np.array([[eps_e, self.hhw * eps_h, self.theta * eps_theta]]).T

            #update of variance

            var = design @ var @ design.T + external
            self.vars[:,:,t] = var

            # update of correlation matrix
            self.rf_correlation[[t], :, :] = correlation_from_covariance(var)

            # update of macro-correlations
            macro_correlation = (np.array([np.sqrt(np.diag(var))])).T
            self.macro_correlation[:, [t]] = macro_correlation

            #update of standardized risks
            self.risk[:, [t]] = z_risk / macro_correlation
            self.risk[:, [t]][macro_correlation == 0] = z_risk[macro_correlation == 0]

        #timer for computation of scenario ends

        timer.stop()

    def rf_correlation_at(self, t):
        return self.rf_correlation[t, :, :]

    def macro_correlation_at(self, t):
        return self.macro_correlation[:, [t]]

    def var_at(self,t):
        return self.vars[:,:,t]

    def risks_at(self, t):
        return self.risk[:, [t]]


