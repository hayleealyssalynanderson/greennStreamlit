import numpy as np
import matplotlib.pyplot as plt
from timer import Timer
from utils import correlation_from_covariance
from scipy.stats import norm

"""This script defines the ScenarioGenerator class, which computes the evolution of a climate trajectory given climate coefficients"""

class ScenarioGenerator:
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
    macro_correlation: array (float)
        macro-correlations
    rf_correlation: array (float)
        array of risk factors correlation matrices
    vars: array (float)
        array of risk factors covariance matrices
    """

    def __init__(self, horizon, alpha, beta, gamma, R, e, p, theta):
        """
        Constructs all the necessary attributes for the ScenarioGenerator object.
        
        Parameters
        ----------
            horizon: int
                horizon in years
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
        """
        self.horizon = horizon
        self.rf_list = ["economic", "physical", "transition"]
        self.nb_rf = len(self.rf_list)
        self.alpha_tilde = alpha
        self.beta = beta
        self.gamma_tilde = gamma
        self.R = R
        self.e = e
        self.p_tilde = p
        self.theta = theta

        #reduction of parameters

        self.alpha = self.alpha_tilde / (1 + self.gamma_tilde)
        self.gamma = self.gamma_tilde / (1 + self.gamma_tilde)
        self.p = self.p_tilde / (1 + self.gamma_tilde)
        self.q = 1 - self.alpha * self.beta - (1+ self.beta) * self.gamma

        self.sigma_squared = (self.alpha + self.gamma)**2 * self.theta**2 + self.e**2 * self.gamma**2 + self.p**2

        self.risk = None
        self.macro_correlation = None
        self.rf_correlation = None

    def simplesum(self, t):
        q = self.q
        return (1-q**t)/(1-q)

    def compute(self):
        """
        Computes all calculations for a random climate trajectory.
        """
        alpha = self.gamma
        beta = self.beta
        gamma = self.gamma
        R = self.R
        e = self.e
        theta = self.theta
    
        q = self.q
        sigma_squared = self.sigma_squared

        #initialization of timer for computation of scenario

        timer = Timer(text="Scenarios generated")
        timer.start()

        #definition of incremental matrix

        design = np.array([[0, 0, 0], [0, self.q, 0], [0, self.beta, 0]])

        #definition of variance of idiosyncratic risk

        external = np.array([[self.e**2, self.gamma*self.e**2, 0],[self.gamma*self.e**2, self.sigma_squared, -(self.alpha + self.gamma)*self.theta**2], [0, -(self.alpha + self.gamma)*self.theta**2, self.theta ** 2]])

        #initialization of risks

        z_risk = np.zeros((self.nb_rf, 1))
        var = np.zeros((self.nb_rf, self.nb_rf))

        #initialization of risk evolution

        self.risk = np.zeros((self.nb_rf,  self.horizon))

        #initialization of cumulative risk evolution

        self.cumulative_rf = np.zeros((self.nb_rf,  self.horizon))

        #initialization of covariance evolution

        self.vars = np.zeros((self.nb_rf, self.nb_rf, self.horizon))

        #initialization of macro-correlations evolutions

        self.macro_correlation = np.zeros((self.nb_rf,  self.horizon))

        #initialization of correlation matrix evolution

        self.rf_correlation = np.zeros((self.horizon, self.nb_rf, self.nb_rf))

        #execution

        for t in range(1, self.horizon):

            #update of risks

            eps_e, eps_p, eps_theta = np.random.normal(size=3)
            z_risk = design @ z_risk + np.array([[self.e*eps_e, -(self.alpha + self.gamma)*self.theta * eps_theta +self.e * self.gamma *eps_e + self.p * eps_p, self.theta * eps_theta]]).T

            #update of cumulative risks

            self.cumulative_rf[:,t] = z_risk.T[0] + self.cumulative_rf[:,t-1]

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

        #calculation of unconditional probability of net-zero transition

        self.unconditional_p_nz = norm.cdf(-gamma*R/sigma_squared*np.sqrt((1+q)/(1-q)))

        #timer for computation of scenario ends

        timer.stop()

    def gdp_compute(self, init = np.zeros(3), delta = 0, disp = "y"): #init is vector of cumulative risk factors at time 0, delta is y^0-y^{-1}

        """
        Returns quantities relevant to the GDP, draws evolution of log GDP.
        
        Parameters
        ----------
            init: array (float)
                initial cumulative risk factor (at time t=0)
            delta: float
                Y_P^0 - Y_P^{-1} (see formula (18) in complementary paper)
        """
        alpha = self.gamma
        beta = self.beta
        gamma = self.gamma
        R = self.R
        e = self.e
        theta = self.theta
    
        q = self.q
        sigma_squared = self.sigma_squared

        #initialization of cumulative growth factors (centered Y tilde)
        
        self.cumulative_growth_factors = np.zeros((self.nb_rf, self.horizon))

        #initialization of associated variance
     
        self.cumulative_growth_var = np.zeros((self.nb_rf, self.nb_rf, self.horizon))

        #initialization of expectations of Y tilde

        self.esp = np.zeros((self.nb_rf, self.horizon))

        #initialization of mu

        self.mu = np.zeros(self.horizon)

        #initialization of s^2

        self.s_squared = np.zeros(self.horizon)

        #logging of esp with formulas in (19)

        self.esp[:,0] = init
        self.esp[:,1:] = np.array([init + [(t+1)*R, (t+1) * gamma * R / (1-q) + (delta - gamma * R / (1-q)) * q * self.simplesum(t+1), beta * delta + t * gamma * R / (1-q) + (delta - gamma * R / (1-q)) * q * self.simplesum(t)] for t in range(self.horizon-1)]).T

        #logging of Y tilde

        self.cumulative_growth_factors = self.esp + self.cumulative_rf

        #logging of associated variance

        for t in range(1, self.horizon):
            self.cumulative_growth_var[:,:,t] = self.cumulative_growth_var[:,:,t-1] + np.array([[e**2, gamma * e**2 * self.simplesum(t), beta * gamma * e**2 * self.simplesum(t-1)],  [gamma * e**2 * self.simplesum(t), sigma_squared * self.simplesum(t)**2, beta * sigma_squared * self.simplesum(t) * self.simplesum(t-1) - (alpha + gamma) * theta**2 * self.simplesum(t)], [beta * gamma * e**2 * self.simplesum(t-1), beta * sigma_squared * self.simplesum(t) * self.simplesum(t-1) - (alpha + gamma) * theta**2 * self.simplesum(t), theta**2 + beta**2 * sigma_squared * self.simplesum(t-1)**2 - 2 * alpha * beta * theta**2 *self.simplesum(t-1)]])

        #log GDP

        self.log_gdp = np.array([[1, -1, -1]])@self.cumulative_growth_factors

        #mu

        self.mu = np.array([[1, -1, -1]])@self.esp

        #s^2

        for t in range(self.horizon):
            self.s_squared[t] = np.array([1,-1,-1])@self.cumulative_growth_var[:,:,t]@np.array([1,-1,-1]).T

        #resizing for plots and exp

        self.log_gdp = np.reshape(self.log_gdp, (self.horizon))
        self.mu = np.reshape(self.mu, (self.horizon))
        self.s_squared = np.reshape(self.s_squared, (self.horizon))

        #logging of median, expected and variance of GDP

        self.med_gdp = np.exp(self.mu)
        self.esp_gdp = np.exp(self.mu + self.s_squared/2)
        self.var_gdp = (np.exp(self.s_squared)-1)*np.exp(2*self.mu+self.s_squared)

        #plotting of GDP evolution

        if disp == "y":
            fig1, (ax1, ax2) = plt.subplots(2,figsize=(16,12))
            fig1.suptitle('log GDP')

            ax1.plot(self.mu, label = "mu")
            ax1.plot(self.log_gdp, label = "log GDP")
            ax1.set_title("Expected and actual evolution of log-GDP")
            ax1.set_xlabel("time (years)")
            ax1.legend()

            ax2.plot(self.s_squared, label = "variance of log GDP")
            ax2.set_title("Evolution of the variance of log GDP")
            ax2.set_xlabel("time (years)")
            ax2.legend()

            plt.show()

    def rf_correlation_at(self, t):
        return self.rf_correlation[t, :, :]

    def macro_correlation_at(self, t):
        return self.macro_correlation[:, [t]]

    def var_at(self,t):
        return self.vars[:,:,t]

    def risks_at(self, t):
        return self.risk[:, [t]]