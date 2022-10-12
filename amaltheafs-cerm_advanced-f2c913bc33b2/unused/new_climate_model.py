import numpy as np
from timer import Timer


def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    _correlation = covariance / outer_v
    _correlation[covariance == 0] = 0
    return _correlation


# This class generates the RF scenarios for 3 risk factors : Economic, Physical, transition
# They are generated following those parameters: alpha, beta, gamma, H, theta
# horizon is the number of years for this simulation.
class NewClimateModel:

    def __init__(self, horizon, alpha, beta, gamma, R, e, p, theta, Y_0=np.zeros((3, 1))):
        self.horizon = horizon
        self.rf_list = ["economic", "physical", "transition"]
        self.nb_rf = len(self.rf_list)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.R = R
        self.e = e
        self.p = p
        self.theta = theta
        self.Y_0 = Y_0

        self.risk = None
        self.macro_correlation = None
        self.rf_correlation = None

    def compute(self):
        timer = Timer(text="Scenarios generated")
        timer.start()

        # t0 init
        design = np.array([[0, 0, 0], [0, 1-self.alpha*self.beta, 0], [0, self.beta, 0]])
        external = np.array([[self.e**2, self.gamma*self.e**2, 0], [self.gamma*self.e**2, self.alpha**2*self.theta**2 + self.e**2*self.gamma**2 + self.p**2 , -self.alpha*self.theta**2], [0, -self.alpha*self.theta**2, self.theta**2]])
        z_risk = self.Y_0
        var = np.zeros((self.nb_rf, self.nb_rf))

        # output init
        self.zrisk = np.zeros((self.nb_rf,  self.horizon))
        self.risk = np.zeros((self.nb_rf,  self.horizon))
        self.vars = np.zeros((self.nb_rf, self.nb_rf, self.horizon))
        self.macro_correlation = np.zeros((self.nb_rf,  self.horizon))
        self.rf_correlation = np.zeros((self.horizon, self.nb_rf, self.nb_rf))

        self.tilde = np.zeros((self.nb_rf, self.horizon))
        self.expectation = np.zeros((self.nb_rf, self.horizon))
        self.var_tilde = np.zeros((self.horizon, self.nb_rf, self.nb_rf))

        q = 1-self.alpha*self.beta

        # first occurrence of climate risks, as model starts at time 0
        #eps_h, eps_theta = np.random.normal(size=2)
        #z_risk = design @ z_risk + np.array([[np.random.normal(), self.hhw * eps_h, self.theta * eps_theta]]).T
        #var = design @ var @ design.T + external
        #self.macro_correlation[:, [0]] = (np.array([np.sqrt(np.diag(var))])).T
        #self.risk[:, [0]] = z_risk / self.macro_correlation[:, [0]]
        #self.risk[:, [0]][self.macro_correlation[:, [0]] == 0] = z_risk[self.macro_correlation[:, [0]] == 0]

        # init of correlation matrix / Risk Factor Correlation
    
        #self.rf_correlation[[0], :, :] = correlation_from_covariance(var)
        # t1 to horizon
        for t in range(1, self.horizon):
            # updates of risks and variance
            eps_e, eps_p, eps_theta = np.random.normal(size=3)
            z_risk = design @ z_risk + np.array([[self.e*eps_e, -self.alpha*self.theta*eps_theta + self.e*self.gamma*eps_e + self.p*eps_p, self.theta * eps_theta]]).T
            self.zrisk[:,t] = np.resize(z_risk,self.nb_rf)
            var = design @ var @ design.T + external

            #update of variance
            self.vars[:,:,t] = var

            # update of rf correlation
            self.rf_correlation[[t], :, :] = correlation_from_covariance(var)

            # update of macro-correlations
            macro_correlation = (np.array([np.sqrt(np.diag(var))])).T
            self.macro_correlation[:, [t]] = macro_correlation

            # standardized risks
            self.risk[:, [t]] = z_risk / macro_correlation
            self.risk[:, [t]][macro_correlation == 0] = z_risk[macro_correlation == 0]

            # update of expectations
            self.expectation[0,t] = t*self.R
            self.expectation[1,t] = (self.gamma*self.R/(self.alpha*self.beta))*t - self.gamma*self.R*(1/(self.alpha*self.beta)-1)*((1-(1-self.alpha*self.beta)**t)/(self.alpha*self.beta))
            self.expectation[2,t] = self.beta*self.expectation[1,t-1]

            # update of tilde
            self.tilde[:,t] = self.zrisk[:,:t+1].sum(axis=1) + self.expectation[:,t]

            #update of vartilde
            S = np.array([[1,0,0],[0,(1-q**t)/(1-q),0],[0, self.beta*(1-q**(t-1))/(1-q),1]])
            self.var_tilde[t] = self.var_tilde[t-1] + S @ external @ S.T

        timer.stop()

        self.gdp = np.exp(np.array([[1,-1,-1]])@self.tilde)

        self.mu = np.array([[1,-1,-1]])@self.expectation

        self.ssigma = np.array([np.sum(np.array([[1,-1,-1]])@self.var_tilde[t]@np.array([[1,-1,-1]]).T) for t in range(self.horizon)])

    def rf_correlation_at(self, t):
        return self.rf_correlation[t, :, :]

    def macro_correlation_at(self, t):
        return self.macro_correlation[:, [t]]

    def var_at(self,t):
        return self.vars[:,:,t]

    def risks_at(self, t):
        return self.risk[:, [t]]

    def gdp_at(self,t):
        return self.gdp[:,t]


