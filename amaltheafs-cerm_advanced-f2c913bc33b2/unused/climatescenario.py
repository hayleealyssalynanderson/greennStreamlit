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
class NewScenarioGenerator:

    def __init__(self, horizon, alpha, beta, gamma, R, e, H, theta, Y_0=np.zeros((3, 1))):
        self.horizon = horizon
        self.rf_list = ["economic", "physical", "transition"]
        self.nb_rf = len(self.rf_list)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.R = R
        self.e = e
        self.hhw = H
        self.theta = theta
        self.Y_0 = Y_0

        self.risk = None
        self.macro_correlation = None
        self.rf_correlation = None

    def compute(self):
        timer = Timer(text="Scenarios generated")
        timer.start()

        # t0 init
        design = np.array([[1, 0, 0], [-self.gamma, 1, -self.alpha], [0, self.beta, 1]])
        external = np.array([[self.e**2, 0, 0], [0, self.hhw ** 2, 0], [0, 0, self.theta ** 2]])
        z_risk = self.Y_0
        var = np.zeros((self.nb_rf, self.nb_rf))

        # output init
        self.risk = np.zeros((self.nb_rf,  self.horizon))
        self.vars = np.zeros((self.nb_rf, self.nb_rf, self.horizon))
        self.macro_correlation = np.zeros((self.nb_rf,  self.horizon))
        self.rf_correlation = np.zeros((self.horizon, self.nb_rf, self.nb_rf))

        # first occurrence of climate risks, as model starts at time 0
        #eps_h, eps_theta = np.random.normal(size=2)
        #z_risk = design @ z_risk + np.array([[np.random.normal(), self.hhw * eps_h, self.theta * eps_theta]]).T
        #var = design @ var @ design.T + external
        #self.macro_correlation[:, [0]] = (np.array([np.sqrt(np.diag(var))])).T
        #self.risk[:, [0]] = z_risk / self.macro_correlation[:, [0]]
        #self.risk[:, [0]][self.macro_correlation[:, [0]] == 0] = z_risk[self.macro_correlation[:, [0]] == 0]

        # init of correlation matrix / Risk Factor Correlation
    
        self.rf_correlation[[0], :, :] = correlation_from_covariance(var)
        # t1 to horizon
        for t in range(1, self.horizon):
            # updates of risks and variance
            eps_e, eps_h, eps_theta = np.random.normal(size=3)
            z_risk = design @ z_risk + np.array([[self.R + self.e * eps_e, self.hhw * eps_h, self.theta * eps_theta]]).T
            var = design @ var @ design.T + external

            #update of variance
            self.vars[:,:,t] = var

            # update of macro-correlations

            design_diff = design - np.eye(3)
            var_diff = (design_diff)@var@(design_diff.T) + external
            var_diff = var_diff@np.array([[1,0,0], [0,1,0],[0,0,1]])

            macro_correlation = (np.array([np.sqrt(np.diag(var_diff))])).T
            self.macro_correlation[:, [t]] = macro_correlation

            # update of rf correlation
            self.rf_correlation[[t], :, :] = correlation_from_covariance(var)

            # standardized risks
            self.risk[:, [t]] = z_risk / macro_correlation
            self.risk[:, [t]][macro_correlation == 0] = z_risk[macro_correlation == 0]
        timer.stop()

    def rf_correlation_at(self, t):
        return self.rf_correlation[t, :, :]

    def macro_correlation_at(self, t):
        return self.macro_correlation[:, [t]]

    def var_at(self,t):
        return self.vars[:,:,t]

    def risks_at(self, t):
        return self.risk[:, [t]]


