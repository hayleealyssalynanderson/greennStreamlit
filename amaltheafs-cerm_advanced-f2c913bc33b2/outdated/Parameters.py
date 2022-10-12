import numpy as np
import copy

from utils import correlation, migmat_to_thresh, thresh_to_migmat
from Ratings import REG_MATRIX_8


def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    _correlation = covariance / outer_v
    _correlation[covariance == 0] = 0
    return _correlation


class Parameters:

    def __init__(self, nb_of_groups, nb_of_ratings, alpha, beta, gamma, H, theta, Y_0=np.zeros((3, 1))):
        # correlation model incorporation
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.hhw = H
        self.theta = theta

        self.design = np.array([[0, 0, 0], [-self.gamma, 1, -self.alpha], [0, self.beta, 0]])
        self.external = np.array([[1, 0, 0], [0, self.hhw ** 2, 0], [0, 0, self.theta ** 2]])
        self.zRisk = Y_0
        self.risk = copy.deepcopy(Y_0)
        self.var = np.zeros((3, 3))

        # first occurrence of climate risks, as model starts at time 0
        eps_h, eps_theta = np.random.normal(size=2)
        self.zRisk = self.design @ self.zRisk + np.array(
            [[np.random.normal(), self.hhw * eps_h, self.theta * eps_theta]]).T
        self.var = self.design @ self.var @ self.design.T + self.external
        self.macro_correlation = (np.array([np.sqrt(np.diag(self.var))])).T
        self.risk = self.zRisk / self.macro_correlation
        self.risk[self.macro_correlation == 0] = self.zRisk[self.macro_correlation == 0]

        # update of correlation matrix
        self.correlation_matrix = correlation_from_covariance(self.var)
        self.init_correlation_matrix = copy.deepcopy(self.correlation_matrix)

        # Portfolio parameters
        self.nb_of_groups = nb_of_groups
        self.nb_of_ratings = nb_of_ratings

        # generation of regulatory migration matrices
        M = np.empty((self.nb_of_groups, self.nb_of_ratings, self.nb_of_ratings))
        thresholds = np.empty((self.nb_of_groups, self.nb_of_ratings, self.nb_of_ratings))
        for g in range(self.nb_of_groups):
            M[g] = REG_MATRIX_8
            thresholds[g] = migmat_to_thresh(M[g])
        self.reg_migration_matrices = copy.deepcopy(M)
        self.migration_matrices = copy.deepcopy(M)
        self.reg_thresholds = copy.deepcopy(thresholds)
        self.thresholds = copy.deepcopy(thresholds)

        # generation of initial micro-correlations
        self.micro_correlation = np.ones((self.nb_of_groups, self.nb_of_ratings, 3))  # modèle univers
        # self.micro_correlation=np.random.normal(size=(self.nb_of_groups,self.nb_of_ratings,3))

        # generation of initial factor_loadings
        a_tilde = np.empty((self.nb_of_groups, self.nb_of_ratings, 3))
        for j in range(3):
            a_tilde[:, :, j] = self.micro_correlation[:, :, j] * self.macro_correlation[j]
        self.init_tilded_factor_loadings = copy.deepcopy(a_tilde)

        a_reg = np.empty((self.nb_of_groups, self.nb_of_ratings, 3))
        R_reg = np.empty((self.nb_of_groups, self.nb_of_ratings))
        for g in range(self.nb_of_groups):
            for i in range(self.nb_of_ratings):
                # correlation model to economic risk
                R_reg[g, i] = correlation(self.reg_migration_matrices[g][i, self.nb_of_ratings - 1])
                a = np.sqrt(R_reg[g, i]) * self.init_tilded_factor_loadings[g, i]
                b = np.sqrt(self.init_tilded_factor_loadings[g, i].T @ (
                            self.correlation_matrix @ self.init_tilded_factor_loadings[g, i]))
                a_reg[g, i] = a / b
        self.reg_factor_loadings = copy.deepcopy(a_reg)
        self.init_factor_loadings = copy.deepcopy(a_reg)
        self.factor_loadings = copy.deepcopy(a_reg)
        self.c_facto = copy.deepcopy(a_reg)

        # TODO: result defined outside init ? To check with Jules
        # self.tilded_factor_loadings, self.c_factor, self.conditional_thresholds

    def update_group_dependent_values(self):
        # update of micro-correlations
        # self.micro_correlation=np.random.normal(size=(self.nb_of_groups,self.nb_of_ratings,3))

        # update of tilde factor loadings
        a_tilde = np.empty((self.nb_of_groups, self.nb_of_ratings, 3))
        for j in range(3):
            a_tilde[:, :, j] = self.micro_correlation[:, :, j] * self.macro_correlation[j]
        self.tilded_factor_loadings = copy.deepcopy(a_tilde)

        # update of c factors
        self.c_factor = self.reg_factor_loadings * self.tilded_factor_loadings / self.init_factor_loadings
        self.c_factor[self.init_factor_loadings == 0] = 0

        # update of factor loadings and thresholds
        for g in range(self.nb_of_groups):
            for k in range(self.nb_of_ratings):
                ratio = 1 + self.c_factor[g, k] @ (self.correlation_matrix @ self.c_factor[g, k].T) - \
                        self.reg_factor_loadings[g, k] @ (
                                    self.init_correlation_matrix @ self.reg_factor_loadings[g, k].T)
                self.factor_loadings[g, k] = self.c_factor[g, k] / np.sqrt(ratio)
                self.thresholds = self.reg_thresholds / np.sqrt(ratio)

        # update of conditional migration matrices
        self.conditional_thresholds = np.empty((self.nb_of_groups, self.nb_of_ratings, self.nb_of_ratings))
        for g in range(self.nb_of_groups):
            for k in range(self.nb_of_ratings):
                ratio = 1 - self.factor_loadings[g, k] @ (self.correlation_matrix @ self.factor_loadings[g, k].T)

                self.conditional_thresholds[g, k] = (self.thresholds[g, k] - np.array(
                    self.factor_loadings[g, k]) @ self.risk) / np.sqrt(ratio)

            self.migration_matrices[g] = thresh_to_migmat(self.conditional_thresholds[g])

    def update_group_non_dependent_values(self):
        # This structures are not group dependant,
        # Should be generated in one run for t times

        # updates of risks and variance
        eps_h, eps_theta = np.random.normal(size=2)
        self.zRisk = self.design @ self.zRisk + \
                     np.array([[np.random.normal(), self.hhw * eps_h, self.theta * eps_theta]]).T
        self.var = self.design @ self.var @ self.design.T + self.external

        # update of correlation matrix
        self.correlation_matrix = correlation_from_covariance(self.var)

        # update of macro-correlations and standardized risks
        self.macro_correlation = (np.array([np.sqrt(np.diag(self.var))])).T
        self.risk = self.zRisk / self.macro_correlation
        self.risk[self.macro_correlation == 0] = self.zRisk[self.macro_correlation == 0]

    def update(self):
        self.update_group_non_dependent_values()
        self.update_group_dependent_values()
