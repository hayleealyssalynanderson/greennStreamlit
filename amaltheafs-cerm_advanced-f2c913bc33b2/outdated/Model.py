import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Model:

    def __init__(self, nb_of_groups, nb_of_ratings, init_parameters, horizon, init_portfolio,  LGD=.5):
        self.data = {}

        self.init_parameters = init_parameters
        self.parameters = copy.deepcopy(self.init_parameters)
        self.parameter_memory = []

        self.init_portfolio = init_portfolio

        self.nb_of_groups = nb_of_groups
        self.nb_of_ratings = nb_of_ratings
        self.ratings = range(1, self.nb_of_ratings+1)
        self.horizon = horizon

        self.LGD = LGD

        self.portfolio = copy.deepcopy(self.init_portfolio)
        self.portfolio_structure = [len(group) for group in self.portfolio]

        self.time = 0

    def reinitialization(self):
        self.portfolio = copy.deepcopy(self.init_portfolio)
        self.parameters = copy.deepcopy(self.init_parameters)
        self.time = 0

    def update(self):
        self.parameter_memory.append(copy.deepcopy(self.parameters))
        if self.time<self.horizon:
            Z = self.parameters.risk
            for g in range(self.nb_of_groups):
                M = self.parameters.migration_matrices[g]
                for loan in self.portfolio[g]:
                    # Just add Loss if rating is not at default
                     if len(loan.rating) == self.time+1:
                        r = loan.rating[-1]
                        # Si rating = "default" = last rating , compute loss
                        if r == self.nb_of_ratings:
                            # loan.loss.append(loan.EAD(self.time)*(1-loan.recovery_rate[-1]))
                            # loan.loss.append(loan.ead(self.time)*self.LGD)
                            loan.loss.append(loan.principal*self.LGD)
                        # Si pas "at default" compute new rating, and set loss at t = 0
                        else:
                            PD = M[r-1, self.nb_of_ratings-1]
                            a = self.parameters.factor_loadings[g, r-1]

                            eps = np.random.normal()
                            X = a@Z+np.sqrt(1-a@self.parameters.correlation_matrix@a.T)*eps

                            # b=self.parameters.collateral_factor_loadings[g,r-1]
                            # eps_tilde=np.random.normal()
                            # RR=norm.cdf(b.T@Z+np.sqrt(1-b.T@(self.parameters.correlation_matrix@b))*eps_tilde)[0]

                            thresholds = self.parameters.thresholds[g, r-1]

                            z = 0
                            while z <= self.nb_of_ratings-1 and thresholds[z] > X:
                                z = z+1

                            loan.rating.append(z)
                            loan.probability_of_default.append(PD)
                            loan.loss.append(0)

            self.parameters.update()
            self.time += 1

    def run(self):
        for t in range(self.horizon):
            self.update()

    def histogram_of_ratings(self):
        L = []
        for g in range(self.nb_of_groups):
            for loan in self.portfolio[g]:
                L.append(loan.rating[-1])
        plt.figure()
        plt.title("Histogram of ratings at time "+str(self.time))
        plt.xlabel("Ratings, "+str(self.nb_of_ratings)+" being default")
        plt.ylabel("Number of loans with associated ratings")
        plt.hist(L, bins=50)

    def evolution_of_ratings(self):
        self.reinitialization()
        T=self.horizon
        for i in range(T):
            self.histogram_of_ratings()
            self.update()
        self.histogram_of_ratings()

    def histogram_of_losses(self):
        L=[]
        for g in range(self.nb_of_groups):
            for loan in self.portfolio[g]:
                L.append(loan.loss[-1])
        plt.figure()
        plt.title("Histogram of losses at time "+str(self.time))
        plt.xlabel("Losses")
        plt.ylabel("Number of loans with associated losses")
        plt.hist(L, bins=50)

    def evolution_of_loss(self):
        self.reinitialization()
        L=[]
        for i in range(self.horizon):
            L.append(self.loss())
            self.update()
        L.append(self.loss())
        plt.figure()
        plt.plot(L)
        plt.title("Evolution of the loss")
        plt.xlabel("Time")
        plt.ylabel("Loss")
        plt.show()

    def expected_loss(self, group="all"):
        L_e=0
        if group=="all":
            for g in range(self.nb_of_groups):
                L_eg=0
                for i in range(self.nb_of_ratings-1):
                    for j in range(self.nb_of_ratings-1):
                        EAD=0
                        for loan in self.portfolio[g]:
                            if loan.rating[0]==i+1:
                                EAD+=loan.EAD(self.time)
                        product=np.eye(self.nb_of_ratings)
                        for t in range(self.time-1):
                            product=product@self.parameter_memory[t].migration_matrices[g]
                        product=product[i,j]*self.parameters.migration_matrices[g,j,self.nb_of_ratings-1]
                        L_eg+=product*self.LGD*EAD
                L_e+=L_eg
            return L_e
        else:
            L_eg=0
            for i in range(self.nb_of_ratings-1):
                for j in range(self.nb_of_ratings-1):
                    EAD=0
                    for loan in self.portfolio[group]:
                        if loan.rating[0]==i+1:
                            EAD+=loan.EAD(self.time)
                    product=np.eye(self.nb_of_ratings)
                    for t in range(self.time-1):
                        product=product@self.parameter_memory[t].migration_matrices[group]
                    product=product[i,j]*self.parameters.migration_matrices[group,j,self.nb_of_ratings-1]
                    L_eg+=product*self.LGD*EAD
            return L_eg

    def evolution_of_expected_loss(self):
        self.reinitialization()
        EL=[]
        for i in range(self.horizon):
            el=self.expected_loss()
            EL.append(el)
            self.update()
        EL.append(self.expected_loss())
        plt.figure()
        plt.plot(EL)
        plt.plot([sum(EL)]*(self.horizon+1))
        plt.title("Evolution of the expected loss")
        plt.xlabel("Time")
        plt.ylabel("Expected Loss")
        plt.show()

    def loss(self):
        loss=[0]*(self.nb_of_groups+1)
        for g in range(self.nb_of_groups):
            for loan in self.portfolio[g]:
                if loan.rating[-1]==self.nb_of_ratings:
                    loss[g]+=loan.loss[-1]
        loss[self.nb_of_groups]=sum(loss[0:self.nb_of_groups])
        return loss

    def evolution_of_loss(self):
        self.reinitialization()
        L=[]
        for i in range(self.horizon):
            L.append(self.loss())
            self.update()
        L.append(self.loss())
        plt.figure()
        plt.plot(L)
        plt.title("Evolution of the loss")
        plt.xlabel("Time")
        plt.ylabel("Loss")
        plt.show()

    def loss_distribution(self, N, disp="yes"):
        G = self.nb_of_groups
        losses = np.empty((N, G+1))
        for i in range(N):
            self.reinitialization()
            self.run()
            losses[i] = self.loss()
        if disp == "yes":
            plt.figure(figsize=(12, 8))
            plt.title("Histogram of losses for a given portfolio of structure "+str(self.portfolio_structure))
            plt.xlabel("Total end loss")
            plt.ylabel("Number of occurrences")
            plt.hist(losses[:, G], bins=max(10, N//10))
        return losses

    def expected_losses_estimator(self, N):
        L=self.loss_distribution(N, disp="no")
        return np.sum(L,axis=0)/N

    def expected_losses_sensitivities(self,N):
        EL=self.expected_losses_estimator(N)
        return EL[0:self.nb_of_groups]/EL[self.nb_of_groups]

    def unexpected_loss_estimator(self,risk,N):
        L=self.loss_distribution(N, disp="no")
        ind=int(np.floor(N*(1-risk)))
        return np.sort(L,axis=0)[ind,self.nb_of_groups]

    def unexpected_loss_sensitivities(self,N,risk,delta):
        UL=self.unexpected_loss_estimator(risk,N)
        L=self.loss_distribution(N, disp="no")
        sensitivities=np.zeros((self.nb_of_groups))
        c=0
        for i in range(N):
            if (1-delta)*UL<L[i,self.nb_of_groups]<(1+delta)*UL:
                c+=1
                sensitivities+=L[i,0:self.nb_of_groups]/L[i,self.nb_of_groups]
        if c<5:
            print("delta is chosen too small")
        return sensitivities/c

    def all_outputs(self,N,risk,delta):
        L=self.loss_distribution(N, disp="no")
        ind=int(np.floor(N*(1-risk)))
        EL=np.sum(L,axis=0)/N
        EL_sensitivities=EL[0:self.nb_of_groups]/EL[self.nb_of_groups]
        UL=np.sort(L,axis=0)[ind,self.nb_of_groups]
        UL_sensitivities=np.zeros((self.nb_of_groups))
        c=0
        for i in range(N):
            if (1-delta)*UL<L[i,self.nb_of_groups]<(1+delta)*UL:
                c+=1
                UL_sensitivities+=L[i,0:self.nb_of_groups]/L[i,self.nb_of_groups]
        if c<5:
            print("delta might be chosen too small")
        UL_sensitivities=UL_sensitivities/c
        outputs={"loss distribution":L,"expected losses":EL,"expected loss sensitivities":EL_sensitivities,"unexpected loss":UL,"unexpected loss sensitivities":UL_sensitivities,"count":c,"risk":risk,"delta":delta}
        return outputs

    def write(self,N,risk,delta):
        outputs=self.all_outputs(N,risk,delta)
        df=pd.DataFrame(dict([ (k,[v]) for k,v in outputs.items()]))
        df.to_csv('/drive/MyDrive/cerm_outputs.csv')