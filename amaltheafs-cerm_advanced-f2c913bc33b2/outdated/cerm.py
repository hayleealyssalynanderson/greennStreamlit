import matplotlib.pyplot as plt
import copy
from Model import Model
from PortfolioGenerator import PortfolioProvider
from Parameters import Parameters
from Portfolio import dump_to_file, load_from_file


nb_of_groups = 3
nb_of_ratings = 8
horizon = 20

filename = 'inputs/portfolio1.dump'

# Create random portfolio & dump it into filename
# portfolioProvider = PortfolioProvider()
# portfolio = portfolioProvider.get_portfolio("Portfolio1", "RANDOM")
# dump_to_file(portfolio, filename)
# quit(0)

portfolio = load_from_file(filename)
portfolio = portfolio.grouped_portfolio()

parameters_no_cc = Parameters(nb_of_groups, nb_of_ratings, .00001, .00001, 0, .001, .001)
parameters_cc = Parameters(nb_of_groups, nb_of_ratings, .01, .01, .001, .2, .2)

model_no_cc = Model(
    nb_of_groups,
    nb_of_ratings,
    parameters_no_cc,
    horizon,
    copy.deepcopy(portfolio)
)

model_cc = Model(
    nb_of_groups,
    nb_of_ratings,
    parameters_cc,
    horizon,
    copy.deepcopy(portfolio)
)

loss_no_cc = model_no_cc.loss_distribution(300)

loss_cc = model_cc.loss_distribution(300)

model_cc.histogram_of_ratings()

portfolio_structure = [len(group) for group in portfolio]
N = 200
loss_no_cc = model_no_cc.loss_distribution(N)
print(loss_no_cc)
loss_cc = model_cc.loss_distribution(N)

plt.figure(figsize=(12, 8))
plt.title("Comparison of losses w and w/o climate change for portfolio of structure "+str(portfolio_structure))
plt.xlabel("Total end loss")
plt.ylabel("Number of occurences")
plt.hist(loss_no_cc[:, nb_of_groups], bins=max(10, N//10), alpha=.5, label="w/o climate change")
plt.hist(loss_cc[:, nb_of_groups], bins=max(10, N//10), alpha=.5, label="w/ climate change")
plt.legend()

plt.figure(figsize=(12, 8))
plt.hist(loss_no_cc[:, nb_of_groups], bins=max(10, N//10), alpha=.5, label="w/o climate change")

plt.figure(figsize=(12, 8))
plt.hist(loss_cc[:, nb_of_groups], bins=max(10, N//10), alpha=.5, label="w/ climate change")

plt.show()