from Portfolio import load_from_file, dump_to_file
from PortfolioGenerator import PortfolioGenerator


NB_LOANS = 100

filename = 'inputs/portfolio{nb_loans}loans.dump'
filename = filename.format(nb_loans=NB_LOANS)
# Create random portfolio & dump it into filename
portfolio = PortfolioGenerator().generate("Portfolio" + str(NB_LOANS), filename, NB_LOANS)
dump_to_file(portfolio, filename)

portfolio = load_from_file(filename)

print(len(portfolio.portfolio))
