import pandas as pd
from Portfolio import load_from_file, show

"""
Prints the input portfolio in filename as a pandas database
"""

filename = 'Work/inputs/portfolio1000loansJules.dump'

portfolio = load_from_file(filename)

table = show(portfolio)
for key in list(table.keys()):
    table[key] = list(table[key])

df = pd.DataFrame.from_dict(table, orient='index')
df.to_csv('portfolio1000Jules.csv')
print(df)
