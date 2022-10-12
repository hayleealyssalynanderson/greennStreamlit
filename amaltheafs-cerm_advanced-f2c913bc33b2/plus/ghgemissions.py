import numpy as np
import csv
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures

#GHG-GDP

total_ghg=[]

with open("Work/ghg_emissions.csv") as file_name:
    reader = csv.reader(file_name, delimiter=',', quotechar='"')
    i=0
    for row in reader:
        row = row[4:]
        ghg_country = []
        for ghg in row:
            if ghg =="":
                ghg_country.append(None)
            else:
                ghg_country.append(float(ghg))
        total_ghg.append(ghg_country)

total_ghg = np.array(total_ghg)
total_ghg = np.where(total_ghg!=None, total_ghg, 0)

years_ghg, total_ghg = total_ghg[0], total_ghg[1:].sum(axis=0)

with open("Work\world-gdp-over-the-last-two-millennia.csv") as file_name:
    next(file_name)
    array = np.loadtxt(file_name, delimiter=",")

years_gdp = array[:,0]
total_gdp = array[:,1]

ghg_data = total_ghg[10:-6].reshape(-1,1)

gdp_data = total_gdp[20:].reshape(-1,1)

poly = PolynomialFeatures(degree=1)
ghg_data = poly.fit_transform(ghg_data)
reg_lasso1 = Lasso()
reg_lasso1.fit(ghg_data,gdp_data)
print(reg_lasso1.coef_)
print(reg_lasso1.score(ghg_data,gdp_data))

#GHG-transition

transition_efforts = 1e9*np.array([30,60,110,140,180,170,230,290,260,230,300,320,380,420,425,450,500])
transition_data = transition_efforts[:-2]

l=len(total_ghg)
for i in range(1,l):
    total_ghg[i]+=total_ghg[i-1]

ghg_data = np.array([total_ghg[44:-3]]).T

poly = PolynomialFeatures(degree=1)
ghg_data = poly.fit_transform(ghg_data)
reg_lasso2 = Lasso()
reg_lasso2.fit(ghg_data,transition_data)
beta = reg_lasso2.coef_

n = len(transition_data)

theta = np.sqrt(sum((transition_data - reg_lasso2.predict(ghg_data))**2)/(n-1))

print(beta, theta)

plt.plot(transition_data)
plt.plot(reg_lasso2.predict(ghg_data))
plt.show()
