from ScenarioGenerator import ScenarioGenerator
from matplotlib import pyplot as plt
from heatmap import heatmap, annotate_heatmap
import numpy as np

# Parameters(nb_of_groups, nb_of_ratings, .00001, .00001, 0, .001, .001)
# Parameters(nb_of_groups, nb_of_ratings, .01, .01, .001, .2, .2)

horizon = 50

# scenarioGenerator_no_cc = ScenarioGenerator(horizon, .00001, .00001, 0, .001, .001)
scenarioGenerator_no_cc = ScenarioGenerator(horizon, .02, 1.5, .005, .3, .1)

scenarioGenerator_no_cc.compute()

risks = scenarioGenerator_no_cc.risk
macro_correlation = scenarioGenerator_no_cc.macro_correlation
print("All risks: ")
print(risks)

print("Risks at 0:")
print(scenarioGenerator_no_cc.risks_at(0))

print("RiskFactor Correlation at 0:")
print(scenarioGenerator_no_cc.rf_correlation_at(0))

print("RiskFactor Correlation at Horizon:")
print(scenarioGenerator_no_cc.rf_correlation_at(horizon-1 ))

climate_risks = scenarioGenerator_no_cc.rf_list

fig1, ax = plt.subplots()
fig1.suptitle("Climate Risks Scenario")
ax.set_xlabel("Horizon")
ax.set_ylabel("°")
for rf in range(len(climate_risks)):
    ax.plot(risks[rf, :], label=climate_risks[rf])
ax.legend()

fig4, ax = plt.subplots()
fig4.suptitle("Macro correlation")
ax.set_xlabel("Horizon")
ax.set_ylabel("°")
for rf in range(len(climate_risks)):
    ax.plot(macro_correlation[rf, :], label=climate_risks[rf])
ax.legend()

fig2, ax2 = plt.subplots()
fig2.suptitle("RF Correlation at 0")
im, cbar = heatmap(scenarioGenerator_no_cc.rf_correlation_at(0), climate_risks, climate_risks, ax=ax2,
                   cmap="coolwarm", cbarlabel="?")
texts = annotate_heatmap(im, valfmt="{x:.6f}")
fig2.tight_layout()

fig3, ax3 = plt.subplots()
fig3.suptitle("RF Correlation at " + str(horizon))
im, cbar = heatmap(scenarioGenerator_no_cc.rf_correlation_at(horizon-1), climate_risks, climate_risks, ax=ax3,
                   cmap="coolwarm", cbarlabel="?")
texts = annotate_heatmap(im, valfmt="{x:.6f}")
fig3.tight_layout()

plt.show()
