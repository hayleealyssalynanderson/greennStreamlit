from clickhouse_driver import Client
from datetime import datetime
import numpy as np

from CERMEngine import CERMEngine
from ScenarioGenerator import ScenarioGenerator
from Portfolio import load_from_file
from Ratings import TEST_8_RATINGS
from timer import Timer
from datetime import date


def iter_portfolio_result(datatset, portfolio_result):
    for loan_result in portfolio_result.portfolio:
        yield {
            "DatasetId": datatset,
            "RunDate": run_date,
            "LoanId": loan_result.loan.id,
            "LoanType": loan_result.loan.type,
            "ISIN": loan_result.loan.borrower.isin,
            "CounterpartyId": loan_result.loan.borrower.id,
            "CounterpartyName": loan_result.loan.borrower.name,
            "LegalType": loan_result.loan.borrower.legal_type,
            "HighLowStakes": loan_result.loan.borrower.stakes,
            "GroupName": loan_result.loan.group(),
            "GeographicRegion": loan_result.loan.borrower.region,
            "IndustrySector": loan_result.loan.borrower.sector,
            "IndustrySubsector": loan_result.loan.borrower.subsector,
            "Rating": loan_result.loan.initialRating,
            "CreationDate": loan_result.loan.creation_date,
            "MaturityDate": loan_result.loan.maturity_date,
            "Principal": loan_result.loan.principal,
            "Status": loan_result.loan.status,
            "TaxonomyStatus": loan_result.loan.taxonomy_status,
            "GAR": loan_result.loan.gar,
            "NbIteration": portfolio_result.nb_iteration,
            "DefaultT": loan_result.t_at_default,
            "Losses": loan_result.losses,
            "EL": loan_result.expected_losses
        }


horizon = 30
filename = 'inputs/portfolio1000loans.dump'
portfolio = load_from_file(filename)
scenario_cc = ScenarioGenerator(horizon, .01, .01, .001, .1, .1)
scenario_cc.compute()
climate_risks = scenario_cc.rf_list
risks = scenario_cc.risk
engine_cc = CERMEngine(portfolio, TEST_8_RATINGS, scenario_cc)
N = 1000  # 1500
run_date = date.today()
engine_cc.compute(N)

portfolio_result = engine_cc.portfolio_result
test_name = "Test1000L*1000N"

client = Client(host='localhost')
timer = Timer(text="CERM Results inserted in Clickhouse.")
timer.start()
client.execute("INSERT INTO database_tenantA.climateMetrics VALUES",
               iter_portfolio_result(test_name, portfolio_result), types_check=True)
timer.stop()

print(client.execute('SELECT COUNT(*), GroupName FROM database_tenantA.climateMetrics GROUP BY GroupName '
                     'ORDER BY GroupName'))


