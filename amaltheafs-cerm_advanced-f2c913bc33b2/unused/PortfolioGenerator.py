from Loan import Loan
from Portfolio import Portfolio
from Ratings import TEST_8_RATINGS
from counterparty import Counterparty
from faker import Faker

import numpy as np
import random
import uuid
import json

MAX_MATURITY = 80
MAX_PRINCIPAL = 100  # *1000
MAX_TAU = .5

NB_GROUPS_MULTIPLIER = 5  # x4

TYPE = ["General", "Project", "Asset"]
STATUS = ["Pending", "Approved", "Rejected"]
TAXONOMY_STATUS = ["Not Defined", "Green", "Not Green"]

fake = Faker()

with open('inputs/counterpartyDetails.json') as json_file:
    data = json.load(json_file)


def get_random_cpty():
    cpty_json = random.choice(data)
    cpty = Counterparty(
        cpty_json["counterparty_id"],
        cpty_json["counterparty_name"],
        cpty_json["ISIN"],
        cpty_json["legal_type"],
        cpty_json["high_low_stakes"],
        cpty_json["geographic_region"],
        cpty_json["industry_sector"],
        cpty_json["industry_subsector"],
        cpty_json["rating"],
        cpty_json["group_name"]
    )
    return cpty


class PortfolioGenerator:

    def generate(self, portfolioId, filename, nb_loans):
        portfolio = Portfolio(portfolioId)

        for i in range(nb_loans):
            counterparty = get_random_cpty()
            group_str = counterparty.group() + str(np.random.randint(1, NB_GROUPS_MULTIPLIER))
            creation_date = fake.date_between(start_date='-5y', end_date='now')

            # maturity_date = fake.date_between(start_date=creation_date, end_date='+30y')
            status = np.random.randint(0, len(STATUS))
            if status == 0:
                taxonomy_status = 0
            else:
                taxonomy_status = np.random.randint(1, len(TAXONOMY_STATUS))
            loan = Loan("Loan" + str(i),
                        TYPE[np.random.randint(0, len(TYPE)-1)],
                        group_str,
                        counterparty,
                        np.random.randint(1, MAX_MATURITY),
                        np.random.randint(1, MAX_PRINCIPAL) * 5000,
                        np.random.randint(1, int(MAX_TAU*100))/100,
                        TEST_8_RATINGS.list()[np.random.randint(0, len(TEST_8_RATINGS.list())-1)],
                        STATUS[status],
                        TAXONOMY_STATUS[taxonomy_status],
                        random.uniform(0, 1),
                        creation_date=creation_date
                        )
            portfolio.add_loan(loan)

        return portfolio

    def _get_from_CSV(self, portfolioId):
        return []


