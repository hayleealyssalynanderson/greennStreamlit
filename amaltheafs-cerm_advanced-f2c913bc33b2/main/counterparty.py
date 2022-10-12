class Counterparty:

    def __init__(self, id, name, isin, legal_type, stakes, region, sector, subsector, rating, group):
        # Inputs
        self.id = id
        self.name = name
        self.isin = isin
        self.legal_type = legal_type
        self.stakes = stakes
        self.region = region
        self.sector = sector
        self.subsector = subsector
        self.rating = rating
        self.__group = group

    def group(self):
        return self.__group



