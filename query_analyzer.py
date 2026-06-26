class QueryAnalyzer:

    def __init__(self, matches):

        self.matches = matches

    def analyze(self):

        measure = None

        dimension = None

        for _, row in self.matches.iterrows():

            dtype = row["DATA_TYPE"]

            column = row["COLUMN_NAME"]

            if dtype == "NUMBER" and measure is None:

                measure = column

            elif dtype == "VARCHAR" and dimension is None:

                dimension = column

        return measure, dimension