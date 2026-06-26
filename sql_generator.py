import pandas as pd


class SQLGenerator:

    def __init__(self, metadata_path):

        self.df = pd.read_csv(metadata_path)

    def get_table(self, column):

        row = self.df[self.df["COLUMN_NAME"] == column]

        if len(row):

            return row.iloc[0]["TABLE_NAME"]

        return None

    def generate(
            self,
            measure,
            dimension=None,
            aggregation="SUM"):

        table = self.get_table(measure)

        if table is None:

            return "Unable to determine table."

        sql = "SELECT\n"

        if dimension:

            sql += f"    {dimension},\n"

        sql += f"    {aggregation}({measure}) AS VALUE\n"

        sql += f"FROM {table}\n"

        if dimension:

            sql += f"GROUP BY {dimension};"

        else:

            sql += ";"

        return sql