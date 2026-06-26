import pandas as pd


class MetadataLoader:

    def __init__(self, file_path):
        self.file_path = file_path

    def load_metadata(self):
        """
        Reads metadata CSV and returns DataFrame.
        """
        df = pd.read_csv(self.file_path)
        return df

    def get_all_columns(self):
        """
        Returns all column names.
        """
        df = self.load_metadata()
        return df["COLUMN_NAME"].tolist()

    def get_all_tables(self):
        """
        Returns all table names.
        """
        df = self.load_metadata()
        return df["TABLE_NAME"].unique().tolist()

    def display_metadata(self):
        df = self.load_metadata()
        print(df)