import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


class MetadataVectorizer:

    def __init__(self, metadata_path):

        self.metadata_path = metadata_path

        self.df = pd.read_csv(metadata_path)

        self.bow = CountVectorizer()

        self.tfidf = TfidfVectorizer()

    def build_corpus(self):

        corpus = []

        for _, row in self.df.iterrows():

            sentence = (
                            f"{row['TABLE_NAME']} "
                            f"{row['COLUMN_NAME']} "
                            f"{row['DESCRIPTION']}")

            corpus.append(sentence.lower())

        return corpus

    def train(self):

        corpus = self.build_corpus()

        bow_matrix = self.bow.fit_transform(corpus)

        tfidf_matrix = self.tfidf.fit_transform(corpus)

        os.makedirs("models", exist_ok=True)

        joblib.dump(self.bow, "models/bow.pkl")

        joblib.dump(self.tfidf, "models/tfidf.pkl")

        return bow_matrix, tfidf_matrix

    def load_models(self):

        bow = joblib.load("models/bow.pkl")

        tfidf = joblib.load("models/tfidf.pkl")

        return bow, tfidf