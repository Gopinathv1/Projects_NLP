import pandas as pd
import joblib

from sklearn.metrics.pairwise import cosine_similarity


class SimilarityEngine:

    def __init__(self, metadata_path):

        self.df = pd.read_csv(metadata_path)

        self.vectorizer = joblib.load("models/tfidf.pkl")

        corpus = []

        for _, row in self.df.iterrows():

            corpus.append(
                f"{row['TABLE_NAME']} {row['COLUMN_NAME']}".lower()
            )

        self.matrix = self.vectorizer.transform(corpus)

    def find_matches(self, user_text, top_n=5):

        user_vector = self.vectorizer.transform([user_text.lower()])

        similarity = cosine_similarity(user_vector, self.matrix)

        similarity = similarity.flatten()

        self.df["Score"] = similarity

        result = self.df.sort_values(
            by="Score",
            ascending=False
        )

        return result.head(top_n)