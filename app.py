from metadata_loader import MetadataLoader
from preprocess import NLPPreprocessor


loader = MetadataLoader("metadata.csv")

print("========== TABLES ==========")

print(loader.get_all_tables())

print()

print("========== COLUMNS ==========")

print(loader.get_all_columns())


# Preprocess a sample question
processor = NLPPreprocessor()

question = "Show total revenue by nation"

tokens = processor.preprocess(question)

print(tokens)

#intent detection
from intent_detector import IntentDetector
intent = IntentDetector()

print(intent.detect_aggregation(tokens))

## Train the models

from vectorizer import MetadataVectorizer

trainer = MetadataVectorizer("metadata.csv")

bow_matrix, tfidf_matrix = trainer.train()

print("Bag of Words Shape:", bow_matrix.shape)

print("TF-IDF Shape:", tfidf_matrix.shape)


#Test Similarity Search

from similarity_engine import SimilarityEngine

engine = SimilarityEngine("metadata.csv")

result = engine.find_matches(
    "Show total sales by country and city"
)

print(result)


#SQL Generation

from sql_generator import SQLGenerator

generator = SQLGenerator("metadata.csv")

query = generator.generate(
            measure="SALES_AMOUNT",
            dimension="REGION",
            aggregation="SUM"
        )

print(query)