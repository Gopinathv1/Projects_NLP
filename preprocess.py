import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from config import SYNONYMS

class NLPPreprocessor:

    def __init__(self):

        self.stop_words = set(stopwords.words("english"))

        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):

        text = text.lower()

        text = re.sub(r'[^a-zA-Z0-9 ]', '', text)

        return text

    def tokenize(self, text):

        return nltk.word_tokenize(text)

    def remove_stopwords(self, words):

        return [w for w in words if w not in self.stop_words]

    def lemmatize(self, words):

        return [self.lemmatizer.lemmatize(word) for word in words]

    def replace_synonyms(self, words):

        mapped = []

        for word in words:

            if word in SYNONYMS:

                mapped.append(SYNONYMS[word])

            else:

                mapped.append(word)

        return mapped

    def preprocess(self, sentence):

        sentence = self.clean_text(sentence)

        tokens = self.tokenize(sentence)

        tokens = self.remove_stopwords(tokens)

        tokens = self.lemmatize(tokens)

        tokens = self.replace_synonyms(tokens)

        return tokens