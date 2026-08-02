from transformers import pipeline

translator = pipeline(
    task="translation",
    model="Helsinki-NLP/opus-mt-fr-en"
)

def translate(text):

    result = translator(text)

    return result[0]['translation_text']