import streamlit as st
from translator import translate

st.title("🌍 AI Language Translator")

text = st.text_area("Enter French Text")

if st.button("Translate"):

    translated = translate(text)

    st.success(translated)