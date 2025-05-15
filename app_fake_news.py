import streamlit as st
import pandas as pd
import re
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

# Set page configuration
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Title and description
st.title("📰 Fake News Detector")
st.markdown("""
This app uses an XGBoost model to classify news as **Real** or **Fake**. 
Enter the news article title and text below, and the model will predict its authenticity.
""")

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Load the saved model and vectorizer
try:
    xgb_model = joblib.load('xgb_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please ensure 'xgb_model.pkl' and 'tfidf_vectorizer.pkl' are in the working directory.")
    st.stop()

# User input
st.subheader("Enter News Article")
title_input = st.text_input("News Title:", placeholder="Enter the article title here")
text_input = st.text_area("News Text:", placeholder="Paste the article text here", height=200)

# Prediction
if st.button("Predict"):
    if not title_input.strip() and not text_input.strip():
        st.warning("Please enter either a title or text to classify.")
    else:
        # Combine title and text (handle empty inputs)
        title = title_input.strip() if title_input.strip() else ""
        text = text_input.strip() if text_input.strip() else ""
        content = title + " " + text if title and text else title or text

        # Clean the combined content
        cleaned_content = clean_text(content)

        # Transform the content using the loaded TF-IDF vectorizer
        content_vector = tfidf_vectorizer.transform([cleaned_content])

        # Predict using the loaded model
        prediction = xgb_model.predict(content_vector)[0]

        # Display result
        label = "Real" if prediction == 0 else "Fake"
        color = "green" if prediction == 0 else "red"
        st.markdown(f"**Prediction**: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)

# Footer
st.markdown("""
---
❤️ using Streamlit and XGBoost.  
Dataset: [Kaggle Fake News Prediction Dataset](https://www.kaggle.com/datasets/rajatkumar30/fake-news)
""")
