import streamlit as st
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load model and tokenizer ---
model = load_model("sentiment_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# --- Constants ---
MAX_LEN = 30

# --- Preprocessing Function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    return text

# Load stopwords
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

# --- Streamlit UI ---
st.title("🧠 Sentiment Analysis")
st.write("Enter a tweet or text snippet to analyze its sentiment.")

user_input = st.text_area("Text Input", "")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        prediction = model.predict(padded)[0][0]
        sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"
        st.markdown(f"**Prediction:** {sentiment}")
        st.caption(f"Confidence: {prediction:.2f}")
