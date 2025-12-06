import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json


# --------- LOAD MODEL ----------
@st.cache_resource
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model


# --------- PREDICT ----------
def predict_emotion(text, tokenizer, model, id2label):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        top_idx = torch.argmax(logits, dim=1).item()
    return id2label[top_idx]


# --------- STREAMLIT UI ----------
st.set_page_config(page_title="Emotion Recognition App", layout="centered")
st.title("🎭 DistilBERT Emotion Recognition")
st.write("Enter text below to detect emotion using your fine-tuned DistilBERT model.")

# Path to your exported model folder
model_folder = "bert_model"  # <-- CHANGE THIS

# Load model
tokenizer, model = load_model(model_folder)

# Map LABEL_0, LABEL_1, ... to actual emotion names
id2label = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    5: "caring",
    6: "confusion",
    7: "curiosity",
    8: "desire",
    9: "disappointment",
    10: "disapproval",
    11: "disgust",
    12: "embarrassment",
    13: "excitement",
    14: "fear",
    15: "gratitude",
    16: "grief",
    17: "joy",
    18: "love",
    19: "nervousness",
    20: "optimism",
    21: "pride",
    22: "realization",
    23: "relief",
    24: "remorse",
    25: "sadness",
    26: "surprise",
    27: "neutral",
}

# Text input
text_input = st.text_area("Enter text:", height=150)

# Analyze button
if st.button("Analyze Emotion"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        label = predict_emotion(text_input, tokenizer, model, id2label)
        st.subheader(f"Predicted Emotion: **{label}**")
