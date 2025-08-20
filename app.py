# Install dependencies first (run in terminal if not installed)
# pip install streamlit transformers torch matplotlib pandas

import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------
# Load Pre-trained Models
# -----------------------

@st.cache_resource
def load_models():
    sentiment_model = pipeline("sentiment-analysis")
    emotion_model = pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        return_all_scores=True
    )
    return sentiment_model, emotion_model

sentiment_analyzer, emotion_analyzer = load_models()

# Helper function for single text analysis
def analyze_text(text):
    sentiment = sentiment_analyzer(text)[0]
    emotions = emotion_analyzer(text)[0]
    emotions_sorted = sorted(emotions, key=lambda x: x['score'], reverse=True)
    return sentiment, emotions_sorted

# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(page_title="AI Sentiment & Emotion Analyzer", page_icon="🧠", layout="centered")

st.title("🧠 Sentiment & Emotion Analyzer")
st.write("Paste any text or upload a file for AI-powered sentiment and emotion insights.")

# -----------------------
# Text Input
# -----------------------
user_text = st.text_area("Enter text here:", height=150)

if st.button("🔍 Analyze"):
    if user_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        sentiment, emotions_sorted = analyze_text(user_text)

        # Display Sentiment
        st.subheader("📌 Sentiment Analysis")
        st.write(f"**Sentiment:** {sentiment['label']}  \n**Confidence:** {round(sentiment['score'], 3)}")

        # Display Emotions
        st.subheader("🎭 Emotion Analysis")
        for emo in emotions_sorted:
            st.write(f"{emo['label']}: {round(emo['score'], 3)}")

        # Plot emotions
        labels = [emo['label'] for emo in emotions_sorted]
        scores = [emo['score'] for emo in emotions_sorted]

        fig, ax = plt.subplots()
        ax.bar(labels, scores, color='skyblue')
        ax.set_title("Emotion Distribution")
        ax.set_ylabel("Confidence Score")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45)
        st.pyplot(fig)

# -----------------------
# File Upload Section
# -----------------------
uploaded_file = st.file_uploader("Upload a file (CSV or test.txt)", type=["txt", "csv"])

if uploaded_file is not None:
    if uploaded_file.name.endswith("test.txt"):
        text_data = uploaded_file.read().decode("utf-8").splitlines()
        df = pd.DataFrame(text_data, columns=["text"])
        st.success(f"Uploaded file: {uploaded_file.name}")
    else:
        df = pd.read_csv(uploaded_file)

        if "text" not in df.columns:
            st.error("CSV must have a 'text' column.")
            df = None
        else:
            df = df[["text"] + ([col for col in df.columns if col.lower() == "date"] or [])]
            st.success(f"Uploaded file: {uploaded_file.name}")

    if df is not None:
        st.write("📄 Loaded Data:", df.head())

        if st.button("🚀 Analyze File"):
            sentiments = []
            dominant_emotions = []

            for text in df["text"].dropna():
                sentiment, emotions_sorted = analyze_text(text)
                sentiments.append(sentiment['label'])
                dominant_emotions.append(emotions_sorted[0]['label'])

            df["sentiment"] = sentiments
            df["dominant_emotion"] = dominant_emotions

            st.subheader("📊 Analysis Results")
            st.dataframe(df)

            # Sentiment Distribution
            st.subheader("📌 Sentiment Distribution")
            st.bar_chart(df["sentiment"].value_counts())

            # Emotion Distribution
            st.subheader("🎭 Dominant Emotion Distribution")
            st.bar_chart(df["dominant_emotion"].value_counts())

            # -----------------------
            # 📈 Trend Analysis (if date exists)
            # -----------------------
            if "date" in df.columns:
                try:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    df = df.dropna(subset=["date"])

                    # Sentiment trend
                    st.subheader("📈 Sentiment Trend Over Time")
                    sentiment_trend = df.groupby("date")["sentiment"].apply(lambda x: x.value_counts().idxmax())
                    st.line_chart(sentiment_trend.value_counts().sort_index())

                    # Emotion trend
                    st.subheader("📈 Emotion Trend Over Time")
                    emotion_trend = df.groupby("date")["dominant_emotion"].apply(lambda x: x.value_counts().idxmax())
                    st.line_chart(emotion_trend.value_counts().sort_index())

                except Exception as e:
                    st.error(f"Could not process date column: {e}")
else:
    st.info("Please upload a file.")
