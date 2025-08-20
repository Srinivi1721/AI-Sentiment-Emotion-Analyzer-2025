from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I am feeling great today!")
# Install dependencies (only run once)
# pip install transformers torch streamlit matplotlib

from transformers import pipeline
import matplotlib.pyplot as plt

# -----------------------
# Step 1: Load Pre-trained Models
# -----------------------

# Sentiment Analysis (Positive/Negative/Neutral)
sentiment_analyzer = pipeline("sentiment-analysis")

# Emotion Classification (HuggingFace pretrained model)
emotion_analyzer = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    return_all_scores=True
)


# -----------------------
# Step 2: Define Analysis Function
# -----------------------

def analyze_text(text):
    # Sentiment
    sentiment = sentiment_analyzer(text)[0]

    # Emotions
    emotions = emotion_analyzer(text)[0]
    emotions_sorted = sorted(emotions, key=lambda x: x['score'], reverse=True)

    print("\n🔹 Input Text:", text)
    print("📌 Sentiment:", sentiment['label'], " | Confidence:", round(sentiment['score'], 3))
    print("\n🎭 Emotions Detected:")
    for emo in emotions_sorted:
        print(f"  {emo['label']}: {round(emo['score'], 3)}")

    # Visualization
    labels = [emo['label'] for emo in emotions_sorted]
    scores = [emo['score'] for emo in emotions_sorted]

    plt.bar(labels, scores, color='skyblue')
    plt.title("Emotion Distribution")
    plt.xticks(rotation=45)
    plt.ylabel("Confidence Score")
    plt.show()


# -----------------------
# Step 3: Test the System
# -----------------------

sample_texts = [
    "I am feeling very happy and excited today!",
    "This is the worst day of my life. I feel so sad.",
    "I’m nervous about my upcoming interview.",
    "The food was okay, not great but not terrible."
]

for txt in sample_texts:
    analyze_text(txt)
