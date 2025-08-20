# 🧠 Sentiment & Emotion Analyzer

An AI-powered web application built with **Streamlit** and **Hugging Face Transformers** that analyzes sentiment and emotions from text.  
It supports both **real-time single text analysis** and **bulk file uploads (TXT/CSV)** with interactive visualizations.

---

## ✨ Features
- 🔍 Sentiment Analysis → Classifies text as Positive / Negative / Neutral with confidence score.  
- 🎭 Emotion Detection → Detects emotions like Joy, Anger, Fear, Sadness, etc.  
- 📊 Interactive Visualizations → Bar charts for emotion distribution.  
- 📂 File Upload Support → Upload `.txt` or `.csv` files for batch analysis.  
- 📈 Trend Analysis → If dataset contains a `date` column, shows sentiment & emotion trends over time.  

---

## 🛠️ Tech Stack
- Python
- Streamlit
- Hugging Face Transformers
- PyTorch
- Matplotlib
- Pandas

---

## 🚀 Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sentiment-emotion-analyzer.git
   cd sentiment-emotion-analyzer
Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

Install dependencies:

pip install -r requirements.txt


If you don’t have a requirements.txt yet, install manually:

pip install streamlit transformers torch matplotlib pandas


Run the app:

streamlit run app.py

📂 File Upload Format

TXT File → One text entry per line.

CSV File → Must contain a text column. Optionally can include a date column for trend analysis.

📸 Screenshots

(Add some screenshots of your app here once running, e.g. analysis results and charts)

💡 Use Cases

Social Media Monitoring

Customer Feedback Analysis

Opinion Mining

Survey Analysis

📜 License

This project is licensed under the MIT License – feel free to use and modify.

👨‍💻 Author

Developed by Srinivas G.
