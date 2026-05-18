# Social Media Sentiment Analyzer 🐦

Analyzes sentiment of any text using both
VADER and TextBlob NLP models.

## Live Demo
[Click here](https://sentiment-analyzer-nlp-gji4c93qtakzjcwepx4d9y.streamlit.app)

## Features
- Single text analysis with 2 models
- Bulk analysis — paste multiple posts
- Word cloud generator with color themes
- Topic comparison — compare sentiment across 2 topics
- Download results as CSV

## Models Used
- VADER — optimized for social media text
- TextBlob — general purpose NLP sentiment

## Tools Used
- Python, Streamlit, VADER, TextBlob, WordCloud
- Pandas, Matplotlib, NumPy

## How to Run Locally
pip install streamlit pandas numpy matplotlib textblob vaderSentiment wordcloud
streamlit run app.py
