import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from vaderSentiment.vaderSentiment import (
    SentimentIntensityAnalyzer)
from wordcloud import WordCloud
from collections import Counter
import re
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Review Analyzer",
    page_icon="⭐",
    layout="wide"
)

st.title("⭐ Product Review Analyzer")
st.markdown("Analyze product reviews to extract "
            "insights, sentiment trends and key themes.")
st.markdown("---")

analyzer = SentimentIntensityAnalyzer()

SAMPLE_REVIEWS = {
    "Smartphone": [
        {"text": "Amazing phone! Camera quality is outstanding and battery lasts all day.", "rating": 5},
        {"text": "Great value for money. Performance is smooth and display is beautiful.", "rating": 5},
        {"text": "Good phone but heating issues after long gaming sessions.", "rating": 3},
        {"text": "Battery drains fast. Camera is decent but not as advertised.", "rating": 2},
        {"text": "Worst phone I ever bought. Screen broke after one week.", "rating": 1},
        {"text": "Excellent build quality. Fast charging is a game changer.", "rating": 5},
        {"text": "Average performance. Nothing special for the price.", "rating": 3},
        {"text": "Love the design and camera. Software updates are slow though.", "rating": 4},
        {"text": "Terrible customer service when I reported the defect.", "rating": 1},
        {"text": "Best smartphone under this budget. Highly recommended!", "rating": 5},
        {"text": "Screen is gorgeous. Face unlock works perfectly every time.", "rating": 4},
        {"text": "Disappointed with the speaker quality. Expected much better.", "rating": 2},
        {"text": "Solid phone with great performance. Camera could be improved.", "rating": 4},
        {"text": "Phone gets very hot during video calls. Unacceptable.", "rating": 2},
        {"text": "Absolutely love this phone. Best purchase of the year!", "rating": 5}
    ],
    "Earphones": [
        {"text": "Crystal clear sound quality. Bass is deep and punchy.", "rating": 5},
        {"text": "Good earphones but the ear tips hurt after long use.", "rating": 3},
        {"text": "Excellent noise cancellation. Perfect for office use.", "rating": 5},
        {"text": "Stopped working after 2 months. Very poor build quality.", "rating": 1},
        {"text": "Amazing sound for the price. Highly recommended budget option.", "rating": 4},
        {"text": "Mic quality is terrible for calls. Music is fine though.", "rating": 2},
        {"text": "Best earphones under 1000 rupees. Exceeded expectations.", "rating": 5},
        {"text": "Connection keeps dropping every few minutes. Very frustrating.", "rating": 1},
        {"text": "Good sound isolation. Comfortable for long hours.", "rating": 4},
        {"text": "Decent product but wire tangles very easily.", "rating": 3}
    ],
    "Laptop": [
        {"text": "Blazing fast performance. SSD makes everything instant.", "rating": 5},
        {"text": "Great laptop for coding and development work.", "rating": 5},
        {"text": "Battery backup is only 4 hours. Very disappointing.", "rating": 2},
        {"text": "Keyboard feel is amazing. Display is bright and sharp.", "rating": 4},
        {"text": "Fan noise is very loud during heavy tasks.", "rating": 3},
        {"text": "Perfect for students. Lightweight and powerful.", "rating": 5},
        {"text": "Trackpad is unresponsive. Had to use external mouse.", "rating": 2},
        {"text": "Excellent build quality. Feels premium and solid.", "rating": 5},
        {"text": "Heats up badly. Thermal throttling affects performance.", "rating": 2},
        {"text": "Good laptop overall. Worth the price for the specs.", "rating": 4}
    ]
}

def get_sentiment(text):
    scores   = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return 'Positive', '#2ecc71', compound
    elif compound <= -0.05:
        return 'Negative', '#e74c3c', compound
    else:
        return 'Neutral', '#f39c12', compound

def star_display(rating):
    return "⭐" * rating + "☆" * (5 - rating)

def get_top_words(reviews, n=15):
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but',
        'in', 'on', 'at', 'to', 'for', 'of',
        'with', 'by', 'is', 'are', 'was', 'were',
        'be', 'been', 'this', 'that', 'it', 'i',
        'my', 'very', 'so', 'after', 'its',
        'not', 'no', 'have', 'has', 'just',
        'get', 'got', 'much', 'good', 'great',
        'bad', 'product', 'really', 'also'
    }
    words = []
    for review in reviews:
        clean = re.sub(
            r'[^a-zA-Z\s]', '',
            review.lower()).split()
        words.extend([
            w for w in clean
            if w not in stop_words
            and len(w) > 3
        ])
    return Counter(words).most_common(n)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Sample Products",
    "📝 Paste Reviews",
    "📁 Upload CSV",
    "🏆 Insights"
])

with tab1:
    st.markdown("### Analyze Sample Product Reviews")

    product      = st.selectbox(
        "Select product:",
        list(SAMPLE_REVIEWS.keys()))
    reviews_data = SAMPLE_REVIEWS[product]
    texts        = [r['text'] for r in reviews_data]
    ratings      = [r['rating'] for r in reviews_data]
    sentiments   = [get_sentiment(t) for t in texts]
    sent_labels  = [s[0] for s in sentiments]
    compounds    = [s[2] for s in sentiments]

    avg_rating   = np.mean(ratings)
    avg_compound = np.mean(compounds)
    pos_count    = sent_labels.count('Positive')
    neg_count    = sent_labels.count('Negative')
    neu_count    = sent_labels.count('Neutral')
    total        = len(reviews_data)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Reviews",    total)
    col2.metric("Avg Rating",       f"{avg_rating:.1f} ⭐")
    col3.metric("Positive 😊",      f"{pos_count} ({pos_count/total*100:.0f}%)")
    col4.metric("Negative 😢",      f"{neg_count} ({neg_count/total*100:.0f}%)")
    col5.metric("Sentiment Score",  f"{avg_compound:.3f}")

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.markdown("#### ⭐ Rating Distribution")
        fig, ax   = plt.subplots(figsize=(6, 4))
        rating_counts = Counter(ratings)
        stars  = [1, 2, 3, 4, 5]
        counts = [rating_counts.get(s, 0) for s in stars]
        colors = ['#e74c3c', '#e67e22', '#f39c12',
                  '#2ecc71', '#27ae60']
        bars   = ax.barh([f"{s} ⭐" for s in stars],
                         counts, color=colors,
                         edgecolor='black')
        for bar, val in zip(bars, counts):
            ax.text(bar.get_width() + 0.1,
                    bar.get_y() + bar.get_height()/2,
                    str(val), va='center',
                    fontweight='bold')
        ax.set_title('Rating Distribution', fontsize=12)
        ax.set_xlabel('Number of Reviews')
        plt.tight_layout()
        st.pyplot(fig)

    with right:
        st.markdown("#### 😊 Sentiment Distribution")
        fig2, ax2   = plt.subplots(figsize=(6, 4))
        sent_counts = [pos_count, neu_count, neg_count]
        sent_colors = ['#2ecc71', '#f39c12', '#e74c3c']
        sent_lbls   = [
            f'Positive\n({pos_count})',
            f'Neutral\n({neu_count})',
            f'Negative\n({neg_count})'
        ]
        non_zero = [
            (c, col, lbl)
            for c, col, lbl in zip(
                sent_counts, sent_colors, sent_lbls)
            if c > 0
        ]
        if non_zero:
            counts_nz, colors_nz, lbls_nz = zip(*non_zero)
            ax2.pie(counts_nz, labels=lbls_nz,
                    colors=colors_nz, autopct='%1.1f%%',
                    startangle=90,
                    wedgeprops={'edgecolor': 'white'})
        ax2.set_title('Sentiment Distribution', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown("#### 📈 Sentiment Trend")
    fig3, ax3  = plt.subplots(figsize=(12, 3))
    bar_colors = [
        '#2ecc71' if s == 'Positive'
        else '#e74c3c' if s == 'Negative'
        else '#f39c12'
        for s in sent_labels
    ]
    ax3.bar(range(1, total + 1), compounds,
            color=bar_colors, edgecolor='black', alpha=0.8)
    ax3.axhline(y=0, color='black', linewidth=1)
    ax3.axhline(y=avg_compound, color='blue',
                linewidth=2, linestyle='--',
                label=f'Avg: {avg_compound:.3f}')
    ax3.set_title(f'Sentiment Score per Review — {product}',
                  fontsize=13)
    ax3.set_xlabel('Review Number')
    ax3.set_ylabel('VADER Compound Score')
    ax3.legend()
    plt.tight_layout()
    st.pyplot(fig3)

    st.markdown("#### 📋 All Reviews")
    for i, (review, rating, sent) in enumerate(
        zip(texts, ratings, sentiments), 1
    ):
        sentiment, color, score = sent
        with st.expander(
            f"Review {i} — "
            f"{star_display(rating)} | {sentiment}"
        ):
            st.markdown(review)
            st.caption(f"Sentiment: {sentiment} "
                       f"| Score: {score:.3f} "
                       f"| Rating: {rating}/5")

with tab2:
    st.markdown("### Paste Your Own Reviews")
    st.markdown("Enter one review per line. "
                "Optionally add rating as `[5]` at the start.")

    custom_input = st.text_area(
        "Paste reviews:",
        placeholder=(
            "[5] Excellent product highly recommend!\n"
            "[3] Average quality nothing special\n"
            "[1] Complete waste of money avoid\n"
            "[4] Good but could be better\n"
            "[5] Amazing fast delivery great quality"
        ),
        height=250
    )
    product_name = st.text_input(
        "Product name:", value="My Product")

    if st.button("📊 Analyze Reviews", type="primary"):
        if custom_input.strip():
            lines = [l.strip() for l in
                     custom_input.split('\n') if l.strip()]
            parsed = []
            for line in lines:
                rating_match = re.match(
                    r'^\[(\d)\]\s*(.*)', line)
                if rating_match:
                    rating = int(rating_match.group(1))
                    text   = rating_match.group(2)
                else:
                    rating = None
                    text   = line
                sent, color, score = get_sentiment(text)
                parsed.append({
                    'Text':      text[:60] + '...'
                                 if len(text) > 60 else text,
                    'Rating':    rating or 'N/A',
                    'Sentiment': sent,
                    'Score':     round(score, 3)
                })

            df  = pd.DataFrame(parsed)
            st.dataframe(df, use_container_width=True,
                         hide_index=True)

            pos = (df['Sentiment'] == 'Positive').sum()
            neg = (df['Sentiment'] == 'Negative').sum()
            neu = (df['Sentiment'] == 'Neutral').sum()
            avg = df['Score'].mean()

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total",       len(df))
            s2.metric("Positive 😊", pos)
            s3.metric("Negative 😢", neg)
            s4.metric("Avg Score",   f"{avg:.3f}")

            st.download_button(
                "⬇️ Download Analysis",
                df.to_csv(index=False),
                f"{product_name}_analysis.csv",
                "text/csv"
            )
        else:
            st.warning("Please enter some reviews!")

with tab3:
    st.markdown("### Upload Reviews CSV")
    st.info("CSV must have a column named "
            "'review', 'text' or 'comment'.")

    uploaded = st.file_uploader(
        "Upload CSV file:", type=['csv'])

    if uploaded:
        try:
            df_upload = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df_upload)} rows!")
            st.dataframe(df_upload.head(),
                         use_container_width=True)

            text_col = None
            for col in ['review', 'text', 'comment',
                        'Review', 'Text', 'Comment']:
                if col in df_upload.columns:
                    text_col = col
                    break

            if text_col:
                if st.button("📊 Analyze CSV",
                             type="primary"):
                    texts_csv = df_upload[
                        text_col].dropna().tolist()
                    results = []
                    for text in texts_csv[:100]:
                        sent, color, score = \
                            get_sentiment(str(text))
                        results.append({
                            'Text':      str(text)[:50],
                            'Sentiment': sent,
                            'Score':     round(score, 3)
                        })
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df,
                                 use_container_width=True,
                                 hide_index=True)
                    pos = (results_df['Sentiment'] ==
                           'Positive').sum()
                    neg = (results_df['Sentiment'] ==
                           'Negative').sum()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Analyzed", len(results_df))
                    c2.metric("Positive", pos)
                    c3.metric("Negative", neg)
            else:
                st.error("No 'review', 'text' or "
                         "'comment' column found.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

with tab4:
    st.markdown("### 🏆 Keyword & Theme Insights")

    insight_product = st.selectbox(
        "Select product for insights:",
        list(SAMPLE_REVIEWS.keys()),
        key="insight_product"
    )

    reviews_i  = [r['text'] for r in
                  SAMPLE_REVIEWS[insight_product]]
    ratings_i  = [r['rating'] for r in
                  SAMPLE_REVIEWS[insight_product]]
    sents_i    = [get_sentiment(t)[0] for t in reviews_i]

    pos_reviews = [r for r, s in zip(reviews_i, sents_i)
                   if s == 'Positive']
    neg_reviews = [r for r, s in zip(reviews_i, sents_i)
                   if s == 'Negative']

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 😊 Positive Keywords")
        if pos_reviews:
            pos_words = get_top_words(pos_reviews, n=10)
            if pos_words:
                words, counts = zip(*pos_words)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(list(words)[::-1],
                        list(counts)[::-1],
                        color='#2ecc71',
                        edgecolor='black')
                ax.set_title('Top Positive Keywords',
                             fontsize=12)
                plt.tight_layout()
                st.pyplot(fig)

    with col2:
        st.markdown("#### 😢 Negative Keywords")
        if neg_reviews:
            neg_words = get_top_words(neg_reviews, n=10)
            if neg_words:
                words, counts = zip(*neg_words)
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.barh(list(words)[::-1],
                         list(counts)[::-1],
                         color='#e74c3c',
                         edgecolor='black')
                ax2.set_title('Top Negative Keywords',
                              fontsize=12)
                plt.tight_layout()
                st.pyplot(fig2)

    st.markdown("#### ☁️ All Reviews Word Cloud")
    all_text = ' '.join(reviews_i)
    clean    = re.sub(r'[^a-zA-Z\s]', '',
                      all_text.lower())

    if len(clean.split()) > 5:
        wc = WordCloud(
            width=800, height=300,
            background_color='white',
            colormap='RdYlGn',
            max_words=80,
            collocations=False
        ).generate(clean)
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        ax3.imshow(wc, interpolation='bilinear')
        ax3.axis('off')
        plt.tight_layout()
        st.pyplot(fig3)

    st.markdown("#### 📊 Rating vs Sentiment Score")
    compounds_i    = [get_sentiment(t)[2] for t in reviews_i]
    colors_scatter = [
        '#2ecc71' if r >= 4
        else '#e74c3c' if r <= 2
        else '#f39c12'
        for r in ratings_i
    ]
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.scatter(ratings_i, compounds_i,
                c=colors_scatter, s=100,
                alpha=0.8, edgecolors='black')
    ax4.set_xlabel('Star Rating')
    ax4.set_ylabel('VADER Compound Score')
    ax4.set_title('Rating vs Sentiment Correlation',
                  fontsize=12)
    ax4.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig4)

st.markdown("---")
st.markdown(
    "Built by **Jyotiraditya** | "
    "Product Review Analyzer | "
    "Powered by VADER Sentiment Analysis"
)