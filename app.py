import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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
    page_title="Sentiment Analyzer",
    page_icon="🐦",
    layout="wide"
)

st.title("🐦 Social Media Sentiment Analyzer")
st.markdown("Analyze sentiment of any text, "
            "topic or batch of social media posts.")
st.markdown("---")

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Helper functions
def analyze_textblob(text):
    blob       = TextBlob(text)
    polarity   = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    if polarity > 0.1:
        sentiment = "Positive"
        emoji     = "😊"
        color     = "#2ecc71"
    elif polarity < -0.1:
        sentiment = "Negative"
        emoji     = "😢"
        color     = "#e74c3c"
    else:
        sentiment = "Neutral"
        emoji     = "😐"
        color     = "#f39c12"
    return {
        'sentiment':     sentiment,
        'emoji':         emoji,
        'color':         color,
        'polarity':      round(polarity, 4),
        'subjectivity':  round(subjectivity, 4)
    }

def analyze_vader(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        sentiment = "Positive"
        emoji     = "😊"
        color     = "#2ecc71"
    elif compound <= -0.05:
        sentiment = "Negative"
        emoji     = "😢"
        color     = "#e74c3c"
    else:
        sentiment = "Neutral"
        emoji     = "😐"
        color     = "#f39c12"
    return {
        'sentiment': sentiment,
        'emoji':     emoji,
        'color':     color,
        'compound':  round(compound, 4),
        'positive':  round(scores['pos'], 4),
        'negative':  round(scores['neg'], 4),
        'neutral':   round(scores['neu'], 4)
    }

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

def get_top_words(texts, n=20):
    all_words = []
    stop_words = {
        'the', 'a', 'an', 'and', 'or',
        'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'is',
        'are', 'was', 'were', 'be', 'been',
        'this', 'that', 'it', 'i', 'you',
        'he', 'she', 'we', 'they', 'my',
        'your', 'his', 'her', 'its', 'our',
        'have', 'has', 'had', 'do', 'did',
        'will', 'would', 'can', 'could',
        'not', 'no', 'so', 'what', 'how'
    }
    for text in texts:
        words = clean_text(text).split()
        all_words.extend([
            w for w in words
            if w not in stop_words
            and len(w) > 2
        ])
    return Counter(all_words).most_common(n)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Single Analysis",
    "📊 Bulk Analysis",
    "☁️ Word Cloud",
    "📈 Comparison"
])

# Tab 1 — Single Analysis
with tab1:
    st.markdown("### Analyze Any Text")

    text_input = st.text_area(
        "Enter text to analyze:",
        placeholder="Paste any tweet, review, "
                    "comment or paragraph...",
        height=150
    )

    model_choice = st.radio(
        "Analysis model:",
        ["Both (recommended)",
         "VADER only",
         "TextBlob only"],
        horizontal=True
    )

    if st.button("🔍 Analyze",
                 type="primary"):
        if text_input.strip():
            st.markdown("---")

            if model_choice in [
                "Both (recommended)",
                "VADER only"
            ]:
                vader = analyze_vader(
                    text_input)
                st.markdown(
                    "#### 🎯 VADER Analysis")
                st.markdown(
                    f"<h3 style='color:"
                    f"{vader['color']}'>"
                    f"{vader['emoji']} "
                    f"{vader['sentiment']}"
                    f"</h3>",
                    unsafe_allow_html=True
                )
                v1, v2, v3, v4 = st.columns(4)
                v1.metric("Compound",
                          vader['compound'])
                v2.metric("Positive",
                          vader['positive'])
                v3.metric("Negative",
                          vader['negative'])
                v4.metric("Neutral",
                          vader['neutral'])

                # VADER score bar
                fig, ax = plt.subplots(
                    figsize=(10, 2))
                scores = [
                    vader['positive'],
                    vader['neutral'],
                    vader['negative']
                ]
                colors = ['#2ecc71',
                          '#f39c12',
                          '#e74c3c']
                labels = ['Positive',
                          'Neutral',
                          'Negative']
                left = 0
                for score, color, label \
                        in zip(scores,
                               colors,
                               labels):
                    ax.barh(
                        ['Sentiment'],
                        [score],
                        left=left,
                        color=color,
                        label=label,
                        height=0.4
                    )
                    if score > 0.05:
                        ax.text(
                            left + score/2,
                            0,
                            f'{score:.2f}',
                            ha='center',
                            va='center',
                            fontsize=10,
                            fontweight='bold',
                            color='white'
                        )
                    left += score
                ax.set_xlim(0, 1)
                ax.set_title(
                    'VADER Score Breakdown',
                    fontsize=12)
                ax.legend(
                    loc='upper right',
                    fontsize=9)
                ax.axis('off')
                plt.tight_layout()
                st.pyplot(fig)

            if model_choice in [
                "Both (recommended)",
                "TextBlob only"
            ]:
                tb = analyze_textblob(
                    text_input)
                st.markdown(
                    "#### 📝 TextBlob Analysis")
                st.markdown(
                    f"<h3 style='color:"
                    f"{tb['color']}'>"
                    f"{tb['emoji']} "
                    f"{tb['sentiment']}"
                    f"</h3>",
                    unsafe_allow_html=True
                )
                t1, t2 = st.columns(2)
                t1.metric(
                    "Polarity",
                    tb['polarity'],
                    help="-1=Negative, "
                         "+1=Positive"
                )
                t2.metric(
                    "Subjectivity",
                    tb['subjectivity'],
                    help="0=Objective, "
                         "1=Subjective"
                )

            # Word analysis
            words = clean_text(
                text_input).split()
            if words:
                st.markdown(
                    "#### 🔤 Word Analysis")
                w1, w2, w3 = st.columns(3)
                w1.metric("Word Count",
                          len(words))
                w2.metric("Unique Words",
                          len(set(words)))
                w3.metric("Avg Word Length",
                          f"{np.mean([len(w) for w in words]):.1f}")
        else:
            st.warning(
                "Please enter some text!")

# Tab 2 — Bulk Analysis
with tab2:
    st.markdown("### Bulk Sentiment Analysis")
    st.markdown(
        "Enter one text per line — "
        "analyze multiple posts at once.")

    bulk_input = st.text_area(
        "Enter texts (one per line):",
        placeholder=(
            "I love this product!\n"
            "This is terrible service.\n"
            "The weather is okay today.\n"
            "Amazing experience, highly recommend!\n"
            "Worst purchase I ever made."
        ),
        height=200
    )

    if st.button("📊 Analyze All",
                 type="primary"):
        if bulk_input.strip():
            lines = [
                l.strip()
                for l in bulk_input.split('\n')
                if l.strip()
            ]

            results = []
            for line in lines:
                vader = analyze_vader(line)
                tb    = analyze_textblob(line)
                results.append({
                    'Text':           line[:60] +
                                      '...' if
                                      len(line) > 60
                                      else line,
                    'VADER':          vader['sentiment'],
                    'TextBlob':       tb['sentiment'],
                    'Compound':       vader['compound'],
                    'Polarity':       tb['polarity'],
                    'Subjectivity':   tb['subjectivity']
                })

            df = pd.DataFrame(results)
            st.dataframe(df,
                         use_container_width=True,
                         hide_index=True)

            # Distribution chart
            fig, axes = plt.subplots(
                1, 2, figsize=(12, 4))

            vader_counts = df['VADER']\
                .value_counts()
            tb_counts    = df['TextBlob']\
                .value_counts()

            sent_colors = {
                'Positive': '#2ecc71',
                'Neutral':  '#f39c12',
                'Negative': '#e74c3c'
            }

            for ax, counts, title in zip(
                axes,
                [vader_counts, tb_counts],
                ['VADER Distribution',
                 'TextBlob Distribution']
            ):
                colors = [
                    sent_colors.get(s, '#95a5a6')
                    for s in counts.index
                ]
                ax.bar(counts.index,
                       counts.values,
                       color=colors,
                       edgecolor='black')
                ax.set_title(title, fontsize=12)
                ax.set_ylabel('Count')

            plt.tight_layout()
            st.pyplot(fig)

            # Summary
            pos = (df['VADER'] ==
                   'Positive').sum()
            neg = (df['VADER'] ==
                   'Negative').sum()
            neu = (df['VADER'] ==
                   'Neutral').sum()

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total Analyzed",
                      len(df))
            s2.metric("Positive 😊", pos)
            s3.metric("Negative 😢", neg)
            s4.metric("Neutral 😐",  neu)

            # Download
            st.download_button(
                "⬇️ Download Results",
                df.to_csv(index=False),
                "sentiment_results.csv",
                "text/csv"
            )
        else:
            st.warning(
                "Please enter some texts!")

# Tab 3 — Word Cloud
with tab3:
    st.markdown("### ☁️ Word Cloud Generator")

    wc_text = st.text_area(
        "Enter text for word cloud:",
        placeholder="Paste any text, article "
                    "or collection of posts...",
        height=150
    )

    col1, col2 = st.columns(2)
    with col1:
        colormap = st.selectbox(
            "Color scheme:",
            ["viridis", "plasma",
             "RdYlGn", "Blues",
             "Reds", "coolwarm"]
        )
    with col2:
        max_words = st.slider(
            "Max words:", 20, 200, 100)

    if st.button("☁️ Generate Word Cloud",
                 type="primary"):
        if wc_text.strip():
            cleaned = clean_text(wc_text)

            if len(cleaned.split()) < 5:
                st.warning(
                    "Please enter more text "
                    "for a meaningful word cloud.")
            else:
                wc = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    colormap=colormap,
                    max_words=max_words,
                    collocations=False
                ).generate(cleaned)

                fig, ax = plt.subplots(
                    figsize=(12, 5))
                ax.imshow(wc,
                          interpolation='bilinear')
                ax.axis('off')
                ax.set_title(
                    'Word Cloud',
                    fontsize=14)
                plt.tight_layout()
                st.pyplot(fig)

                # Top words
                top_words = get_top_words(
                    [wc_text], n=15)
                if top_words:
                    st.markdown(
                        "### 🔤 Top 15 Words")
                    words_df = pd.DataFrame(
                        top_words,
                        columns=['Word', 'Count']
                    )
                    fig2, ax2 = plt.subplots(
                        figsize=(10, 4))
                    ax2.barh(
                        words_df['Word'][::-1],
                        words_df['Count'][::-1],
                        color='#3498db',
                        edgecolor='black'
                    )
                    ax2.set_title(
                        'Most Frequent Words',
                        fontsize=12)
                    ax2.set_xlabel('Count')
                    plt.tight_layout()
                    st.pyplot(fig2)
        else:
            st.warning(
                "Please enter some text!")

# Tab 4 — Comparison
with tab4:
    st.markdown(
        "### 📈 Compare Sentiment "
        "Across Topics")

    st.markdown(
        "Enter topic name and sample "
        "texts to compare sentiment.")

    col1, col2 = st.columns(2)

    with col1:
        topic1 = st.text_input(
            "Topic 1 name:",
            value="Product A"
        )
        texts1 = st.text_area(
            "Topic 1 reviews:",
            value=(
                "Great product love it!\n"
                "Amazing quality worth every penny\n"
                "Best purchase ever made\n"
                "Highly recommend to everyone\n"
                "Could be better honestly"
            ),
            height=150
        )

    with col2:
        topic2 = st.text_input(
            "Topic 2 name:",
            value="Product B"
        )
        texts2 = st.text_area(
            "Topic 2 reviews:",
            value=(
                "Terrible product waste of money\n"
                "Very disappointed with quality\n"
                "Never buying again horrible\n"
                "Broke after one day use\n"
                "Not what I expected at all"
            ),
            height=150
        )

    if st.button("📊 Compare Topics",
                 type="primary"):
        topics   = {
            topic1: texts1,
            topic2: texts2
        }
        comp_results = {}

        for topic, texts in topics.items():
            lines = [
                l.strip()
                for l in texts.split('\n')
                if l.strip()
            ]
            scores = [
                analyze_vader(l)['compound']
                for l in lines
            ]
            comp_results[topic] = {
                'mean':    np.mean(scores),
                'scores':  scores,
                'lines':   lines
            }

        # Comparison chart
        fig, axes = plt.subplots(
            1, 2, figsize=(12, 5))

        topic_names = list(comp_results.keys())
        mean_scores = [
            comp_results[t]['mean']
            for t in topic_names
        ]
        bar_colors  = [
            '#2ecc71' if s >= 0
            else '#e74c3c'
            for s in mean_scores
        ]

        bars = axes[0].bar(
            topic_names, mean_scores,
            color=bar_colors,
            edgecolor='black'
        )
        for bar, val in zip(bars, mean_scores):
            axes[0].text(
                bar.get_x() +
                bar.get_width()/2,
                bar.get_height() +
                (0.01 if val >= 0 else -0.03),
                f'{val:.3f}',
                ha='center',
                fontweight='bold'
            )
        axes[0].axhline(
            y=0, color='black',
            linewidth=1)
        axes[0].set_title(
            'Average Sentiment Score',
            fontsize=12)
        axes[0].set_ylabel(
            'VADER Compound Score')
        axes[0].set_ylim(-1, 1)

        # Score distributions
        for topic, color in zip(
            topic_names,
            ['#3498db', '#e74c3c']
        ):
            axes[1].hist(
                comp_results[topic]['scores'],
                bins=10,
                alpha=0.6,
                color=color,
                label=topic,
                edgecolor='black'
            )
        axes[1].axvline(
            x=0, color='black',
            linewidth=1,
            linestyle='--'
        )
        axes[1].set_title(
            'Score Distribution',
            fontsize=12)
        axes[1].set_xlabel(
            'VADER Compound Score')
        axes[1].set_ylabel('Count')
        axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig)

        # Winner
        winner = max(
            comp_results,
            key=lambda x:
            comp_results[x]['mean']
        )
        loser  = min(
            comp_results,
            key=lambda x:
            comp_results[x]['mean']
        )
        st.success(
            f"✅ **{winner}** has more "
            f"positive sentiment "
            f"(score: "
            f"{comp_results[winner]['mean']:.3f})"
        )
        st.error(
            f"❌ **{loser}** has more "
            f"negative sentiment "
            f"(score: "
            f"{comp_results[loser]['mean']:.3f})"
        )

st.markdown("---")
st.markdown(
    "Built by **Jyotiraditya** | "
    "Social Media Sentiment Analyzer | "
    "Powered by VADER + TextBlob"
)