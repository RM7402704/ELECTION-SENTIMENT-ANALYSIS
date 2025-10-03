# tweet_sentiment_analysis.py
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"  # Add this line

# 0) Required downloads / setup for NLTK (first run)
import nltk
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')     # needed by TextBlob sometimes

# 1) Imports
import os
import re
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings('ignore')
plt.style.use('dark_background')  # global style for matplotlib charts

# 2) Utility functions: cleaning and sentiment helpers
lemmatizer = WordNetLemmatizer()
nltk_stopwords = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """
    Clean tweet text:
      - remove urls
      - lowercase
      - keep only letters a-z
      - tokenize, remove stopwords, lemmatize
    Returns cleaned string.
    """
    if pd.isna(text):
        return ""
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', str(text))
    # Lowercase
    text = text.lower()
    # Replace anything other than a-z with a space
    text = re.sub('[^a-z]', ' ', text)
    # Tokenize
    words = text.split()
    # Lemmatize and remove stopwords
    words = [lemmatizer.lemmatize(w) for w in words if w not in nltk_stopwords]
    return " ".join(words)


def get_subjectivity(text: str) -> float:
    """Return TextBlob subjectivity (0.0 - 1.0)."""
    if not text:
        return 0.0
    return TextBlob(text).sentiment.subjectivity


def get_polarity(text: str) -> float:
    """Return TextBlob polarity (-1.0 - 1.0)."""
    if not text:
        return 0.0
    return TextBlob(text).sentiment.polarity


def get_analysis_label(polarity: float) -> str:
    """Map polarity float to label."""
    if polarity > 0.0:
        return "positive"
    elif polarity < 0.0:
        return "negative"
    else:
        return "neutral"


def safe_read_csv(path: str) -> pd.DataFrame:
    """Read CSV with common fallbacks."""
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")
    # Use engine and lineterminator for robustness
    return pd.read_csv(path, lineterminator='\n', encoding='utf-8', low_memory=False)


def ensure_column(df: pd.DataFrame, col: str, default=None):
    """Ensure a column exists in df; if missing, create with default."""
    if col not in df.columns:
        df[col] = default
    return df


# 3) Load datasets (update these file paths if needed)
TRUMP_CSV = "hashtag_donaldtrump.csv"
BIDEN_CSV = "hashtag_joebiden.csv"

trump = safe_read_csv(TRUMP_CSV)
biden = safe_read_csv(BIDEN_CSV)

# 4) Quick peek and info (prints)
print("== Trump head ==")
print(trump.head(3))
print("\nColumns (trump):", list(trump.columns))
print("\n== Biden head ==")
print(biden.head(3))
print("\nColumns (biden):", list(biden.columns))

print("\nShapes:", "trump:", trump.shape, "biden:", biden.shape)
print("\nTrump info:")
print(trump.info())
print("\nBiden info:")
print(biden.info())

# 5) Normalize and add candidate column (lowercase candidate label)
trump = trump.copy()
biden = biden.copy()
trump['candidate'] = 'trump'
biden['candidate'] = 'biden'

# 6) Combine
data = pd.concat([trump, biden], ignore_index=True, sort=False)
print("\nCombined dataset shape:", data.shape)

# 7) Drop pure-empty rows (if entire row is NA) and reset index
data.dropna(how='all', inplace=True)
data.reset_index(drop=True, inplace=True)
print("After dropping all-nulls shape:", data.shape)

# 8) Ensure required columns exist before operations
for col in ['tweet', 'country', 'likes']:
    ensure_column(data, col, default=np.nan)

# 9) Exploratory Data Analysis (simple counts and plots)
# Number of tweets per candidate
tweets_count = data.groupby('candidate')['tweet'].count().reset_index().rename(columns={'tweet': 'tweet_count'})
print("\nTweets per candidate:")
print(tweets_count)

# Plotly bar for tweet counts
color_map = {'trump': '#0a84ff', 'biden': '#ff2d55'}  # blue for trump, pink/red for biden (customizable)
fig = px.bar(tweets_count, x='candidate', y='tweet_count',
             color='candidate', color_discrete_map=color_map,
             labels={'candidate': 'Candidate', 'tweet_count': 'Number of Tweets'},
             title='Number of Tweets per Candidate')
fig.show(color_map=color_map)

# Likes comparison (sum). If likes isn't numeric, convert safely
data['likes'] = pd.to_numeric(data['likes'], errors='coerce').fillna(0)
likes_comparison = data.groupby('candidate')['likes'].sum().reset_index()
print("\nLikes per candidate:")
print(likes_comparison)

fig2 = px.bar(likes_comparison, x='candidate', y='likes', color='candidate',
              color_discrete_map=color_map,
              labels={'candidate': 'Candidate', 'likes': 'Total Likes'},
              title='Comparison of Likes')
fig2.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white')
fig2.show()

# Top 10 countries by tweet count (if country column exists and not all NaN)
if data['country'].notna().any():
    top10countries = data.groupby('country')['tweet'].count().sort_values(ascending=False).reset_index().head(10)
    print("\nTop 10 countries by tweet count:")
    print(top10countries)

    fig3 = px.bar(top10countries, x='country', y='tweet', template='plotly_dark',
                  color_discrete_sequence=px.colors.qualitative.Dark24_r,
                  title='Top 10 Countries by Tweet Count')
    fig3.show()

    # Tweet counts per candidate in those top 10 countries
    tweet_df = data.groupby(['country', 'candidate'])['tweet'].count().reset_index(name='tweet_count')
    tweeters = tweet_df[tweet_df['country'].isin(top10countries['country'])]
    fig4 = px.bar(tweeters, x='country', y='tweet_count', color='candidate',
                  labels={'country': 'Country', 'tweet_count': 'Number of Tweets', 'candidate': 'Candidate'},
                  title='Tweet Counts for Each Candidate in Top 10 Countries',
                  template='plotly_dark', barmode='group', color_discrete_map=color_map)
    fig4.show()
else:
    print("Country column missing or all NaN — skipping country-level EDA.")


# Compute subjectivity and polarity
data['subjectivity'] = data['cleantext'].apply(get_subjectivity)
data['polarity'] = data['cleantext'].apply(get_polarity)
data['analysis'] = data['polarity'].apply(get_analysis_label)

# 11) Save processed master file for later use
processed_out = "processed_tweets_trump_biden.csv"
data.to_csv(processed_out, index=False, encoding='utf-8')
print(f"\nProcessed data saved to: {processed_out}")

# 12) Candidate-specific analysis functions
def candidate_report(df: pd.DataFrame, candidate_name: str, country_filter: str = None, top_n_words: int = 40):
    """
    Produce candidate-wise summary, plot sentiment distribution, and wordcloud.
    - country_filter: if provided, filter to that country (exact match).
    """
    candidate_df = df[df['candidate'] == candidate_name].copy()
    if country_filter:
        candidate_df = candidate_df[candidate_df['country'].str.upper() == country_filter.upper()]

    if candidate_df.empty:
        print(f"No tweets found for {candidate_name} (after applying filters).")
        return

    print(f"\n==== Report for {candidate_name.upper()} ====")
    print("Total tweets:", candidate_df.shape[0])
    print("Sentiment distribution (counts):")
    print(candidate_df['analysis'].value_counts())
    print("Average polarity:", candidate_df['polarity'].mean())
    print("Average subjectivity:", candidate_df['subjectivity'].mean())

    # Matplotlib bar for sentiment distribution (percentage)
    pct = (candidate_df['analysis'].value_counts(normalize=True) * 100).reindex(['positive', 'neutral', 'negative']).fillna(0)
    colors = ['orange', 'blue', 'red']
    plt.figure(figsize=(7, 5))
    pct.plot.bar(color=colors)
    plt.ylabel("% of tweets")
    plt.title(f"Distribution of Sentiments towards {candidate_name.capitalize()}")
    plt.show()

    # Word cloud based on cleaned text
    text_for_wc = " ".join(candidate_df['cleantext'].dropna().astype(str).tolist())
    if text_for_wc.strip():
        wc = WordCloud(background_color='black',
                       stopwords=STOPWORDS.union(nltk_stopwords),
                       width=1600, height=800, max_words=top_n_words, max_font_size=200,
                       colormap="viridis").generate(text_for_wc)
        plt.figure(figsize=(12, 10))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"WordCloud for {candidate_name.capitalize()}", pad=20)
        plt.show()
    else:
        print("Not enough textual content for wordcloud.")

    # Return sample of analysis
    sample_cols = ['tweet', 'cleantext', 'polarity', 'subjectivity', 'analysis']
    sample = candidate_df[sample_cols].head(10)
    print("\nSample processed tweets:")
    print(sample.to_string(index=False))
    # Save candidate csv
    outname = f"processed_{candidate_name}_tweets.csv"
    candidate_df.to_csv(outname, index=False, encoding='utf-8')
    print(f"Saved {candidate_name} processed tweets to {outname}")


# 13) Generate reports for Trump and Biden (optionally filter to US)
# Example: candidate_report(data, 'trump', country_filter='US')
candidate_report(data, 'trump', country_filter='US')   # Trump's US tweets analysis
candidate_report(data, 'biden', country_filter='US')   # Biden's US tweets analysis

# 14) (Optional) Overall comparison plot of polarity distributions across candidates
plt.figure(figsize=(10, 6))
for cand in data['candidate'].unique():
    subset = data[data['candidate'] == cand]['polarity'].dropna()
    if subset.shape[0] > 0:
        subset.plot(kind='kde', label=cand)  # density plot
plt.legend()
plt.title("Polarity Distribution by Candidate (KDE)")
plt.xlabel("Polarity")
plt.show()

print("\nAll done. Processed files were saved, and candidate reports generated. You can change country_filter in candidate_report() to analyze other countries.")