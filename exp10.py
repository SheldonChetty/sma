# Install once
# pip install pandas matplotlib vaderSentiment wordcloud

import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv("Dataset/exp10.csv")

print("Dataset:\n", df.head())

# ---------------- SENTIMENT ANALYSIS ---------------- #
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

df['sentiment'] = df['review'].apply(get_sentiment)

print("\nSentiment Results:\n", df[['review','sentiment']])

# ---------------- SENTIMENT COUNT ---------------- #
counts = df['sentiment'].value_counts()

print("\nSentiment Count:\n")
print(counts.to_string())

# ---------------- BAR GRAPH ---------------- #
plt.bar(counts.index, counts.values)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# ---------------- NEGATIVE REVIEWS ---------------- #
df_neg = df[df['sentiment'] == "Negative"]

# ---------------- WORD CLOUD ---------------- #
text = " ".join(df_neg['review'])

wc = WordCloud(width=800, height=400, background_color='black').generate(text)

plt.imshow(wc)
plt.axis("off")
plt.title("Negative Review WordCloud")
plt.show()