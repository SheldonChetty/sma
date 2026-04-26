import pandas as pd
import re
import string
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv("exp5.csv")

TEXT_COLUMN = "tweet"
LABEL_COLUMN = "label"

# ---------------- CLEANING FUNCTION ---------------- #
def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Handle hashtags (#word -> word)
    text = re.sub(r"#", "", text)

    # Remove emojis / special unicode
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Keep only alphabets
    text = re.sub(r"[^a-z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Apply cleaning
df["clean_text"] = df[TEXT_COLUMN].apply(clean_text)

# ---------------- SPLIT DATA ---------------- #
X = df["clean_text"]
y = df[LABEL_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- VECTORIZE ---------------- #
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------- TRAIN MODEL ---------------- #
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_vec, y_train)

# ---------------- EVALUATION ---------------- #
y_pred = model.predict(X_test_vec)

print("\n===== MODEL PERFORMANCE =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------- PREDICT FULL DATASET ---------------- #
X_all_vec = vectorizer.transform(df["clean_text"])
df["sentiment"] = model.predict(X_all_vec)

# ---------------- MAP LABELS ---------------- #
label_mapping = {
    0: "negative",
    1: "positive",
    2: "neutral"
}

df["sentiment"] = df["sentiment"].map(label_mapping)

# ---------------- COUNT RESULTS ---------------- #
counts = df["sentiment"].value_counts()

print("\n===== SENTIMENT COUNT =====")
print("Positive :", counts.get("positive", 0))
print("Negative :", counts.get("negative", 0))
print("Neutral  :", counts.get("neutral", 0))

# ---------------- SAVE OUTPUT ---------------- #
df.to_csv("sentiment_output.csv", index=False)
print("\n✅ Output saved as 'sentiment_output.csv'")
