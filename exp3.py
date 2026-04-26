#pip install pandas emoji nltk
import pandas as pd
import re
import emoji
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords (run once)
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# ---------- CLEANING FUNCTION ----------
def clean_text(text):
    if pd.isna(text):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # 3. Remove Mentions (@username)
    text = re.sub(r'@\w+', '', text)

    # 4. Remove Emojis
    text = emoji.replace_emoji(text, replace='')

    # 5. Handle Hashtags (#word → word)
    text = re.sub(r'#', '', text)

    # 6. Remove Punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 7. Remove Extra Whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    # 8. Remove Stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# ---------- READ FILE ----------
df = pd.read_csv("Dataset/Book1.csv")   # change filename if needed

# Apply cleaning
df['cleaned_text'] = df['text'].apply(clean_text)

# ---------- SAVE OUTPUT ----------
df.to_csv("cleaned_output.csv", index=False)

print("✅ Cleaning complete! Saved as cleaned_output.csv")