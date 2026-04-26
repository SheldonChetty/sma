import pandas as pd
import re
import emoji
import string

# ---------- CLEANING FUNCTION ----------
def clean_text(text):
    if pd.isna(text):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # 3. Remove Emojis
    text = emoji.replace_emoji(text, replace='')

    # 4. Handle Hashtags (remove # but keep word)
    text = re.sub(r'#', '', text)

    # 5. Remove Punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 6. Remove Extra Whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# ---------- READ FILE ----------
# For CSV (change column name if needed)
df = pd.read_csv("Book1.csv")   # Make sure your file is in same folder

# Apply cleaning on a column (change 'text' to your column name)
df['cleaned_text'] = df['text'].apply(clean_text)

# ---------- SAVE OUTPUT ----------
df.to_csv("cleaned_output.csv", index=False)

print("✅ Cleaning complete! Saved as cleaned_output.csv")