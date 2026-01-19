import re
import pandas as pd
from nltk.tokenize import word_tokenize

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zà-ÿ0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str):
    return word_tokenize(text)

def load_reviews(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)

    if "review_comment_message" not in df.columns:
        raise ValueError("column 'review_comment_message' missing. Check csv-file.")

    df = df[["review_id", "order_id", "review_score", "review_comment_message"]].copy()
    df["clean_text"] = df["review_comment_message"].apply(clean_text)

    df = df[df["clean_text"].str.len() >= 10].copy()

    df["tokens"] = df["clean_text"].apply(tokenize)
    return df

if __name__ == "__main__":
    df = load_reviews(r"data\raw\olist_order_reviews_dataset.csv")
    print(df.head(3))
    print("Texte Gesamt: ", len(df))
