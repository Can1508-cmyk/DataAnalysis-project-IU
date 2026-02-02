import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zà-ÿ0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def top_terms(components, feature_names, top_n=12):
    topics = []
    for comp in components:
        top_idx = np.argsort(comp)[::-1][:top_n]
        topics.append([feature_names[i] for i in top_idx])
    return topics


def run_lsa(texts, n_topics=8, max_features=5000):
    tfidf = TfidfVectorizer(
        max_features=max_features,
        min_df=5,
        ngram_range=(1, 2),
    )
    X = tfidf.fit_transform(texts)

    svd = TruncatedSVD(n_components=n_topics, random_state=42)
    svd.fit(X)

    feats = tfidf.get_feature_names_out()
    return top_terms(svd.components_, feats, top_n=12)


def run_lda(texts, n_topics=8, max_features=5000):
    bow = CountVectorizer(
        max_features=max_features,
        min_df=5,
        ngram_range=(1, 2),
    )
    X = bow.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="batch",
    )
    lda.fit(X)

    feats = bow.get_feature_names_out()
    return top_terms(lda.components_, feats, top_n=12)


def save_topics(topics, out_path, title):
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(title + "\n\n")
        for i, terms in enumerate(topics, start=1):
            f.write(f"Topic {i}: " + ", ".join(terms) + "\n")


def plot_topic(topic_words, topic_id, out_dir="outputs/figures"):
    os.makedirs(out_dir, exist_ok=True)

    # einfache Ranggewichtung (höchstes Wort = höchste Relevanz)
    weights = list(range(len(topic_words), 0, -1))

    plt.figure(figsize=(8, 4))
    plt.barh(topic_words[::-1], weights[::-1])
    plt.xlabel("Relative Relevanz")
    plt.title(f"LDA Topic {topic_id}")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"topic_{topic_id}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Abbildung gespeichert: {out_path}")


if __name__ == "__main__":
    path = r"data\raw\olist_order_reviews_dataset.csv"
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    texts = df["review_comment_message"].dropna().astype(str).tolist()
    texts = [clean_text(t) for t in texts]
    texts = [t for t in texts if len(t) >= 10]

    n_topics = 8

    lsa_topics = run_lsa(texts, n_topics=n_topics)
    lda_topics = run_lda(texts, n_topics=n_topics)

    save_topics(lsa_topics, r"outputs\topics_lsa.txt", "LSA Topics (TF-IDF+SVD)")
    save_topics(lda_topics, r"outputs\topics_lda.txt", "LDA Topics (BoW+LDA)")

    # Visualisierung ausgewählter Topics (z.B. 3 Stück)
    plot_topic(lda_topics[0], topic_id=1)
    plot_topic(lda_topics[1], topic_id=2)
    plot_topic(lda_topics[2], topic_id=3)

    print("done. output:")
    print(" - outputs\\topics_lsa.txt")
    print(" - outputs\\topics_lda.txt")
