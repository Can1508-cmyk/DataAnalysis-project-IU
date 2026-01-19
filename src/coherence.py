import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zà-ÿ0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def top_words_per_topic(lda_model, feature_names, top_n=10):
    topics = []
    for comp in lda_model.components_:
        top_idx = np.argsort(comp)[::-1][:top_n]
        topics.append([feature_names[i] for i in top_idx])
    return topics


if __name__ == "__main__":
    path = r"data\raw\olist_order_reviews_dataset.csv"
    df = pd.read_csv(path, encoding="utf-8")

    texts = df["review_comment_message"].dropna().astype(str).tolist()
    texts = [clean_text(t) for t in texts if len(t) >= 10]

    tokenized_texts = [t.split() for t in texts]

    dictionary = Dictionary(tokenized_texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)

    k_values = [4, 6, 8, 10, 12]

    print("Coherence (c_v) pro topic count K:\n")

    for k in k_values:
        vectorizer = CountVectorizer(min_df=5, ngram_range=(1, 2), max_features=5000)
        X = vectorizer.fit_transform(texts)

        lda = LatentDirichletAllocation(
            n_components=k,
            random_state=42,
            learning_method="batch"
        )
        lda.fit(X)

        feature_names = vectorizer.get_feature_names_out()
        topics = top_words_per_topic(lda, feature_names, top_n=10)

        cm = CoherenceModel(
            topics=topics,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence="c_v"
        )
        coherence = cm.get_coherence()

        print(f"K={k:2d} coherence={coherence:.4f}")
