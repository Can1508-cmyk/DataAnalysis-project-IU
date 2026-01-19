# Olist Reviews NLP Topic Modeling

This project applies NLP techniques to a dataset of e-commerce customer reviews (Olist, Kaggle)
to extract dominant topics using TF-IDF, LSA, and LDA.

## Setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

## Data
Place the Kaggle Olist CSV files in data/raw/.

## Run
python src/preprocess.py
python src/topics.py
python src/coherence.py

## Output
See outputs/topics_lda.txt and outputs/topics_lsa.txt.
