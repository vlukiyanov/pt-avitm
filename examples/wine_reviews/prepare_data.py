from os.path import isfile, join
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import save_npz
import textacy

max_features = 10000


def process(text):
    return textacy.preprocess.preprocess_text(
        text,
        fix_unicode=True,
        lowercase=True,
        transliterate=True,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=True,
        no_currency_symbols=True,
        no_punct=True,
        no_contractions=True,
        no_accents=True,
    )


print("Verifying data")
for item in ["winemag-data-130k-v2.csv"]:
    item = join("data_input", item)
    name = item.split(".")[0]
    if not isfile(item):
        raise ValueError('No input file "%s"' % item)

print("Loading data")
df = pd.read_csv(join("data_input", "winemag-data-130k-v2.csv"))

print("Cleaning data")
df["processed_text"] = df["description"].apply(process)

print("Vectorising data")
vectorizer = CountVectorizer(
    stop_words="english", max_features=max_features, max_df=0.9
)
term_document = vectorizer.fit_transform(df["processed_text"])

print("Saving data")
# TODO validation
length = term_document.shape[0]
save_npz(file="data/train.txt.npz", matrix=term_document[25000:, :].astype(np.float32))
save_npz(file="data/test.txt.npz", matrix=term_document[:25000, :].astype(np.float32))

with open("data/vocab.pkl", "wb") as f:
    pickle.dump(vectorizer.vocabulary_, f)
