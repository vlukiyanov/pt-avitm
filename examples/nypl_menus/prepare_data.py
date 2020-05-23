from os.path import isfile, join
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import textacy

max_features = 7500


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
for item in ["Menu.csv", "MenuItem.csv", "Dish.csv", "MenuPage.csv"]:
    item = join("data_input", item)
    name = item.split(".")[0]
    if not isfile(item):
        raise ValueError('No input file "%s"' % item)

print("Loading data")
menu = pd.read_csv(join("data_input", "Menu.csv")).rename(columns={"id": "menu_id"})
menu_item = pd.read_csv(join("data_input", "MenuItem.csv")).rename(
    columns={"id": "menu_item_id"}
)
dish = pd.read_csv(join("data_input", "Dish.csv")).rename(columns={"id": "dish_id"})
menu_page = pd.read_csv(join("data_input", "MenuPage.csv")).rename(
    columns={"id": "menu_page_id"}
)

print("Merging data")
merged = pd.merge(
    pd.merge(menu, menu_page, on="menu_id"),
    pd.merge(menu_item, dish, on="dish_id"),
    on="menu_page_id",
)

print("Cleaning data")
texts = (
    merged.groupby(["menu_id"])
    .agg({"name_y": lambda x: " ".join(x)})
    .rename(columns={"name_y": "text"})
)

texts["processed_text"] = texts["text"].apply(process)

print("Vectorising data")
vectorizer = CountVectorizer(
    stop_words="english", max_features=max_features, max_df=0.9
)
term_document = vectorizer.fit_transform(texts["processed_text"])

print("Saving data")
# TODO validation
np.save("data/test.txt.npy", term_document[16000:, :].todense())
np.save("data/train.txt.npy", term_document[:16000, :].todense())

with open("data/vocab.pkl", "wb") as f:
    pickle.dump(vectorizer.vocabulary_, f)
