import json

import pandas as pd


def get_corpus(name):
    with open(f'../data/corpora/{name}.md', 'r', encoding='utf-8') as file:
        corpus = file.read()
    return corpus


def get_questions_df(name):
    df = pd.read_csv("../data/questions_df.csv")
    df["references"] = df["references"].apply(json.loads)
    df = df[df["corpus_id"] == name]
    df = df.reset_index(drop=True)
    return df
