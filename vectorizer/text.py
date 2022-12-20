import pandas as pd
import os
import numpy as np
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer


class TextVectorizer:
    def __init__(self, use_w2v: bool = True):
        self.use_w2v = use_w2v
        self.tfidf = None

    @staticmethod
    def __get_doc2vec(x: pd.Series):
        google_w2v = KeyedVectors.load(os.path.join(".", "google_word2vec", "google_w2v_100k.bin"),
                                       mmap='r')
        corpus_w2v = []
        for doc in x:
            doc_w2v = []
            for token in doc.split():
                try:
                    doc_w2v.append(list(google_w2v[token]))
                except:
                    pass
            if len(doc_w2v) != 0:
                doc_w2v = np.array(doc_w2v)
                if doc_w2v.ndim == 1:
                    corpus_w2v.append(doc_w2v)
                else:
                    corpus_w2v.append(doc_w2v.mean(axis=0))
            else:
                corpus_w2v.append(np.array([0] * 300))
        return np.array(corpus_w2v)

    def vectorize(self, x: pd.Series, dataset: str = "train"):
        x = x.copy()
        if not self.use_w2v:
            if dataset == "train":
                self.tfidf = TfidfVectorizer()
                self.tfidf.fit(x)
            x = self.tfidf.transform(x).copy()
        else:
            x = self.__get_doc2vec(x).copy()
        return x
