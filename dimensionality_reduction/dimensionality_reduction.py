import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler


class DimensionalityReduction:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.pca = None
        self.tsvd = None
        self.minmax = None
        self.maxabs = None

    def reduce_dimensions(self, x, dataset: str = "train"):
        x = x.copy()
        if not isinstance(x, pd.DataFrame) and not isinstance(x, np.ndarray):
            if dataset == "train":
                self.maxabs = MaxAbsScaler()
                self.maxabs.fit(x)
                x = self.maxabs.transform(x).copy()
                self.tsvd = TruncatedSVD(n_components=self.n_components, random_state=42)
                self.tsvd.fit(x)
            if dataset != "train":
                x = self.maxabs.transform(x).copy()
            x = self.tsvd.transform(x).copy()
        else:
            if dataset == "train":
                self.minmax = MinMaxScaler()
                self.minmax.fit(x)
                x = self.minmax.transform(x).copy()
                self.pca = PCA(n_components=self.n_components, random_state=42)
                self.pca.fit(x)
            if dataset != "train":
                x = self.minmax.transform(x).copy()
            x = self.pca.transform(x).copy()
        return x
