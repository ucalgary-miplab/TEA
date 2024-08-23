import pickle

from sklearn.decomposition import PCA, KernelPCA


class DimRed:
    def __init__(self, ncomps, method="PCA"):
        self.ncomps = ncomps

        if method == "PCA":
            self.model = PCA(n_components=ncomps)
        elif method == "KPCA":
            self.model = KernelPCA(n_components=ncomps)
        else:
            print("Undefined method")

    def fit(self, X):
        self.model.fit(X)

    def save(self, path):
        pickle.dump(self.model, open(path, "wb"))

    def load(self, path):
        self.model = pickle.load(open(path, "rb"))

    def transform(self, X):
        return self.model.transform(X)

    def inverse_transform(self, X):
        return self.model.inverse_transform(X)
