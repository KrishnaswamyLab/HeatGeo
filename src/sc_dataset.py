import numpy as np
import scanpy as sc
import sys
import os
from dotenv import load_dotenv
from src import DATA_DIR
from sklearn.model_selection import train_test_split

load_dotenv()

if "SHARE_DATA" in os.environ:
    DATA_DIR = os.environ["SHARE_DATA"]


def tnet_dataset(path, embed_name="pcs", label_name="sample_labels", max_dim=100):
    a = np.load(path, allow_pickle=True)
    return a[embed_name][:, :max_dim], a[label_name], np.unique(a[label_name])


class PBMC:
    def __init__(self, n_points, random_state, train_fold = True) -> None:
        import scanpy as sc
        
        sc.settings.datasetdir = os.path.join(DATA_DIR)
        adata = sc.datasets.pbmc3k_processed()

        train_idx, test_idx = train_test_split(np.arange(adata.shape[0]), test_size=0.5, random_state=42)
        if train_fold:
            adata = adata[train_idx]
        else:
            adata = adata[test_idx]

        sc.pp.subsample(adata, n_obs=n_points, random_state=random_state)
        self.X = adata.X

        labels = adata.obs["louvain"]
        labels = labels.replace(
            to_replace=[
                "CD4 T cells",
                "CD14+ Monocytes",
                "B cells",
                "CD8 T cells",
                "NK cells",
                "FCGR3A+ Monocytes",
                "Dendritic cells",
                "Megakaryocytes",
            ],
            value=[0, 1, 2, 3, 4, 5, 6, 7],
        )

        self.labels = labels




class EBData:
    def __init__(self, n_points, random_state) -> None:
        data_file = "eb_velocity_v5.npz"
        np.random.seed(random_state)
        X, labels, _ = tnet_dataset(os.path.join(DATA_DIR, data_file))
        if n_points > X.shape[0]:
            print("Warning: n_points > X.shape[0]")
            self.X = X
            self.labels = labels
        else:
            rand_idx = np.random.choice(np.arange(X.shape[0]), size=n_points, replace=False)
            self.X = X[rand_idx]
            self.labels = labels[rand_idx]

class IPSC:
    def __init__(self, n_points, random_state):
        self.n_points = n_points
        data_file = "ipscData.mat"
        import scipy.io
        df = scipy.io.loadmat(os.path.join(DATA_DIR, "ipsc",data_file))
        rand_idx = np.random.choice(np.arange(df['data'].shape[0]), size=n_points, replace=False)
        self.X = df['data'][rand_idx]
        self.labels = df['data_time'][rand_idx,0]

class AdataTraj:
    def __init__(self, data_file, n_points, random_state=42):
        adata = sc.read_h5ad(os.path.join(DATA_DIR, data_file))
        self.n_points = n_points
        self.random_state = random_state
        self.adata_pr = self._process(adata)
        self.X = None
        self.labels = None

    def _process(self, adata):
        pass


class MNIST:
    def __init__(self, n_points, random_state = 42, train_fold = True):
        from sklearn.datasets import load_digits
        (self.X, self.labels) = load_digits(return_X_y=True)
        train_idx, test_idx = train_test_split(np.arange(self.X.shape[0]), test_size=0.5, random_state=42)

        m, s = np.mean(self.X[train_idx]), np.std(self.X[train_idx])
    
        if train_fold:
            self.X = (self.X[train_idx] - m) /s
            self.labels = self.labels[train_idx]
        else:
            self.X = (self.X[test_idx] -m) / s
            self.labels = self.labels[test_idx]

class MultiCiteSeq(AdataTraj):
    def __init__(self, data_file, n_points=2000, random_state=42):
        super().__init__(data_file, n_points, random_state)
        self.X = self.adata_pr.obsm["X_pca"]  # Using the PCA embedding
        self.labels = self.adata_pr.obs["day"].astype("category")

    def _process(self, adata):
        sc.pp.subsample(adata, n_obs=self.n_points, random_state=self.random_state)
        return adata


def adata_dataset(path, embed_name="X_pca", label_name="day", max_dim=100):
    adata = sc.read_h5ad(path)
    labels = adata.obs[label_name].astype("category")
    ulabels = labels.cat.categories
    return adata.obsm[embed_name][:, :max_dim], labels, ulabels


def load_dataset(path, max_dim=100):
    if path.endswith("h5ad"):
        return adata_dataset(path, max_dim=max_dim)
    if path.endswith("npz"):
        return tnet_dataset(path, max_dim=max_dim)
    raise NotImplementedError()
