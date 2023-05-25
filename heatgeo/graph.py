import scanpy as sc
import graphtools as gt
import pygsp
from typing import Union
import umap
from graphtools.matrix import set_diagonal, to_array
from scipy import sparse
from sklearn.preprocessing import normalize
import numpy as np


#### Functions to compute diffusion operator and affinity on a pygsp graph ####


def diff_op(graph):
    """
    Compute the diffusion operator for a pygsp graph.
    """
    assert isinstance(graph, pygsp.graphs.Graph)
    K = set_diagonal(graph.W, 1)
    diff_op_ = normalize(K, norm="l1", axis=1)
    return diff_op_


def kernel_degree(graph):
    """
    Compute the kernel degree for a pygsp graph.
    """
    assert isinstance(graph, pygsp.graphs.Graph)
    K = set_diagonal(graph.W, 1)
    return to_array(K.sum(axis=1)).reshape(-1, 1)


def diff_aff(graph):
    """
    Compute the diffusion affinity for a pygsp graph.
    """
    assert isinstance(graph, pygsp.graphs.Graph)
    K = set_diagonal(graph.W, 1)
    row_degrees = kernel_degree(graph)

    if sparse.issparse(K):
        # diagonal matrix
        degrees = sparse.csr_matrix(
            (
                1 / np.sqrt(row_degrees.flatten()),
                np.arange(len(row_degrees)),
                np.arange(len(row_degrees) + 1),
            )
        )
        return degrees @ K @ degrees
    else:
        col_degrees = row_degrees.T
        return (K / np.sqrt(row_degrees)) / np.sqrt(col_degrees)


###------------------------Graphs Classes ----------------------------###


def get_knn_graph(X, knn=5, **kwargs):
    return pygsp.graphs.NNGraph(X, k=knn)


def get_alpha_decay_graph(X, knn=5, decay=40.0, anisotropy=0, n_pca=None, **kwargs):
    return gt.Graph(
        X,
        knn=knn,
        decay=decay,
        anisotropy=anisotropy,
        n_pca=n_pca,
        use_pygsp=True,
        random_state=42,
    ).to_pygsp()


def get_scanpy_graph(X, knn=5, **kwargs):
    adata = sc.AnnData(X)
    sc.pp.neighbors(adata, n_neighbors=knn)
    w = adata.obsp["connectivities"]
    return pygsp.graphs.Graph(w)


def get_umap_graph(X, knn=5, **kwargs):  # knn default to 15 in UMAP
    umap_op = umap.UMAP(n_neighbors=knn, metric="euclidean")
    umap_op.fit(X)
    w = umap_op.graph_.toarray()
    return pygsp.graphs.Graph(w)
