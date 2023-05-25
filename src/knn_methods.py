from sklearn.neighbors import NearestNeighbors
import numpy as np
import scipy
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

class KNNClassifier:
    """KNN classifier for a given distance method."""

    def __init__(self, model) -> None:
        self.dist = None
        self.model = model

    def fit(self, data):
        self.model.fit(data)
        self.model.metric_computation(data)
        dist = self.model.dist
        self.dist = dist

    def evaluate(self, ground_dist, ks=[5]):
        assert self.dist is not None
        #Evaluating on the distance matrix where we set the diagonal to 0.
        dist = self.dist.copy()
        np.fill_diagonal(dist,0)
        return evaluate(dist, ground_dist, ks=ks)
    
    def fit_transform(self,data):
        emb = self.model.fit_transform(data)
        self.dist = squareform(pdist(emb))


def corrs(d1, d2):
    """Average spearman, and pearson correlation across points."""
    spearman_corrs = []
    pearson_corrs = []
    for i in range(len(d1)):
        s_correlation, pval = scipy.stats.spearmanr(d1[i], d2[i])
        p_correlation, pval = scipy.stats.pearsonr(d1[i], d2[i])
        spearman_corrs.append(s_correlation)
        pearson_corrs.append(p_correlation)
    spearman_corrs = np.array(spearman_corrs)
    return np.mean(spearman_corrs), np.mean(pearson_corrs)


def precision_at_k(pred, true, k=10):
    assert np.all(np.sum(true, axis=1) == k)
    assert np.all(np.sum(pred, axis=1) == k)
    return np.sum((pred + true) == 2) / (k * true.shape[0])


def evaluate(pred, true, ks=[1, 5, 10, 100, 500]):
    """
    Args:
        pred: dists
        true: dists
    returns:
        results: (p@K, spearmancorr, persoR, and norm between the two matrix
    """
    M = true.shape[0]
    neigh_pred = NearestNeighbors(
        n_neighbors=min(M - 1, ks[-1]), algorithm="auto", metric="precomputed"
    )
    neigh_true = NearestNeighbors(
        n_neighbors=min(M - 1, ks[-1]), algorithm="auto", metric="precomputed"
    )
    neigh_pred.fit(pred)
    neigh_true.fit(true)
    ps = []
    for k in ks:
        if k >= M:
            continue
        adj_pred = neigh_pred.kneighbors_graph(n_neighbors=k)
        adj_true = neigh_true.kneighbors_graph(n_neighbors=k)
        ps.append(precision_at_k(adj_pred, adj_true, k))
    c = corrs(pred, true)
    norm_f, norm_inf = np.linalg.norm(pred - true, ord="fro"), np.linalg.norm(
        pred - true, ord=np.inf
    )

    return (
        *c,
        *ps,
        norm_f / np.linalg.norm(true, ord="fro"),
        norm_inf / np.linalg.norm(true, ord=np.inf),
        norm_f / (M**2),
        norm_inf / (M**2),
    )



# MODIFIED FROM SKLEARN

def eval_k_means(kmeans, data, labels):
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    results = []

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
        )
    ]    
    return results