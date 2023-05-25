# Modification of the function from PHATE. Working only for distance matrix.

from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import scipy.spatial
from sklearn import manifold
import s_gd2


def classic(D, n_components=2, random_state=None):
    """Fast CMDS using random SVD
    Parameters
    ----------
    D : array-like, shape=[n_samples, n_samples]
        pairwise distances
    n_components : int, optional (default: 2)
        number of dimensions in which to embed `D`
    random_state : int, RandomState or None, optional (default: None)
        numpy random state
    Returns
    -------
    Y : array-like, embedded data [n_sample, ndim]
    """
    D = D**2
    D = D - D.mean(axis=0)[None, :]
    D = D - D.mean(axis=1)[:, None]
    pca = PCA(
        n_components=n_components, svd_solver="randomized", random_state=random_state
    )
    Y = pca.fit_transform(D)
    return Y


def sgd(D, w=None, n_components=2, random_state=None, init=None):
    """Metric MDS using stochastic gradient descent
    Parameters
    ----------
    D : array-like, shape=[n_samples, n_samples]
        pairwise distances
    n_components : int, optional (default: 2)
        number of dimensions in which to embed `D`
    random_state : int or None, optional (default: None)
        numpy random state
    init : array-like or None
        Initialization algorithm or state to use for MMDS
    Returns
    -------
    Y : array-like, embedded data [n_sample, ndim]
    """
    N = D.shape[0]
    D = squareform(D, checks=False)
    # Metric MDS from s_gd2
    print("Using s_gd2 for MDS.", w)
    Y = s_gd2.mds_direct(N, D, w=w, init=init, random_seed=random_state)
    return Y


def smacof(
    D,
    n_components=2,
    metric=True,
    init=None,
    random_state=None,
    verbose=0,
    max_iter=3000,
    eps=1e-6,
    n_jobs=1,
):
    """Metric and non-metric MDS using SMACOF
    Parameters
    ----------
    D : array-like, shape=[n_samples, n_samples]
        pairwise distances
    n_components : int, optional (default: 2)
        number of dimensions in which to embed `D`
    metric : bool, optional (default: True)
        Use metric MDS. If False, uses non-metric MDS
    init : array-like or None, optional (default: None)
        Initialization state
    random_state : int, RandomState or None, optional (default: None)
        numpy random state
    verbose : int or bool, optional (default: 0)
        verbosity
    max_iter : int, optional (default: 3000)
        maximum iterations
    eps : float, optional (default: 1e-6)
        stopping criterion
    Returns
    -------
    Y : array-like, shape=[n_samples, n_components]
        embedded data
    """
    # Metric MDS from sklearn
    Y, _ = manifold.smacof(
        D,
        n_components=n_components,
        metric=metric,
        max_iter=max_iter,
        eps=eps,
        random_state=random_state,
        n_jobs=n_jobs,
        n_init=1,
        init=init,
        verbose=verbose,
    )
    return Y

    # initialize all by CMDS


def embed_MDS(
    X,
    ndim=2,
    seed=2,
    solver="sgd",
    how="metric",
    input_is_dist=True,
    distance_metric="euclidean",
    mds_weights=None,
):

    X_dist = X if input_is_dist else squareform(pdist(X, distance_metric))

    Y_classic = classic(X_dist, n_components=ndim, random_state=seed)
    # metric is next fastest

    if ndim > 2 and solver == "sgd":
        print("Changed MDS solver to `smacof`, `sge` not implemented for 2<dim.")
        solver = "smacof"

    if solver == "sgd":
        # use sgd2 if it is available
        Y = sgd(X_dist, w=mds_weights ,n_components=ndim, random_state=seed, init=Y_classic)
        # sgd2 currently only supports n_components==2
    elif solver == "smacof":
        Y = smacof(
            X_dist, n_components=ndim, random_state=seed, init=Y_classic, metric=True
        )
    else:
        raise RuntimeError
    if how == "metric":
        # re-orient to classic
        _, Y, _ = scipy.spatial.procrustes(Y_classic, Y)
        return Y

    # nonmetric is slowest
    Y = smacof(X_dist, n_components=ndim, random_state=seed, init=Y, metric=False)
    # re-orient to classic
    _, Y, _ = scipy.spatial.procrustes(Y_classic, Y)
    return Y
