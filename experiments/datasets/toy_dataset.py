from scipy.stats import special_ortho_group
from sklearn.metrics import pairwise_distances

import graphtools
import itertools
import numpy as np
import pygsp
import sklearn.datasets as skd
import phate
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import pdist, squareform

# Modified from https://github.com/atong01/MultiscaleEMD and https://homepages.ecs.vuw.ac.nz/~marslast/Code/Ch6/lle.py


class Dataset:
    """Dataset class for Optimal Transport."""

    def __init__(self):
        super().__init__()
        self.X = None
        self.labels = None
        self.graph = None

    def get_labels(self):
        return self.labels

    def get_data(self):
        return self.X

    def get_graph(self):
        """Create a graphtools graph if does not exist."""
        if self.graph is None:
            self.graph = graphtools.Graph(self.X, use_pygsp=True)
        return self.graph

    def standardize_data(self):
        """Standardize data putting it in a unit box around the origin.
        This is necessary for quadtree type algorithms
        """
        X = self.X
        minx = np.min(self.X, axis=0)
        maxx = np.max(self.X, axis=0)
        self.std_X = (X - minx) / (maxx - minx)
        return self.std_X

    def rotate_to_dim(self, dim):
        """Rotate dataset to a different dimensionality."""
        self.rot_mat = special_ortho_group.rvs(dim)[: self.X.shape[1]]
        self.high_X = np.dot(self.X, self.rot_mat)
        return self.high_X


class SwissRoll(Dataset):
    def __init__(
        self,
        n_points=100,
        manifold_noise=0.05,
        width=1,
        random_state=42,
        rotate=False,
        rotate_dim=None,
        n_clusters=5,
        clustered = False,
        train_fold = None
    ):
        super().__init__()

        np.random.seed(random_state)

        if clustered:
            #mv = np.concatenate((np.random.multivariate_normal(mean = [7,0.5],cov = [[1.5,0],[0,0.05]],size = int(n_points/2)),
            #        np.random.multivariate_normal(mean = [12,0.5],cov = [[1.5,0],[0,0.05]],size = n_points-int(n_points/2))))

            t = np.concatenate((7+ 1. * np.random.normal(size = int(n_points/2)),
                    12 + 1.*np.random.normal(size = n_points-int(n_points/2))))[None,:]
            h = width * np.random.rand(1, n_points)
            self.clusters = np.concatenate((np.zeros(int(n_points/2)),np.ones(n_points-int(n_points/2))))
        else:
            t = 3 * np.pi / 2 * (1 + 2 * np.random.rand(1, n_points))
            h = width * np.random.rand(1, n_points)
        
        data = np.concatenate((t * np.cos(t), h, t * np.sin(t)))
        self.X = data.T

        if rotate and rotate_dim is not None:
            self.X = self.rotate_to_dim(rotate_dim)
            self.X = self.X + manifold_noise * np.random.randn(n_points, rotate_dim)
        else:
            self.X = self.X + manifold_noise * np.random.randn(n_points, 3)

        self.t = t
        self.h = h
        # define n_clusters labels depending on self.t
        np.linspace(t.min(), t.max(), n_clusters)
        self.labels = np.digitize(t, np.linspace(t.min(), t.max(), n_clusters))
        self.labels = self.labels.reshape(-1)

    def get_geodesic(self):
        # true_coords = np.stack([self.means[:, 1], self.t / 10], axis=1)
        true_coords = np.concatenate((self.t, self.h)).T
        geodesic_dist = pairwise_distances(true_coords, metric="euclidean")
        return geodesic_dist


# Swiss Roll class where the manifold is in 3D, i.e. the geodesic distance is computed on by taking into account the
# stretching of t.


class SwissRollStretch(SwissRoll):
    """Swiss roll class, but the geodesic distance account for the streching of t."""

    def _unroll_t(self, t):
        t = t.flatten()
        return 0.5 * ((np.sqrt(t**2 + 1) * t) + np.arcsinh(t)).reshape(1, -1)

    def get_geodesic(self):
        u_t = self._unroll_t(self.t)
        true_coords = np.concatenate((u_t, self.h)).T
        geodesic_dist = pairwise_distances(true_coords, metric="euclidean")
        return geodesic_dist


class TreePhate(Dataset):
    """Dataset from PHATE, modified for our purposes.
    The geodeisc distances are computed from a manifold without noise and the Data are a noisy version."""

    def __init__(
        self,
        n_dim: int = 10,
        n_points: int = 200,
        n_branch: int = 10,
        manifold_noise: float = 4,
        knn_geodesic: int = 10,  # Number of nearest neighbors to use for the shortest path distance.
        random_state=42,
        clustered = None,
        train_fold = None
    ):
        super().__init__()
        self.n_dim = n_dim
        self.n_points = n_points
        self.n_branch = n_branch
        self.knn_geodesic = knn_geodesic

        # The manifold witout noise
        self.gt_X, self.labels = phate.tree.gen_dla(
            n_dim=self.n_dim,
            n_branch=self.n_branch,
            branch_length=self.n_points,
            sigma=0,
            seed=random_state,
        )

        # The noisy manifold
        noise = np.random.normal(0, 10*manifold_noise, self.gt_X.shape)
        self.X = self.gt_X + noise

        self.labels = np.array([i // n_points for i in range(n_branch * n_points)])

    def get_geodesic(self):
        graph = pygsp.graphs.NNGraph(self.gt_X, k=self.knn_geodesic)
        euc_dist = squareform(pdist(self.gt_X))
        A = graph.A.toarray()
        euc_dist[A == 0] = 0
        geodesic_dist = shortest_path(euc_dist, method="auto", directed=False)
        return geodesic_dist
