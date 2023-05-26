# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/other_emb.ipynb.

# %% auto 0
__all__ = ['RandWalkGeo', 'DiffusionMap', 'ShortestPath', 'PhateBasic']

# %% ../nbs/other_emb.ipynb 2
from .embedding import BaseEmb, new_heatgeo
import heatgeo.graph as graph_utils
import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import Union
from scipy.sparse.csgraph import shortest_path
from pygsp.graphs import NNGraph

try:
    # Optional dependencies
    import phate
except ImportError as imp_err:
    phate = imp_err

# %% ../nbs/other_emb.ipynb 3
class RandWalkGeo(new_heatgeo):
    """ HeatGeo with a random walk matrix instead of Heat kernel."""

    _valid_methods = [
        "exact",
        "affinity",
        "symmetric",
    ]  

    def __init__(
        self,
        knn: int,
        anisotropy: int = 0,
        decay: int = 40,
        n_pca: int = 40,
        tau: int = 10,
        emb_dim: int = 2,
        filter_method: str = "exact",
        order: int = 32,
        lap_type: str = "normalized",
        log_normalize: bool = False,
        scale_factor: float = 1,
        denoising: bool = False,
        n_ref: int = 50,
        n_svd: int = 50,
        graph_type: str = "alpha",
    ):

        super().__init__(
            knn=knn,
            anisotropy=anisotropy,
            decay=decay,
            n_pca=n_pca,
            tau=tau,
            emb_dim=emb_dim,
            filter_method=filter_method,
            order=order,
            lap_type=lap_type,
            log_normalize=log_normalize,
            scale_factor=scale_factor,
            denoising=denoising,
            n_ref=n_ref,
            n_svd=n_svd,
            graph_type=graph_type,
        )

        if filter_method not in self._valid_methods:
            raise ValueError("method must be one of {}".format(self._valid_methods))

    def metric_computation(self, data):
        # TODO: wrap this in one classe/method adding the Cheb. approximation
        # similar to the Heat_filter.
        if self.filter_method == "exact":
            # P = self.graph.diff_op.toarray()
            P = graph_utils.diff_op(self.graph).toarray()
            diffusion = np.linalg.matrix_power(P, self.tau)
        elif self.filter_method == "affinity":
            # A = self.graph.diff_aff.toarray()
            A = graph_utils.diff_aff(self.graph).toarray()
            diffusion = np.linalg.matrix_power(A, self.tau)
        elif self.filter_method == "symmetric":
            # D = self.graph.kernel_degree.squeeze()
            # P = self.graph.diff_op.toarray()
            D = graph_utils.kernel_degree(self.graph).squeeze()
            P = graph_utils.diff_op(self.graph).toarray()
            Pt = np.linalg.matrix_power(P, self.tau)
            diffusion = np.diag(D**0.5) @ Pt @ np.diag(D**-0.5)

        self.dist = (
            np.sqrt(-4 * self.tau * np.log(diffusion + 1e-16)) ** self.scale_factor
        )

        if self.check_symmetric():
            self.symetrize_dist()

# %% ../nbs/other_emb.ipynb 4
class DiffusionMap(BaseEmb):
    """Diffusion Map embedding with different graph construction."""
    def __init__(
        self,
        knn: int = 0,
        decay: int = 40,
        n_pca: int = 40,
        tau: float = 1,
        emb_dim: int = 2,
        anisotropy: int = 0,
        graph_type: str = "alpha",
        **kwargs
    ):
        super().__init__(
            knn=knn,
            decay=decay,
            n_pca=n_pca,
            anisotropy=anisotropy,
            tau=tau,
            emb_dim=emb_dim,
            graph_type=graph_type,
        )

    def metric_computation(self, data):
        # P = self.graph.P.toarray()
        P = graph_utils.diff_op(self.graph).toarray()
        eval, evec = np.linalg.eig(P)
        eval, evec = np.real(eval), np.real(evec)
        eval = eval**self.tau
        emb = eval[None, :] * evec
        self.dist = squareform(pdist(emb))

    def diffusion_emb(self, data):
        # P = self.graph.P.toarray()
        P = graph_utils.diff_op(self.graph).toarray()
        eval, evec = np.linalg.eig(P)
        eval = eval**self.tau
        emb = eval[None, :] * evec
        return emb

    def fit_transform(
        self,
        data,
    ) -> np.ndarray:
        self.fit(data)
        P = graph_utils.diff_op(self.graph).toarray()
        eval, evec = np.linalg.eig(P)
        eval, evec = np.real(eval), np.real(evec)
        eval = eval**self.tau
        order_eval = np.argsort(np.abs(eval))[::-1]
        self.emb = (
            eval[None, order_eval[: self.emb_dim]] * evec[:, order_eval[: self.emb_dim]]
        )
        return self.emb

# %% ../nbs/other_emb.ipynb 5
class ShortestPath(BaseEmb):
    """Shortest path embedding with different graph construction."""
    def __init__(
        self,
        knn: int,
        anisotropy: int = 0,
        decay: int = 40,
        n_pca: int = 40,
        graph_type: str = "alpha",
        **kwargs
    ):
        super().__init__(
            knn=knn,
            anisotropy=anisotropy,
            decay=decay,
            n_pca=n_pca,
            graph_type=graph_type,
        )

    def fit(self, data):
        self.graph = NNGraph(data, k=self.knn)

    def metric_computation(self, data):
        euc_dist = squareform(pdist(data))
        A = self.graph.A.toarray()
        euc_dist[A == 0] = 0
        self.dist = shortest_path(euc_dist, method="auto", directed=False)

# %% ../nbs/other_emb.ipynb 6
class PhateBasic(BaseEmb):
    """Wrapper for PHATE."""

    def __init__(
        self,
        knn: int,
        anisotropy: int = 0,
        decay: int = 40,
        n_pca: int = 40,
        tau: Union[int, str] = "auto",
        emb_dim: int = 2,
    ):
        if isinstance(phate, ImportError):
            raise ImportError(
                "Install phate to use this embedding. "
                "You can install it with `pip install phate` or the dev. version of heatgeo."
            )
        super().__init__(knn, anisotropy, decay, n_pca, tau=tau, emb_dim=emb_dim)

    def fit(self, data):
        self.phate_op = phate.PHATE(
            knn=self.knn,
            n_components=self.emb_dim,
            anisotropy=self.anisotropy,
            n_pca=self.n_pca,
            verbose=False,
            t=self.tau,
        )
        self.phate_op.fit(data)

    def metric_computation(self, data):
        """Compute the potential distance matrix."""
        potential = self.phate_op.diff_potential
        self.dist = squareform(pdist(potential))
