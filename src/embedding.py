import graphtools as gt
import numpy as np
import pygsp
from src.mds import embed_MDS
import scprep
import scipy
from scipy.spatial.distance import jensenshannon
from numpy.linalg import matrix_power
from src.utils import interpolate, time_entropy, get_optimal_heat
import phate

# from MultiscaleEMD import DiffusionCheb
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import pinv
from math import pi
from src.cheb import expm_multiply
from scipy.sparse.linalg import eigsh
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils.extmath import randomized_svd
from src.filter_approx import Heat_Euler
from scipy.sparse.csgraph import shortest_path
from pygsp.graphs import NNGraph
from typing import Union
from src.filter_approx import Heat_filter
from scipy.cluster.vq import vq
import src.graph as graph_utils


class BaseEmb:
    """Base class for embedding methods.
    Arguments
    ---------
    knn: int
        Number of nearest neighbors to use for the graph.
    anisotropy: int
        Anisotropy parameter for the graph.
    decay: int
        Decay parameter for the kernel.
    n_pca: int
        Number of principal components to use for knn estimation.
    tau: int or str
        Diffusion time of the diffusion operator on the graph.
    emb_dim: int
        Dimension of the embedding.
    order: int
        Order of the Chebyshev approximation.
    random_state: int
        Random state for the embedding.
    scale_factor: int
        Power when computing the distance matrix.
    tau_min: float
        Minimum diffusion time for the diffusion operator.
    tau_max: float
        Maximum diffusion time for the diffusion operator.
    n_tau: int
        Number of diffusion times for the multiscale diffusion operator.
    n_landmarks: int
        Number of landmarks to summarize the data.
    solver: str
        Solver to use for MDS, `"sgd"` or `"smacof"`.
    lap_type: str
        Type of Laplacian to use for the graph `"normalized"` or `"combinatorial"`.
    filter_method: str
        Method to use for Heat approx. `"pygsp"` or `"euler"`, `"mar"`.
    graph_type: str
        Type of graph to use for the embedding `"knn"` or `"alpha"` or `scanpy`.
    """

    # NOTE / TODO: probably too many parameters, keep only the ones needed for all classes.

    def __init__(
        self,
        knn: int,
        anisotropy: int = 0,
        decay: int = 40,
        n_pca: int = 40,
        tau: Union[int, str] = "auto",
        emb_dim: int = 2,
        order: int = 32,
        random_state: int = 42,
        scale_factor: float = 2.0,
        tau_min: float = 0.1,
        tau_max: float = 1.0,
        n_tau: int = 1,
        n_landmarks: Union[int, None] = None,
        solver: str = "sgd",
        lap_type: str = "normalized",
        filter_method: str = "pygsp",
        graph_type: str = "alpha",
        mds_weights: Union[str,None] =  None,
    ):
        super().__init__()
        self.knn = knn
        self.dist = None
        self.graph = None
        self.anisotropy = anisotropy
        self.decay = decay
        self.emb = None
        self.n_pca = n_pca
        self.tau = tau
        self.emb_dim = emb_dim
        self.order = order
        self.random_state = random_state
        self.scale_factor = scale_factor
        self.tau_min = tau_min
        self._tau_max = tau_max
        self._n_tau = n_tau
        self.n_landmarks = n_landmarks
        self.solver = solver
        self.lap_type = lap_type
        self.filter_method = filter_method
        self.graph_type = graph_type
        self.mds_weights = mds_weights

    def fit(self, data) -> None:
        if isinstance(data, (gt.graphs.kNNPyGSPGraph, pygsp.graphs.Graph)):
            self.graph = data
        elif self.graph_type == "knn": # simple knn graph
            self.graph = graph_utils.get_knn_graph(data, self.knn)
        elif self.graph_type == "alpha": # alpha-decay graph uses in PHATE
            self.graph = graph_utils.get_alpha_decay_graph(data, knn=self.knn, decay=self.decay, anisotropy=self.anisotropy, n_pca=self.n_pca)
        elif self.graph_type == "scanpy": # knn graph used in Scanpy.
            self.graph = graph_utils.get_scanpy_graph(data, self.knn)
        elif self.graph_type == "umap": # knn graph used in UMAP.
            self.graph = graph_utils.get_umap_graph(data, self.knn)
        else:
            raise ValueError("Graph type not recognized.")

        self.graph.compute_laplacian(lap_type=self.lap_type)

    def symetrize_dist(self) -> None:
        self.dist = 0.5 * (self.dist + self.dist.T)

    # TODO: make it one function
    def check_symmetric(self) -> bool:
        return np.allclose(self.dist, self.dist.T, atol=10e-6, rtol=10e-6)

    def metric_computation(self, data):
        raise NotImplementedError

    def fit_transform(
        self,
        data,
    ) -> np.ndarray:
        self.fit(data)
        self.metric_computation(data)
        self.emb = embed_MDS(self.dist, self.emb_dim, solver=self.solver, mds_weights=self.mds_weights)
        return self.emb

    # NOTE: WIP: the idea is to get the geodesic from the euclidean embedding.
    def geodesic_euc(
        self, data, labels_0, labels_1, nsteps=20, euc_dim=50, emb_dim=2, **kwargs
    ) -> tuple:
        emb_euc = self.fit_transform(data, ndim=euc_dim, **kwargs)
        paths, labels = interpolate(emb_euc[labels_0, :], emb_euc[labels_1, :], nsteps)
        full_emb = np.concatenate((emb_euc, paths), axis=0)
        return (
            embed_MDS(
                full_emb, ndim=emb_dim, input_is_dist=False, distance_metric="euclidean", mds_weights=self.mds_weights
            ),
            labels,
        )

    def scatterplot(self, data, labels, title="Embedding", legend=False) -> None:
        emb = self.fit_transform(data)
        scprep.plot.scatter2d(emb, c=labels, title=title, legend=legend)

    def reset_emb(self):
        self.emb = None

    def get_relative_dist(self) -> np.ndarray:
        if self.dist is None:
            raise NameError("Fit and Compute the distance first.")
        return self.dist.sum(axis=1)


class PhateBasic(BaseEmb):
    """Wrapper for the phate algorithm."""

    def __init__(
        self,
        knn: int,
        anisotropy: int = 0,
        decay: int = 40,
        n_pca: int = 40,
        tau: Union[int, str] = "auto",
        emb_dim: int = 2,
    ):
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


class EucDist(BaseEmb):
    def __init__(
        self,
        knn: int = 10,
        anisotropy: int = 0,
        decay: int = 40,
        n_pca: int = 40,
        emb_dim: int = 2,
    ):
        super().__init__(knn, anisotropy, decay, n_pca, emb_dim=emb_dim)

    def metric_computation(self, data):
        self.dist = squareform(pdist(data))


# working on a new version that includes all types of heat kernel approximations.
# TODO: ADD LANDAMARKS
class new_HeatGeo(BaseEmb):
    def __init__(
        self,
        knn: int,
        anisotropy: int = 0,
        decay: int = 40,
        n_pca: int = 40,
        tau: int = 10,
        emb_dim: int = 2,
        filter_method: str = "pygsp",
        order: int = 32,
        lap_type: str = "normalized",
        tau_min: float = 0.1,
        tau_max: float = 200,
        n_tau: int = 1,
        log_normalize: bool = False,
        scale_factor: float = 1.0,
        denoising: bool = False,
        n_ref: int = 50,
        n_svd: int = 50,
        graph_type: str = "alpha",
        truncation_type: Union[str,None] = None,
        truncation_arg: Union[str,None] = None,
        treshold_type: Union[str,None] = None, # "min" or "max"
        harnack_regul: float = 0,  #Harnack regularization parameter, between 0 and 1.
        norm_treshold: bool = True,
        mds_weights_type: Union[str,None] = None, # "heat_kernel", "inv_dist","gaussian_dist"
        mds_weights_args: Union[str,None] = None,
        denoise_regul: float = 0.0,
    ):
        """
        truncation_type = None, heat_truncation (truncating the heat kernel)
            Cases : 
                - None : no truncation
                - "heat_truncation" : truncation of the heat kernel. In this case, the heat kernel is truncated such that all values below truncation_arg*max(heat_kernel) are set to 0.
                - "dist_truncation" : truncation of the distance matrix. In this case, the distance matrix is normalized to [0,1] and values above truncation_arg are set to 1.
        """
        super().__init__(
            knn,
            anisotropy,
            decay,
            n_pca,
            tau=tau,
            emb_dim=emb_dim,
            lap_type=lap_type,
            filter_method=filter_method,
            order=order,
            tau_min=tau_min,
            tau_max=tau_max,
            n_tau=n_tau,
            scale_factor=scale_factor,
            graph_type=graph_type
        )
        self.log_normalize = log_normalize
        self.n_ref = n_ref
        self.denoising = denoising
        self.n_svd = n_svd
        self._n_tau = n_tau
        self._tau_max = tau_max
        self.truncation_type = truncation_type
        self.truncation_arg = truncation_arg
        self.harnack_regul = harnack_regul
        self.treshold_type = treshold_type
        self.norm_treshold = norm_treshold
        self.mds_weights_type = mds_weights_type
        self.mds_weights_args = mds_weights_args
        self.denoise_regul = denoise_regul

        #assert self.harnack_regul<=1 and self.harnack_regul>=0, "Harnack regularization parameter must be between 0 and 1."

    @property
    def n_tau(self):
        if self._n_tau == 1 and self.tau == "auto":
            return 10  # default value.
        else:
            return self._n_tau

    @property
    def tau_max(self):
        if self._tau_max == 1.0 and self.tau == "auto":
            return 50  # default value.
        else:
            return self._tau_max

    def metric_computation(self, data):
        N = self.graph.W.shape[0]
        eye = np.eye(N)
        
        if self.tau == "auto":
            heat_kernel, self.opt_tau, self.entro_H = get_optimal_heat(self, self.tau_max, self.n_tau)
            tau = [self.opt_tau]
        else:
            if (
            self.filter_method in ["pygsp", "mar"] and self.n_tau > 1
            ):  # NOTE: for Multiscale only works with pygsp and mar.
                tau = np.geomspace(self.tau_min, self.tau_max, self.n_tau)
            else:
                tau = [self.tau]

            filter = Heat_filter(self.graph, tau, self.order, self.filter_method)
            heat_kernel = filter(eye).reshape(N, N, -1)
            
            self.entro_H = time_entropy(heat_kernel)

        if self.truncation_type == "heat_truncation": # truncating the heat kernel directly.
                threshold = self.truncation_arg * heat_kernel.max()
                heat_kernel[heat_kernel < self.truncation_arg] = 0
        
        if len(heat_kernel.shape)==2: # NOTE currently works for (n_points,n_points,n_tau).
            heat_kernel = heat_kernel[:,:,np.newaxis]
        
        heat_kernel[heat_kernel < 0] = 0 # TODO: move this somewhere else.

        if self.log_normalize:
            # with the Euler method the diffusion time is really tau/order.
            den = (
                [t / self.order for t in tau]
                if self.filter_method == "euler"
                else tau
            )

            distance = [
                (np.log(heat_kernel[:, :, i] + 1e-16) / np.log(den[i] + 1e-16))
                ** self.scale_factor
                for i in range(len(tau))
            ]
        else:
              # NOTE/TODO: this could be default with a weight parameter to interpolate between the two.
            s_distance = [
                    (-4 * tau[i] * np.log(heat_kernel[:, :, i] + 1e-16))
                    + self.harnack_regul*(4 * tau[i])
                    * np.log(
                        (1 / 2)
                        * (
                            np.diag(heat_kernel[:, :, i])
                            + np.diag(heat_kernel[:, :, i]).reshape(-1, 1)
                        )
                        + 1e-16
                    )
                for i in range(len(tau))
            ]

            s_distance = np.array(s_distance)

            s_distance[s_distance < 0] = 0

            distance = np.sqrt(s_distance) ** self.scale_factor


        if self.n_tau > 1 and self.tau != "auto":
            weights = 1 - tau / tau.sum()
            w_t = weights.sum()
            weights = weights / w_t if w_t > 0 else None
        else:
            weights = None

        self.dist = np.average(distance, axis=0, weights=weights)

        if self.denoising:
            self.selective_denoising(self.dist)

        if self.truncation_type == "dist_truncation":
            if self.norm_treshold:
                self.dist = self.dist - self.dist.min()
                self.dist = self.dist / self.dist.max()
                assert (self.dist - self.dist.T).max() < 1e-10 # checking almost symmetric matrix.
                if self.treshold_type == "max":
                    self.dist[self.dist > self.truncation_arg] = 1
                elif self.treshold_type == "min":
                    self.dist[self.dist < self.truncation_arg] = 0
                self.dist = np.maximum(self.dist, self.dist.T)
            else:
                dist_min = self.dist.min()
                dist_max = self.dist.max()
                qt = np.quantile(self.dist, self.truncation_arg)
                assert (self.dist - self.dist.T).max() < 1e-10 # checking almost symmetric matrix.
                if self.treshold_type == "max":
                    self.dist[self.dist > qt] = dist_max
                elif self.treshold_type == "min":
                    self.dist[self.dist < qt] = dist_min
                self.dist = np.maximum(self.dist, self.dist.T)
            
        if self.mds_weights_type == "heat_kernel":
            self.mds_weights = scipy.spatial.distance.squareform(heat_kernel[:, :, 0]+1e-5, checks=False)
        if self.mds_weights_type == "inv_dist":
            self.mds_weights = scipy.spatial.distance.squareform(1.0/(self.dist**self.mds_weights_args), checks=False)
        if self.mds_weights_type == "gaussian_dist":
            dist_min = self.dist.min()
            dist_max = self.dist.max()
            dist_norm = (self.dist - dist_min) / (dist_max - dist_min)
            self.mds_weights = scipy.spatial.distance.squareform(np.exp(-dist_norm**2/self.mds_weights_args), checks=False)

    def selective_denoising(self, data):
        """Selectively denoise the graph based on the number of reference observation.
        The reference observations are found via spectral clustering. The final distance matrix is based on the
        relative distance to the reference observations.
        Arguments
        ---------
        data: np.ndarray
            Data matrix to cluster in `n_ref` reference observations.

        """
        
        # _, _, VT = randomized_svd(
        #     data,
        #     n_components=self.n_svd,
        #     random_state=self.random_state,
        # )
        # kmeans = MiniBatchKMeans(
        #     self.n_ref,
        #     init_size=3 * self.n_ref,
        #     batch_size=5000,
        #     random_state=self.random_state,
        # )
        # proj_data = data @ VT.T
        # kmeans.fit(proj_data)
        # closest, distances = vq(
        #     kmeans.cluster_centers_, proj_data
        # )  # indices of the closest points to the cluster centers.
        # mask = np.ones(data.shape[0], dtype=bool)
        # mask[closest] = False
        # self.dist[:, mask] = 0
        denoise_dist = squareform(pdist(self.dist))
        assert self.denoise_regul <= 1 and self.denoise_regul >= 0
        self.dist = (1-self.denoise_regul)*self.dist + self.denoise_regul*denoise_dist



class RandWalkGeo(new_HeatGeo):

    _valid_methods = [
        "exact",
        "affinity",
        "symmetric",
    ]  # TODO: add cheb. approximation.

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
            knn = knn,
            anisotropy = anisotropy,
            decay = decay,
            n_pca = n_pca,
            tau = tau,
            emb_dim = emb_dim,
            filter_method = filter_method,
            order = order,
            lap_type = lap_type,
            log_normalize = log_normalize,
            scale_factor = scale_factor,
            denoising = denoising,
            n_ref = n_ref,
            n_svd = n_svd,
            graph_type = graph_type
        )

        if filter_method not in self._valid_methods:
            raise ValueError("method must be one of {}".format(self._valid_methods))
        

    def metric_computation(self, data):
        # TODO: wrap this in one classe/method adding the Cheb. approximation
        # similar to the Heat_filter.
        if self.filter_method == "exact":
            #P = self.graph.diff_op.toarray()
            P = graph_utils.diff_op(self.graph).toarray()
            diffusion = np.linalg.matrix_power(P, self.tau)
        elif self.filter_method == "affinity":
            #A = self.graph.diff_aff.toarray()
            A = graph_utils.diff_aff(self.graph).toarray()
            diffusion = np.linalg.matrix_power(A, self.tau)
        elif self.filter_method == "symmetric":
            #D = self.graph.kernel_degree.squeeze()
            #P = self.graph.diff_op.toarray()
            D = graph_utils.kernel_degree(self.graph).squeeze()
            P = graph_utils.diff_op(self.graph).toarray()
            Pt = np.linalg.matrix_power(P, self.tau)
            diffusion = np.diag(D**0.5) @ Pt @ np.diag(D**-0.5)

        self.dist = np.sqrt(-4*self.tau*np.log(diffusion + 1e-16)) ** self.scale_factor

        if self.check_symmetric():
            self.symetrize_dist()

        #self.dist = self.dist * (1 - np.eye(self.dist.shape[0]))


class EmbHeatPHATE(BaseEmb):
    """PHATE with the heat kernel approximated by Chebyshev polynomials (pygsp)."""

    def __init__(
        self,
        knn: int,
        anisotropy: int = 0,
        decay: int = 40,
        n_pca: int = 40,
        tau: Union[float,str] = 10,
        emb_dim: int = 2,
        order: int = 32,
        lap_type: str = "normalized",
        filter_method: str = "euler",
        tau_max: int = 200, #Only used if tau=="auto"
        n_tau : int = 10, #Only used if tau=="auto"
        graph_type: str = "alpha",
    ):
        super().__init__(knn = knn, 
                        anisotropy = anisotropy,
                        decay = decay,
                        n_pca = n_pca,
                        tau=tau,
                        emb_dim=emb_dim,
                        lap_type=lap_type, 
                        order=order, 
                        filter_method=filter_method, 
                        graph_type = graph_type)
        
        self.tau_max = tau_max
        self.n_tau = n_tau

    def metric_computation(self, data, order=32, **kwargs):
        if self.tau == "auto":
            heat_kernel, self.opt_tau, self.entro_H = get_optimal_heat(self, self.tau_max, self.n_tau)
            heat_kernel = heat_kernel.squeeze(2) # Remove the last dimension (N,N,1).
        else:
            N = self.graph.W.shape[0]
            eye = np.eye(N)
            filter = Heat_filter(self.graph, self.tau, self.order, self.filter_method)
            heat_kernel = filter(eye).reshape(N, N)
            heat_kernel[heat_kernel < 0] = 0

        log_heat = -np.log(heat_kernel + 1e-16)
        log_heat
        self.dist = squareform(pdist(log_heat))

    def fit_transform(self, data, solver="sgd", **kwargs):
        self.solver = solver
        return super().fit_transform(data)


# NOTE: Depracated soon changing to new_HeatGeo, which includes all types of heat kernel approximations and the possiblity to log normalize.
class CraneEmb(BaseEmb):
    """Similar to Crane Geodesic in heat Lemma 1, in the appendix."""

    def __init__(self, knn: int, anisotropy: int = 0, decay: int = 40, n_pca: int = 40):
        super().__init__(knn, anisotropy, decay, n_pca)

    def metric_computation(self, data, tau, order, **kwargs):
        N = self.graph.W.shape[0]
        heat_op = Heat_Euler(self.graph.L, tau, order)

        eye = np.eye(N)
        diffusion = np.eye(N)
        for i in range(N):
            diffusion[i, :] = heat_op(eye[i, :])
        self.dist = np.log(diffusion + 1e-16) / np.log(tau / order)

        if not self.check_symmetric():
            print("Distance matrix is not symmetric.")
            self.symetrize_dist()


# NOTE: Depracated soon changing to new_HeatGeo, which includes all types of heat kernel approximations.
class EmbHeatGeo(BaseEmb):
    def __init__(
        self,
        knn,
        anisotropy=0,
        decay=40,
        n_svd=50,
        random_state=42,
        n_landmark=None,
        n_pca=40,
    ):
        super().__init__(knn, anisotropy, decay, n_pca=n_pca)
        self.n_svd = n_svd
        self.random_state = random_state
        self.n_landmark = n_landmark

    def get_landmark(self, data):

        _, _, VT = randomized_svd(
            data,
            n_components=self.n_svd,
            random_state=self.random_state,
        )
        kmeans = MiniBatchKMeans(
            self.n_landmark,
            init_size=3 * self.n_landmark,
            batch_size=10000,
            random_state=self.random_state,
        )
        kmeans.fit(data @ VT.T)
        centroid = kmeans.cluster_centers_ @ VT
        self.graph = gt.Graph(
            centroid,
            use_pygsp=True,
            n_pca=self.n_pca,
            anisotropy=self.anisotropy,
            decay=self.decay,
            knn=self.knn,
        )

    def metric_computation(
        self,
        data,
        tau_min=0.5,
        tau_max=50,
        n_tau=10,
        order=10,
        scale_factor=2,
        **kwargs
    ):
        n = self.graph.W.shape[0]
        self.graph.estimate_lmax()
        tau = np.geomspace(tau_min, tau_max, n_tau)
        filt = pygsp.filters.Heat(self.graph, tau=tau)
        heat_kernel = filt.filter(np.eye(n), order=order).reshape(
            n, n, -1
        )  # shape is (n,n,n_tau)
        heat_kernel[heat_kernel < 0] = 0
        multi_geo = [
            np.sqrt(-4 * tau[i] * np.log(heat_kernel[:, :, i] + 1e-16)) ** scale_factor
            for i in range(len(tau))
        ]
        weights = 1 - tau / tau.sum()
        w_t = weights.sum()
        weights = weights / w_t if w_t > 0 else None
        self.dist = np.average(multi_geo, axis=0, weights=weights)

        if not self.check_symmetric():
            print("Distance matrix is not symmetric.")
            self.symetrize_dist()

    def fit_transform(self, data, ndim=2, solver="sgd", **kwargs):
        if self.n_landmark is not None:
            self.n_landmark = (
                None if self.n_landmark > data.shape[0] else self.n_landmark
            )

        if self.n_landmark is not None:
            self.get_landmark(data)
        else:
            self.fit(data)
        self.metric_computation(data, **kwargs)
        self.emb = embed_MDS(self.dist, ndim, solver=solver)
        if self.n_landmark is not None:
            transitions = self.graph.extend_to_data(data)
            self.emb = self.graph.interpolate(self.emb, transitions)
        return self.emb


# NOTE: Depracated soon changing to new_HeatGeo, which includes all types of heat kernel approximations.
class EmbMarHeatGeo(BaseEmb):
    def __init__(self, knn, anisotropy=0, decay=40, n_pca=40):
        super().__init__(knn, anisotropy, decay, n_pca=n_pca)

    def metric_computation(
        self,
        data,
        lap="norm",
        tau_min=0.5,
        tau_max=50,
        n_tau=10,
        order=32,
        scale_factor=2,
        **kwargs
    ):
        tau = np.geomspace(tau_min, tau_max, n_tau)
        if lap == "norm":
            self.graph.compute_laplacian("normalized")
            phi = eigsh(self.graph.L, k=1, return_eigenvectors=False)[0] / 2
        elif lap == "comb":
            self.graph.compute_laplacian("combinatorial")
            phi = eigsh(self.graph.L, k=1, return_eigenvectors=False)[0] / 2
            self.graph.L = (1 / phi) * self.graph.L
            # tau = phi * tau # NOTE we can also rescale the time
        else:
            raise NotImplementedError
        multi_dist = np.stack(
            expm_multiply(
                self.graph.L, np.eye(self.graph.W.shape[0]), phi, tau=tau, K=order
            )
        )
        multi_dist[multi_dist < 0] = 0
        multi_geo = [
            np.sqrt(-4 * np.log(multi_dist[i, :, :] + 1e-16) / np.log(tau[i]))
            for i in range(len(tau)) ** scale_factor
        ]
        multi_geo = np.stack(multi_geo)
        weights = 1 - tau / tau.sum()
        w_t = weights.sum()
        weights = weights / w_t if w_t > 0 else None
        self.dist = np.average(multi_geo, axis=0, weights=weights)

        if not self.check_symmetric():
            print("Distance matrix is not symmetric.")
            self.symetrize_dist()

    def fit_transform(self, data, ndim=2, solver="sgd", **kwargs):
        return super().fit_transform(data, ndim, solver, **kwargs)


class EmbJSD(BaseEmb):
    def __init__(self, knn, anisotropy=0, decay=40):
        super().__init__(knn, anisotropy, decay)

    def metric_computation(self, data, tau=1, method="random_walk", order=32):
        n = data.shape[0]
        if method.lower() == "random_walk":
            Pt = (
                matrix_power(self.graph.P.toarray(), tau)
                if tau > 1
                else self.graph.P.toarray()
            )
        elif method.lower() == "chebyshev":
            self.graph.estimate_lmax()
            filt = pygsp.filters.Heat(self.graph, tau=tau)
            Pt = filt.filter(np.eye(self.graph.W.shape[0]), order=order)
        dist = np.eye(n)
        for i in range(n):
            for j in range(i, n):
                Pi = Pt[i, :]
                Pj = Pt[j, :]
                dist[i, j] = dist[j, i] = jensenshannon(Pi, Pj)
        self.dist = dist

    def fit_transform(self, data, ndim=2, solver="sgd", **kwargs):
        return super().fit_transform(data, ndim, solver, **kwargs)


class EmbHyp(BaseEmb):
    def __init__(self, knn: int, anisotropy: int = 0, decay: int = 40, n_pca: int = 40):
        super().__init__(knn, anisotropy, decay, n_pca)

    def metric_computation(self, data, tau=1, order=32):
        # Pt = (
        #     matrix_power(self.graph.P.toarray(), tau)
        #     if tau > 1
        #     else self.graph.P.toarray()
        # )
        # NOTE using the heat kernel approximation instead of Random walk.
        self.graph.estimate_lmax()
        filt = pygsp.filters.Heat(self.graph, tau=tau)
        Pt = filt.filter(np.eye(self.graph.W.shape[0]), order=order)
        den = 1 - scipy.linalg.norm(Pt, axis=1) ** 2
        den = den[:, None] * den[None, :]
        num = squareform(pdist(Pt) ** 2)
        self.dist = np.arccosh(1 + 2 * num / den)

    def fit_transform(self, data, ndim=2, solver="sgd", **kwargs):
        return super().fit_transform(data, ndim, solver, **kwargs)


class EmbGraphMMD(BaseEmb):
    def __init__(self, knn: int, anisotropy: int = 0, decay: int = 40, n_pca: int = 40):
        super().__init__(knn, anisotropy, decay, n_pca)

    def metric_computation(self, data, order=32, T=None):
        self.graph.estimate_lmax()
        if T is None:
            T = self.graph.L.trace()
        filt = pygsp.filters.Filter(
            self.graph, kernels=[lambda x: [i**-0.5 if i > 0 else 0 for i in x]]
        )
        embeddings = (
            filt.filter(self.graph.P.toarray(), method="chebyshev", order=order)
            * (T**0.5)
        ).T
        self.dist = squareform(pdist(embeddings, metric="minkowski", p=2.0))

    def fit_transform(self, data, ndim=2, solver="sgd", **kwargs):
        return super().fit_transform(data, ndim, solver, **kwargs)


class EmbCTD(BaseEmb):  # TODO use an approximation for the pseudo inverse
    def __init__(self, knn: int, anisotropy: int = 0, decay: int = 40, n_pca: int = 40):
        super().__init__(knn, anisotropy, decay, n_pca)

    def metric_computation(self, data):
        L = self.graph.L
        L_p = pinv(L.toarray())
        diag = np.diag(L_p)
        self.dist = diag[None, :] + diag[:, None] - 2 * L_p

    def fit_transform(self, data, ndim=2, solver="sgd", **kwargs):
        return super().fit_transform(data, ndim, solver, **kwargs)


class EmbDer(BaseEmb):
    def __init__(self, knn: int, anisotropy: int = 0, decay: int = 40, n_pca: int = 40):
        super().__init__(knn, anisotropy, decay, n_pca)

    def metric_computation(self, data, epsilon):
        n = data.shape[0]
        mat = np.eye(n)
        for i in range(n):
            K_x = lambda y: deriv_gauss(x=data[i, :], y=y, epsilon=epsilon)
            mat[i, :] = K_x(data)
        self.dist = squareform(pdist(mat, metric="minkowski", p=2.0))

    def fit_transform(self, data, ndim=2, solver="sgd", **kwargs):
        return super().fit_transform(data, ndim, solver, **kwargs)


# TODO double check the implementation
class DiffusionMap(BaseEmb):
    def __init__(
        self, 
        knn: int = 0,
        decay: int = 40, 
        n_pca: int = 40, 
        tau: float = 1, 
        emb_dim:int = 2, 
        anisotropy: int = 0, 
        graph_type: str = "alpha",
        **kwargs
    ):
        super().__init__(knn=knn, 
                         decay=decay, 
                         n_pca=n_pca, 
                         anisotropy=anisotropy, 
                         tau=tau, 
                         emb_dim=emb_dim, 
                         graph_type=graph_type)

    def metric_computation(self, data):
        #P = self.graph.P.toarray()
        P = graph_utils.diff_op(self.graph).toarray()
        eval, evec = np.linalg.eig(P)
        eval, evec = np.real(eval), np.real(evec)
        eval = eval**self.tau
        emb = eval[None, :] * evec
        self.dist = squareform(pdist(emb))

    def diffusion_emb(self, data):
        #P = self.graph.P.toarray()
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


class ShortestPath(BaseEmb):
    def __init__(
        self, knn: int, 
        anisotropy: int = 0, 
        decay: int = 40, 
        n_pca: int = 40, 
        graph_type: str = "alpha",
        **kwargs
    ):
        super().__init__(knn = knn,
                         anisotropy= anisotropy, 
                         decay = decay, 
                         n_pca = n_pca,
                         graph_type = graph_type)

    def fit(self, data):
        self.graph = NNGraph(data, k=self.knn)

    def metric_computation(self, data):
        euc_dist = squareform(pdist(data))
        A = self.graph.A.toarray()
        euc_dist[A == 0] = 0
        self.dist = shortest_path(euc_dist, method="auto", directed=False)


def deriv_gauss(x, y, epsilon=1):
    return (
        (1 / np.sqrt(2 * pi * epsilon))
        * np.exp(-0.5 * np.linalg.norm((x - y), axis=1) ** 2 / epsilon)
        * np.linalg.norm((x - y), axis=1)
    )
