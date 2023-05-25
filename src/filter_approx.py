import numpy as np
from sksparse.cholmod import cholesky
import scipy
import pygsp
from src.cheb import expm_multiply
from scipy.sparse.linalg import eigsh
from typing import Union

# implicit Euler discretization of the heat equation.
# Code inspired by Heitz et al, 2020
# https://github.com/matthieuheitz/2020-JMIV-ground-metric-learning-graphs/blob/main_release/ml_kinterp2.py


class Heat_Euler:
    """Implicit Euler discretization of the heat equation using Chelesky prefactorization."""

    def __init__(self, L, t, K):
        """L: Laplacian N X N
        t: time regularization param.
        S: number of steps."""
        N = L.shape[0]
        Id = scipy.sparse.eye(N)

        # Backward Euler (implicit Euler)
        
        M = Id + (t / K) * L

        # Using Cholesky prefactorization
        Mpre = cholesky(M.tocsc(), ordering_method="metis")

        self.solve = Mpre.solve_A
        self.Mpre = Mpre
        self.M = M
        self.N = N
        self.K = K

    def __call__(self, b):
        """Discretization of the heat equation with init. condition b."""
        u_l = np.zeros([self.K + 1, self.N])
        u_l[0] = b
        for i in range(1, self.K + 1):
            u_l[i] = self.solve(u_l[i - 1])
        return u_l[self.K]


class Heat_filter:
    """Wrapper for the approximation of the heat kernel.

    Arguments
    ---------
    graph:
    t: diffusion time.
    order: number of steps or order of the polynomial approximation.
    method: 'euler', 'pygsp', 'mar'.


    Returns
    -------
    Callable that takes as input a vector b and returns its diffusion.
    """

    _valid_methods = ["euler", "pygsp", "mar", "lowrank", "exact"]

    def __init__(self, graph, tau, order, method="euler"):
        self.graph = graph
        self.tau = tau
        self.order = order
        self.method = method

        # if isinstance(tau, (list,int)):

        if method not in self._valid_methods:
            raise ValueError("method must be one of {}".format(self._valid_methods))

        if method == "euler":
            if isinstance(self.tau,list):
                self.tau = np.array(self.tau)
            if isinstance(self.tau,Union[int,float]):
                self.tau = np.array([self.tau])
            if len(self.tau)==1:
                self._filter = [Heat_Euler(graph.L, self.tau[0], order)]
            else:
                self._filter = [Heat_Euler(graph.L, self.tau[i], order) for i in range(len(self.tau))]

        elif method == "pygsp":
            graph.estimate_lmax()
            self._filter = pygsp.filters.Heat(graph, tau)

        elif method == "mar":
            self.phi = eigsh(self.graph.L, k=1, return_eigenvectors=False)[0] / 2

        elif method == "lowrank":
            N = graph.L.shape[0]
            eval, evec = scipy.sparse.linalg.eigsh(self.graph.L, k=order, which="SM")
            # L = self.graph.L.toarray()
            # L = (L + L.T)/2 # The matrix is already symmetric, but this is to avoid numerical errors.
            # # eval, evec = scipy.linalg.eigh(L, subset_by_index=[0,order])
            self._filter = evec @ np.diag(np.exp(-self.tau * eval)) @ evec.T

        elif method == "exact":
            eval, evec = scipy.linalg.eigh(graph.L.toarray())
            self._filter = evec @ np.diag(np.exp(-self.tau * eval)) @ evec.T

    def __call__(self, b):

        if self.method == "euler":
            N = b.shape[0]
            diff = np.zeros((N,N,len(self.tau)))
            for i in range(N):
                for i_f, f in enumerate(self._filter):
                    diff[i, :,i_f] = f(b[i, :])
            return diff

        elif self.method == "pygsp":
            return self._filter.filter(b, order=self.order)

        elif self.method == "mar":
            diff = expm_multiply(self.graph.L, b, self.phi, tau=self.tau, K=self.order)
            diff = np.stack(diff)
            if diff.ndim == 3:  # shape n_tau x n x n
                return diff.transpose(2, 1, 0)  # shape n x n x n_tau
            else:
                return diff

        elif self.method == "lowrank":
            return self._filter @ b

        elif self.method == "exact":
            return self._filter @ b
