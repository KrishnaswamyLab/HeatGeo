import numpy as np
from src.filter_approx import Heat_filter
from kneed import KneeLocator


def interpolate(x0, x1, n_steps):
    assert x0.shape == x1.shape
    n, dim = x0.shape
    times = np.linspace(0, 1, n_steps)[1:-1]
    path = []
    labels = []
    for t in times:
        xt = (1 - t) * x0 + t * x1
        path.append(xt)
        labels.append(np.arange(1, n + 1))

    return np.array(path).reshape(-1, dim), np.array(labels).reshape(-1)


def time_entropy(H):
    """
    Argurments
    ----------
    H: np.array (n_nodes, n_nodes, n_times)
    Returns
    -------
    Entropy of the kernel at each time step.
    """
    n_nodes, _, n_times = H.shape
    entropy = []
    for i in range(n_times):
        H_i = H[:,:,i]
        # H_i = H_i / np.sum(H_i)
        entropy.append(-np.sum(H_i * np.log(H_i+1e-10)))
    entropy = np.array(entropy)
    return entropy


def get_optimal_heat(emb_op, tau_max: float = 50, n_tau:int = 20):
    """
    Select the optimal tau for the heat kernel.

    Optimal tau is found using Checbyshev approximation.

    Arguments
    ---------
    emb_op: BaseEmb
        Embedding operator.
    tau_max: float
        Maximum value of tau to consider (Default 50).
    n_tau: int
        Number of tau values to consider (Default 20).
    
    Returns
    -------
    H: np.array (n_nodes, n_nodes)
        Heat kernel with optimal tau.
    tau: float
        Optimal tau.
    entropy: float
        Entropy of the heat kernel with optimal tau.
    """
    taus = np.linspace(0.05, tau_max, n_tau)
    
    H = Heat_filter(graph=emb_op.graph, tau=taus, order=emb_op.order, method="mar")(np.eye(emb_op.graph.N))
    H[H<0]=0
    entro_H = time_entropy(H)
    kneedle = KneeLocator(taus, entro_H, S=0.5, curve="concave")
    idx = np.where(taus==kneedle.knee)[0]
    
    if emb_op.filter_method == "mar":
        H_opt = H[...,idx]
    else:
        H_opt = Heat_filter(graph = emb_op.graph, tau = taus[idx],  order = emb_op.order, method = emb_op.filter_method)(np.eye(emb_op.graph.N))
    print("Optimal tau: ", taus[idx])
    return H_opt, taus[idx], entro_H[idx]