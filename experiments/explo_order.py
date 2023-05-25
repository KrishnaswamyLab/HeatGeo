from omegaconf import DictConfig
from omegaconf import OmegaConf

import hydra
import os
import pandas as pd
import time
from hydra.utils import instantiate

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, rand_score, adjusted_rand_score
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from src.knn_methods import eval_k_means
import os


@hydra.main(version_base=None, config_path="config", config_name="explo_order")
def main(cfg: DictConfig) -> None:
    # Version 0 is initial tests
    version = 0
    print(OmegaConf.to_yaml(cfg))

    
    model_name = cfg.model_name
    if model_name in  ["heat_geo","rand_geo", "heat_phate"]:
        model_name = model_name+"_"+cfg.model.filter_method

    if cfg.model.harnack:
        order_list = np.arange(30, 56 ,1)
    else:
        order_list = np.arange(1, 26 ,1)
    ds = instantiate(cfg.data)(random_state = 42) 

    os.makedirs("embeddings", exist_ok=True)
    embeddings = []
    for order in order_list:
        emb_op = instantiate(cfg.model, order=order)
        emb = emb_op.fit_transform(ds.X)
        embeddings.append(emb)
        np.save(f"embeddings/emb_{cfg.dataset_name}_{model_name}_{order}.npy", emb)

    fig, axs = plt.subplots(5,5, figsize=(20, 20))
    axs = axs.ravel()
    for i, ax in enumerate(axs):
        ax.scatter(embeddings[i][:,0],embeddings[i][:,1],c=ds.labels)
        ax.set_title("order = {}".format(order_list[i]))
        ax.axis('off')
    name = "{}_{}_{}_{}.png".format(cfg.dataset_name, model_name, cfg.model.tau, cfg.model.knn)
    fig.suptitle(name, y=0.92, fontsize=20)
    fig.savefig(name, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()

