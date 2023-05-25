from omegaconf import DictConfig
from omegaconf import OmegaConf

import hydra
import os
import pandas as pd
import time
from experiments.datasets.toy_dataset import SwissRoll
from experiments.evaluation.knn_methods import KNNClassifier
from hydra.utils import instantiate
from sklearn.metrics.pairwise import pairwise_distances
from experiments.evaluation.emd import eval_interpolation, sinkhorn_mccann
import numpy as np


# The goal of this task is to see if the embeddings methods preserve a time-ordering of the datasets. 
# We use scRNA-seq datasets with cell differention. 

# TESTED ON EB, MultiCiteSeq with op_train_multi_targets_0.h5ad, and wot_v1.h5ad


@hydra.main(version_base=None, config_path="config", config_name="interpolation")
def main(cfg: DictConfig) -> None:
    # Version 0 is initial tests
    version = 0
    print(OmegaConf.to_yaml(cfg))
    columns=[
            "Method",
            "Seed",
            "Holdout",
            "W1",
            "W2",
            "MMD RBF",
            "MMD Linear",
        ]
    
    model_name = cfg.model_name
    if model_name in  ["heat_geo","rand_geo", "heat_phate"]:
        model_name = model_name+"_"+cfg.model.filter_method


    df = pd.DataFrame(columns=columns)
    for seed in range(cfg.n_seeds):
        model = instantiate(cfg.model)
        ds = instantiate(cfg.data)(random_state = 42+seed)
        data, labels = ds.X, ds.labels
        unique_time = np.unique(labels)

        # embedding all dataset and normalizing to compare embeddings
        emb = model.fit_transform(data)
        norm_emb = (emb - emb.mean(0))/emb.std(0)

        # loop over holdout timepoint
        for idx_ho in range(1,len(unique_time[:-1])):
            t_init = unique_time[idx_ho-1]
            t_hold = unique_time[idx_ho]
            t_final = unique_time[idx_ho+1]

            mask_init = labels==t_init
            mask_hold = labels==t_hold
            mask_final = labels==t_final

            emb_init = norm_emb[mask_init]
            emb_hold = norm_emb[mask_hold]
            emb_final = norm_emb[mask_final]

            n_pred = int(0.5*(mask_init.sum() + mask_final.sum()))

            pred_ho = sinkhorn_mccann(emb_init, emb_hold, emb_final, n_points=n_pred)
            res_pred = eval_interpolation(pred_ho, emb_hold, n_dim=emb.shape[1])

            res = [model_name, seed, t_hold, *res_pred]

            df_run = pd.DataFrame([res], columns=columns)
            df = pd.concat([df,df_run], ignore_index=True)



    df.to_pickle(
        f"ot_pred_{cfg.dataset_name}_{model_name}_{version}.pkl"
    )

if __name__ == "__main__":
    main()