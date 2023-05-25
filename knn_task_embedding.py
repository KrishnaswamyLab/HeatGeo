from omegaconf import DictConfig
from omegaconf import OmegaConf

import hydra
import os
import pandas as pd
import time
from src.dataset import SwissRoll
from src.knn_methods import KNNClassifier
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig) -> None:
    # Version 0 is initial tests
    version = 0
    print(OmegaConf.to_yaml(cfg))
    ks = [5, 10,20,30,40,50]
    columns=[
            "Method",
            "Seed",
            "# group",
            "SpearmanR",
            "PearsonR",
            *[f"P@{k}" for k in ks],
            "Norm Fro",
            "Norm inf",
            "Norm Fro N2",
            "Norm inf N2",
            "time(s)",
        ]
    
    model_name = cfg.model_name
    if model_name in  ["heat_geo","rand_geo", "heat_phate"]:
        model_name = model_name+"_"+cfg.model.filter_method


    df = pd.DataFrame(columns=columns)
    for seed in range(cfg.n_seeds):
        model = instantiate(cfg.model)
        ds = instantiate(cfg.data)(random_state = 42+seed)
        data, labels = ds.X, ds.labels
        ground_dist = ds.get_geodesic()
        start_time = time.time()
        knn_exp = KNNClassifier(model)
        knn_exp.fit_transform(data)
        res = knn_exp.evaluate(ground_dist, ks=ks)
        end_time = time.time() - start_time
        results = [[model_name, 42+seed, cfg.data.n_points, *res, end_time]]
        df_run = pd.DataFrame(results, columns=columns)
        df = pd.concat([df,df_run], ignore_index=True)

    df.to_pickle(
        f"final_{cfg.dataset_name}_{model_name}_{version}.pkl"
    )

if __name__ == "__main__":
    main()