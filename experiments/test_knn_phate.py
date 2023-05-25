from omegaconf import DictConfig
from omegaconf import OmegaConf

import hydra
import os
import pandas as pd
import time
from experiments.datasets.toy_dataset import SwissRoll
from experiments.evaluation.knn_methods import KNNClassifier
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig) -> None:
    # Version 0 is initial tests
    version = 0
    print(OmegaConf.to_yaml(cfg))
    ks = [5, 10, 20,30,40,50]
    columns=[
            "Method", 
            "tau",
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
    df = pd.DataFrame(columns=columns)
    for seed in range(cfg.n_seeds):
        model = instantiate(cfg.model)
        ds = SwissRoll(cfg.data.n_points, cfg.data.manifold_noise, cfg.data.width, random_state=cfg.data.random_state+seed)
        data, labels = ds.X, ds.labels
        ground_dist = ds.get_geodesic()
        start_time = time.time()
        knn_exp = KNNClassifier(model)
        knn_exp.fit(data, **cfg.fit_args)
        res = knn_exp.evaluate(ground_dist, ks=ks)
        end_time = time.time() - start_time
        results = [[cfg.model_name, cfg.model.tau, cfg.data.random_state+seed, cfg.data.n_points, *res, end_time]]
        df_run = pd.DataFrame(results, columns=columns)
        df = pd.concat([df,df_run], ignore_index=True)

    df.to_pickle(
        f"final_{cfg.model_name}_{version}.pkl"
    )

if __name__ == "__main__":
    main()