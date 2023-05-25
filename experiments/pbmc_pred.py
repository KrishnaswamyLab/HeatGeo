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
from src.knn_methods import eval_k_means


@hydra.main(version_base=None, config_path="config", config_name="main_pred")
def main(cfg: DictConfig) -> None:
    # Version 0 is initial tests
    version = 0
    print(OmegaConf.to_yaml(cfg))
    columns=[
            "Method",
            "Seed",
            "Accuracy (t)",
            "Rand Score (t)",
            "Adjusted Rand Score (t)",
            "Homogeneity",
            "Completeness",
            "V-measure",
            "Adjusted Rand Score",
            "Adjusted Mutual Info Score",
            "Silhouette Score",
            "time(s)",
        ]
    
    model_name = cfg.model_name
    if model_name in  ["heat_geo","rand_geo", "heat_phate"]:
        model_name = model_name+"_"+cfg.model.filter_method


    # Load and prepare data
    pbmc = sc.datasets.pbmc3k_processed()
    data = pbmc.X
    labels = pbmc.obs["louvain"]
    labels = labels.replace(to_replace=['CD4 T cells', 'CD14+ Monocytes', 'B cells', 'CD8 T cells', 'NK cells', 'FCGR3A+ Monocytes', 'Dendritic cells', 'Megakaryocytes'],
    value=[0,1,2,3,4,5,6,7])
    labels = np.array(labels)

    df = pd.DataFrame(columns=columns)
    for seed in range(cfg.n_seeds):
        model = instantiate(cfg.model)
        start_time = time.time()
        emb = model.fit_transform(data)
        end_time = time.time() - start_time
        data_train, data_test, label_train, label_test = train_test_split(emb, labels, test_size=0.33, random_state=42+seed)

        #KNN Classifier
        knn = KNeighborsClassifier(n_neighbors=cfg.knn.n_neighbors)
        knn.fit(data_train, label_train)
        pred = knn.predict(data_test)
        accuracy = accuracy_score(label_test,pred)
        rand_idx = rand_score(label_test,pred)
        adj_rand_idx = adjusted_rand_score(label_test,pred)

        #KMeans Clustering
        kmeans = KMeans(n_clusters=8, random_state=42+seed)
        res_kmean = eval_k_means(kmeans, data_train, label_train)

        results = [[model_name, 42+seed, accuracy, rand_idx, adj_rand_idx] + res_kmean + [end_time]]
        df_run = pd.DataFrame(results, columns=columns)
        df = pd.concat([df,df_run], ignore_index=True)

    df.to_pickle(
        f"pred_{cfg.dataset_name}_{model_name}_{version}.pkl"
    )

if __name__ == "__main__":
    main()

