import pandas as pd
import os
import yaml

import os
os.environ['PATH'] += ":/vast/palmer/apps/avx2/software/texlive/20220321-GCC-10.2.0/bin/x86_64-linux/"

from experiments.datasets.toy_dataset import SwissRoll
from experiments.datasets.sc_dataset import IPSC, EBData, PBMC
from heatgeo.embedding import HeatGeo
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

import scanpy as sc
pbmc = sc.datasets.pbmc3k_processed()
data = pbmc.X
labels = pbmc.obs["louvain"]
labels = labels.replace(to_replace=['CD4 T cells', 'CD14+ Monocytes', 'B cells', 'CD8 T cells', 'NK cells', 'FCGR3A+ Monocytes', 'Dendritic cells', 'Megakaryocytes'],
value=[0,1,2,3,4,5,6,7]
)

mar_order_embs = []

for order in [5,10,20,30,40]:

    model = HeatGeo(tau = "auto", order = order, knn = 5, filter_method = "mar", log_normalize = False, emb_dim=2, harnack_regul = 0.) #mds_weights_type="heat_kernel", scale_factor=2)  

    emb = model.fit_transform(data)

    mar_order_embs.append(emb)

np.save("mar_order_embs.npy", mar_order_embs)

edede

harnack_embs = []
for harnack_regul in [0,0.5,1.0,1.5,2]:
    print(harnack_regul)
    model = HeatGeo(tau = "auto", order = 30, knn=5, filter_method = "euler", log_normalize = False, emb_dim=2, harnack_regul = harnack_regul) #mds_weights_type="heat_kernel", scale_factor=2)  

    emb = model.fit_transform(data)

    harnack_embs.append(emb)

np.save("harnack_embs.npy", harnack_embs)

knn_embs = []
for knn in [3,5,10,15,20]:

    model = HeatGeo(tau = "auto", order = 30, knn=knn, filter_method = "euler", log_normalize = False, emb_dim=2, harnack_regul = 0.) #mds_weights_type="heat_kernel", scale_factor=2)  

    emb = model.fit_transform(data)

    knn_embs.append(emb)

np.save("knn_embs.npy", knn_embs)

order_embs = []

for order in [10,20,30,40,50]:

    model = HeatGeo(tau = "auto", order = order, knn = 5, filter_method = "euler", log_normalize = False, emb_dim=2, harnack_regul = 0.) #mds_weights_type="heat_kernel", scale_factor=2)  

    emb = model.fit_transform(data)

    order_embs.append(emb)

np.save("order_embs.npy", order_embs)

tau_embs = []

for tau in [1,5,10,20, "auto"]:

    model = HeatGeo(tau = tau, order = 30, knn = 5, filter_method = "euler", log_normalize = False, emb_dim=2, harnack_regul = 0.) #mds_weights_type="heat_kernel", scale_factor=2)  

    emb = model.fit_transform(data)

    tau_embs.append(emb)

np.save("tau_embs.npy", tau_embs)