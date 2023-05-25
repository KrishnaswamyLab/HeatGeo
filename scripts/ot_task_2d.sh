#!/bin/bash

EMB_DIM=2

python ot_task.py -m model=heat_geo,heat_phate \
    model.knn=5,10 \
    model.tau=auto \
    model.emb_dim=$EMB_DIM \
    model.filter_method=mar \
    data=multi_cite data.data_file=op_cite_inputs_0.h5ad,op_train_multi_targets_0.h5ad,wot_v1.h5ad \
    hydra.job.name=mc_heat &
python ot_task.py -m model=heat_geo,heat_phate \
    model.knn=5,10 \
    model.tau=auto \
    model.emb_dim=$EMB_DIM \
    model.filter_method=mar \
    data=eb \
    hydra.job.name=eb_heat &
python ot_task.py -m model=phate \
    model.knn=5,10 \
    model.emb_dim=$EMB_DIM \
    data=multi_cite data.data_file=op_cite_inputs_0.h5ad,op_train_multi_targets_0.h5ad,wot_v1.h5ad \
    hydra.job.name=mc_phate &
python ot_task.py -m model=phate \
    model.knn=5,10 \
    model.emb_dim=$EMB_DIM \
    data=eb \
    hydra.job.name=eb_phate &
python ot_task.py -m model=diff_map \
    model.knn=5,10 \
    model.tau=10,20 \
    model.emb_dim=$EMB_DIM \
    data=multi_cite data.data_file=op_cite_inputs_0.h5ad,op_train_multi_targets_0.h5ad,wot_v1.h5ad \
    hydra.job.name=mc_dm &
python ot_task.py -m model=diff_map \
    model.knn=5,10 \
    model.tau=10,20 \
    model.emb_dim=$EMB_DIM \
    data=eb \
    hydra.job.name=eb_dm&
python ot_task.py -m model=umap \
    model.n_components=$EMB_DIM \
    model.n_neighbors=5,10 \
    data=multi_cite data.data_file=op_cite_inputs_0.h5ad,op_train_multi_targets_0.h5ad,wot_v1.h5ad \
    hydra.job.name=mc_umap &
python ot_task.py -m model=umap \
    model.n_components=$EMB_DIM \
    model.n_neighbors=5,10 \
    data=eb \
    hydra.job.name=eb_umap &
python ot_task.py -m model=tsne \
    model.n_components=$EMB_DIM \
    data=multi_cite data.data_file=op_cite_inputs_0.h5ad,op_train_multi_targets_0.h5ad,wot_v1.h5ad \
    hydra.job.name=mc_tsne &
python ot_task.py -m model=tsne \
    model.n_components=$EMB_DIM \
    data=eb \
    hydra.job.name=eb_tsne &    
python ot_task.py -m model=heat_geo model.knn=5,10 model.tau=auto model.emb_dim=$EMB_DIM \
    model.filter_method=mar data=multi_cite data.data_file=op_cite_inputs_0.h5ad,op_train_multi_targets_0.h5ad,wot_v1.h5ad \
    hydra.job.name=mc_heat \
    model.harnack_regul=0.1,0.5,1.0,1.5 &
python ot_task.py -m model=heat_geo model.knn=5,10 model.tau=auto model.emb_dim=$EMB_DIM \
    model.filter_method=mar data=eb \
    hydra.job.name=mc_heat \
    model.harnack_regul=0.1,0.5,1.0,1.5 &
python ot_task.py -m model=heat_geo model.knn=5,10 model.tau=auto model.emb_dim=$EMB_DIM \
    model.filter_method=mar data=multi_cite data.data_file=op_cite_inputs_0.h5ad,op_train_multi_targets_0.h5ad,wot_v1.h5ad \
    hydra.job.name=mc_heat model.mds_weights_type=heat_kernel model.scale_factor=2 &
python ot_task.py -m model=heat_geo model.knn=5,10 model.tau=auto model.emb_dim=$EMB_DIM \
    model.filter_method=mar data=eb \
    hydra.job.name=mc_heat model.mds_weights_type=heat_kernel model.scale_factor=2 &
