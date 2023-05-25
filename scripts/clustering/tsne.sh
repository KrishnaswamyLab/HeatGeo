#!/bin/bash

#SBATCH --job-name=tsne_clustering
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

cd ~/project/metric_embeddings
module load miniconda
conda activate metric_embedding

python  cluster_pred.py -m launcher=mccleary experiment=clustering/tsne data_sweeper=cluster_data_sc 