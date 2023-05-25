#!/bin/bash

#SBATCH --job-name=umap_embed
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err


cd ~/project/metric_embeddings
module load miniconda
conda activate umap

python knn_task_embedding.py -m launcher=mccleary name=umap_embed experiment=umap data_sweeper=toy_data