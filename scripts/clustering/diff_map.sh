#!/bin/bash

#SBATCH --job-name=diff_map_clustering
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

cd ~/project/metric_embeddings
module load miniconda
conda activate metric_embedding

python  cluster_pred.py -m launcher=mccleary experiment=clustering/diff_map data_sweeper=cluster_data 