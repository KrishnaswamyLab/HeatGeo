#!/bin/bash

#SBATCH --job-name=phate_embed
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

cd ~/project/metric_embeddings
module load miniconda
conda activate metric_embedding

python knn_task_embedding.py -m name=phate_embed launcher=mccleary experiment=phate data_sweeper=toy_data