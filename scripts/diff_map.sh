#!/bin/bash

#SBATCH --job-name=diff_map
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

cd ~/project/metric_embeddings
module load miniconda
conda activate metric_embedding

python knn_task.py -m launcher=mccleary experiment=diff_map data_sweeper=toy_data
