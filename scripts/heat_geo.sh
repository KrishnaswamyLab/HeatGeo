#!/bin/bash

#SBATCH --job-name=heat_geo
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

cd ~/project/metric_embeddings
module load miniconda
conda activate metric_embedding

python  knn_task.py -m launcher=mccleary experiment=heat_geo data_sweeper=toy_data 