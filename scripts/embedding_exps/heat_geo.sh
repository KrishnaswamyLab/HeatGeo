#!/bin/bash

#SBATCH --job-name=heat_geo_embed
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

cd ~/project/metric_embeddings
module load miniconda
conda activate metric_embedding

python knn_task_embedding.py -m launcher=mccleary name=heat_geo_embed experiment=heat_geo data_sweeper=toy_data
