#!/bin/bash
sbatch ./scripts/clustering/phate.sh
sbatch ./scripts/clustering/heat_geo.sh
sbatch ./scripts/clustering/umap.sh
sbatch ./scripts/clustering/tsne.sh
sbatch ./scripts/clustering/diff_map.sh