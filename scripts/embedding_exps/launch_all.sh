#!/bin/bash
sbatch ./scripts/embedding_exps/diff_map.sh
sbatch ./scripts/embedding_exps/heat_phate.sh
sbatch ./scripts/embedding_exps/phate.sh
sbatch ./scripts/embedding_exps/heat_geo.sh
sbatch ./scripts/embedding_exps/rand_geo.sh
sbatch ./scripts/embedding_exps/shortest_path.sh
sbatch ./scripts/embedding_exps/tsne.sh
sbatch ./scripts/embedding_exps/umap.sh