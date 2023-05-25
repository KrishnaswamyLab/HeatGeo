#!/bin/bash
sbatch ./scripts/sweep_knn/diff_map.sh
sbatch ./scripts/sweep_knn/heat_phate.sh
sbatch ./scripts/sweep_knn/phate.sh
sbatch ./scripts/sweep_knn/heat_geo.sh
sbatch ./scripts/sweep_knn/rand_geo.sh
sbatch ./scripts/sweep_knn/shortest_path.sh