#!/bin/bash
sbatch ./scripts/diff_map.sh
sbatch ./scripts/heat_phate.sh
sbatch ./scripts/phate.sh
sbatch ./scripts/heat_geo.sh
sbatch ./scripts/rand_geo.sh
sbatch ./scripts/shortest_path.sh