#!/bin/bash
sbatch ./scripts/sweep_tau/diff_map.sh
sbatch ./scripts/sweep_tau/heat_phate.sh
sbatch ./scripts/sweep_tau/phate.sh
sbatch ./scripts/sweep_tau/heat_geo.sh
sbatch ./scripts/sweep_tau/rand_geo.sh