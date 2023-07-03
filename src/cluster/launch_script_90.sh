#!/bin/bash

#SBATCH -J CONT_90
#SBATCH --output /mnt/beegfs/abruley/CONTEXT/OUT_90.log
#SBATCH --error /mnt/beegfs/abruley/CONTEXT/ERROR_90.log

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=50G
#SBATCH --time=7-00:00:00
#SBATCH --partition=type_2

module load python/conda

eval "$(conda shell.bash hook)"

cd /trinity/home/abruley/STAGE_M2/FIERLENIUS/

conda activate fierlenius

python src/scope_tree/context_extraction.py 90
