#!/bin/bash
#SBATCH --job-name=adni_fc_extract
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/projects/adresearch/moghaddami/gnn-connectivity-matrix/slurm_logs/fc_extract_%j.out
#SBATCH --error=/projects/adresearch/moghaddami/gnn-connectivity-matrix/slurm_logs/fc_extract_%j.err

cd /projects/adresearch/moghaddami/gnn-connectivity-matrix
uv run src/gnn_connectivity_matrix/09-extract-fc-matrices.py --workers 16
