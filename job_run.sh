#!/bin/bash
#SBATCH --job-name=corafull_graphfsl_taskns
#SBATCH --output=/scratch/data/bikash_rs/vivek/TaskNS-GraphFSL/logs/%x_%j.out
#SBATCH --error=/scratch/data/bikash_rs/vivek/TaskNS-GraphFSL/logs/%x_%j.err
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --qos=fatqos
#SBATCH -D /scratch/data/bikash_rs/vivek/TaskNS-GraphFSL/

# Create logs directory
mkdir -p logs

# Load CUDA module (adjust version based on your system)
# module load cuda/11.8

# Activate virtual environment
source taskns-env/bin/activate

python train.py --dataset corafull --way 5 --shot 3 --qry 15 --episodes 2000
# python create_cora_split.py