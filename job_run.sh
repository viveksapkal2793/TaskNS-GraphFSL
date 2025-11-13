#!/bin/bash
#SBATCH --job-name=ood_graphfsl_taskns
#SBATCH --output=/scratch/data/bikash_rs/vivek/TaskNS-GraphFSL/logs/%x_%j.out
#SBATCH --error=/scratch/data/bikash_rs/vivek/TaskNS-GraphFSL/logs/%x_%j.err
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --qos=fatqos
#SBATCH -D /scratch/data/bikash_rs/vivek/TaskNS-GraphFSL/

# Create logs directory
mkdir -p logs

# Load CUDA module (adjust version based on your system)
# module load cuda/11.8

# Activate virtual environment
source taskns-env/bin/activate

# python train.py --use_cuda --dataset dblp --way 5 --shot 3 --qry 15 --episodes 2000
# python create_cora_split.py
python train.py --use_cuda --dataset Amazon_clothing --external_ood_path Amazon_electronics --external_ood_ratio 0.5 --way 5 --shot 3 --qry 15 --episodes 2000