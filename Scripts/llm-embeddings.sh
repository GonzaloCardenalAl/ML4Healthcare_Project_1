#!/bin/bash

#SBATCH --job-name=representation_learning
#SBATCH --output=%x%j.out
#SBATCH --ntasks=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --gres=gpumem:40G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=40G

# Load modules
#module load stack/2024-04
module load stack/2024-06 cuda/12.1.1 python_cuda/3.11.6
source /cluster/scratch/gcardenal/ml4h_p1_env/bin/activate

# Debugging
conda list
pip list
module list

echo "Testing Slurm Variables..."
env | grep SLURM
nvidia-smi

# Node networking section
head_node_ip=$(hostname --ip-address)
echo Node IP: $head_node_ip
export LOGLEVEL=INFO

scontrol show job $SLURM_JOB_ID

# Run model
if [ "${SLURM_JOB_NUM_NODES:-1}" -eq 1 ]; then
    # Single node
    echo "Running in standalone mode..."

    srun torchrun \
        --standalone \
        --nproc_per_node=${SLURM_NTASKS:-1} \
        llm-embeddings.py

else
    # Multi-node
    echo "Running in distributed mode..."
        
    srun torchrun \
        --standalone \
        --nproc_per_node=${SLURM_NTASKS:-1} \
        llm-embeddings.py
    
fi