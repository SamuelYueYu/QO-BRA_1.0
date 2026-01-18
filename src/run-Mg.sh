#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 15:00:00
#SBATCH -C gpu
#SBATCH -c 128
#SBATCH --gpus-per-task=4
#SBATCH -q regular
#SBATCH -J Mg
#SBATCH --output=%x.out
#SBATCH -A m410

module load python

# Quantum circuit configuration
NUM_QUBITS=9
REPS=2

# Loss function weights (lambdas)
# Note: λ_Fidelity is always 1.0 (not configurable)
LAMBDA_MMD=1.0          # Maximum MMD loss weight (reached after warmup)
LAMBDA_ESM=0.1          # Maximum ESM loss weight (reached after warmup)

# Annealing Configuration (default: MAX_EPOCHS//2 = 2500 for both)
# WARMUP_MMD=2500       # Epochs to increase MMD lambda from 0 to max
WARMUP_ESM=500       # Epochs to increase ESM lambda from 0 to max

# ESM Configuration
ESM_K=32                # Number of masked positions for ESM
ESM_MODEL=esm2_t6_8M_UR50D  # ESM model (esm2_t6_8M_UR50D, esm2_t33_650M_UR50D, etc.)

# Training
# Add --warmup-mmd and --warmup-esm to override defaults (MAX_EPOCHS//2)
srun python train.py Mg \
    --num-qubits $NUM_QUBITS \
    --reps $REPS \
    --lambda-mmd $LAMBDA_MMD \
    --lambda-esm $LAMBDA_ESM \
    --esm-k $ESM_K \
    --esm-model $ESM_MODEL \
    --mode 0

# Generation
srun python gen.py Mg \
    --num-qubits $NUM_QUBITS \
    --reps $REPS \
    --lambda-mmd $LAMBDA_MMD \
    --lambda-esm $LAMBDA_ESM \
    --esm-k $ESM_K \
    --esm-model $ESM_MODEL \
    --mode 1