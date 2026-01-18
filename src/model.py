"""
QOBRA - Model Module

This module defines the quantum circuit models used in the QOBRA system.
It creates three main quantum circuits:
1. Encoder model - Transforms molecular sequences into latent quantum states
2. Decoder model - Reconstructs sequences from latent states (INDEPENDENT PARAMETERS)
3. Full model - Complete autoencoder for training (encoder + decoder)

The quantum circuits use parameterized gates that are optimized during training
to learn meaningful representations of molecular sequences and their functional patterns.

TWO-PHASE TRAINING ARCHITECTURE:
- Phase 1: Encoder trained on MMD loss (latent space matching)
- Phase 2: Decoder trained on Fidelity + ESM losses (encoder frozen)
- Encoder: Parameterized by xe (encoder parameters)
- Decoder: Parameterized by xd (decoder parameters) - INDEPENDENT from encoder
"""

import os, pickle
from ansatz import *
from coding import *

from qiskit.visualization import circuit_drawer
from qiskit_algorithms.utils import algorithm_globals

# Set random seed for reproducible quantum circuit behavior
# This ensures consistent results across different runs
algorithm_globals.random_seed = 0

# =============================================
# ENCODER MODEL DEFINITION
# =============================================
# The encoder transforms input molecular sequences into quantum latent representations
# It consists of:
# 1. Input feature map (fm_i) - Encodes classical sequence data as quantum amplitudes
# 2. Parameterized ansatz (e) - Trainable quantum circuit for learning representations

qc_e = QuantumCircuit(num_tot)  # Create quantum circuit with required number of qubits

# Add input feature map layer
# This layer encodes the classical molecular sequence data into quantum amplitudes
qc_e = qc_e.compose(fm_i.assign_parameters(i_params))

# Add barrier for visual clarity (separates encoding from processing)
qc_e.barrier()

# Add encoder ansatz layer
# This parameterized quantum circuit learns to compress sequence information
qc_e = qc_e.compose(e.assign_parameters(e_params))

# =============================================
# FULL AUTOENCODER MODEL DEFINITION
# =============================================
# The full model is used for training and includes both encoder and decoder
# Architecture: Input → Encoder → Latent State → Decoder → Output
# The goal is to reconstruct the input sequence through the quantum latent space
#
# NOTE: Decoder uses INDEPENDENT parameters (d_params)

qc_ed = QuantumCircuit(num_tot)  # Create training circuit

# ENCODER PATH
# Add input feature map to encode classical sequence data
qc_ed = qc_ed.compose(fm_i.assign_parameters(i_params))

# Add encoder ansatz to compress information into latent space
qc_ed = qc_ed.compose(e.assign_parameters(e_params))

# Add barrier to separate encoder from decoder
qc_ed.barrier()

# DECODER PATH (INDEPENDENT PARAMETERS)
# Add decoder ansatz with its own parameters
# Decoder is trained separately in phase 2
qc_ed = qc_ed.compose(d.assign_parameters(d_params))

# =============================================
# DECODER MODEL DEFINITION
# =============================================
# The decoder generates new molecular sequences from quantum latent representations
# It's used for de novo sequence generation after training
# Architecture: Latent State → Decoder → Output Sequence
#
# NOTE: Decoder uses INDEPENDENT parameters (d_params)

qc_d = QuantumCircuit(num_tot)  # Create decoder circuit

# Add latent feature map layer
# This encodes latent variables (sampled from learned distribution) as quantum amplitudes
# Only uses the first num_latent qubits for latent representation
qc_d = qc_d.compose(fm_l.assign_parameters(l_params), range(num_latent))

# Add barrier for visual clarity
qc_d.barrier()

# Add decoder ansatz with independent parameters
# Uses trained decoder parameters (from phase 2)
qc_d = qc_d.compose(d.assign_parameters(d_params))