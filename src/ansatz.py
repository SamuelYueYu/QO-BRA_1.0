"""
QOBRA (Quantum Operator-Based Real-Amplitude autoencoder) - Ansatz Module

This module sets up the quantum circuit structure and parameters for the QOBRA system.
It defines the quantum ansatz (parameterized quantum circuits) used for encoding and
decoding molecular sequences. The current implementation demonstrates on protein sequences
but the framework is designed for general molecular design applications.

Key concepts:
- Ansatz: A parameterized quantum circuit that can be trained
- Feature maps: Quantum circuits that encode classical molecular data into quantum states
- Parameters: Trainable variables in the quantum circuits
"""

import os, difflib, sys, argparse
folder_name=os.getcwd()

# =============================================
# COMMAND LINE ARGUMENT PARSING
# =============================================

parser = argparse.ArgumentParser(
    description='QOBRA - Quantum Operator-Based Real-Amplitude Autoencoder',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Required positional arguments
parser.add_argument('metals', nargs='+', type=str,
                    help='Metal types to analyze (e.g., Ca Mg Zn)')

# Quantum circuit parameters
parser.add_argument('--num-qubits', '-q', type=int, default=9,
                    help='Total number of qubits in the circuit')
parser.add_argument('--reps', '-r', type=int, default=2,
                    help='Number of repetitions/layers in the ansatz')

# Loss function lambda weights (all default to 1.0)
parser.add_argument('--lambda-mmd', type=float, default=1.0,
                    help='Weight for MMD loss (encoder training, phase 1)')
parser.add_argument('--lambda-fidelity', type=float, default=1.0,
                    help='Weight for Fidelity loss (decoder training, phase 2)')
parser.add_argument('--lambda-esm', type=float, default=1.0,
                    help='Weight for ESM loss (decoder training, phase 2)')

# ESM configuration
parser.add_argument('--esm-k', type=int, default=32,
                    help='Number of positions to mask per sequence for ESM')
parser.add_argument('--esm-model', type=str, default='esm2_t6_8M_UR50D',
                    help='ESM model name (e.g., esm2_t6_8M_UR50D, esm2_t33_650M_UR50D)')
parser.add_argument('--warmup-esm', type=int, default=None,
                    help='Number of iterations to linearly anneal ESM lambda from 0 to max (default: None = no warmup)')
parser.add_argument('--esm-step-interval', type=int, default=None,
                    help='Iterations to wait before turning on ESM lambda (0 -> MAX step function)')
parser.add_argument('--esm-subset-size', type=int, default=100,
                    help='Number of sequences to use for ESM loss computation (default: 100)')

# Mode switch
parser.add_argument('--mode', '-m', type=int, choices=[0, 1], default=0,
                    help='0 = training mode, 1 = inference/generation mode')

# De novo generation parameters
parser.add_argument('--num-denovo', type=int, default=5,
                    help='Number of de novo sequences to generate per seed folder (default: 5)')
parser.add_argument('--sample-batches', type=int, default=100,
                    help='Number of sample batches/seeds to generate (default: 100)')

# Parse arguments
args = parser.parse_args()

# Extract parsed values
keys_target = args.metals
num_tot = args.num_qubits
r = args.reps

# Loss function weights (all default to 1.0)
# Phase 1 (encoder training): MMD loss
lambda_mmd_max = args.lambda_mmd

# Phase 2 (decoder training): Fidelity + ESM losses
lambda_fidelity_max = args.lambda_fidelity
esm_lambda_max = args.lambda_esm

# ESM configuration
esm_K = args.esm_k
esm_model_name = args.esm_model
esm_warmup = args.warmup_esm  # Number of iterations for ESM lambda warmup (None = no warmup)
esm_step_interval = args.esm_step_interval  # Iterations before ESM turns on (None = use linear warmup)

# Mode switch
switch = args.mode

# De novo generation parameters
num_denovo = args.num_denovo
sample_batches = args.sample_batches
esm_subset_size = args.esm_subset_size

# Create descriptive name for metal combination (used in output paths)
metals = "".join(keys_target)

# Navigate to the training data directory where molecular sequence data is stored
os.chdir('Training data')

# Import quantum computing libraries
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.parametertable import ParameterView

# Import quantum machine learning components
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import BlueprintCircuit, RealAmplitudes
num_trash = 0  # Number of unused qubits (currently 0)
num_latent = num_tot - num_trash  # Number of qubits used for latent representation

# Calculate dimension sizes for quantum states
dim_tot = 2**num_tot  # Total dimension of quantum state space
dim_latent = 2**num_latent  # Dimension of latent space

# Define parameter counts for different circuit components
num_feature = 2**num_tot  # Number of features for input encoding
num_encode = (r+1)*num_tot  # Number of encoding parameters (depends on repetitions)
num_decode = num_encode  # Number of decoding parameters (symmetric)

# Create feature maps (quantum circuits that encode classical data)
# fm_i: Feature map for input sequences (maps classical sequence to quantum state)
fm_i = RawFeatureVector(num_feature)
# Create named parameters for input feature map (used for sequence encoding)
i_params = [Parameter(fr'$\iota_{{{i}}}$') for i in range(num_feature)]

# fm_l: Feature map for latent space (maps latent variables to quantum state)
fm_l = RawFeatureVector(dim_latent)
# Create named parameters for latent feature map (used for generation)
l_params = [Parameter(fr'$\lambda_{{{i}}}$') for i in range(dim_latent)]

def ansatz(num_qubits, r, prefix):
    """
    Creates a parameterized quantum circuit (ansatz) for training.
    
    This function generates a RealAmplitudes circuit which is a common ansatz
    for variational quantum algorithms. It consists of rotation gates and
    entangling gates repeated in layers.
    
    Parameters:
    - num_qubits: Number of qubits in the circuit
    - r: Number of repetitions/layers in the ansatz
    - prefix: String prefix for parameter names
    
    Returns:
    - RealAmplitudes circuit with full entanglement pattern
    """
    return RealAmplitudes(num_qubits, entanglement="full", reps=r, parameter_prefix=prefix)

# Create encoder and decoder ansatz circuits
# e: Encoder ansatz (transforms input sequences to latent representation)
e = ansatz(num_tot, r, "e")
# d: Decoder ansatz (transforms latent representation back to sequences)
# Decoder has INDEPENDENT parameters for two-phase training
d = ansatz(num_tot, r, "d").inverse()

# Create named parameters for encoder and decoder circuits
# e_params: Parameters for the encoder circuit (trainable in phase 1)
e_params = [Parameter(fr'$\epsilon_{{{i}}}$') for i in range(num_encode)]
# d_params: Parameters for the decoder circuit (trainable in phase 2)
d_params = [Parameter(fr'$\delta_{{{i}}}$') for i in range(num_decode)]

# Note: n_params (noise parameters) are commented out - could be used for
# noise modeling in future versions
#n_params = [Parameter(fr'$\nu_{{{i}}}$') for i in range(num_latent)]