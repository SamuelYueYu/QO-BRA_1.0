"""
QOBRA - Encoding Module

This module handles quantum state encoding and unitary matrix operations:
1. Pre-computing sequence encodings
2. Batch unitary matrix multiplication
3. Encoder and decoder unitary computation
4. Full autoencoder batch operations

These functions provide GPU-accelerated quantum circuit simulation
via direct matrix multiplication instead of circuit execution.
"""

import time
import numpy as np
import torch
from qiskit.quantum_info import Operator

# Import everything from config (which imports from count)
# This gives us: dim_tot, num_encode, e, d, head, ctf, etc.
from config import *
from utils import get_device

# =============================================
# CACHING FOR PRE-ENCODED STATES
# =============================================

# Cache for pre-encoded sequence states (computed once before training)
_cached_train_states = None
_cached_test_states = None

def get_cached_states():
    """Get the cached train and test states."""
    return _cached_train_states, _cached_test_states

def set_cached_states(train_states, test_states):
    """Set the cached train and test states."""
    global _cached_train_states, _cached_test_states
    _cached_train_states = train_states
    _cached_test_states = test_states

# =============================================
# PRE-COMPUTE SEQUENCE ENCODINGS
# =============================================

def precompute_encoded_states(train_seqs, test_seqs):
    """
    Pre-compute encoded amplitude vectors for all sequences.
    
    This function is called ONCE before training starts, eliminating
    redundant encoding computations on every iteration.
    
    Parameters:
    - train_seqs: List of training protein sequences
    - test_seqs: List of test protein sequences
    
    Returns:
    - train_states: Pre-encoded training states (N, dim_tot)
    - test_states: Pre-encoded test states (M, dim_tot)
    """
    global _cached_train_states, _cached_test_states
    
    print("Pre-computing sequence encodings (one-time cost)...")
    start = time.time()
    
    _cached_train_states = np.array([
        encode_amino_acid_sequence(seq, ctf=ctf, head=head, 
                                   max_len=dim_tot-1, vec_len=dim_tot)
        for seq in train_seqs
    ])
    _cached_test_states = np.array([
        encode_amino_acid_sequence(seq, ctf=ctf, head=head, 
                                   max_len=dim_tot-1, vec_len=dim_tot)
        for seq in test_seqs
    ])
    
    elapsed = time.time() - start
    print(f"Pre-computed {len(train_seqs)} train + {len(test_seqs)} test encodings in {elapsed:.2f}s")
    
    return _cached_train_states, _cached_test_states

# =============================================
# ENCODER UNITARY COMPUTATION
# =============================================

def get_encoder_unitary(p):
    """
    Compute the encoder unitary matrix for given parameters.
    
    This extracts the unitary transformation matrix from the RealAmplitudes
    ansatz, which can then be applied to all input states via batch
    matrix multiplication.
    
    Parameters:
    - p: Encoder parameters (array of length num_encode)
    
    Returns:
    - U: Unitary matrix (dim_tot × dim_tot) as numpy array
    """
    # Get the actual parameters from the encoder circuit (e.g., e[0], e[1], ...)
    # These are different from e_params which use $\epsilon_{i}$ naming
    circuit_params = list(e.parameters)
    
    # Create parameter dictionary mapping circuit parameters to values
    param_dict = {circuit_params[j]: p[j] for j in range(len(circuit_params))}
    encoder_circuit = e.assign_parameters(param_dict)
    
    # Get unitary matrix using Qiskit's Operator class
    U = Operator(encoder_circuit).data
    return U

# =============================================
# DECODER UNITARY COMPUTATION
# =============================================

def get_decoder_unitary(p_decoder):
    """
    Compute the decoder unitary matrix for given decoder parameters.
    
    The decoder has INDEPENDENT parameters from the encoder.
    This allows for two-phase training: encoder on MMD, then decoder on Fidelity+ESM.
    
    Parameters:
    - p_decoder: Decoder parameters (array of length num_decode)
    
    Returns:
    - U_d: Decoder unitary matrix (dim_tot × dim_tot) as numpy array
    """
    # Get the actual parameters from the decoder circuit (d[0], d[1], ...)
    circuit_params = list(d.parameters)
    
    # Create parameter dictionary mapping circuit parameters to decoder values
    param_dict = {circuit_params[j]: p_decoder[j] for j in range(len(circuit_params))}
    decoder_circuit = d.assign_parameters(param_dict)
    
    # Get unitary matrix using Qiskit's Operator class
    U_d = Operator(decoder_circuit).data
    return U_d

# =============================================
# BATCH ENCODING OPERATIONS
# =============================================

def batch_latent_encode(p, train_states, test_states):
    """
    Batch encode all sequences using GPU-accelerated matrix multiplication.
    
    Instead of simulating the quantum circuit for each sequence individually,
    this function:
    1. Computes the encoder unitary matrix ONCE
    2. Applies it to ALL sequences via batch matrix multiplication
    
    This provides massive speedup: O(1) unitary computation + O(N) matmul
    vs O(N) circuit simulations.
    
    Parameters:
    - p: Encoder parameters
    - train_states: Pre-encoded training states (N, dim_tot)
    - test_states: Pre-encoded test states (M, dim_tot)
    
    Returns:
    - train_encode: Latent representations of training sequences
    - test_encode: Latent representations of test sequences
    """
    device = get_device()
    
    # Compute encoder unitary ONCE per iteration
    U = get_encoder_unitary(p)
    
    # Move to GPU for fast matrix multiplication
    U_t = torch.tensor(U, device=device, dtype=torch.complex64)
    train_t = torch.tensor(train_states, device=device, dtype=torch.complex64)
    test_t = torch.tensor(test_states, device=device, dtype=torch.complex64)
    
    # Batch matrix multiplication: output = input @ U^T
    # For quantum states: |ψ_out⟩ = U |ψ_in⟩
    # With row vectors: output_row = input_row @ U^T
    train_encode = (train_t @ U_t.T).real.cpu().numpy()
    test_encode = (test_t @ U_t.T).real.cpu().numpy()
    
    return train_encode, test_encode

def batch_autoencoder(p_encoder, p_decoder, input_states):
    """
    Apply full autoencoder (encoder + decoder) to input states.
    
    The decoder has INDEPENDENT parameters from the encoder.
    
    Parameters:
    - p_encoder: Encoder parameters
    - p_decoder: Decoder parameters
    - input_states: Input states (N, dim_tot)
    
    Returns:
    - latent_states: States after encoder (N, dim_tot), complex
    - reconstructed_states: States after decoder (N, dim_tot), complex
    """
    device = get_device()
    
    # Get encoder and decoder unitaries
    U_e = get_encoder_unitary(p_encoder)
    U_d = get_decoder_unitary(p_decoder)
    
    # Move to GPU
    U_e_t = torch.tensor(U_e, device=device, dtype=torch.complex64)
    U_d_t = torch.tensor(U_d, device=device, dtype=torch.complex64)
    input_t = torch.tensor(input_states, device=device, dtype=torch.complex64)
    
    # Apply encoder: latent = input @ U_e^T
    latent_states = input_t @ U_e_t.T
    
    # Apply decoder: reconstructed = latent @ U_d^T
    reconstructed_states = latent_states @ U_d_t.T
    
    return latent_states.cpu().numpy(), reconstructed_states.cpu().numpy()

# =============================================
# LEGACY ENCODING FUNCTIONS
# =============================================

def latent_rep(x, p):
    """
    Compute the latent representation of a molecular sequence using the quantum encoder.
    
    NOTE: This function is kept for backward compatibility and single-sequence use cases.
    For batch processing during training, use batch_latent_encode() instead.
    
    Parameters:
    - x: Input molecular sequence encoded as quantum amplitudes
    - p: Encoder parameters (fixed during this computation)
    
    Returns:
    - Real-valued latent representation vector
    """
    from qiskit.quantum_info import Statevector
    
    # Create parameter dictionary for the quantum circuit
    # Map input features to their corresponding amplitudes
    param_dict = {i_params[j]: x[j] for j in range(num_feature)}
    
    # Add encoder parameters to the parameter dictionary
    param_dict.update({e_params[j]: p[j] for j in range(num_encode)})
    
    # Execute the encoder quantum circuit with the given parameters
    q = qc_e.assign_parameters(param_dict)
    psi = Statevector.from_instruction(q)
    
    # Return the real part of the quantum state amplitudes
    return np.real(psi.data)

def latent_encode(p, train_input, test_input):
    """
    Encode training and test molecular sequences into latent representations.
    
    OPTIMIZED VERSION: Uses pre-cached encoded states and batch matrix multiplication.
    
    Parameters:
    - p: Encoder parameters
    - train_input: Training molecular sequences (used only if cache is empty)
    - test_input: Test molecular sequences (used only if cache is empty)
    
    Returns:
    - train_encode: Encoded training sequences
    - test_encode: Encoded test sequences
    """
    global _cached_train_states, _cached_test_states
    
    # Use cached states if available, otherwise compute them
    if _cached_train_states is None or _cached_test_states is None:
        precompute_encoded_states(train_input, test_input)
    
    # Use optimized batch encoding with unitary matrix multiplication
    train_encode, test_encode = batch_latent_encode(p, _cached_train_states, _cached_test_states)
    
    return train_encode, test_encode

