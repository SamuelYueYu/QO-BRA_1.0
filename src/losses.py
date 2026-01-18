"""
QOBRA - Loss Functions Module

This module contains all loss computation functions:
1. Reconstruction Fidelity Loss - Quantum state fidelity F = |⟨ψ|φ⟩|²
2. MMD (Maximum Mean Discrepancy) Loss - Latent space distribution matching
3. Target distribution generation
4. Sequence decoding utilities

These functions compute the individual loss components that are
combined in the main autoencoder_loss function.
"""

import numpy as np
import torch

# Import everything from config (which imports from count)
# This gives us: make_target, decode_amino_acid_sequence, etc.
from config import *
from utils import get_device

# =============================================
# CACHING FOR ESM AND FIDELITY LOSSES
# =============================================

# ESM loss caching
_esm_iteration_counter = 0
_cached_esm_train_loss = 0.0
_cached_esm_test_loss = 0.0

# ESM baseline (computed once on original training sequences)
_esm_baseline_loss = None  # None = not yet computed

# Fidelity loss caching
_cached_fidelity_train_loss = 0.0
_cached_fidelity_test_loss = 0.0

# Test set caching
_cached_k_test = 0.0

# Target distribution caching
_cached_target = None
_cached_target_n = 0

def get_esm_cache():
    """Get ESM loss cache values."""
    return _esm_iteration_counter, _cached_esm_train_loss, _cached_esm_test_loss

def set_esm_cache(counter, train_loss, test_loss):
    """Set ESM loss cache values."""
    global _esm_iteration_counter, _cached_esm_train_loss, _cached_esm_test_loss
    _esm_iteration_counter = counter
    _cached_esm_train_loss = train_loss
    _cached_esm_test_loss = test_loss

def get_esm_baseline():
    """Get cached ESM baseline loss (computed on original training sequences)."""
    return _esm_baseline_loss

def set_esm_baseline(baseline):
    """Set ESM baseline loss."""
    global _esm_baseline_loss
    _esm_baseline_loss = baseline

def get_fidelity_cache():
    """Get fidelity loss cache values."""
    return _cached_fidelity_train_loss, _cached_fidelity_test_loss

def set_fidelity_cache(train_loss, test_loss):
    """Set fidelity loss cache values."""
    global _cached_fidelity_train_loss, _cached_fidelity_test_loss
    _cached_fidelity_train_loss = train_loss
    _cached_fidelity_test_loss = test_loss

def get_test_cache():
    """Get test loss cache value."""
    return _cached_k_test

def set_test_cache(k_test):
    """Set test loss cache value."""
    global _cached_k_test
    _cached_k_test = k_test

# =============================================
# RECONSTRUCTION FIDELITY LOSS
# =============================================

def compute_fidelity_loss(input_states, reconstructed_states):
    """
    Compute the reconstruction fidelity loss.
    
    Fidelity measures how close two quantum states are:
    F(ψ, φ) = |⟨ψ|φ⟩|²
    
    Loss = 1 - mean(F) so that perfect reconstruction gives 0 loss.
    
    Parameters:
    - input_states: Original input states (N, dim_tot), complex
    - reconstructed_states: States after encoder→decoder (N, dim_tot), complex
    
    Returns:
    - Mean infidelity loss (1 - fidelity)
    """
    device = get_device()
    
    # Element-wise operations are memory efficient
    X_t = torch.as_tensor(input_states, dtype=torch.complex64, device=device)
    Y_t = torch.as_tensor(reconstructed_states, dtype=torch.complex64, device=device)
    
    # Ensure states are normalized
    X_t /= torch.linalg.norm(X_t, dim=1, keepdim=True)
    Y_t /= torch.linalg.norm(Y_t, dim=1, keepdim=True)
    
    # Element-wise inner products: ⟨X[i]|Y[i]⟩ for each i
    inner_products = torch.sum(X_t * torch.conj(Y_t), dim=1)
    fidelities = torch.abs(inner_products) ** 2
    loss = 1 - torch.mean(fidelities)
    
    return loss.item()

# =============================================
# MMD (KERNEL) LOSS
# =============================================

def compute_kernel_loss(X, Y):
    """
    GPU-accelerated kernel-based Maximum Mean Discrepancy (MMD) loss.
    
    This implementation uses PyTorch for GPU acceleration, computing all
    pairwise dot products in a single batched matrix multiplication.
    No multiprocessing needed - GPU handles the parallelism.
    
    The SWAP kernel is: k(x, y) = 1 - |<x, y>|^2
    
    Parameters:
    - X: First distribution (encoded protein sequences), shape (N, D)
    - Y: Second distribution (target latent distribution), shape (M, D)
    
    Returns:
    - Mean MMD loss value (scalar)
    """
    device = get_device()
    
    # Convert numpy arrays to PyTorch tensors on GPU
    X_t = torch.tensor(X, device=device, dtype=torch.float32)
    Y_t = torch.tensor(Y, device=device, dtype=torch.float32)
    
    # Compute all pairwise dot products in one batched operation
    # X_t: (N, D), Y_t: (M, D) -> dot_products: (N, M)
    dot_products = torch.mm(X_t, Y_t.T)
    
    # Apply SWAP kernel: k(x, y) = 1 - |<x, y>|^2
    kernel_values = 1 - dot_products ** 2
    
    # Return mean kernel loss
    return kernel_values.mean().item()

# =============================================
# TARGET DISTRIBUTION
# =============================================

def get_cached_target(n, mu, std):
    """
    Get cached target distribution or generate new one.
    
    The target distribution has the same parameters every iteration,
    so we cache it to avoid regeneration.
    
    Parameters:
    - n: Number of samples needed
    - mu: Mean of Gaussian distribution
    - std: Standard deviation
    
    Returns:
    - Cached or newly generated target distribution
    """
    global _cached_target, _cached_target_n
    
    if _cached_target is None or _cached_target_n != n:
        _cached_target = make_target(n, mu, std)
        _cached_target_n = n
    
    return _cached_target

# =============================================
# SEQUENCE DECODING
# =============================================

def decode_states_to_sequences(states, ftc, head):
    """
    Decode quantum states to amino acid sequences (batch version).
    
    Parameters:
    - states: Quantum states (N, dim_tot), complex
    - ftc: Frequency to character mapping
    - head: Head amplitude for normalization
    
    Returns:
    - List of decoded sequences
    """
    sequences = []
    states_real = np.real(states)
    
    for i in range(len(states)):
        seq = decode_amino_acid_sequence(states_real[i], ftc, head)
        sequences.append(seq)
    return sequences

# =============================================
# ESM BASELINE COMPUTATION
# =============================================

def compute_esm_baseline(train_sequences, K=32, model_name="esm2_t6_8M_UR50D"):
    """
    Compute ESM baseline loss on ALL original training sequences.
    
    This establishes the "floor" ESM loss that represents the biological
    plausibility of real training data. The normalized ESM loss during
    training is: normalized = raw - baseline, so achieving training-level
    quality gives a loss of 0.
    
    This should be called ONCE at the start of Phase 2 (decoder training).
    Uses ALL training sequences for accurate baseline estimation.
    
    Parameters:
    - train_sequences: List of original training sequences (not reconstructed)
    - K: Number of positions to mask per sequence for ESM
    - model_name: ESM model to use
    
    Returns:
    - baseline_loss: Mean ESM loss on original training sequences
    """
    from esm_loss import esm_loss
    
    # Filter empty sequences
    valid_sequences = [seq for seq in train_sequences if len(seq) > 0]
    
    if len(valid_sequences) == 0:
        print("Warning: No valid sequences for ESM baseline computation")
        return 0.0
    
    print(f"Computing ESM baseline on ALL {len(valid_sequences)} original training sequences...")
    baseline = esm_loss(valid_sequences, K=K, model_name=model_name)
    
    # Cache the baseline
    set_esm_baseline(baseline)
    
    print(f"ESM baseline loss: {baseline:.4f}")
    print("  (This is the 'floor' loss representing natural protein quality)")
    print("  (Normalized ESM loss = raw - baseline, target is 0)")
    
    return baseline