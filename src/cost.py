"""
QOBRA - Cost Function Module

This module implements the loss functions for two-phase training:

PHASE 1 - ENCODER TRAINING:
- Loss: MMD only (latent space matching to Gaussian)
- Optimizes encoder parameters

PHASE 2 - DECODER TRAINING:
- Loss: Fidelity + ESM (reconstruction quality + biological plausibility)
- Encoder frozen, optimizes decoder parameters

ARCHITECTURE:
- Encoder: Parameterized by xe, maps input → latent space
- Decoder: Parameterized by xd (INDEPENDENT), maps latent space → output

This module imports from:
- config.py: Configuration constants and settings
- utils.py: Device management, checkpointing, progress bar
- encoding.py: Sequence encoding and unitary operations
- losses.py: Individual loss computation functions
- plotting.py: Visualization functions
"""

import time
import numpy as np
import torch
from tqdm import tqdm

# Import everything from config (which imports from count)
# This gives us all upstream variables: num_encode, S, xe, xd, etc.
# Also includes: init_batch_iterator, get_batch_iterator, EpochBatchIterator
from config import *

# Import from split modules
from utils import (
    get_device, save_checkpoint, 
    get_progress_bar, update_progress_bar,
    init_progress_bar, close_progress_bar
)
from encoding import (
    precompute_encoded_states, get_cached_states, set_cached_states,
    get_encoder_unitary, get_decoder_unitary,
    batch_latent_encode, batch_autoencoder, latent_rep, latent_encode
)
from losses import (
    compute_fidelity_loss, compute_kernel_loss, get_cached_target,
    decode_states_to_sequences,
    get_esm_cache, set_esm_cache,
    get_fidelity_cache, set_fidelity_cache,
    get_test_cache, set_test_cache,
    get_esm_baseline
)
from plotting import (
    plot, plot_hist, plot_extended, plot_loss_curves,
    trains_k, tests_k, trains_esm, tests_esm, trains_fidelity, tests_fidelity,
    append_encoder_losses, append_decoder_losses, 
    get_encoder_iteration, get_decoder_iteration,
    reset_decoder_tracking
)

# Import ESM loss function
from esm_loss import esm_loss

# =============================================
# ESM SUBSET RANDOM SELECTION
# =============================================
# Seeded RNG for reproducible random subset selection across runs
_esm_rng = np.random.RandomState(seed=12345)

def get_random_esm_indices(n_total, subset_size):
    """
    Get random indices for ESM subset selection.
    
    Uses a seeded RNG so that the same sequence of random selections
    is reproducible across runs of train.py.
    
    Parameters:
    - n_total: Total number of sequences available
    - subset_size: Number of sequences to select
    
    Returns:
    - Array of randomly selected indices
    """
    actual_size = min(subset_size, n_total)
    return _esm_rng.choice(n_total, size=actual_size, replace=False)

# =============================================
# PHASE 1: ENCODER LOSS (MMD ONLY)
# =============================================

def encoder_loss(p_encoder, train_input, test_input):
    """
    Compute the encoder loss function for phase 1 training.
    
    PHASE 1: Train encoder on MMD loss only.
    Goal: Learn encoder that maps inputs to a well-structured Gaussian latent space.
    
    Loss: L = λ_mmd * L_MMD
    
    Parameters:
    - p_encoder: Encoder parameters (length = num_encode)
    - train_input: Training protein sequences
    - test_input: Test protein sequences
    
    Returns:
    - MMD loss value (scalar)
    """
    s = time.time()
    iteration = get_encoder_iteration()
    
    # Get cached states
    _cached_train_states, _cached_test_states = get_cached_states()
    _cached_k_test = get_test_cache()
    
    # Ensure cached states exist
    if _cached_train_states is None or _cached_test_states is None:
        _cached_train_states, _cached_test_states = precompute_encoded_states(train_input, test_input)
    
    # =============================================
    # MINI-BATCHING
    # =============================================
    batch_iterator = get_batch_iterator()
    
    if MINI_BATCH_ENABLED and batch_iterator is not None:
        batch_indices, is_epoch_start = batch_iterator.get_batch_indices()
        train_states_batch = _cached_train_states[batch_indices]
        n = len(batch_indices)
        
        if is_epoch_start and iteration > 0:
            progress = batch_iterator.get_progress()
            tqdm.write(f"--- Epoch {progress['epoch']} started (shuffled) ---")
    else:
        train_states_batch = _cached_train_states
        n = len(_cached_train_states)
    
    device = get_device()
    
    # =============================================
    # COMPUTE ENCODER UNITARY AND ENCODE
    # =============================================
    U_e = get_encoder_unitary(p_encoder)
    U_e_t = torch.tensor(U_e, device=device, dtype=torch.complex64)
    
    train_t = torch.tensor(train_states_batch, device=device, dtype=torch.complex64)
    latent_states = train_t @ U_e_t.T
    
    # =============================================
    # MMD LOSS
    # =============================================
    train_encode = latent_states.real.cpu().numpy()
    target = make_target(n, mu, std)
    k_train = compute_kernel_loss(train_encode, target)
    
    # =============================================
    # TEST SET EVALUATION (PERIODIC)
    # =============================================
    k_test = _cached_k_test
    test_encode = None
    
    if iteration % TEST_COMPUTE_FREQUENCY == 0:
        test_t = torch.tensor(_cached_test_states, device=device, dtype=torch.complex64)
        test_latent = test_t @ U_e_t.T
        test_encode = test_latent.real.cpu().numpy()
        test_target = make_target(len(_cached_test_states), mu, std)
        k_test = compute_kernel_loss(test_encode, test_target)
        set_test_cache(k_test)
    
    # Store loss values for plotting
    append_encoder_losses(k_train, k_test)
    
    # =============================================
    # TOTAL LOSS
    # =============================================
    total_loss = LAMBDA_MMD * k_train
    total_test_loss = LAMBDA_MMD * k_test
    
    # =============================================
    # CHECKPOINT SAVING
    # =============================================
    verbose_checkpoint = iteration % CHECKPOINT_VERBOSE_FREQUENCY == 0 or iteration < 5
    save_checkpoint('encoder', iteration, p_encoder, total_loss, total_test_loss, S, verbose=verbose_checkpoint)
    
    # Logging
    elapsed = time.time() - s
    if verbose_checkpoint:
        log_msg = (f"[Encoder] Iter {iteration:4d} | "
                   f"MMD: {k_train:.4f}/{k_test:.4f} | "
                   f"Time: {elapsed:.2f}s")
        tqdm.write(log_msg)
    
    # Progress bar
    _pbar = get_progress_bar()
    if _pbar is not None:
        postfix = {'mmd': f'{k_train:.4f}'}
        update_progress_bar(postfix)
    
    # Plotting (periodic)
    if iteration % PLOT_FREQUENCY == 0:
        if train_encode is None:
            train_encode = latent_states.real.cpu().numpy()
            target = make_target(n, mu, std)
        if test_encode is None:
            test_t = torch.tensor(_cached_test_states, device=device, dtype=torch.complex64)
            test_encode = (test_t @ U_e_t.T).real.cpu().numpy()
        plot_extended(train_encode, test_encode, target)
    
    return total_loss


# =============================================
# PHASE 2: DECODER LOSS (FIDELITY + ESM)
# =============================================

# Frozen encoder unitary (set when phase 2 starts)
_frozen_encoder_unitary = None
_frozen_encoder_unitary_t = None

def set_frozen_encoder(p_encoder):
    """
    Freeze the encoder parameters for phase 2 training.
    Computes and caches the encoder unitary.
    """
    global _frozen_encoder_unitary, _frozen_encoder_unitary_t
    _frozen_encoder_unitary = get_encoder_unitary(p_encoder)
    device = get_device()
    _frozen_encoder_unitary_t = torch.tensor(_frozen_encoder_unitary, device=device, dtype=torch.complex64)
    print(f"Encoder frozen with {len(p_encoder)} parameters")

def get_frozen_encoder_unitary():
    """Get the frozen encoder unitary (tensor on device)."""
    return _frozen_encoder_unitary_t

def decoder_loss(p_decoder, train_input, test_input):
    """
    Compute the decoder loss function for phase 2 training.
    
    PHASE 2: Train decoder on Fidelity + ESM losses.
    Encoder is FROZEN; only decoder parameters are optimized.
    Goal: Learn decoder that reconstructs biologically plausible sequences.
    
    Loss: L = λ_fidelity * L_fidelity + λ_esm * L_ESM
    
    Parameters:
    - p_decoder: Decoder parameters (length = num_decode)
    - train_input: Training protein sequences
    - test_input: Test protein sequences
    
    Returns:
    - Combined Fidelity + ESM loss value (scalar)
    """
    s = time.time()
    iteration = get_decoder_iteration()
    
    # Get cached states
    _cached_train_states, _cached_test_states = get_cached_states()
    _esm_iteration_counter, _cached_esm_train_loss, _cached_esm_test_loss = get_esm_cache()
    _cached_fidelity_train_loss, _cached_fidelity_test_loss = get_fidelity_cache()
    
    # Ensure cached states exist
    if _cached_train_states is None or _cached_test_states is None:
        _cached_train_states, _cached_test_states = precompute_encoded_states(train_input, test_input)
    
    # =============================================
    # MINI-BATCHING
    # =============================================
    batch_iterator = get_batch_iterator()
    
    if MINI_BATCH_ENABLED and batch_iterator is not None:
        batch_indices, is_epoch_start = batch_iterator.get_batch_indices()
        train_states_batch = _cached_train_states[batch_indices]
        n = len(batch_indices)
        
        if is_epoch_start and iteration > 0:
            progress = batch_iterator.get_progress()
            tqdm.write(f"--- Epoch {progress['epoch']} started (shuffled) ---")
    else:
        train_states_batch = _cached_train_states
        n = len(_cached_train_states)
    
    device = get_device()
    
    # =============================================
    # APPLY FROZEN ENCODER + DECODER
    # =============================================
    U_e_t = get_frozen_encoder_unitary()
    U_d = get_decoder_unitary(p_decoder)
    U_d_t = torch.tensor(U_d, device=device, dtype=torch.complex64)
    
    train_t = torch.tensor(train_states_batch, device=device, dtype=torch.complex64)
    
    # Apply frozen encoder
    latent_states = train_t @ U_e_t.T
    
    # Apply decoder
    reconstructed_states = latent_states @ U_d_t.T
    
    # =============================================
    # FIDELITY LOSS
    # =============================================
    fidelity_train = compute_fidelity_loss(
        train_states_batch,
        reconstructed_states.cpu().numpy()
    )
    
    # =============================================
    # TEST SET EVALUATION (PERIODIC) - compute first for ESM test loss
    # =============================================
    fidelity_test = _cached_fidelity_test_loss
    test_reconstructed = None
    
    if iteration % TEST_COMPUTE_FREQUENCY == 0:
        test_t = torch.tensor(_cached_test_states, device=device, dtype=torch.complex64)
        test_latent = test_t @ U_e_t.T
        test_reconstructed = test_latent @ U_d_t.T
        fidelity_test = compute_fidelity_loss(
            _cached_test_states,
            test_reconstructed.cpu().numpy()
        )
        set_fidelity_cache(fidelity_train, fidelity_test)
    
    # =============================================
    # ESM LOSS (with baseline normalization)
    # =============================================
    esm_train_loss_raw = _cached_esm_train_loss
    esm_test_loss_raw = _cached_esm_test_loss
    
    # Get the current ESM lambda with warmup annealing
    lambda_esm_current = get_esm_lambda(iteration)
    
    # Get baseline for normalization (computed on original training sequences)
    esm_baseline = get_esm_baseline()
    if esm_baseline is None:
        esm_baseline = 0.0  # Fallback if baseline not computed
    
    if LAMBDA_ESM_MAX > 0:
        if _esm_iteration_counter % ESM_COMPUTE_FREQUENCY == 0:
            # ESM train loss: from reconstructed training sequences
            recon_states_np = reconstructed_states.cpu().numpy()
            esm_indices = get_random_esm_indices(len(recon_states_np), ESM_SUBSET_SIZE)
            decoded_sequences = decode_states_to_sequences(
                recon_states_np[esm_indices], ftc, head
            )
            decoded_sequences = [seq for seq in decoded_sequences if len(seq) > 0]
            
            if len(decoded_sequences) > 0:
                esm_train_loss_raw = esm_loss(decoded_sequences, K=esm_K, model_name=esm_model_name)
            else:
                esm_train_loss_raw = 1.0
            
            # ESM test loss: from reconstructed TEST sequences (not original!)
            if test_reconstructed is not None:
                test_recon_np = test_reconstructed.cpu().numpy()
                test_esm_indices = get_random_esm_indices(len(test_recon_np), ESM_SUBSET_SIZE)
                decoded_test_sequences = decode_states_to_sequences(
                    test_recon_np[test_esm_indices], ftc, head
                )
                decoded_test_sequences = [seq for seq in decoded_test_sequences if len(seq) > 0]
                
                if len(decoded_test_sequences) > 0:
                    esm_test_loss_raw = esm_loss(decoded_test_sequences, K=esm_K, model_name=esm_model_name)
                else:
                    esm_test_loss_raw = 1.0
            else:
                # Compute test reconstruction if not already done
                test_t = torch.tensor(_cached_test_states, device=device, dtype=torch.complex64)
                test_latent = test_t @ U_e_t.T
                test_recon = test_latent @ U_d_t.T
                test_recon_np = test_recon.cpu().numpy()
                test_esm_indices = get_random_esm_indices(len(test_recon_np), ESM_SUBSET_SIZE)
                decoded_test_sequences = decode_states_to_sequences(
                    test_recon_np[test_esm_indices], ftc, head
                )
                decoded_test_sequences = [seq for seq in decoded_test_sequences if len(seq) > 0]
                
                if len(decoded_test_sequences) > 0:
                    esm_test_loss_raw = esm_loss(decoded_test_sequences, K=esm_K, model_name=esm_model_name)
                else:
                    esm_test_loss_raw = 1.0
            
            set_esm_cache(_esm_iteration_counter + 1, esm_train_loss_raw, esm_test_loss_raw)
        else:
            set_esm_cache(_esm_iteration_counter + 1, _cached_esm_train_loss, _cached_esm_test_loss)
    
    # Apply baseline normalization: normalized = max(0, raw - baseline)
    # This makes 0 the target (achieving training-level biological plausibility)
    esm_train_normalized = max(0.0, esm_train_loss_raw - esm_baseline)
    esm_test_normalized = max(0.0, esm_test_loss_raw - esm_baseline)
    
    # Store loss values for plotting (use normalized values)
    esm_train_val = esm_train_normalized if LAMBDA_ESM_MAX > 0 else 0.0
    esm_test_val = esm_test_normalized if LAMBDA_ESM_MAX > 0 else 0.0
    append_decoder_losses(fidelity_train, fidelity_test, esm_train_val, esm_test_val, lambda_esm_current)
    
    # =============================================
    # TOTAL LOSS (using normalized ESM loss)
    # =============================================
    total_loss = LAMBDA_FIDELITY * fidelity_train + lambda_esm_current * esm_train_normalized
    total_test_loss = LAMBDA_FIDELITY * fidelity_test + lambda_esm_current * esm_test_normalized
    
    # =============================================
    # CHECKPOINT SAVING
    # =============================================
    verbose_checkpoint = iteration % CHECKPOINT_VERBOSE_FREQUENCY == 0 or iteration < 5
    save_checkpoint('decoder', iteration, p_decoder, total_loss, total_test_loss, S, verbose=verbose_checkpoint)
    
    # Logging
    elapsed = time.time() - s
    if verbose_checkpoint:
        log_msg = (f"[Decoder] Iter {iteration:4d} | "
                   f"Fid: {fidelity_train:.4f}/{fidelity_test:.4f}")
        if LAMBDA_ESM_MAX > 0:
            # Show normalized ESM loss (raw - baseline), with baseline reference
            log_msg += f" | ESM: {esm_train_normalized:.4f}/{esm_test_normalized:.4f}"
            log_msg += f" (raw={esm_train_loss_raw:.4f}, base={esm_baseline:.4f})"
        log_msg += f" | Time: {elapsed:.2f}s"
        tqdm.write(log_msg)
    
    # Progress bar (show normalized ESM loss)
    _pbar = get_progress_bar()
    if _pbar is not None:
        postfix = {'fid': f'{fidelity_train:.4f}'}
        if LAMBDA_ESM_MAX > 0:
            postfix['esm'] = f'{esm_train_normalized:.4f}'
        update_progress_bar(postfix)
    
    # Plotting (periodic)
    if iteration % PLOT_FREQUENCY == 0:
        plot_loss_curves()
    
    return total_loss


# =============================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================

def autoencoder_loss(p, train_input, test_input):
    """
    Backward compatible wrapper.
    Uses encoder_loss for encoder-only optimization.
    """
    return encoder_loss(p, train_input, test_input)

def e_loss(p, train_input, test_input):
    """
    Backward compatible wrapper for encoder_loss.
    """
    return encoder_loss(p, train_input, test_input)
