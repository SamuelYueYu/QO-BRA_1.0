"""
QOBRA - Plotting Module

This module contains all visualization and plotting functions:
1. Histogram plots for latent space distributions
2. Loss curve plots during training
3. Extended multi-panel plots

TWO-PHASE TRAINING:
- Phase 1 (Encoder): Tracks MMD loss
- Phase 2 (Decoder): Tracks Fidelity + ESM losses

These functions create visualizations to monitor training progress
and analyze the learned representations.
"""

# Set matplotlib backend BEFORE importing pyplot
# 'Agg' is a non-GUI backend that works in threads and headless environments
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# Import everything from config (which imports from count)
# This gives us: S, dim_latent, LAMBDA_ESM, etc.
from config import *

# =============================================
# GLOBAL TRACKING LISTS - PHASE 1 (ENCODER)
# =============================================

# Encoder training tracks MMD loss only
trains_k, tests_k = [], []

def get_encoder_iteration():
    """Get the current encoder iteration number."""
    return len(trains_k)

def append_encoder_losses(k_train, k_test):
    """Append encoder loss values (MMD only)."""
    trains_k.append(k_train)
    tests_k.append(k_test)

# =============================================
# GLOBAL TRACKING LISTS - PHASE 2 (DECODER)
# =============================================

# Decoder training tracks Fidelity + ESM losses
trains_fidelity, tests_fidelity = [], []
trains_esm, tests_esm = [], []
lambda_esm_history = []  # Track lambda_esm annealing

def get_decoder_iteration():
    """Get the current decoder iteration number."""
    return len(trains_fidelity)

def append_decoder_losses(fidelity_train, fidelity_test, esm_train=None, esm_test=None, lambda_esm=None):
    """Append decoder loss values (Fidelity + ESM) and lambda_esm."""
    trains_fidelity.append(fidelity_train)
    tests_fidelity.append(fidelity_test)
    if esm_train is not None:
        trains_esm.append(esm_train)
    if esm_test is not None:
        tests_esm.append(esm_test)
    if lambda_esm is not None:
        lambda_esm_history.append(lambda_esm)

def reset_decoder_tracking():
    """Reset decoder tracking lists for phase 2 start."""
    global trains_fidelity, tests_fidelity, trains_esm, tests_esm, lambda_esm_history
    trains_fidelity, tests_fidelity = [], []
    trains_esm, tests_esm = [], []
    lambda_esm_history = []

# =============================================
# BACKWARD COMPATIBILITY
# =============================================

def get_loss_history():
    """Get all loss history lists."""
    return {
        'trains_k': trains_k,
        'tests_k': tests_k,
        'trains_fidelity': trains_fidelity,
        'tests_fidelity': tests_fidelity,
        'trains_esm': trains_esm,
        'tests_esm': tests_esm,
        'lambda_esm_history': lambda_esm_history,
    }

def append_losses(k_train, k_test, fidelity_train=None, fidelity_test=None, 
                  esm_train=None, esm_test=None):
    """Backward compatible append function."""
    trains_k.append(k_train)
    tests_k.append(k_test)
    if fidelity_train is not None:
        trains_fidelity.append(fidelity_train)
    if fidelity_test is not None:
        tests_fidelity.append(fidelity_test)
    if esm_train is not None:
        trains_esm.append(esm_train)
    if esm_test is not None:
        tests_esm.append(esm_test)

def get_current_iteration():
    """Get the current iteration (encoder phase)."""
    return len(trains_k)

# =============================================
# HISTOGRAM PLOTTING
# =============================================

def plot_hist(train, test, target):
    """
    Plot histograms of latent space distributions for training monitoring.
    
    This function creates visualizations to monitor the training progress by
    showing how well the encoded sequences match the target distribution.
    It plots both the full distributions and a detailed view of the first component.
    
    Parameters:
    - train: Training set latent representations
    - test: Test set latent representations  
    - target: Target distribution samples
    """
    plt.figure(figsize=(5, 5))
    
    # Plot histogram of latent space amplitudes (excluding first component)
    plt.hist(train[:, 1:].flatten(), density=True, bins=dim_latent, 
             color='r', alpha=1, label='Train')
    plt.hist(test[:, 1:].flatten(), density=True, bins=dim_latent, 
             color='g', alpha=.3, label='Test')
    plt.hist(target[:, 1:].flatten(), density=True, bins=dim_latent, 
             color='b', alpha=.2, label='Target')
    
    plt.title("Frequency of state amplitudes")
    plt.xlabel("State amplitudes")
    plt.ylabel("Frequency")
    plt.legend()
    
    # Add inset plot for the first component (most important)
    inset_ax = plt.axes([0.125, 0.679, 0.2, 0.2])
    
    # Plot histograms of the first component (head amplitudes)
    inset_ax.hist(abs(train[:, 0]).flatten(), density=True, bins=dim_latent, 
                  color='r', alpha=1)
    inset_ax.hist(abs(test[:, 0]).flatten(), density=True, bins=dim_latent, 
                  color='g', alpha=.3)
    inset_ax.hist(target[:, 0].flatten(), density=True, bins=dim_latent, 
                  color='b', alpha=.2)
    
    # Format inset plot
    inset_ax.tick_params(axis='both', labelsize=8)
    inset_ax.yaxis.tick_right()
    
    plt.savefig(f"{S}/{S}-hist.png", dpi=300, bbox_inches='tight')
    plt.close()

# =============================================
# LOSS CURVE PLOTTING
# =============================================
    
def plot(train, test, target):
    """
    Generate all training progress plots (encoder phase).
    
    Parameters:
    - train: Training set latent representations
    - test: Test set latent representations
    - target: Target distribution samples
    """
    plot_hist(train, test, target)
    
    # Plot MMD loss evolution
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    ax.plot(range(len(trains_k)), trains_k, 'b-', label="Train MMD")
    ax.plot(range(len(tests_k)), tests_k, 'b:', label="Test MMD")
    ax.set_title("Phase 1: Encoder MMD Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MMD")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{S}/{S}-encoder.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

# =============================================
# LOSS CURVE PLOTTING (supports both phases)
# =============================================

def plot_loss_curves():
    """
    Plot loss curves for both phases in a 2x2 layout.
    
    Phase 1: MMD loss (encoder)
    Phase 2: Fidelity + ESM losses (decoder)
    Plus: Lambda_ESM annealing schedule
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing
    
    # Plot 1: MMD Loss (Encoder)
    if len(trains_k) > 0:
        axes[0].plot(range(len(trains_k)), trains_k, 'b-', label="Train")
        axes[0].plot(range(len(tests_k)), tests_k, 'b:', label="Test")
        axes[0].legend()
    axes[0].set_title("Phase 1: MMD Loss (Encoder)")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("MMD Loss")
    
    # Plot 2: Fidelity Loss (Decoder)
    if len(trains_fidelity) > 0:
        axes[1].plot(range(len(trains_fidelity)), trains_fidelity, 'g-', label="Train")
        axes[1].plot(range(len(tests_fidelity)), tests_fidelity, 'g:', label="Test")
        axes[1].legend()
    axes[1].set_title("Phase 2: Fidelity Loss (Decoder)")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Fidelity Loss")
    
    # Plot 3: ESM Loss (Decoder) - Normalized (raw - baseline)
    if len(trains_esm) > 0:
        axes[2].plot(range(len(trains_esm)), trains_esm, 'r-', label="Train")
        axes[2].plot(range(len(tests_esm)), tests_esm, 'r:', label="Test")
        axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3, label="Baseline (target)")
        axes[2].legend()
    axes[2].set_title("Phase 2: Normalized ESM Loss (raw - baseline)")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Normalized ESM Loss")
    
    # Plot 4: Lambda_ESM Annealing
    if len(lambda_esm_history) > 0:
        axes[3].plot(range(len(lambda_esm_history)), lambda_esm_history, 'm-', linewidth=2)
    axes[3].set_title("Lambda_ESM Annealing Schedule")
    axes[3].set_xlabel("Iteration")
    axes[3].set_ylabel("λ_ESM")
    
    plt.tight_layout()
    plt.savefig(f"{S}/{S}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================
# EXTENDED PLOTTING (encoder phase)
# =============================================

def plot_extended(train, test, target):
    """
    Extended plotting with latent space histograms and loss curves.
    
    Parameters:
    - train: Training set latent representations
    - test: Test set latent representations
    - target: Target distribution samples
    """
    plot_hist(train, test, target)
    plot_loss_curves()
