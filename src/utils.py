"""
QOBRA - Utilities Module

This module contains utility functions for:
1. GPU device management
2. Checkpoint saving and loading
3. Progress bar management
4. Other helper functions

These utilities are shared across the training pipeline.
"""

import os
import pickle
import torch
from tqdm import tqdm

# =============================================
# GPU DEVICE CONFIGURATION
# =============================================
_device = None

def get_device():
    """Get the best available device (GPU if available, else CPU)."""
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device("cuda")
            print(f"GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
        else:
            _device = torch.device("cpu")
            print("Running on CPU (no GPU detected)")
    return _device

# =============================================
# TQDM PROGRESS BAR
# =============================================

# Global progress bar (initialized in train.py)
_pbar = None

def init_progress_bar(max_epochs=500):
    """Initialize the tqdm progress bar for training."""
    global _pbar
    _pbar = tqdm(total=max_epochs, desc="Training", unit="epoch", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    return _pbar

def close_progress_bar():
    """Close the progress bar."""
    global _pbar
    if _pbar is not None:
        _pbar.close()
        _pbar = None

def get_progress_bar():
    """Get the current progress bar instance."""
    return _pbar

def update_progress_bar(postfix_dict):
    """Update progress bar with new postfix values."""
    global _pbar
    if _pbar is not None:
        _pbar.update(1)
        _pbar.set_postfix(postfix_dict)

# =============================================
# CHECKPOINT SAVING
# =============================================

def save_checkpoint(name, iteration, params, train_loss, test_loss, output_dir, verbose=True):
    """
    Save a checkpoint of the training state.
    
    Parameters:
    - name: Checkpoint name (e.g., 'encoder', 'decoder', 'autoencoder')
    - iteration: Current iteration/epoch number
    - params: Parameter values to save
    - train_loss: Current training loss
    - test_loss: Current test loss
    - output_dir: Directory to save checkpoints
    - verbose: Whether to print checkpoint message
    """
    checkpoint = {
        'iteration': iteration,
        'params': params,
        'train_loss': train_loss,
        'test_loss': test_loss,
    }
    
    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = f"{output_dir}/checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Save checkpoint
    checkpoint_path = f"{checkpoint_dir}/{name}_iter{iteration:04d}.pkl"
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    # Also save as 'latest' for easy resumption
    latest_path = f"{checkpoint_dir}/{name}_latest.pkl"
    with open(latest_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    if verbose:
        tqdm.write(f"Saved {name} checkpoint at iteration {iteration}")

def load_checkpoint(name, output_dir, iteration=None):
    """
    Load a checkpoint from disk.
    
    Parameters:
    - name: Checkpoint name (e.g., 'encoder', 'decoder', 'autoencoder')
    - output_dir: Directory containing checkpoints
    - iteration: Specific iteration to load (None = load latest)
    
    Returns:
    - Checkpoint dictionary or None if not found
    """
    checkpoint_dir = f"{output_dir}/checkpoints"
    
    if iteration is not None:
        checkpoint_path = f"{checkpoint_dir}/{name}_iter{iteration:04d}.pkl"
    else:
        checkpoint_path = f"{checkpoint_dir}/{name}_latest.pkl"
    
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    return None