"""
QOBRA - Configuration Module

This module contains all configuration constants and settings for the QOBRA training system.
Centralizing configuration makes it easy to tune hyperparameters and adjust training behavior.

Configuration categories:
1. Loss function weights (lambdas) for two-phase training
2. Caching and frequency settings
3. Mini-batching configuration
4. Progress bar settings

TWO-PHASE TRAINING:
- Phase 1: Encoder trained on MMD loss (latent space matching)
- Phase 2: Decoder trained on Fidelity + ESM losses (encoder frozen)

This module imports from count.py to get access to all upstream variables
(from ansatz.py, model.py, inputs.py, count.py) via the import chain.
"""

# Import everything from count to get access to all upstream variables
# This includes: dim_tot, num_encode, S, mu, std, etc.
from count import *

# =============================================
# LOSS FUNCTION WEIGHTS (TWO-PHASE TRAINING)
# =============================================

# Phase 1: Encoder training - MMD loss only
LAMBDA_MMD = lambda_mmd_max  # MMD weight for encoder training

# Phase 2: Decoder training - Fidelity + ESM losses
LAMBDA_FIDELITY = lambda_fidelity_max  # Fidelity weight for decoder training
LAMBDA_ESM_MAX = esm_lambda_max  # Maximum ESM weight for decoder training
ESM_WARMUP = esm_warmup  # Number of iterations for ESM lambda warmup (None = no warmup)
ESM_STEP_INTERVAL = esm_step_interval  # Iterations before ESM turns on (0 -> MAX step function)
ESM_SUBSET_SIZE = esm_subset_size  # Number of sequences for ESM loss computation

def get_esm_lambda(iteration):
    """
    Get the ESM lambda value with optional annealing (linear or step).
    
    Parameters:
    - iteration: Current training iteration (0-indexed)
    
    Returns:
    - Current ESM lambda value
    
    Annealing modes:
    - No warmup (ESM_WARMUP=None): Returns LAMBDA_ESM_MAX immediately
    - Step function (ESM_STEP_INTERVAL set): 0 for first ESM_STEP_INTERVAL iterations, then LAMBDA_ESM_MAX
    - Linear warmup (ESM_STEP_INTERVAL=None): Linearly anneals from 0 to LAMBDA_ESM_MAX over ESM_WARMUP iterations
    """
    # Step function: 0 for first N iterations, then jump to MAX
    if ESM_STEP_INTERVAL is not None and ESM_STEP_INTERVAL > 0:
        if iteration < ESM_STEP_INTERVAL:
            return 0.0
        return LAMBDA_ESM_MAX
    
    # No warmup: return MAX immediately
    if ESM_WARMUP is None or ESM_WARMUP <= 0:
        return LAMBDA_ESM_MAX
    
    # After warmup complete: return MAX
    if iteration >= ESM_WARMUP:
        return LAMBDA_ESM_MAX
    
    # Linear warmup: lambda = (iteration / warmup) * lambda_max
    return (iteration / ESM_WARMUP) * LAMBDA_ESM_MAX

# Backward compatibility: LAMBDA_ESM is now the max value
LAMBDA_ESM = LAMBDA_ESM_MAX

# =============================================
# TQDM PROGRESS BAR SETTINGS
# =============================================

MAX_EPOCHS_ENCODER = 5000  # Default epochs for encoder (phase 1)
MAX_EPOCHS_DECODER = 5000  # Default epochs for decoder (phase 2)

# =============================================
# CACHING AND FREQUENCY SETTINGS
# =============================================

# ESM loss computation frequency (reduces ESM overhead significantly)
ESM_COMPUTE_FREQUENCY = 1  # Compute ESM loss every N iterations

# Test set computation frequency (test loss only needed for monitoring)
TEST_COMPUTE_FREQUENCY = 1  # Compute test loss every N iterations

# Plotting frequency (reduce overhead from frequent plot generation)
PLOT_FREQUENCY = 20  # Generate plots every N iterations

# Checkpoint saving frequency for verbose output
CHECKPOINT_VERBOSE_FREQUENCY = 20  # Print checkpoint message every N iterations

# =============================================
# MINI-BATCHING CONFIGURATION
# =============================================

# Mini-batching configuration (reduces per-iteration computation)
MINI_BATCH_SIZE = 6000  # Number of sequences per mini-batch
MINI_BATCH_ENABLED = True  # Set to False to use full dataset

# =============================================
# EPOCH-BASED BATCH ITERATOR
# =============================================

class EpochBatchIterator:
    """
    Proper epoch-based mini-batch iterator.
    
    Standard training practice:
    1. Shuffle dataset at the start of each epoch
    2. Partition into sequential mini-batches
    3. Each sample seen exactly once per epoch
    """
    
    def __init__(self, n_samples, batch_size, seed=42):
        """
        Initialize the batch iterator.
        
        Parameters:
        - n_samples: Total number of training samples
        - batch_size: Size of each mini-batch
        - seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        
        # Calculate number of batches per epoch
        self.n_batches = int(np.ceil(n_samples / batch_size))
        
        # Initialize state
        self.current_epoch = 0
        self.current_batch = 0
        self.shuffled_indices = None
        
        # Start first epoch
        self._start_new_epoch()
    
    def _start_new_epoch(self):
        """Shuffle indices for a new epoch."""
        self.shuffled_indices = self.rng.permutation(self.n_samples)
        self.current_batch = 0
        self.current_epoch += 1
    
    def get_batch_indices(self):
        """
        Get indices for the next mini-batch.
        
        Returns:
        - batch_indices: Array of indices for this batch
        - is_epoch_start: True if this is the first batch of a new epoch
        """
        is_epoch_start = (self.current_batch == 0)
        
        # Calculate batch boundaries
        start = self.current_batch * self.batch_size
        end = min(start + self.batch_size, self.n_samples)
        
        # Get batch indices
        batch_indices = self.shuffled_indices[start:end]
        
        # Move to next batch
        self.current_batch += 1
        
        # Check if epoch is complete
        if self.current_batch >= self.n_batches:
            self._start_new_epoch()
        
        return batch_indices, is_epoch_start
    
    def get_progress(self):
        """Get current epoch and batch progress."""
        return {
            'epoch': self.current_epoch,
            'batch': self.current_batch,
            'n_batches': self.n_batches,
            'progress': self.current_batch / self.n_batches
        }

# Global batch iterator (initialized when training starts)
_batch_iterator = None

def init_batch_iterator(n_samples, batch_size=MINI_BATCH_SIZE, seed=42):
    """Initialize the global batch iterator."""
    global _batch_iterator
    _batch_iterator = EpochBatchIterator(n_samples, batch_size, seed)
    print(f"Batch iterator: {_batch_iterator.n_batches} batches/epoch, {batch_size} samples/batch")
    return _batch_iterator

def get_batch_iterator():
    """Get the global batch iterator."""
    return _batch_iterator

