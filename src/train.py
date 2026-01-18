"""
QOBRA - Training Module

This module implements TWO-PHASE training for the QOBRA quantum autoencoder:

PHASE 1 - ENCODER TRAINING:
- Loss: MMD only (latent space matching to Gaussian)
- Optimizes encoder parameters
- Goal: Learn encoder that maps inputs to well-structured latent space

PHASE 2 - DECODER TRAINING:
- Loss: Fidelity + ESM (reconstruction + biological plausibility)
- Encoder FROZEN, optimizes decoder parameters
- Goal: Learn decoder that reconstructs biologically valid sequences

ARCHITECTURE:
- Encoder: Parameterized by xe, maps input → latent space
- Decoder: Parameterized by xd (INDEPENDENT), maps latent space → output

OPTIMIZATIONS APPLIED:
1. Pre-computed sequence encodings (one-time cost before training)
2. Batch unitary matrix multiplication (GPU-accelerated)
3. Reduced ESM loss computation frequency
4. Cached target distribution
5. Mini-batching for faster iterations
"""

# Set matplotlib backend BEFORE any imports (must be first)
# 'Agg' is a non-GUI backend that works in threads and headless environments
import matplotlib
matplotlib.use('Agg')

from cost import *
from scipy.optimize import minimize
from qiskit_algorithms.optimizers import COBYLA

# =============================================
# PRE-COMPUTATION PHASE (ONE-TIME COST)
# =============================================

print("=" * 60)
print("QOBRA Two-Phase Training")
print("=" * 60)
print(f"Training samples: {len(train_seqs)}")
print(f"Test samples: {len(test_seqs)}")
print(f"Qubits: {num_tot}, Dimension: {dim_tot}")
print("-" * 60)
print("Parameters:")
print(f"  Encoder: {num_encode} parameters")
print(f"  Decoder: {num_decode} parameters (independent)")
print("-" * 60)
print(f"Phase 1 (Encoder): λ_MMD = {LAMBDA_MMD}")
print(f"Phase 2 (Decoder): λ_Fidelity = {LAMBDA_FIDELITY}, λ_ESM = {LAMBDA_ESM}")
print("-" * 60)
print("Optimization settings:")
if MINI_BATCH_ENABLED:
    n_batches_per_epoch = int(np.ceil(len(train_seqs) / MINI_BATCH_SIZE))
    print(f"  Mini-batching: {MINI_BATCH_SIZE} samples/batch")
    print(f"  Batches/epoch: {n_batches_per_epoch}")
else:
    print(f"  Mini-batching: disabled")
print(f"  Max epochs (encoder): {MAX_EPOCHS_ENCODER}")
print(f"  Max epochs (decoder): {MAX_EPOCHS_DECODER}")
print(f"  Test loss frequency: every {TEST_COMPUTE_FREQUENCY} iterations")
print(f"  Plot frequency: every {PLOT_FREQUENCY} iterations")
if LAMBDA_ESM > 0:
    print(f"  ESM frequency: every {ESM_COMPUTE_FREQUENCY} iterations")
print("=" * 60)

# Pre-compute encoded states
precompute_encoded_states(train_seqs, test_seqs)

# Initialize epoch-based batch iterator
if MINI_BATCH_ENABLED:
    init_batch_iterator(len(train_seqs), batch_size=MINI_BATCH_SIZE, seed=42)

# Pre-generate target distribution
print("Pre-generating target distribution...")
_ = get_cached_target(len(train_seqs), mu, std)
print("Target distribution cached.")

# =============================================
# PHASE 1: ENCODER TRAINING (MMD ONLY)
# =============================================

print("\n" + "=" * 60)
print("PHASE 1: ENCODER TRAINING (MMD Loss)")
print("=" * 60)
print(f"Loss: L = {LAMBDA_MMD} * L_MMD")
print(f"Optimizing {num_encode} encoder parameters")
print("-" * 60)

start_phase1 = time.time()

# Initialize encoder optimizer
opt_encoder = COBYLA(maxiter=MAX_EPOCHS_ENCODER)

# Initialize progress bar for phase 1
print()
pbar = init_progress_bar(max_epochs=MAX_EPOCHS_ENCODER)

# Create partial function for encoder loss
f_encoder = partial(encoder_loss, train_input=train_seqs, test_input=test_seqs)

# Optimize encoder
try:
    encoder_result = opt_encoder.minimize(fun=f_encoder, x0=xe)
finally:
    close_progress_bar()

print()

# Extract optimized encoder parameters
xe_opt = encoder_result.x

# Save encoder parameters
with open(f'{S}/opt-e-{S}.pkl', 'wb') as F:
    pickle.dump(xe_opt, F)

elapsed_phase1 = (time.time() - start_phase1) / 3600
print(f"Phase 1 completed in {elapsed_phase1:.2f} h")
print(f"Saved encoder parameters to {S}/opt-e-{S}.pkl")

# =============================================
# PHASE 2: DECODER TRAINING (FIDELITY + ESM)
# =============================================

print("\n" + "=" * 60)
print("PHASE 2: DECODER TRAINING (Fidelity + ESM)")
print("=" * 60)
print(f"Loss: L = {LAMBDA_FIDELITY} * L_Fidelity + {LAMBDA_ESM} * L_ESM")
print(f"Encoder FROZEN, optimizing {num_decode} decoder parameters")
print("-" * 60)

start_phase2 = time.time()

# Freeze the encoder with optimized parameters
set_frozen_encoder(xe_opt)

# Compute ESM baseline on ALL original training sequences (one-time cost)
if LAMBDA_ESM_MAX > 0:
    from losses import compute_esm_baseline
    esm_baseline = compute_esm_baseline(
        train_seqs, 
        K=esm_K, 
        model_name=esm_model_name
    )
    print(f"ESM baseline established: {esm_baseline:.4f}")
    print("-" * 60)

# Reset batch iterator for phase 2
if MINI_BATCH_ENABLED:
    init_batch_iterator(len(train_seqs), batch_size=MINI_BATCH_SIZE, seed=43)

# Reset decoder loss tracking
reset_decoder_tracking()

# Initialize decoder optimizer
opt_decoder = COBYLA(maxiter=MAX_EPOCHS_DECODER)

# Initialize progress bar for phase 2
print()
pbar = init_progress_bar(max_epochs=MAX_EPOCHS_DECODER)

# Create partial function for decoder loss
f_decoder = partial(decoder_loss, train_input=train_seqs, test_input=test_seqs)

# Optimize decoder
try:
    decoder_result = opt_decoder.minimize(fun=f_decoder, x0=xd)
finally:
    close_progress_bar()

print()

# Extract optimized decoder parameters
xd_opt = decoder_result.x

# Save decoder parameters
with open(f'{S}/opt-d-{S}.pkl', 'wb') as F:
    pickle.dump(xd_opt, F)

elapsed_phase2 = (time.time() - start_phase2) / 3600
print(f"Phase 2 completed in {elapsed_phase2:.2f} h")
print(f"Saved decoder parameters to {S}/opt-d-{S}.pkl")

# Update global parameters
xe = xe_opt
xd = xd_opt

# =============================================
# PERFORMANCE EVALUATION
# =============================================

print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

start_eval = time.time()

# Initialize results file
file = open(f"{S}/Results-{S}.txt", "w")
file.write("Dataset\tSize\tR\n")
file.close()

# =============================================
# TRAINING SET EVALUATION
# =============================================

file = open(f"{S}/R-{S}.txt", "w")
file.write("TRAINING SET\n")
file.close()

output(train_seqs, head, "Train", encoder_params=xe, decoder_params=xd)

# =============================================
# TEST SET EVALUATION
# =============================================

file = open(f"{S}/R-{S}.txt", "a")
file.write("TEST SET\n")
file.close()

output(test_seqs, head, "Test", encoder_params=xe, decoder_params=xd)

# =============================================
# FINAL PERFORMANCE REPORTING
# =============================================

elapsed_eval = (time.time() - start_eval) / 60
print(f"\nEvaluation completed in {elapsed_eval:.2f} min")

total_time = elapsed_phase1 + elapsed_phase2 + elapsed_eval/60
print(f"\nTotal training time: {total_time:.2f} h")
print(f"  Phase 1 (Encoder): {elapsed_phase1:.2f} h")
print(f"  Phase 2 (Decoder): {elapsed_phase2:.2f} h")
print(f"  Evaluation: {elapsed_eval:.2f} min")
