"""
QOBRA - Count Module

This module handles token frequency analysis, target distribution generation, and
sequence reconstruction evaluation. It creates the mapping between molecular tokens and
quantum amplitudes, generates target distributions for training, and evaluates
the performance of the trained quantum autoencoder.

Key functionality:
- Token frequency analysis and quantum amplitude mapping
- Target distribution generation for optimal latent space structure
- Sequence reconstruction evaluation and comparison
- Performance metrics calculation and visualization
- Statistical analysis of training results

The module is crucial for both training (creating target distributions) and
evaluation (measuring reconstruction quality). Current implementation uses amino
acids as tokens but is designed for general molecular sequence applications.
"""

from functools import partial
import torch
import time

from inputs import *
from scipy.stats import norm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Target distribution parameters for latent space
# These create a well-structured Gaussian distribution in latent space
mu, std = 0., 1/np.sqrt(1.5*dim_tot)  # Mean=0, std scaled by quantum dimension

# GPU device for target generation
_target_device = None

def get_target_device():
    """Get the device for target generation (GPU if available)."""
    global _target_device
    if _target_device is None:
        if torch.cuda.is_available():
            _target_device = torch.device("cuda")
        else:
            _target_device = torch.device("cpu")
    return _target_device

def make_target(n, mu, std):
    """
    GPU-accelerated target distribution sampling.
    
    This function creates samples from a multivariate Gaussian distribution
    that serves as the target for the latent space. Uses GPU batch sampling
    with rejection for efficiency.
    
    The first component is always positive (representing the "head" amplitude),
    while the remaining components follow a Gaussian distribution. This structure
    ensures that the generated samples are valid quantum state amplitudes.
    
    Parameters:
    - n: Number of samples to generate
    - mu: Mean of the Gaussian distribution
    - std: Standard deviation of the Gaussian distribution
    
    Returns:
    - Array of target latent space samples (n × dim_latent)
    """
    device = get_target_device()
    
    target = []
    batch_size = n * 3  # Oversample to account for rejections
    
    while len(target) < n:
        # Generate large batch of samples on GPU
        t = torch.randn(batch_size, dim_latent - 1, device=device) * std + mu
        
        # Calculate norms of tail components
        norms = torch.linalg.norm(t, dim=1)
        
        # Reject samples outside unit sphere
        valid_mask = norms < 1
        valid_t = t[valid_mask]
        
        if valid_t.shape[0] > 0:
            # Calculate head components to ensure unit norm
            valid_norms = torch.linalg.norm(valid_t, dim=1)
            heads = torch.sqrt(1 - valid_norms ** 2)
            
            # Create complete target vectors [head, tail_components]
            samples = torch.cat([heads.unsqueeze(1), valid_t], dim=1)
            
            # Move to CPU and convert to numpy
            target.extend(samples.cpu().numpy())
    
    return np.array(target[:n])

# =============================================
# QUANTUM AMPLITUDE MAPPING SYSTEM
# =============================================
# Create mapping between amino acid tokens and quantum amplitudes

# Generate amplitude values using cumulative sum approach
# This creates a non-uniform distribution that reflects amino acid frequencies
keys = [(-1)**i * (.5 + sum(xt[:i])) for i in range(Len)]
keys = sorted(keys)  # Sort to ensure consistent ordering

# Calculate head amplitude (normalization constant)
# This ensures proper quantum state normalization
head = np.max(np.abs(keys)) * (.75*num_tot - 3)

# Create bidirectional mappings between characters and amplitudes
ctf = {ks[i]: keys[i] for i in range(Len)}     # Character to frequency mapping
ftc = {keys[i]: ks[i] for i in range(Len)}     # Frequency to character mapping

# Save the character-to-frequency mapping for reference
F = open(f"{S}/PDBcodes-{S}.txt", "a")
F.write(f"{ctf}")
F.close()

# =============================================
# ENCODER AND DECODER PARAMETER INITIALIZATION
# =============================================
# Initialize or load encoder AND decoder parameters based on training mode
# TWO-PHASE TRAINING: Encoder and decoder have INDEPENDENT parameters

if switch == 0:
    # Training mode: Initialize with random parameters
    algorithm_globals.random_seed = 42
    xe = algorithm_globals.random.random(e.num_parameters)
    
    # Initialize decoder with different random seed
    algorithm_globals.random_seed = 43
    xd = algorithm_globals.random.random(d.num_parameters)
    
    print(f"Initialized {len(xe)} encoder + {len(xd)} decoder parameters")
else:
    # Inference mode: Load pre-trained parameters
    with open(f'{S}/opt-e-{S}.pkl', 'rb') as F:
        xe = pickle.load(F)
    
    # Load decoder parameters
    decoder_file = f'{S}/opt-d-{S}.pkl'
    if os.path.exists(decoder_file):
        with open(decoder_file, 'rb') as F:
            xd = pickle.load(F)
        print(f"Loaded encoder and decoder parameters from {S}/")
    else:
        # Fallback: initialize decoder randomly
        algorithm_globals.random_seed = 43
        xd = algorithm_globals.random.random(d.num_parameters)
        print(f"Loaded encoder parameters, initialized decoder randomly")

# =============================================
# TOKEN FREQUENCY VISUALIZATION
# =============================================
# Create visualization of token frequencies in Gaussian order

# Extract tokens and their frequencies in Gaussian order
cnt_x3 = list(gauss_sort.keys())
cnt_y3 = [cnt[x] for x in cnt_x3]

# Create bar chart of token frequencies
fig, ax = plt.subplots(figsize=(10, 5))
clear_output(wait=True)
bars = ax.bar(keys, cnt_y3, width=1.8, color='g')

# Add amplitude values as labels on each bar
i = 0
for bar in bars:
    height = bar.get_height()  # Get the height of each bar
    ax.text(
        bar.get_x() + bar.get_width()/2,  # X position at center of bar
        height,                            # Y position slightly above bar
        f'{keys[i]:.1f}',                 # Amplitude value label
        ha='center',                       # Horizontal alignment
        va='bottom',                       # Vertical alignment
        fontsize=4.5                       # Font size
    )
    i += 1

# Format and save the plot
ax.set_title('Frequencies of token numbers', fontsize=16)
ax.set_xticks(keys, cnt_x3)
ax.tick_params(axis='x', labelsize=8)
ax.set_xlabel("Token values", fontsize=16)
ax.set_ylabel("Frequency", fontsize=16)
fig.savefig(f"{S}/tokens-{metals}.png", dpi=300, bbox_inches='tight')

def output(seqs, h, Set, encoder_params=None, decoder_params=None):
    """
    Evaluate sequence reconstruction performance of the trained autoencoder.
    
    OPTIMIZED VERSION:
    - Batch unitary computation (compute autoencoder unitary once)
    - GPU-accelerated matrix multiplication for all sequences
    - Multiprocessing for sequence decoding
    
    Parameters:
    - seqs: List of molecular sequences to evaluate
    - h: Head amplitude for normalization
    - Set: Dataset name (e.g., "Train", "Test")
    - encoder_params: Trained encoder parameters (if None, uses module-level xe)
    - decoder_params: Trained decoder parameters (if None, uses module-level xd)
    """
    import torch
    from qiskit.quantum_info import Operator
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    import multiprocessing as mp
    
    # Use provided parameters or fall back to module-level globals
    p_encoder = encoder_params if encoder_params is not None else xe
    p_decoder = decoder_params if decoder_params is not None else xd
    
    print(f"\nEvaluating {Set} set ({len(seqs)} sequences)...")
    start_time = time.time()
    
    # ============================================
    # BATCH ENCODE ALL SEQUENCES
    # ============================================
    print("  Encoding sequences...")
    states = np.array([
        encode_amino_acid_sequence(seq, ctf=ctf, head=h, 
                                   max_len=max_len, vec_len=dim_tot)
        for seq in seqs
    ])
    
    # ============================================
    # COMPUTE AUTOENCODER UNITARY (ONCE)
    # ============================================
    # The autoencoder uses INDEPENDENT encoder and decoder parameters
    # Autoencoder = Decoder @ Encoder
    print("  Computing encoder and decoder unitaries...")
    
    # Get encoder unitary
    encoder_circuit_params = list(e.parameters)
    encoder_dict = {encoder_circuit_params[j]: p_encoder[j] for j in range(len(encoder_circuit_params))}
    encoder_circuit = e.assign_parameters(encoder_dict)
    U_encoder = Operator(encoder_circuit).data
    
    # Get decoder unitary (INDEPENDENT parameters)
    decoder_circuit_params = list(d.parameters)
    decoder_dict = {decoder_circuit_params[j]: p_decoder[j] for j in range(len(decoder_circuit_params))}
    decoder_circuit = d.assign_parameters(decoder_dict)
    U_decoder = Operator(decoder_circuit).data
    
    # Autoencoder unitary: U_decoder @ U_encoder
    U_autoencoder = U_decoder @ U_encoder
    
    # ============================================
    # BATCH MATRIX MULTIPLICATION (GPU)
    # ============================================
    print("  Applying autoencoder (batch GPU)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move to GPU
    states_t = torch.tensor(states, device=device, dtype=torch.complex64)
    U_t = torch.tensor(U_autoencoder, device=device, dtype=torch.complex64)
    
    # Batch apply autoencoder: output = input @ U^T
    output_states = (states_t @ U_t.T).cpu().numpy()
    
    # Input states are just the original (no transformation needed)
    input_states = states
    
    # ============================================
    # PARALLEL DECODING WITH MULTIPROCESSING
    # ============================================
    print("  Decoding sequences (parallel)...")
    
    # Prepare data for parallel processing
    decode_args = [
        (np.real(input_states[i]), np.real(output_states[i]), ftc, h, dim_tot)
        for i in range(len(seqs))
    ]
    
    # Use ThreadPoolExecutor for I/O-bound decoding (GIL-friendly for dict lookups)
    num_workers = min(mp.cpu_count(), 16)
    
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_decode_and_compare, decode_args))
    
    # ============================================
    # WRITE RESULTS
    # ============================================
    print("  Writing results...")
    file = open(f"{S}/Results-{S}.txt", "a")
    file1 = open(f"{S}/R-{S}.txt", "a")
    
    success_count = 0
    for original_seq, output_seq, ratio in results:
        # Find sequence terminators
        idx1 = original_seq.find("X")
        idx2 = output_seq.find("X")
        
        # Write sequences
        if idx1 != -1:
            file1.write(f"I: {original_seq[:idx1]}\n")
        else:
            file1.write(f"I: {original_seq}\n")
            
        if idx2 != -1:
            file1.write(f"O: {output_seq[:idx2]}\n")
        else:
            file1.write(f"O: {output_seq}\n")
        
        file1.write(f"Sequence ratio of correct matches: {ratio}\n")
        file1.write("*" * dim_tot + '\n')
        
        if ratio == 1:
            success_count += 1
    
    success_rate = success_count / len(seqs)
    file.write(f"{Set}\t{len(seqs)}\t{success_rate:.3f}\n")
    
    file.close()
    file1.close()
    
    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.2f}s (success rate: {success_rate:.3f})")


def _decode_and_compare(args):
    """
    Helper function for parallel decoding.
    Decodes input and output states and computes similarity.
    
    Parameters:
    - args: Tuple of (input_state, output_state, ftc, head, dim_tot)
    
    Returns:
    - Tuple of (original_seq, output_seq, similarity_ratio)
    """
    input_state, output_state, ftc, h, dim_tot = args
    
    # Decode both states to sequences
    original_seq = _decode_sequence(input_state, ftc, h, dim_tot)
    output_seq = _decode_sequence(output_state, ftc, h, dim_tot)
    
    # Calculate similarity
    ratio = seq_match_ratio(original_seq, output_seq)
    return original_seq, output_seq, ratio


def _decode_sequence(code, ftc, h, dim_tot):
    """
    Decode a quantum state vector to amino acid sequence.
    Optimized version of decode_amino_acid_sequence for parallel execution.
    
    Parameters:
    - code: State vector (real part)
    - ftc: Frequency to character mapping
    - h: Head amplitude
    - dim_tot: Dimension of state vector
    
    Returns:
    - Decoded amino acid sequence string
    """
    ret = ""
    
    # Handle edge case
    if abs(code[0]) < 1e-10:
        return ""
    
    factor = h / code[0]
    ftc_keys = np.array(list(ftc.keys()))
    
    for i in range(1, dim_tot):
        key_val = code[i] * factor
        
        # Find closest matching amplitude using vectorized operation
        diffs = np.abs(ftc_keys - key_val)
        min_idx = np.argmin(diffs)
        key = ftc_keys[min_idx]
        
        if key not in ftc:
            return ""
        ret += ftc[key]
    return ret