"""
QOBRA - Generation Functions Module

This module contains utility functions for de novo molecular sequence generation,
visualization, and statistical analysis. It provides tools for:

- Quantum sequence generation from latent space
- Visualization script generation (current example: PyMOL for proteins)
- Statistical analysis and comparison of generated vs training sequences
- Data visualization and plotting utilities
- File I/O operations for sequence storage

These functions support the main generation pipeline by providing specialized
tools for sequence analysis, visualization, and validation across different
molecular domains.
"""

# Set matplotlib backend BEFORE importing matplotlib (via count.py)
# 'Agg' is a non-GUI backend that works in threads and headless environments
import matplotlib
matplotlib.use('Agg')

import string, urllib3
import sys, pickle, os
from count import *

# Import path for esm_eval module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'esm_eval'))

# =============================================
# INITIALIZATION AND SETUP
# =============================================

# Get alphabet for chain naming (A, B, C, ..., T)
# Used to assign unique identifiers to protein chains
alphabets = list(string.ascii_uppercase)

# PyMOL color palette for visualization
# These colors will be used to highlight different chains and binding sites
colors = ["red", "green", "yellow", "blue", 
          "salmon", "marine", "grey", "black"]

# Import Statevector for quantum circuit execution
from qiskit.quantum_info import Statevector

def generate_batch(lcodes):
    """
    Generate molecular sequences from a batch of latent codes.
    
    GPU-OPTIMIZED VERSION:
    1. Pre-compute decoder unitary matrix ONCE
    2. Batch all latent codes into a GPU tensor
    3. Single batched matrix multiplication on GPU (massive speedup)
    4. Parallel sequence decoding using ThreadPoolExecutor
    
    This is ~100-1000x faster than the previous version which ran
    individual Statevector operations per sequence.
    
    Parameters:
    - lcodes: Array of latent space vectors (N x dim_latent)
    
    Returns:
    - List of decoded molecular sequence strings
    """
    from concurrent.futures import ThreadPoolExecutor
    import multiprocessing as mp
    from qiskit.quantum_info import Operator
    import torch
    
    lcodes = np.array(lcodes)
    n_samples = len(lcodes)
    
    print(f"Generating {n_samples} sequences in GPU-batched mode...")
    
    # ============================================
    # STEP 1: COMPUTE DECODER UNITARY ONCE
    # ============================================
    print("  Computing decoder unitary matrix...")
    d_circuit_params = list(d.parameters)
    decoder_param_dict = {d_circuit_params[j]: xd[j] for j in range(num_decode)}
    decoder_circuit = d.assign_parameters(decoder_param_dict)
    U_decoder = Operator(decoder_circuit).data
    
    # ============================================
    # STEP 2: PREPARE ALL LATENT STATES
    # ============================================
    print("  Preparing latent states...")
    initial_states = np.zeros((n_samples, dim_tot), dtype=np.float64)
    
    for i, lcode in enumerate(lcodes):
        # Normalize the latent code
        lcode = np.array(lcode, dtype=np.float64)
        norm = np.linalg.norm(lcode)
        if norm > 0:
            lcode = lcode / norm
        
        # Pad to full state dimension
        initial_states[i, :len(lcode)] = lcode
    
    # ============================================
    # STEP 3: GPU BATCH MATRIX MULTIPLICATION
    # ============================================
    print("  Applying decoder (GPU batch)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move to GPU
    states_t = torch.tensor(initial_states, device=device, dtype=torch.complex128)
    U_t = torch.tensor(U_decoder, device=device, dtype=torch.complex128)
    
    # Batch apply decoder: output = input @ U^T
    # This single GPU operation replaces N individual Statevector.evolve() calls
    output_states = (states_t @ U_t.T).cpu().numpy()
    
    # Extract real parts (RealAmplitudes ansatz produces real outputs)
    output_real = output_states.real
    
    print(f"  GPU computation complete on {device}")
    
    # ============================================
    # STEP 4: PARALLEL SEQUENCE DECODING (CPU)
    # ============================================
    print("  Decoding sequences (parallel CPU)...")
    num_workers = min(mp.cpu_count(), 16)
    
    # Prepare decode arguments
    decode_args = [(output_real[i], ftc, head) for i in range(n_samples)]
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        sequences = list(executor.map(_decode_single, decode_args))
    
    print(f"Batch generation complete: {n_samples} sequences")
    return sequences


def _decode_single(args):
    """
    Helper function for parallel decoding of a single sequence.
    
    Parameters:
    - args: Tuple of (output_state, ftc, head)
    
    Returns:
    - Decoded amino acid sequence string
    """
    o_sv, ftc, head = args
    return decode_amino_acid_sequence(o_sv, ftc, head)

def generate(lcode):
    """
    Generate a molecular sequence from a latent space code using the quantum decoder.
    
    GPU-OPTIMIZED: Uses pre-computed decoder unitary and GPU matrix multiplication
    for single sequence generation. For batch generation, use generate_batch() which
    is even more efficient.
    
    Parameters:
    - lcode: Latent space vector (numpy array of length dim_latent)
    
    Returns:
    - Decoded molecular sequence string
    """
    from qiskit.quantum_info import Operator
    import torch
    
    # Normalize the latent code to ensure valid quantum state amplitudes
    lcode = np.array(lcode, dtype=np.float64)
    norm = np.linalg.norm(lcode)
    if norm > 0:
        lcode = lcode / norm
    
    # Pad latent code to full state dimension if needed (dim_latent -> dim_tot)
    initial_state = np.zeros(dim_tot, dtype=np.float64)
    initial_state[:len(lcode)] = lcode
    
    # Build decoder unitary
    d_circuit_params = list(d.parameters)
    decoder_param_dict = {d_circuit_params[j]: xd[j] for j in range(num_decode)}
    decoder_circuit = d.assign_parameters(decoder_param_dict)
    U_decoder = Operator(decoder_circuit).data
    
    # Apply decoder using GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_t = torch.tensor(initial_state, device=device, dtype=torch.complex128)
    U_t = torch.tensor(U_decoder, device=device, dtype=torch.complex128)
    
    # Apply decoder: output = input @ U^T
    output_state = (state_t @ U_t.T).cpu().numpy()
    
    # Get the quantum state amplitudes (real part preserves sign information)
    # RealAmplitudes ansatz produces real-valued outputs, so imaginary part is negligible
    o_sv = output_state.real
    
    # Decode the quantum state back to amino acid sequence
    return decode_amino_acid_sequence(o_sv, ftc, head)

def pml(s, HL, idx, prot_folder):
    """
    Generate visualization script and sequence files for structure visualization.
    
    This function creates visualization commands (current example: PyMOL for proteins)
    to visualize molecular structures with highlighted functional sites. It generates
    both a sequence file and a visualization script that can be used to visualize
    the molecular structure with color-coded elements and functional sites.
    
    Parameters:
    - s: Molecular sequence string
    - HL: Dictionary mapping segment IDs to functional site positions
    - idx: Sequence index for file naming
    - prot_folder: Output directory for generated files
    """
    # Ensure the output directory exists
    if not os.path.exists(prot_folder):
        os.makedirs(prot_folder)
    
    # =============================================
    # SEQUENCE FILE GENERATION
    # =============================================
    # Save the protein sequence to a text file
    
    file = open(f"{prot_folder}/{idx}.txt", "w")
    file.write(s)
    file.close()
    
    # =============================================
    # PYMOL SCRIPT GENERATION
    # =============================================
    # Generate PyMOL commands for structure visualization
    
    i = 0
    file = open(f"{prot_folder}/{idx}.pml", "w")
    
    # Process each protein chain
    for ID in HL.keys():
        # Set base color for this chain (cycling through color palette)
        file.write(f"color {colors[int(i%len(colors))]}, chain {ID} and elem C\n")
        
        # Create residue selection string for binding sites
        resi = f'{HL[ID][0]}'  # First binding site
        for res in HL[ID][1:]:
            resi += f"+{res}"   # Add additional binding sites
        
        # Create PyMOL selection for binding sites in this chain
        file.write(f"select res{ID}, chain {ID} and resi {resi}\n")
        
        # Show binding sites as stick representation
        file.write(f"show sticks, res{ID}\n")
        
        # =============================================
        # AMINO ACID TYPE COLOR CODING
        # =============================================
        # Color amino acids by their chemical properties
        
        # Hydrophobic amino acids (non-polar)
        file.write(f"color lime, res{ID} and resn ALA+VAL+LEU+ILE+MET+PHE+TRP+PRO+GLY and elem C\n")
        
        # Polar amino acids (uncharged)
        file.write(f"color orange, res{ID} and resn SER+THR+CYS+ASN+GLN+TYR and elem C\n")
        
        # Positively charged amino acids (basic)
        file.write(f"color magenta, res{ID} and resn LYS+ARG+HIS and elem C\n")
        
        # Negatively charged amino acids (acidic)
        file.write(f"color cyan, res{ID} and resn ASP+GLU and elem C\n")
        
        # Add residue labels for identification
        file.write(f"label res{ID}, resn\n")
        i += 1
    
    # Set label color for visibility
    file.write(f"set label_color, yellow")
    file.close()
    
    print(f"PDB sequence saved to {prot_folder}")

# Mapping dictionary for plot property names
# Maps internal property names to more descriptive labels
mp = {'binding sites': 'plus', 'chain numbers': 'chains', 'length': 'len'}

def Plot(real, gen, s, folder, seed):
    """
    Generate comparative plots between training and generated sequence properties.
    
    This function creates statistical comparison plots showing how well the
    generated sequences match the training data distribution for various
    properties like sequence length, functional site count, and segment numbers.
    
    Parameters:
    - real: Training data values
    - gen: Generated data values  
    - s: Property name string ('length', 'binding sites', 'chain numbers')
    - folder: Output directory for plots
    - seed: Random seed for file naming
    """
    # Count occurrences of each property value
    counts_real = Counter(real)
    counts_gen = Counter(gen)
    
    # Initialize list for relative ratio calculations
    relative_ratios = []
    
    # =============================================
    # SEQUENCE LENGTH ANALYSIS
    # =============================================
    # Special handling for sequence length (continuous variable)
    
    if s == 'length':
        # Create bins for length distribution
        # Use 8-amino acid bins for grouping
        bins = [i*8 for i in range(dim_tot//8+1)]
        
        # Initialize bin counts
        counts_real_bin = [0] * (len(bins)-1)
        counts_gen_bin = [0] * (len(bins)-1)
        
        # Distribute counts into bins
        for i in range(len(bins)-1):
            # Count training sequences in this bin
            for k in counts_real.keys():
                if bins[i] < k < bins[i+1]:
                    counts_real_bin[i] += counts_real[k]
            
            # Count generated sequences in this bin
            for k in counts_gen.keys():
                if bins[i] < k < bins[i+1]:
                    counts_gen_bin[i] += counts_gen[k]
        
        # Normalize counts to get frequencies
        sum_real = sum(counts_real_bin)
        sum_gen = sum(counts_gen_bin)
        
        norm_counts_real = [i/sum_real for i in counts_real_bin]
        norm_counts_gen = [i/sum_gen for i in counts_gen_bin]
    
        # Calculate relative ratios for statistical analysis
        for i in range(len(norm_counts_real)):
            real_freq = norm_counts_real[i]
            gen_freq = norm_counts_gen[i]
            
            if real_freq != 0:
                relative_ratios.append(gen_freq/real_freq)
        
        # Calculate statistical measures
        mean_relative_ratio = np.mean(relative_ratios)
        std_relative_ratio = np.std(relative_ratios)
        
        # Calculate bin widths for plotting
        bin_widths = np.diff(bins)
        
        # Create the length distribution plot
        plt.figure(figsize=(4, 4))
        clear_output(wait=True)
        
        # Plot training and generated distributions
        plt.bar(bins[:-1], norm_counts_real, log=True, alpha=1, 
                width=bin_widths, align='edge', label='Training')
        plt.bar(bins[:-1], norm_counts_gen, log=True, alpha=.6, 
                width=bin_widths, align='edge', label='De novo')
        plt.xlabel("Length")
    
    # =============================================
    # DISCRETE PROPERTY ANALYSIS
    # =============================================
    # Handle discrete properties (binding sites, chain numbers)
    
    else:
        # Normalize counts to get frequencies
        sum_real = sum(list(counts_real.values()))
        sum_gen = sum(list(counts_gen.values()))
        
        norm_counts_real = {k: v/sum_real for k, v in counts_real.items()}
        norm_counts_gen = {k: v/sum_gen for k, v in counts_gen.items()}
        
        # Extract keys and values for plotting
        x_real, y_real = list(norm_counts_real.keys()), list(norm_counts_real.values())
        x_gen, y_gen = list(norm_counts_gen.keys()), list(norm_counts_gen.values())
        
        x_real, y_real = np.array(x_real), np.array(y_real)
        x_gen, y_gen = np.array(x_gen), np.array(y_gen)
    
        # Calculate relative ratios for each property value
        for k in x_real:
            real_freq = norm_counts_real[k]
            gen_freq = norm_counts_gen.get(k, 0)  # Default to 0 if not present
            relative_ratios.append(gen_freq/real_freq)
        
        # Calculate statistical measures
        mean_relative_ratio = np.mean(relative_ratios)
        std_relative_ratio = np.std(relative_ratios)
        
        # Create the discrete property plot
        plt.figure(figsize=(4, 4))
        clear_output(wait=True)
        
        # Plot training and generated distributions
        plt.bar(x_real, y_real, log=True, alpha=1, label='Training')
        plt.bar(x_gen, y_gen, log=True, alpha=.6, label='De novo')
        plt.xlabel("Count")
    
    # =============================================
    # PLOT FORMATTING AND SAVING
    # =============================================
    # Format and save the comparison plot
    
    plt.title(f"Distribution of {s}")
    plt.yscale('log')  # Log scale for better visibility of differences
    plt.ylabel("Frequency")
    plt.legend()
    
    # Save the plot with descriptive filename
    plt.savefig(f"{folder}/{mp[s]}-{seed}-{S}.png", dpi=300, bbox_inches='tight')
    
    # =============================================
    # STATISTICAL RESULTS STORAGE
    # =============================================
    # Store statistical comparison results
    
    # Save relative ratio statistics for later analysis
    data_to_store = {'mean': mean_relative_ratio, 'std': std_relative_ratio}
    with open(f'{folder}/RR-{mp[s]}-{seed}-{S}.pkl', 'wb') as F:
        pickle.dump(data_to_store, F)


def clean_sequence_for_esm_scoring(sequence):
    """
    Clean a QOBRA sequence for ESM scoring.
    
    Removes binding site markers (+), chain separators (:), and terminators (X).
    Only keeps valid amino acid characters for ESM compatibility.
    Concatenates all chains into a single sequence for scoring.
    
    This is consistent with clean_sequence_for_esm() in esm_loss.py but returns
    a single concatenated string instead of a list of chains.
    
    Parameters:
    - sequence: QOBRA format sequence (may contain +, :, X markers)
    
    Returns:
    - Clean amino acid sequence string (concatenated chains)
    """
    # Valid amino acid characters (same as esm_loss.py)
    VALID_AA = 'ACDEFGHIKLMNPQRSTVWY'
    
    # Truncate at terminator (consistent with esm_loss.py)
    if 'X' in sequence:
        sequence = sequence[:sequence.index('X')]
    
    # Split by chain separator and clean each chain
    raw_chains = sequence.split(':')
    
    # Clean each chain: keep only valid amino acids
    clean_chains = []
    for chain in raw_chains:
        clean_chain = ''.join(c for c in chain if c in VALID_AA)
        if clean_chain:  # Only keep non-empty chains
            clean_chains.append(clean_chain)
    
    # Concatenate all chains for single-sequence scoring
    return ''.join(clean_chains)


def compute_esm_scores(sequences, scorer=None, model_name="esm2_t6_8M_UR50D", batch_size=8):
    """
    Compute ESM log-likelihood scores for a list of sequences.
    
    Parameters:
    - sequences: List of QOBRA format sequences
    - scorer: Optional pre-initialized ESMScorer (for reuse across calls)
    - model_name: ESM model name (default: esm2_t6_8M_UR50D for speed)
    - batch_size: Batch size for GPU processing
    
    Returns:
    - List of mean log-likelihood scores (one per sequence)
    """
    from esm_scoring import ESMScorer
    
    # Clean sequences for ESM
    cleaned_seqs = [clean_sequence_for_esm_scoring(s) for s in sequences]
    
    # Filter out empty sequences
    valid_indices = [i for i, s in enumerate(cleaned_seqs) if len(s) > 0]
    valid_seqs = [cleaned_seqs[i] for i in valid_indices]
    
    if not valid_seqs:
        return [float('-inf')] * len(sequences)
    
    # Use provided scorer or create new one
    if scorer is None:
        scorer = ESMScorer(model_name=model_name)
    
    # Create sequence IDs and source files for batch processing
    seq_ids = [f"seq_{i}" for i in range(len(valid_seqs))]
    source_files = ["denovo"] * len(valid_seqs)
    
    # Compute scores
    scores = scorer.compute_log_likelihood_batch(
        sequences=valid_seqs,
        sequence_ids=seq_ids,
        source_files=source_files,
        show_progress=True,
        batch_size=batch_size
    )
    
    # Map scores back to original indices
    result_scores = [float('-inf')] * len(sequences)
    for idx, orig_idx in enumerate(valid_indices):
        result_scores[orig_idx] = scores[idx].mean_log_likelihood
    
    return result_scores


def select_best_by_esm(sequences, scorer=None, model_name="esm2_t6_8M_UR50D", batch_size=8):
    """
    Select the sequence with ESM score closest to 0 from a list.
    
    ESM log-likelihood scores are typically negative, with values closer to 0
    indicating higher biological plausibility.
    
    Parameters:
    - sequences: List of QOBRA format sequences
    - scorer: Optional pre-initialized ESMScorer (for reuse across calls)
    - model_name: ESM model name (default: esm2_t6_8M_UR50D for speed)
    - batch_size: Batch size for GPU processing
    
    Returns:
    - Tuple of (best_sequence, best_score, best_index, all_scores)
    """
    if not sequences:
        return None, float('-inf'), -1, []
    
    # Compute ESM scores for all sequences
    scores = compute_esm_scores(sequences, scorer=scorer, model_name=model_name, batch_size=batch_size)
    
    # Find sequence with score closest to 0 (smallest absolute value)
    best_idx = -1
    best_score = float('-inf')
    
    for i, score in enumerate(scores):
        # Score closest to 0 means smallest absolute value
        # Since scores are typically negative, we want the least negative
        if score > best_score:
            best_score = score
            best_idx = i
    
    best_sequence = sequences[best_idx] if best_idx >= 0 else None
    
    return best_sequence, best_score, best_idx, scores


def compile_best_sequences(seed_folders, output_path, model_name="esm2_t6_8M_UR50D", batch_size=32):
    """
    For each seed folder, select the best sequence by ESM score and compile into output file.
    
    GPU-OPTIMIZED: Scores ALL sequences from ALL folders in ONE batched GPU call,
    then maps scores back to folders for selection. This maximizes GPU utilization.
    
    Parameters:
    - seed_folders: List of tuples (seed_number, list_of_sequences)
    - output_path: Path to output file (e.g., "Training data/denovo-qobra-Zn2.txt")
    - model_name: ESM model name for scoring
    - batch_size: Batch size for GPU processing (default: 32 for better GPU utilization)
    
    Returns:
    - List of selected (seed, sequence, score) tuples
    """
    from esm_scoring import ESMScorer
    
    print(f"\n{'='*70}")
    print("SELECTING BEST SEQUENCES BY ESM SCORE (GPU-OPTIMIZED)")
    print(f"{'='*70}")
    print(f"Output file: {output_path}")
    print(f"ESM model: {model_name}")
    print(f"Batch size: {batch_size}")
    
    # =========================================
    # STEP 1: Collect ALL sequences from ALL folders
    # =========================================
    all_sequences = []      # Flat list of all sequences
    sequence_to_folder = [] # Maps each sequence index to (folder_idx, local_idx)
    folder_info = []        # Store (seed, original_sequences) for each folder
    
    for folder_idx, (seed, sequences) in enumerate(seed_folders):
        folder_info.append((seed, sequences))
        for local_idx, seq in enumerate(sequences):
            all_sequences.append(seq)
            sequence_to_folder.append((folder_idx, local_idx))
    
    total_sequences = len(all_sequences)
    print(f"\nTotal sequences to score: {total_sequences} across {len(seed_folders)} folders")
    
    if total_sequences == 0:
        print("Warning: No sequences to process")
        return []
    
    # =========================================
    # STEP 2: Score ALL sequences in ONE batched GPU call
    # =========================================
    print(f"\nInitializing ESM scorer...")
    scorer = ESMScorer(model_name=model_name)
    
    print(f"\nScoring all {total_sequences} sequences in batched GPU call...")
    
    # Clean all sequences
    cleaned_seqs = [clean_sequence_for_esm_scoring(s) for s in all_sequences]
    
    # Filter out empty sequences and track valid indices
    valid_indices = [i for i, s in enumerate(cleaned_seqs) if len(s) > 0]
    valid_seqs = [cleaned_seqs[i] for i in valid_indices]
    
    print(f"  Valid sequences after cleaning: {len(valid_seqs)}")
    
    # Create sequence IDs for batch processing
    seq_ids = [f"seq_{i}" for i in range(len(valid_seqs))]
    source_files = ["denovo"] * len(valid_seqs)
    
    # ONE GPU call for ALL sequences
    scores_result = scorer.compute_log_likelihood_batch(
        sequences=valid_seqs,
        sequence_ids=seq_ids,
        source_files=source_files,
        show_progress=True,
        batch_size=batch_size
    )
    
    # Map scores back to original indices
    all_scores = [float('-inf')] * total_sequences
    for result_idx, orig_idx in enumerate(valid_indices):
        all_scores[orig_idx] = scores_result[result_idx].mean_log_likelihood
    
    # =========================================
    # STEP 3: Select best from each folder (CPU - fast)
    # =========================================
    print(f"\nSelecting best sequence from each folder...")
    
    # Group scores by folder
    folder_scores = [[] for _ in range(len(seed_folders))]
    for seq_idx, (folder_idx, local_idx) in enumerate(sequence_to_folder):
        folder_scores[folder_idx].append((local_idx, all_scores[seq_idx]))
    
    selected_sequences = []
    
    for folder_idx, (seed, sequences) in enumerate(folder_info):
        if not sequences:
            print(f"  [Seed {seed}] Warning: No sequences")
            continue
        
        scores_for_folder = folder_scores[folder_idx]
        
        # Find best (highest score = closest to 0)
        best_local_idx = -1
        best_score = float('-inf')
        
        for local_idx, score in scores_for_folder:
            if score > best_score:
                best_score = score
                best_local_idx = local_idx
        
        if best_local_idx >= 0:
            best_seq = sequences[best_local_idx]
            selected_sequences.append((seed, best_seq, best_score))
            
            # Show score distribution for this folder
            valid_folder_scores = [s for _, s in scores_for_folder if s > float('-inf')]
            if valid_folder_scores:
                print(f"  [Seed {seed}] Best: seq {best_local_idx}, score={best_score:.4f}, "
                      f"range=[{min(valid_folder_scores):.4f}, {max(valid_folder_scores):.4f}]")
        else:
            print(f"  [Seed {seed}] Warning: Could not select sequence")
    
    # Write selected sequences to output file
    print(f"\n[Writing] Compiling {len(selected_sequences)} sequences to {output_path}")
    
    with open(output_path, 'w') as f:
        f.write("De novo sequences (best by ESM score per seed)\n")
        f.write(f"ESM model: {model_name}\n")
        f.write("=" * 70 + "\n\n")
        
        for seed, seq, score in selected_sequences:
            # Truncate at terminator X (not just strip trailing X)
            if 'X' in seq:
                seq_clean = seq[:seq.index('X')]
            else:
                seq_clean = seq
            f.write(f"Seed {seed} | ESM score: {score:.4f}\n")
            f.write(f"{seq_clean}\n")
            f.write("*" * 70 + "\n\n")
    
    print(f"Done! Wrote {len(selected_sequences)} sequences to {output_path}")
    print(f"{'='*70}\n")
    
    return selected_sequences