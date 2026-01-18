"""
QOBRA - Inputs Module

This module handles the preprocessing and preparation of molecular sequence data for training.
It processes raw molecular sequence data, creates training/test splits, and prepares the data
structures needed for quantum sequence encoding. The current implementation demonstrates
on protein sequences but is designed for general molecular applications.

Key functionality:
- Loading and preprocessing molecular sequence data
- Creating training/test splits
- Token frequency analysis and encoding preparation
- Data augmentation and sequence filtering
- Statistical analysis of sequence properties
"""

# Set matplotlib backend BEFORE importing pyplot
# 'Agg' is a non-GUI backend that works in threads and headless environments
import matplotlib
matplotlib.use('Agg')

from model import *
from collections import Counter
from scipy.special import gamma

import matplotlib.pyplot as plt
from IPython.display import clear_output

from qiskit.quantum_info import Statevector
from qiskit_algorithms.utils import algorithm_globals

# Training mode switch is now parsed via argparse in ansatz.py
# 0 = training mode, 1 = inference mode
# (switch is imported from ansatz via model.py)

# Print ESM configuration if enabled
if esm_lambda_max > 0:
    warmup_str = f", warmup={esm_warmup}" if esm_warmup is not None else ""
    print(f"ESM loss enabled: lambda_max={esm_lambda_max}, K={esm_K}, model={esm_model_name}{warmup_str}")

# Create descriptive filename including all parser variables for experiment tracking
S = (f"{metals}_q{num_tot}_r{r}"
     f"_mmd{lambda_mmd_max}_fid{lambda_fidelity_max}_esm{esm_lambda_max}"
     f"_K{esm_K}_sub{esm_subset_size}_{esm_model_name}"
     f"_warm{esm_warmup}_step{esm_step_interval}")

# Create output directory for this experiment
if not os.path.exists(f"{S}"):
    os.makedirs(f"{S}")

# Note: 'metals' is now defined in ansatz.py as "".join(keys_target)
# Draw the circuit with the custom font sizes
circuit_drawer(qc_e, output='mpl', style="clifford", filename=f'{S}/qc_e.png')
circuit_drawer(qc_ed, output='mpl', style="clifford", filename=f'{S}/qc_ed.png')
circuit_drawer(qc_d, output='mpl', style="clifford", filename=f'{S}/qc_d.png')

def cnts(seqs, folder='', seed=0, is_denovo=True):
    """
    Analyze and count various properties of molecular sequences.
    
    This function performs comprehensive statistical analysis of molecular sequences,
    counting token frequencies, segment lengths, functional sites, and other
    structural properties. Results are used for training validation and generation quality assessment.
    
    Parameters:
    - seqs: List of molecular sequences to analyze
    - folder: Output directory for saving results
    - seed: Random seed for reproducibility
    - is_denovo: Boolean flag indicating if these are generated sequences
    
    Returns:
    - cnt: Dictionary of token frequencies
    - dn_cnts: List of segment counts per sequence
    - len_cnts: List of sequence lengths
    - plus_cnts: List of functional site counts per sequence
    """
    cnt, dn_cnts = {}, []
    len_cnts, plus_cnts = [], []
    
    # Process each sequence individually
    for s in seqs:
        # Find sequence terminator 'X' and truncate if present
        idx = s.find("X")
        s1 = s[:idx+1] if idx != -1 else s
        
        # Count sequence tokens (excluding special characters)
        len_s1 = len(''.join([c for c in s1 if c != "+"]))
        
        # Save generated sequences to file for later analysis
        if is_denovo:
            file = open(f"{folder}/denovo-{seed}.txt", "a")
            file.write(f"{s1[:-1]}\n")  # Remove terminator for output
            file.write("*" * dim_tot + '\n')  # Add separator
            file.close()
        
        # Count special characters in sequence
        char_cnts = Counter(s1)
        len_cnts.append(len_s1)           # Total token count
        dn_cnts.append(char_cnts[':'])    # Number of segments
        plus_cnts.append(char_cnts['+'])  # Number of functional sites
        
        # Count individual tokens and functional sites
        single_cnt = {}
        for i in range(len(s1)):
            if i < len(s1)-1 and (s1[i+1] == "+" or s1[i+1] == "X"):
                # Handle functional sites and terminators
                single_cnt[s1[i:i+2]] = single_cnt.get(s1[i:i+2], 0) + 1
            elif s1[i] != "+" and s1[i] != "X":
                # Handle regular tokens
                single_cnt[s1[i]] = single_cnt.get(s1[i], 0) + 1
        
        # Normalize counts by sequence length
        for k in single_cnt.keys():
            cnt[k] = cnt.get(k, 0) + single_cnt[k]/len_s1
    
    # Average counts across all sequences
    for k in cnt.keys():
        cnt[k] /= len(seqs)
    
    return cnt, dn_cnts, len_cnts, plus_cnts

# Create metal-specific sequence dictionary
# This will map sequences to their corresponding metal types
metal_dct = {}  # For future multi-ion support
for k in keys_target:
    lst, PDBcodes = prep(k)  # Get sequences for this metal type
    for i in lst:
        if "++" not in i:  # Skip sequences with consecutive binding sites
            l = metal_dct.get(i, [])
            l.append(k)
            metal_dct[i] = l

# Set maximum sequence length and processing parameters
max_len = dim_tot - 1  # Leave space for terminator
cap, processes = 6000, 6  # Maximum training sequences and parallel processes

# Create training and test sets from the sequence data
seqs = list(metal_dct.keys())
all_input_seqs, all_test_seqs = [], []

# Split sequences into training (80%) and test (20%) sets
#random.shuffle(seqs)  # Commented out for reproducibility
for i in range(len(seqs)):
    s = seqs[i].replace("\n", ":")  # Normalize line endings
    if i % 5 > 0:
        all_input_seqs.append(s)  # 80% for training
    else:
        all_test_seqs.append(s)   # 20% for testing

# Filter and prepare final training and test sets
train_seqs, test_seqs = [], []
F = open(f"{S}/PDBcodes-{S}.txt", "w")

# Process training sequences
for s in all_input_seqs:
    seg = crop(s, max_len)  # Crop to maximum length
    char_cnts = Counter(seg)
    
    # Only include sequences with binding sites and within capacity limit
    if 0 < char_cnts['+'] and len(train_seqs) < cap:
        train_seqs.append(seg)
        code = PDBcodes[s.replace(':', '\n')]
        
        # Save PDB code and sequence information
        F.write(f"{code}\n")
        if seg.find('X') > -1:
            F.write(f"{seg[:seg.find('X') + 1]}\n")
        else: 
            F.write(f"{seg}\n")
        F.write("-" * max_len + '\n')
F.close()

# Set test set size proportional to training set
train_size = len(train_seqs)
test_size = train_size // 5

# Process test sequences
for s in all_test_seqs:
    seg = crop(s, max_len)
    char_cnts = Counter(seg)
    if 0 < char_cnts['+'] and len(test_seqs) < test_size:
        test_seqs.append(seg)

# Analyze training data for token frequency distribution
cnt, dn_cnts_real, len_cnts_real, plus_cnts_real = cnts(train_seqs, is_denovo=False)
max_dn = np.max(dn_cnts_real)  # Maximum number of chains in dataset

# Add small random noise to prevent identical frequencies
# This helps with numerical stability in quantum encoding
srtd = sorted(list(cnt.values()), reverse=True)
for k in cnt.keys():
    cnt[k] += random.random() * srtd[-1] * 1e-3

# Re-sort after adding noise
srtd = sorted(list(cnt.values()), reverse=True)

# Create reverse mapping from frequencies to tokens
rev_cnt = {cnt[k]: k for k in cnt.keys()}

# Create Gaussian-distributed frequency ordering
# This special ordering optimizes quantum state preparation
gsrtd = [srtd[0]]  # Start with highest frequency
for i in range(1, len(srtd)):
    if i % 2 == 0:
        gsrtd.append(srtd[i])      # Add to end
    else:
        gsrtd = [srtd[i]] + gsrtd  # Add to beginning

# Create mapping dictionaries for different orderings
dec_sort = {rev_cnt[k]: k for k in srtd}      # Decreasing order
gauss_sort = {rev_cnt[k]: k for k in gsrtd}   # Gaussian order

# Extract token list in Gaussian order (optimal for quantum encoding)
ks = list(gauss_sort.keys())
Len = len(ks)

# Initialize encoding parameters
xt = np.array([1. for i in range(Len-1)])  # Uniform spacing initially
cnt_y1 = [cnt[x] if x in cnt.keys() else 0 for x in cnt_x]  # Frequency vector