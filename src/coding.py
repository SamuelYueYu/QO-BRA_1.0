"""
QOBRA - Coding Module

This module handles the encoding and decoding of molecular sequences for quantum processing.
It converts molecular sequences into quantum-compatible representations and vice versa.
Key functionality includes:
- Molecular sequence preprocessing and validation
- Quantum state encoding/decoding
- Functional site analysis
- Sequence similarity calculations

The current implementation works with protein sequences as an example, where functional
sites are marked with '+' and segments are separated by ':'. The framework is designed
to be adaptable to other molecular sequence types.
"""

import numpy as np
import random
from ansatz import *

# Minimum length threshold for protein chain segments
threshold = 4
# Concatenated string of target sequence types for file naming
metals = ''.join(keys_target)

# Standard 20 amino acid codes used in protein sequences (current example)
keys_aa = ['A', 'C', 'D', 'E', 'F', 
           'G', 'H', 'I', 'K', 'L', 
           'M', 'N', 'P', 'Q', 'R', 
           'S', 'T', 'V', 'W', 'Y']

# Create comprehensive token list for sequence encoding
# This includes both regular amino acids and metal-binding amino acids (marked with '+')
cnt_x = []
# Add regular amino acids
for aa in keys_aa:
    cnt_x.append(aa)
# Add metal-binding amino acids (marked with '+')
for aa in keys_aa:
    cnt_x.append(aa + "+")

# Add special sequence markers
cnt_x.append(':')    # Chain separator
cnt_x.append(':X')   # Sequence terminator

def LCS(str1, str2):
    """
    Calculate the Longest Common Subsequence (LCS) between two strings.
    
    This function finds the length of the longest contiguous matching substring
    between two input strings. Used for sequence similarity analysis.
    
    Parameters:
    - str1, str2: Input strings to compare
    
    Returns:
    - Length of the longest common subsequence
    """
    same = False
    s = difflib.SequenceMatcher(None, str1, str2)
    match = s.find_longest_match(0, len(str1), 0, len(str2))
    return match.size

def issame(str1, str2):
    """
    Determine if two protein sequences are significantly similar.
    
    This function uses the LCS algorithm to check if two sequences share
    enough common subsequences to be considered similar. A sequence is
    considered similar if >10% of each sequence matches.
    
    Parameters:
    - str1, str2: Protein sequences to compare
    
    Returns:
    - Boolean indicating if sequences are similar
    """
    same = False
    match = LCS(str1, str2)
    # Consider sequences similar if >10% of each sequence matches
    if match/len(str1) > .1 and match/len(str2) > .1:
        same = True
    return same

# =============================================
# FAST NOVELTY CHECKING WITH K-MER FINGERPRINTS
# =============================================
# These functions enable O(1) pre-filtering before expensive LCS computation

def get_kmers(seq, k=4):
    """
    Extract all k-mers from a sequence.
    
    Parameters:
    - seq: Input sequence string
    - k: Length of k-mers (default 4)
    
    Returns:
    - Set of k-mer strings
    """
    if len(seq) < k:
        return set()
    return {seq[i:i+k] for i in range(len(seq) - k + 1)}

def build_kmer_index(sequences, k=4):
    """
    Build an inverted index from k-mers to sequence indices.
    
    This enables fast candidate retrieval for novelty checking.
    Only sequences sharing k-mers with the query need LCS comparison.
    
    Parameters:
    - sequences: List of training sequences
    - k: K-mer length (default 4)
    
    Returns:
    - kmer_to_seqs: Dict mapping k-mers to sets of sequence indices
    - seq_kmers: List of k-mer sets for each sequence
    """
    kmer_to_seqs = {}
    seq_kmers = []
    
    for idx, seq in enumerate(sequences):
        kmers = get_kmers(seq, k)
        seq_kmers.append(kmers)
        for kmer in kmers:
            if kmer not in kmer_to_seqs:
                kmer_to_seqs[kmer] = set()
            kmer_to_seqs[kmer].add(idx)
    
    return kmer_to_seqs, seq_kmers

def check_novelty_fast(query_seq, training_seqs, kmer_index=None, seq_kmers=None, k=4, min_shared_ratio=0.05):
    """
    Fast novelty check using k-mer filtering + early-termination LCS.
    
    This is much faster than checking LCS against all training sequences:
    1. Extract k-mers from query
    2. Find candidate sequences that share k-mers (using inverted index)
    3. Only run LCS on candidates with significant k-mer overlap
    4. Early termination: return False as soon as a match is found
    
    Parameters:
    - query_seq: The generated sequence to check
    - training_seqs: List of training sequences
    - kmer_index: Precomputed k-mer to sequence index mapping
    - seq_kmers: Precomputed k-mer sets for training sequences
    - k: K-mer length
    - min_shared_ratio: Minimum k-mer overlap ratio to consider as candidate
    
    Returns:
    - Boolean indicating if sequence is novel (True = novel, False = similar to training)
    """
    if len(query_seq) < k:
        return True  # Very short sequences are considered novel
    
    query_kmers = get_kmers(query_seq, k)
    
    if kmer_index is None or seq_kmers is None:
        # Fallback to slow method if index not provided
        return not any(issame(seq, query_seq) for seq in training_seqs)
    
    # Find candidate sequences that share k-mers
    candidate_indices = set()
    for kmer in query_kmers:
        if kmer in kmer_index:
            candidate_indices.update(kmer_index[kmer])
    
    # Score candidates by k-mer overlap
    candidates_with_scores = []
    for idx in candidate_indices:
        shared = len(query_kmers & seq_kmers[idx])
        # Quick rejection: if too few shared k-mers, skip LCS
        if shared / max(len(query_kmers), len(seq_kmers[idx])) >= min_shared_ratio:
            candidates_with_scores.append((idx, shared))
    
    # Sort by overlap (most overlapping first for faster early termination)
    candidates_with_scores.sort(key=lambda x: -x[1])
    
    # Check candidates with LCS (early termination)
    for idx, _ in candidates_with_scores:
        if issame(training_seqs[idx], query_seq):
            return False  # Found a match - not novel
    
    return True  # No matches found - novel sequence

def prep(metal):
    """
    Prepare and filter molecular sequences for a specific sequence type.
    
    This function processes molecular sequence data files for a given type,
    removing duplicate and highly similar sequences to create a clean dataset.
    It also maintains a mapping between sequences and their identifiers.
    
    Parameters:
    - metal: Sequence type string (e.g., 'Ca', 'Mg', 'Zn' for metal-binding proteins)
    
    Returns:
    - ret: List of unique molecular sequences
    - PDBcodes: Dictionary mapping sequences to their identifiers
    """
    ret, temp, PDBcodes = [], [], {}
    folder_name = f"{metal}_bind"
    
    # Read all protein sequence files for this metal type
    for filename in os.listdir(folder_name):
        if filename[0] != '.':  # Skip hidden files
            f = f"{folder_name}/{filename}"
            
            with open(f,'r') as file:
                # Read the contents line by line
                lines = file.readlines()
            
            # Concatenate all lines into a single sequence string
            s = ''
            for i in range(len(lines)):
                if len(lines[i]) > 0:
                    s += lines[i]
            temp.append(s)
            
            # Store PDB code (first 4 characters of filename)
            PDBcodes[s] = filename[:4]
    
    # Remove highly similar sequences to avoid redundancy
    i = 0
    while i < len(temp)-1:
        lcs = LCS(temp[i], temp[i+1])  # Calculate similarity
        if len(temp[i]) == 0 or len(temp[i+1]) == 0:
            i += 1
        elif lcs/len(temp[i]) < .1 and lcs/len(temp[i+1]) < .1:
            # Sequences are sufficiently different - keep both
            ret.append(temp[i])
            i += 1
        else:
            # Sequences are too similar - keep only the shorter one
            if len(temp[i+1]) < len(temp[i]): 
                del temp[i]
            else: 
                del temp[i+1]
    return ret, PDBcodes

def crop(sequence, max_len):
    """
    Crop and pad protein sequences to a standardized length.
    
    This function intelligently crops protein sequences to fit within the
    maximum length constraint while preserving important binding sites.
    It prioritizes keeping metal-binding regions (marked with '+') and
    ensures minimum chain lengths.
    
    Parameters:
    - sequence: Input protein sequence string
    - max_len: Maximum allowed sequence length
    
    Returns:
    - Cropped sequence string with terminator 'X'
    """
    random.seed(1)  # Ensure reproducible cropping
    seq = sequence + "X"  # Add terminator
    
    # Count amino acids (excluding special characters)
    t = len(''.join([c for c in sequence if (c != "+" and c != "X")]))
    
    # If sequence is already short enough, pad it by repeating
    if t < max_len:
        s = seq
        t = len(''.join([c for c in s if (c != "+" and c != "X")]))
        
        # Repeat sequence until reaching maximum length
        while t < max_len:
            s += seq[:min(len(seq), max_len-t)]
            t = len(''.join([c for c in s if (c != "+" and c != "X")]))
        return s
    
    # For longer sequences, intelligently crop around binding sites
    # Find binding sites (+) and chain boundaries (:)
    idx = seq.find("+")      # First binding site
    idx_n = seq.find(":")    # First chain boundary
    
    # Generate random offsets for cropping window
    d1 = random.randint(threshold, dim_tot//2)
    d2 = random.randint(threshold, dim_tot//2)
    
    # Adjust cropping window based on binding sites and chain boundaries
    if idx_n > idx:
        idx = max(0, idx-random.randint(1,9))
    else:
        idx = max(0, idx-d1)
        idx = min(idx, idx_n-threshold)
        if seq[idx] == "+":
            idx -= 1
        idx_n = seq.find(":", idx_n+1)
    
    # Ensure proper sequence boundaries
    idx_n = min(idx_n, len(seq)-2) + 1
    if idx_n == len(seq)-1:
        idx_n += 1
    
    # Extract initial sequence segment
    s = seq[max(0, idx):idx_n]
    t = len(''.join([c for c in s if (c != "+" and c != "X")]))
    
    # Extend sequence if too short
    if t < max_len:
        # Add amino acids from the left
        i = max(0, idx)-1
        while t < max_len-d2 and i > -1:
            if seq[i] != "+":
                s = seq[i] + s
                i -= 1
            else:
                s = seq[i-1:i+1] + s  # Include binding site marker
                i -= 2
            t = len(''.join([c for c in s if (c != "+" and c != "X")]))
        
        # Add amino acids from the right
        j = idx_n
        while t < max_len and j < len(seq):
            if seq[j+1] != "+" and seq[j+1] != "X":
                s = s + seq[j]
                j += 1
            else:
                s = s + seq[j:j+2]  # Include binding site marker
                j += 2
            t = len(''.join([c for c in s if (c != "+" and c != "X")]))
            
        # Final extension from the left if still needed
        while t < max_len and i > -1:
            if seq[i] != "+":
                s = seq[i] + s
                i -= 1
            else:
                s = seq[i-1:i+1] + s
                i -= 2
            t = len(''.join([c for c in s if (c != "+" and c != "X")]))
        return s
    
    # If sequence is still too long, truncate while preserving structure
    n, i = len(s), 0
    for j in range(max_len):
        if i >= n: break
        elif i < n-1 and (s[i+1] == "+" or s[i+1] == "X"):
            i += 2  # Skip binding site markers
        else:
            i += 1
    
    t = len(''.join([c for c in s[:i] if (c != "+" and c != "X")]))
    return s[:i]
  
def amps(sequence, ctf, max_len):
    """
    Convert amino acid sequence to probability amplitudes.
    
    This function transforms a protein sequence into a list of probability
    amplitudes using a character-to-frequency mapping. Each amino acid
    (and binding site marker) is mapped to a specific amplitude value.
    
    Parameters:
    - sequence: Input protein sequence
    - ctf: Character-to-frequency mapping dictionary
    - max_len: Maximum sequence length
    
    Returns:
    - List of probability amplitudes corresponding to the sequence
    """
    prob_vector, n, i = [], len(sequence), 0
    
    # Process sequence character by character
    while i < n:
        if i < n-1 and (sequence[i+1] == "+" or sequence[i+1] == "X"):
            # Handle binding site markers or terminators
            prob_vector.append(ctf[sequence[i:i+2]])
            i += 2
        else:
            # Handle regular amino acids
            prob_vector.append(ctf[sequence[i]])
            i += 1
    return prob_vector

def encode_amino_acid_sequence(sequence, ctf, head, max_len, vec_len):
    """
    Encode a protein sequence into a normalized quantum state vector.
    
    This function converts a protein sequence into a quantum state representation
    suitable for quantum circuit processing. The sequence is first converted to
    amplitudes, then normalized to create a valid quantum state.
    
    Parameters:
    - sequence: Input protein sequence
    - ctf: Character-to-frequency mapping
    - head: Normalization constant for the first amplitude
    - max_len: Maximum sequence length
    - vec_len: Length of output vector
    
    Returns:
    - Normalized quantum state vector
    """
    # Convert sequence to amplitude values
    state_vector = amps(sequence, ctf, max_len)
    s = len(state_vector)
    
    # Add head amplitude and normalize
    state_vector = [head] + state_vector
    state_vector = np.array(state_vector)
    state_vector /= np.linalg.norm(state_vector)  # Normalize to unit vector
    return state_vector

def decode_amino_acid_sequence(code, ftc, head):
    """
    Decode a quantum state vector back into a protein sequence.
    
    This function reverses the encoding process, converting quantum state
    amplitudes back into amino acid sequences. It matches each amplitude
    to the closest known amino acid encoding.
    
    Parameters:
    - code: Quantum state amplitude vector
    - ftc: Frequency-to-character mapping dictionary
    - head: Reference amplitude for normalization
    
    Returns:
    - Decoded protein sequence string
    """
    ret = ""
    
    # Calculate scaling factor from the first amplitude
    factor = head/code[0].real
    
    # Decode each amplitude to the corresponding amino acid
    for i in range(1, dim_tot):
        diff = 10**6  # Large initial difference
        key = code[i].real * factor
        c = key
        
        # Find the closest matching amplitude in the dictionary
        for k in ftc.keys():
            if abs(k - c) < diff:
                diff = abs(k - c)
                key = k
        
        # Return empty string if no valid match found
        if key not in ftc.keys():
            return ""
        ret += ftc[key]
    return ret

def seq_match_ratio(s1, s2):
    """
    Calculate the similarity ratio between two protein sequences.
    
    This function computes the fraction of matching characters between
    two sequences, providing a measure of sequence similarity.
    
    Parameters:
    - s1, s2: Protein sequences to compare
    
    Returns:
    - Average similarity ratio between the sequences
    """
    score, i = 0, 0
    
    # Count matching characters position by position
    for i in range(min(len(s1), len(s2))):
        if s1[i] == s2[i]:
            score += 1
    
    # Return average of both sequence similarity ratios
    return np.mean(np.array([score/len(s1), score/len(s2)]))