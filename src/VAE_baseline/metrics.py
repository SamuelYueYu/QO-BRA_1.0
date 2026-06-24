"""
PyTorch Lightning Bolts VAE - Metrics Module

Metrics for evaluating the VAE model:
- N (Novelty): fraction of generated sequences not similar to training
- U (Uniqueness): fraction of unique generated sequences
- V (Validity): fraction passing biological validity checks
- R (Reconstruction rate): fraction of perfectly reconstructed sequences
- RR (Relative Ratio): distribution similarity between generated and training
"""

import re
import difflib
import numpy as np
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


# =============================================================================
# SEQUENCE SIMILARITY (for Novelty computation)
# =============================================================================

def longest_common_substring(str1: str, str2: str) -> int:
    """
    Calculate the Longest Common Substring (LCS) length between two strings.
    """
    if not str1 or not str2:
        return 0
    s = difflib.SequenceMatcher(None, str1, str2)
    match = s.find_longest_match(0, len(str1), 0, len(str2))
    return match.size


def is_similar(str1: str, str2: str, threshold: float = 0.1) -> bool:
    """
    Determine if two sequences are significantly similar.
    
    Two sequences are considered similar if the LCS length is greater than
    threshold fraction of BOTH sequence lengths.
    """
    if not str1 or not str2:
        return False
    
    lcs_len = longest_common_substring(str1, str2)
    return (lcs_len / len(str1) > threshold) and (lcs_len / len(str2) > threshold)


# =============================================================================
# N, U, V METRICS
# =============================================================================

PATTERN_3_CONSECUTIVE = r".\+.\+.\+"        # 3 consecutive binding sites
PATTERN_5_CONSECUTIVE = r".\+.\+.\+.\+.\+"  # 5 consecutive binding sites
MIN_CHAIN_LENGTH = 4


def _quick_similar_check(str1: str, str2: str, threshold: float = 0.1) -> bool:
    """
    Quick check if two sequences could possibly be similar.
    Returns False if definitely not similar (can skip expensive LCS).
    Returns True if might be similar (need full LCS check).
    """
    len1, len2 = len(str1), len(str2)
    if len1 == 0 or len2 == 0:
        return False
    
    # If lengths differ by more than 10x, LCS can't meet threshold
    if len1 > len2 * 10 or len2 > len1 * 10:
        return False
    
    # Quick character set overlap check
    # If they share very few characters, likely not similar
    set1 = set(str1)
    set2 = set(str2)
    overlap = len(set1 & set2)
    if overlap < 5:  # Very few common characters
        return False
    
    return True  # Might be similar, need full LCS


def _check_novelty_single(args):
    """
    Helper function for parallel novelty computation.
    Checks if a single generated sequence is novel (not similar to any training sequence).
    """
    gen_seq, training_seqs_cleaned, training_set, similarity_threshold = args
    
    # Clean generated sequence: truncate at X terminator (QOBRA style)
    term_idx = gen_seq.find('X')
    gen_clean = gen_seq[:term_idx] if term_idx != -1 else gen_seq
    
    # Quick exact match check (O(1))
    if gen_clean in training_set:
        return 0  # Not novel (exact match)
    
    # Check against all training sequences for similarity
    for train_clean in training_seqs_cleaned:
        # Quick pre-filter to avoid expensive LCS
        if not _quick_similar_check(gen_clean, train_clean, similarity_threshold):
            continue
        
        if is_similar(gen_clean, train_clean, similarity_threshold):
            return 0  # Not novel
    
    return 1  # Novel


def compute_novelty(generated_seqs: list, training_seqs: list,
                    similarity_threshold: float = 0.1, n_workers: int = None) -> float:
    """
    Compute novelty rate (N): fraction of generated sequences not similar to training.
    
    A sequence is novel if it is NOT similar to ANY training sequence.
    Similarity is determined by LCS: similar if >10% of both sequences match (QOBRA style).
    
    Uses multiprocessing for acceleration.
    """
    if not generated_seqs:
        return 0.0
    
    # Pre-clean training sequences once (avoid repeated work)
    training_seqs_cleaned = []
    for train_seq in training_seqs:
        term_idx = train_seq.find('X')
        training_seqs_cleaned.append(train_seq[:term_idx] if term_idx != -1 else train_seq)
    
    # Build a set of training sequences for O(1) exact match lookup
    training_set = set(training_seqs_cleaned)
    
    # Determine number of workers
    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), len(generated_seqs))
    
    # Prepare arguments for parallel processing
    args_list = [(gen_seq, training_seqs_cleaned, training_set, similarity_threshold) 
                 for gen_seq in generated_seqs]
    
    # Use multiprocessing for parallel computation
    if n_workers > 1 and len(generated_seqs) > 10:
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(_check_novelty_single, args_list))
            novel_count = sum(results)
        except Exception:
            # Fallback to sequential if multiprocessing fails
            novel_count = sum(_check_novelty_single(args) for args in args_list)
    else:
        # Sequential for small batches (avoid overhead)
        novel_count = sum(_check_novelty_single(args) for args in args_list)
    
    return novel_count / len(generated_seqs)


def is_valid_sequence(seq: str, min_chain_length: int = MIN_CHAIN_LENGTH) -> bool:
    """
    Check if a single sequence passes biological validity checks.
    
    Returns True if sequence is valid, False otherwise.
    """
    # Clean sequence: truncate at X terminator (QOBRA style)
    term_idx = seq.find('X')
    s = seq[:term_idx] if term_idx != -1 else seq
    
    # Check 1: Must have binding sites
    if '+' not in s:
        return False
    
    # Check 2: No consecutive chain separators
    if '::' in s:
        return False
    
    # Check 3: No 5+ consecutive binding sites
    match5 = re.findall(PATTERN_5_CONSECUTIVE, s)
    if len(match5) > 0:
        return False
    
    # Check 4: Less than 2 occurrences of 3 consecutive binding sites
    match3 = re.findall(PATTERN_3_CONSECUTIVE, s)
    if len(match3) >= 2:
        return False
    
    # Check 5: All chains meet minimum length
    s_no_plus = ''.join([c for c in s if c != '+'])
    chains = s_no_plus.split(':')
    chains = [c for c in chains if c]
    
    if ':' not in s_no_plus:
        return len(s_no_plus) >= min_chain_length
    else:
        return all(len(c) >= min_chain_length for c in chains)


def compute_uniqueness(generated_seqs: list, valid_seqs: list = None) -> float:
    """
    Compute uniqueness rate (U): QOBRA-style uniqueness among valid sequences.
    
    QOBRA checks uniqueness against the 'valid' list (sequences that are novel+unique+valid).
    If valid_seqs is not provided, falls back to checking uniqueness among all generated.
    """
    if not generated_seqs:
        return 0.0
    
    # If no valid_seqs provided, use all generated (fallback behavior)
    seqs_to_check = valid_seqs if valid_seqs is not None else generated_seqs
    
    if not seqs_to_check:
        return 0.0
    
    # Clean sequences: truncate at X terminator (QOBRA style)
    cleaned = []
    for seq in seqs_to_check:
        term_idx = seq.find('X')
        cleaned.append(seq[:term_idx] if term_idx != -1 else seq)
    
    unique_seqs = set(cleaned)
    return len(unique_seqs) / len(generated_seqs)


def compute_validity(generated_seqs: list, min_chain_length: int = MIN_CHAIN_LENGTH) -> float:
    """
    Compute validity rate (V): fraction passing biological validity checks.
    
    Validity criteria (QOBRA style):
    1. Must have binding sites ('+' present)
    2. No consecutive chain separators ('::')
    3. No 5+ consecutive binding sites
    4. Less than 2 occurrences of 3 consecutive binding sites
    5. All chains meet minimum length requirement (threshold=4)
    """
    if not generated_seqs:
        return 0.0
    
    valid_count = sum(1 for seq in generated_seqs 
                      if is_valid_sequence(seq, min_chain_length))
    
    return valid_count / len(generated_seqs)


def get_valid_sequences(generated_seqs: list, min_chain_length: int = MIN_CHAIN_LENGTH) -> list:
    """
    Get list of valid sequences from generated sequences.
    """
    return [seq for seq in generated_seqs if is_valid_sequence(seq, min_chain_length)]


def compute_nuv(generated_seqs: list, training_seqs: list,
                min_chain_length: int = MIN_CHAIN_LENGTH,
                similarity_threshold: float = 0.1) -> dict:
    """
    Compute all NUV metrics: Novelty, Uniqueness, Validity rates (QOBRA style).
    
    - N: Fraction of sequences not similar to any training sequence
    - U: Fraction of unique sequences among valid sequences (QOBRA style)
    - V: Fraction of sequences passing biological validity checks
    """
    # Get valid sequences for QOBRA-style uniqueness calculation
    valid_seqs = get_valid_sequences(generated_seqs, min_chain_length)
    
    return {
        'N': compute_novelty(generated_seqs, training_seqs, similarity_threshold),
        'U': compute_uniqueness(generated_seqs, valid_seqs),
        'V': compute_validity(generated_seqs, min_chain_length),
    }


# =============================================================================
# R (RECONSTRUCTION RATE) METRIC
# =============================================================================

def compute_reconstruction_rate(original_seqs: list, reconstructed_seqs: list,
                                 tolerance: float = 0.0) -> float:
    """
    Compute reconstruction rate (R): fraction of perfectly reconstructed sequences.
    """
    if not original_seqs or len(original_seqs) != len(reconstructed_seqs):
        return 0.0
    
    perfect_count = 0
    
    for orig, recon in zip(original_seqs, reconstructed_seqs):
        if tolerance == 0.0:
            if orig == recon:
                perfect_count += 1
        else:
            min_len = min(len(orig), len(recon))
            if min_len == 0:
                continue
            
            mismatches = sum(1 for i in range(min_len) if orig[i] != recon[i])
            mismatches += abs(len(orig) - len(recon))
            
            if mismatches / len(orig) <= tolerance:
                perfect_count += 1
    
    return perfect_count / len(original_seqs)


def compute_token_accuracy(original_tokens: np.ndarray,
                           reconstructed_tokens: np.ndarray,
                           token_to_value: dict,
                           value_to_token: dict) -> float:
    """
    Compute token-level accuracy after decoding to discrete tokens.
    """
    token_values = np.array(list(token_to_value.values()))
    
    total_correct = 0
    total_tokens = 0
    
    for i in range(original_tokens.shape[0]):
        for j in range(original_tokens.shape[1]):
            orig_val = original_tokens[i, j]
            recon_val = reconstructed_tokens[i, j]
            
            if orig_val < -0.5:  # Skip padding
                break
            
            orig_val_clamped = max(0.0, min(1.0, orig_val))
            recon_val_clamped = max(0.0, min(1.0, recon_val))
            
            orig_idx = np.argmin(np.abs(token_values - orig_val_clamped))
            recon_idx = np.argmin(np.abs(token_values - recon_val_clamped))
            
            if orig_idx == recon_idx:
                total_correct += 1
            total_tokens += 1
    
    return total_correct / total_tokens if total_tokens > 0 else 0.0


# =============================================================================
# RR (RELATIVE RATIO) METRICS
# =============================================================================

def extract_tokens(sequence: str) -> dict:
    """
    Extract and count tokens from a sequence (QOBRA-style).
    
    QOBRA logic (from cnts function):
    - If next char is '+' or 'X', count the pair (e.g., A+, :X)
    - Otherwise, if current char is not '+' or 'X', count single char
    - Use while loop to properly skip chars after pairing
    
    Returns a dictionary of token counts for this sequence.
    """
    # Find terminator X and truncate (include X)
    idx = sequence.find('X')
    s1 = sequence[:idx+1] if idx != -1 else sequence
    
    single_cnt = {}
    i = 0
    while i < len(s1):
        if i < len(s1) - 1 and (s1[i+1] == '+' or s1[i+1] == 'X'):
            # Handle functional sites (A+) and terminators (:X)
            token = s1[i:i+2]
            single_cnt[token] = single_cnt.get(token, 0) + 1
            i += 2  # Skip both characters
        elif s1[i] != '+' and s1[i] != 'X':
            # Handle regular tokens (single char)
            single_cnt[s1[i]] = single_cnt.get(s1[i], 0) + 1
            i += 1
        else:
            # Skip orphan '+' or 'X' (shouldn't happen in valid sequences)
            i += 1
    
    return single_cnt


def compute_aa_frequencies(sequences: list) -> dict:
    """
    Compute amino acid/token frequency distribution (QOBRA-style).
    
    QOBRA approach:
    1. For each sequence, count tokens and normalize by sequence length
    2. Average normalized frequencies across all sequences
    """
    if not sequences:
        return {}
    
    cnt = {}
    
    for seq in sequences:
        # Find terminator X and truncate
        idx = seq.find('X')
        s1 = seq[:idx+1] if idx != -1 else seq
        
        # Compute sequence length (excluding '+')
        len_s1 = len(''.join([c for c in s1 if c != '+']))
        if len_s1 == 0:
            continue
        
        # Get token counts for this sequence
        single_cnt = extract_tokens(seq)
        
        # Normalize counts by sequence length and accumulate
        for k in single_cnt.keys():
            cnt[k] = cnt.get(k, 0) + single_cnt[k] / len_s1
    
    # Average across all sequences
    for k in cnt.keys():
        cnt[k] /= len(sequences)
    
    return cnt


def compute_sequence_lengths(sequences: list) -> list:
    """
    Compute sequence lengths (QOBRA-style: exclude only '+').
    
    Parameters:
    - sequences: List of sequence strings
    
    Returns:
    - lengths: List of sequence lengths
    """
    lengths = []
    for seq in sequences:
        # Find terminator X (QOBRA uses X, not :X)
        idx = seq.find('X')
        s = seq[:idx+1] if idx != -1 else seq
        
        # Count tokens excluding only '+' (QOBRA style)
        length = len(''.join([c for c in s if c != '+']))
        lengths.append(length)
    return lengths


def compute_binding_site_counts(sequences: list) -> list:
    """Count binding sites per sequence (QOBRA-style: plus_cnts)."""
    counts = []
    for seq in sequences:
        # Find terminator X (QOBRA uses X, not :X)
        idx = seq.find('X')
        s = seq[:idx+1] if idx != -1 else seq
        counts.append(Counter(s)['+'])
    return counts


def compute_chain_counts(sequences: list) -> list:
    """
    Count chain separators per sequence (QOBRA-style: count ':' directly).
    
    QOBRA uses: dn_cnts.append(char_cnts[':'])
    """
    counts = []
    for seq in sequences:
        # Find terminator X (QOBRA uses X, not :X)
        idx = seq.find('X')
        s = seq[:idx+1] if idx != -1 else seq
        
        # Count ':' characters (QOBRA style - counts separators, not chains)
        counts.append(Counter(s)[':'])
    return counts


def compute_relative_ratio_discrete(train_values: list, gen_values: list) -> dict:
    """Compute relative ratio for discrete property values."""
    train_counts = Counter(train_values)
    gen_counts = Counter(gen_values)
    
    train_total = sum(train_counts.values())
    gen_total = sum(gen_counts.values())
    
    if train_total == 0 or gen_total == 0:
        return {'mean': 0.0, 'std': 0.0}
    
    train_freq = {k: v / train_total for k, v in train_counts.items()}
    gen_freq = {k: v / gen_total for k, v in gen_counts.items()}
    
    relative_ratios = []
    for k in train_freq:
        if train_freq[k] > 0:
            gen_f = gen_freq.get(k, 0)
            relative_ratios.append(gen_f / train_freq[k])
    
    if not relative_ratios:
        return {'mean': 0.0, 'std': 0.0}
    
    return {
        'mean': float(np.mean(relative_ratios)),
        'std': float(np.std(relative_ratios)),
    }


def compute_relative_ratio_length(train_lengths: list, gen_lengths: list,
                                   bin_size: int = 8, max_len: int = 256) -> dict:
    """Compute relative ratio for sequence lengths using binning."""
    bins = list(range(0, max_len + bin_size, bin_size))
    
    train_binned = [0] * (len(bins) - 1)
    gen_binned = [0] * (len(bins) - 1)
    
    for length in train_lengths:
        for i in range(len(bins) - 1):
            if bins[i] <= length < bins[i + 1]:
                train_binned[i] += 1
                break
    
    for length in gen_lengths:
        for i in range(len(bins) - 1):
            if bins[i] <= length < bins[i + 1]:
                gen_binned[i] += 1
                break
    
    train_total = sum(train_binned)
    gen_total = sum(gen_binned)
    
    if train_total == 0 or gen_total == 0:
        return {'mean': 0.0, 'std': 0.0}
    
    train_freq = [c / train_total for c in train_binned]
    gen_freq = [c / gen_total for c in gen_binned]
    
    relative_ratios = []
    for i in range(len(train_freq)):
        if train_freq[i] > 0:
            relative_ratios.append(gen_freq[i] / train_freq[i])
    
    if not relative_ratios:
        return {'mean': 0.0, 'std': 0.0}
    
    return {
        'mean': float(np.mean(relative_ratios)),
        'std': float(np.std(relative_ratios)),
    }


def compute_relative_ratio_aa(train_seqs: list, gen_seqs: list) -> dict:
    """Compute relative ratio for amino acid frequencies."""
    train_freq = compute_aa_frequencies(train_seqs)
    gen_freq = compute_aa_frequencies(gen_seqs)
    
    if not train_freq or not gen_freq:
        return {'mean': 0.0, 'std': 0.0}
    
    relative_ratios = []
    for token in train_freq:
        if train_freq[token] > 0:
            gen_f = gen_freq.get(token, 0)
            relative_ratios.append(gen_f / train_freq[token])
    
    if not relative_ratios:
        return {'mean': 0.0, 'std': 0.0}
    
    return {
        'mean': float(np.mean(relative_ratios)),
        'std': float(np.std(relative_ratios)),
    }


def compute_all_relative_ratios(train_seqs: list, gen_seqs: list,
                                 max_len: int = 256) -> dict:
    """Compute all relative ratio metrics."""
    rr_aa = compute_relative_ratio_aa(train_seqs, gen_seqs)
    
    train_lens = compute_sequence_lengths(train_seqs)
    gen_lens = compute_sequence_lengths(gen_seqs)
    rr_length = compute_relative_ratio_length(train_lens, gen_lens, max_len=max_len)
    
    train_plus = compute_binding_site_counts(train_seqs)
    gen_plus = compute_binding_site_counts(gen_seqs)
    rr_binding = compute_relative_ratio_discrete(train_plus, gen_plus)
    
    train_chains = compute_chain_counts(train_seqs)
    gen_chains = compute_chain_counts(gen_seqs)
    rr_chains = compute_relative_ratio_discrete(train_chains, gen_chains)
    
    return {
        'RR_aa': rr_aa,
        'RR_length': rr_length,
        'RR_binding_sites': rr_binding,
        'RR_chains': rr_chains,
    }


def compute_generation_metrics(generated_seqs: list, training_seqs: list,
                                reconstructed_seqs: list = None,
                                original_seqs: list = None,
                                max_len: int = 256) -> dict:
    """
    Compute all generation quality metrics.
    """
    nuv = compute_nuv(generated_seqs, training_seqs)
    
    if reconstructed_seqs is not None and original_seqs is not None:
        R = compute_reconstruction_rate(original_seqs, reconstructed_seqs)
    else:
        R = None
    
    rr = compute_all_relative_ratios(training_seqs, generated_seqs, max_len)
    
    result = {
        'N': nuv['N'],
        'U': nuv['U'],
        'V': nuv['V'],
    }
    
    if R is not None:
        result['R'] = R
    
    result.update(rr)
    
    return result

