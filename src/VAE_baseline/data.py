"""
PyTorch Lightning Bolts VAE - Data Module

Handles data loading and preprocessing for the sequence VAE.
Uses DISCRETE token IDs (integers) for proper cross-entropy reconstruction.

Uses the same train/test split logic as QOBRA for fair comparison:
- 80/20 split using modulo 5 (every 5th sequence goes to test)
- Cap on maximum training sequences (default 6000)
- Filter for sequences with binding sites
- No shuffling for reproducibility
- Cyclic padding to fill 512 positions
"""

import os
import random
import difflib
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Standard 20 amino acid codes
AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 
               'G', 'H', 'I', 'K', 'L', 
               'M', 'N', 'P', 'Q', 'R', 
               'S', 'T', 'V', 'W', 'Y']

# Special characters in sequence encoding
BINDING_MARKER = '+'   # Marks metal-binding residues
CHAIN_SEPARATOR = ':'  # Separates protein chains
TERMINATOR = 'X'       # Sequence terminator


def LCS(str1: str, str2: str) -> int:
    """
    Calculate the Longest Common Subsequence (LCS) between two strings.
    
    This function finds the length of the longest contiguous matching substring
    between two input strings. Used for sequence similarity analysis.
    
    Parameters:
    - str1, str2: Input strings to compare
    
    Returns:
    - Length of the longest common subsequence
    """
    s = difflib.SequenceMatcher(None, str1, str2)
    match = s.find_longest_match(0, len(str1), 0, len(str2))
    return match.size


def remove_similar_sequences(sequences: list) -> list:
    """
    Remove highly similar sequences to avoid redundancy.
    
    This matches QOBRA's deduplication logic in coding.py:prep()
    
    Parameters:
    - sequences: List of sequence strings
    
    Returns:
    - ret: List of unique/dissimilar sequences
    """
    if len(sequences) <= 1:
        return sequences
    
    temp = list(sequences)  # Make a copy
    ret = []
    
    i = 0
    while i < len(temp) - 1:
        lcs = LCS(temp[i], temp[i + 1])
        
        if len(temp[i]) == 0 or len(temp[i + 1]) == 0:
            i += 1
        elif lcs / len(temp[i]) < 0.1 and lcs / len(temp[i + 1]) < 0.1:
            # Sequences are sufficiently different - keep both
            ret.append(temp[i])
            i += 1
        else:
            # Sequences are too similar - keep only the shorter one
            if len(temp[i + 1]) < len(temp[i]):
                del temp[i]
            else:
                del temp[i + 1]
    
    # Don't forget the last sequence
    if temp:
        ret.append(temp[-1])
    
    return ret


def build_token_list():
    """
    Build comprehensive token list for sequence encoding.
    
    Returns:
    - tokens: List of all possible tokens (AA, AA+, :, :X)
    """
    tokens = []
    # Regular amino acids
    for aa in AMINO_ACIDS:
        tokens.append(aa)
    # Metal-binding amino acids (marked with '+')
    for aa in AMINO_ACIDS:
        tokens.append(aa + BINDING_MARKER)
    # Special tokens
    tokens.append(CHAIN_SEPARATOR)
    tokens.append(CHAIN_SEPARATOR + TERMINATOR)
    return tokens


def build_token_mappings():
    """
    Build token-to-ID mappings using integer indices.
    
    Returns:
    - token_to_id: Dict mapping token strings to integer IDs
    - id_to_token: Dict mapping integer IDs to token strings
    - vocab_size: Number of unique tokens
    """
    tokens = build_token_list()
    vocab_size = len(tokens)
    
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    id_to_token = {idx: token for idx, token in enumerate(tokens)}
    
    return token_to_id, id_to_token, vocab_size


def read_sequences_from_folder(folder_path: str):
    """
    Read all protein sequences from a folder.
    
    Each file may contain multiple chains (one chain per line).
    The chains are concatenated into a single string, keeping newlines
    as chain separators (which get converted to ':' later).
    
    This matches QOBRA's data loading logic in coding.py:prep()
    
    Parameters:
    - folder_path: Path to folder containing .txt sequence files
    
    Returns:
    - sequences: List of protein sequence strings (with newlines for chain separators)
    - filenames: List of corresponding filenames (PDB codes)
    """
    sequences = []
    filenames = []
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Data folder not found: {folder_path}")
    
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.txt') and not filename.startswith('.'):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Concatenate all lines into a single sequence string
            # This preserves newlines as chain separators (matching QOBRA)
            s = ''
            for line in lines:
                if len(line) > 0:
                    s += line
            
            if s:
                sequences.append(s)
                filenames.append(filename[:-4])  # Remove .txt
    
    return sequences, filenames


def crop(sequence: str, max_len: int) -> str:
    """
    Crop a sequence to maximum length, matching QOBRA's cropping logic.
    
    Finds a valid crop window that includes binding sites if possible.
    Always adds :X terminator if there's room.
    
    Parameters:
    - sequence: Protein sequence string
    - max_len: Maximum sequence length
    
    Returns:
    - cropped: Cropped sequence string with :X terminator
    """
    # First, handle sequences shorter than max_len
    if len(sequence) <= max_len - 2:  # Leave room for :X
        # Add :X terminator if not already present
        if not sequence.endswith(':X'):
            return sequence + ':X'
        return sequence
    elif len(sequence) <= max_len:
        # Just fits, no room for :X
        return sequence
    
    # For longer sequences, need to crop
    # Find positions of binding sites
    plus_positions = [i for i, c in enumerate(sequence) if c == '+']
    
    if not plus_positions:
        # No binding sites, just take first max_len-2 characters and add :X
        cropped = sequence[:max_len - 2]
        return cropped + ':X'
    
    # Try to include as many binding sites as possible
    # Start from the first binding site position
    first_plus = plus_positions[0]
    
    # Calculate start position to include binding sites
    # Try to center around the first binding site
    start = max(0, first_plus - max_len // 4)
    
    # Ensure we don't exceed the sequence length
    if start + max_len > len(sequence):
        start = max(0, len(sequence) - max_len)
    
    # Leave room for :X terminator
    cropped = sequence[start:start + max_len - 2]
    
    # Add terminator
    if not cropped.endswith(':X'):
        cropped = cropped + ':X'
    
    return cropped[:max_len]


def tokenize_sequence(sequence: str) -> list:
    """
    Parse a sequence string into a list of tokens.
    
    Parameters:
    - sequence: Protein sequence string
    
    Returns:
    - tokens: List of token strings
    """
    tokens = []
    i = 0
    
    while i < len(sequence):
        if i < len(sequence) - 1 and sequence[i + 1] in [BINDING_MARKER, TERMINATOR]:
            token = sequence[i:i + 2]
            i += 2
        else:
            token = sequence[i]
            i += 1
        tokens.append(token)
    
    return tokens


def sequence_to_ids(sequence: str, token_to_id: dict, max_len: int):
    """
    Convert a protein sequence to a vector of discrete token IDs.
    
    Uses cyclic padding (QOBRA-style): if sequence is shorter than max_len,
    repeat from the beginning until all positions are filled.
    The :X terminator marks where the original sequence ends.
    
    Parameters:
    - sequence: Protein sequence string
    - token_to_id: Dictionary mapping tokens to integer IDs
    - max_len: Maximum sequence length (for padding)
    
    Returns:
    - token_ids: numpy array of shape (max_len,) with integer IDs
    - seq_len: Original sequence length (before padding)
    """
    # Parse tokens
    tokens = tokenize_sequence(sequence)
    
    # Convert to IDs
    token_ids = []
    for token in tokens:
        if token in token_to_id:
            token_ids.append(token_to_id[token])
        else:
            # Unknown token: use first token (A) as fallback
            token_ids.append(0)
    
    # Store original length
    seq_len = len(token_ids)
    
    # If empty sequence, fill with zeros
    if seq_len == 0:
        return np.zeros(max_len, dtype=np.int64), 0
    
    # Cyclic padding: repeat sequence from beginning until max_len is reached
    result = []
    idx = 0
    while len(result) < max_len:
        result.append(token_ids[idx % len(token_ids)])
        idx += 1
    
    return np.array(result[:max_len], dtype=np.int64), seq_len


def ids_to_sequence(token_ids: np.ndarray, id_to_token: dict, seq_len: int = None) -> str:
    """
    Convert token IDs back to a protein sequence string.
    
    Parameters:
    - token_ids: numpy array of integer token IDs
    - id_to_token: Dictionary mapping IDs to token strings
    - seq_len: Original sequence length (if known, truncates to this)
    
    Returns:
    - sequence: Decoded protein sequence string
    """
    sequence = ""
    for i, tid in enumerate(token_ids):
        if seq_len is not None and i >= seq_len:
            break
        tid = int(tid)
        if tid in id_to_token:
            sequence += id_to_token[tid]
        else:
            sequence += "?"
    
    return sequence


def get_effective_sequence(sequence: str) -> str:
    """
    Get the effective sequence up to the :X terminator.
    
    In QOBRA-style encoding, :X marks where the original sequence ends.
    Everything after is cyclic padding and should be ignored for evaluation.
    
    Parameters:
    - sequence: Full decoded sequence string
    
    Returns:
    - effective: Sequence up to and including :X (or full sequence if no terminator)
    """
    # Find :X terminator
    term_idx = sequence.find(':X')
    if term_idx != -1:
        return sequence[:term_idx + 2]  # Include the :X
    
    # Also check for just X (single terminator)
    x_idx = sequence.find('X')
    if x_idx != -1:
        return sequence[:x_idx + 1]
    
    return sequence


def compare_sequences_effective(original: str, reconstructed: str) -> tuple:
    """
    Compare two sequences considering only the effective part (up to :X).
    
    Parameters:
    - original: Original sequence string
    - reconstructed: Reconstructed sequence string
    
    Returns:
    - match: True if effective parts match exactly
    - orig_eff: Effective part of original
    - recon_eff: Effective part of reconstructed (same length as orig_eff)
    """
    orig_eff = get_effective_sequence(original)
    
    # For reconstruction comparison, compare same number of tokens
    # as the original effective length
    recon_eff = reconstructed[:len(orig_eff)]
    
    return orig_eff == recon_eff, orig_eff, recon_eff


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for protein sequences encoded as discrete token IDs.
    """
    
    def __init__(self, sequences: list, token_to_id: dict, max_len: int):
        """
        Initialize dataset.
        
        Parameters:
        - sequences: List of protein sequence strings
        - token_to_id: Dictionary mapping tokens to integer IDs
        - max_len: Maximum sequence length
        """
        self.sequences = sequences
        self.token_to_id = token_to_id
        self.max_len = max_len
        
        # Pre-compute token IDs and lengths
        self.token_ids = []
        self.seq_lens = []
        for seq in sequences:
            ids, slen = sequence_to_ids(seq, token_to_id, max_len)
            self.token_ids.append(ids)
            self.seq_lens.append(slen)
        
        self.token_ids = np.stack(self.token_ids, axis=0)
        self.seq_lens = np.array(self.seq_lens, dtype=np.int64)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (torch.from_numpy(self.token_ids[idx]), 
                torch.tensor(self.seq_lens[idx], dtype=torch.long))


def load_data(data_dir: str, metal: str, max_len: int = 512, 
              cap: int = 6000, seed: int = 42):
    """
    Load and prepare data for training, using QOBRA's train/test split logic.
    
    Split logic (matching QOBRA):
    - 80/20 split using i % 5 (every 5th sequence goes to test)
    - Cap on maximum training sequences (default 6000)
    - Filter for sequences with binding sites ('+')
    - Test size is proportional to training size (train_size // 5)
    - No shuffling for reproducibility
    
    Parameters:
    - data_dir: Path to Training data directory
    - metal: Metal type subfolder (Ca, Mg, or Zn)
    - max_len: Maximum sequence length
    - cap: Maximum number of training sequences (default 6000)
    - seed: Random seed (not used for split, kept for API compat)
    
    Returns:
    - train_dataset: Training dataset
    - test_dataset: Test dataset
    - token_to_id: Token to ID mapping
    - id_to_token: ID to token mapping
    - vocab_size: Vocabulary size
    """
    folder_path = os.path.join(data_dir, f"{metal}_bind")
    all_sequences, filenames = read_sequences_from_folder(folder_path)
    
    if len(all_sequences) == 0:
        raise ValueError(f"No sequences found in {folder_path}")
    
    print(f"Loaded {len(all_sequences)} total sequences from {folder_path}")
    
    # Remove highly similar sequences (matching QOBRA's prep() function)
    all_sequences = remove_similar_sequences(all_sequences)
    print(f"After deduplication: {len(all_sequences)} unique sequences")
    
    # Build token mappings
    token_to_id, id_to_token, vocab_size = build_token_mappings()
    print(f"Vocabulary size: {vocab_size} tokens (discrete IDs)")
    
    # =========================================================================
    # QOBRA-style train/test split
    # =========================================================================
    
    # Split sequences into training (80%) and test (20%) sets using modulo 5
    # No shuffling for reproducibility (matching QOBRA)
    all_input_seqs = []  # 80% for training candidates
    all_test_seqs = []   # 20% for test candidates
    
    for i, seq in enumerate(all_sequences):
        # Normalize: replace newlines with chain separator (matching QOBRA)
        s = seq.replace("\n", ":")
        
        if i % 5 > 0:
            all_input_seqs.append(s)  # 80% for training
        else:
            all_test_seqs.append(s)   # 20% for testing
    
    print(f"Split: {len(all_input_seqs)} training candidates, {len(all_test_seqs)} test candidates")
    
    # Filter and prepare final training set
    train_seqs = []
    for s in all_input_seqs:
        seg = crop(s, max_len)  # Crop to max_len (matching QOBRA)
        char_cnts = Counter(seg)
        
        # Only include sequences with binding sites and within capacity limit
        if char_cnts['+'] > 0 and len(train_seqs) < cap:
            train_seqs.append(seg)
    
    # Set test set size proportional to training set (matching QOBRA)
    train_size = len(train_seqs)
    test_size = train_size // 5
    
    # Filter and prepare final test set
    test_seqs = []
    for s in all_test_seqs:
        seg = crop(s, max_len)  # Crop to max_len (matching QOBRA)
        char_cnts = Counter(seg)
        
        if char_cnts['+'] > 0 and len(test_seqs) < test_size:
            test_seqs.append(seg)
    
    print(f"Final: {len(train_seqs)} training, {len(test_seqs)} test sequences")
    print(f"  (cap={cap}, filtered for binding sites)")
    
    # Create datasets
    train_dataset = SequenceDataset(train_seqs, token_to_id, max_len)
    test_dataset = SequenceDataset(test_seqs, token_to_id, max_len)
    
    return train_dataset, test_dataset, token_to_id, id_to_token, vocab_size


def get_dataloader(dataset: Dataset, batch_size: int = 256, shuffle: bool = True, 
                   num_workers: int = 0):
    """
    Create DataLoader for training.
    
    Parameters:
    - dataset: Training dataset
    - batch_size: Batch size
    - shuffle: Whether to shuffle data
    - num_workers: Number of data loading workers
    
    Returns:
    - dataloader: DataLoader
    """
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False
    )
