"""
Sequence loading utilities for ESM evaluation.

Provides functions to read, validate, and prepare protein sequences for ESM scoring.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Standard amino acids (20 canonical + rare amino acids)
STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# Extended set including rare amino acids sometimes seen in sequences
EXTENDED_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYBZXUO")


def validate_sequence(sequence: str, allow_extended: bool = False) -> bool:
    """
    Validate that a sequence contains only valid amino acid characters.
    
    Args:
        sequence: Amino acid sequence string (uppercase).
        allow_extended: If True, allows extended amino acid characters (B, Z, X, U, O).
    
    Returns:
        True if sequence is valid, False otherwise.
    """
    valid_chars = EXTENDED_AMINO_ACIDS if allow_extended else STANDARD_AMINO_ACIDS
    return all(char in valid_chars for char in sequence)


def is_skippable_line(line: str) -> bool:
    """
    Check if a line should be skipped during sequence loading.
    
    Skips:
        - Empty lines
        - Comment lines (starting with #)
        - Header lines (e.g., "De novo sequences")
        - Separator lines (rows of asterisks)
        - PDB ID lines (short alphanumeric identifiers like "1a0b")
    
    Args:
        line: Raw line from file.
    
    Returns:
        True if line should be skipped, False otherwise.
    """
    stripped = line.strip()
    if not stripped:
        return True
    # Skip comment lines
    if stripped.startswith('#'):
        return True
    # Skip separator lines (all asterisks)
    if all(c == '*' for c in stripped):
        return True
    # Skip header lines (contain only letters and spaces, no valid amino acid pattern)
    # Headers like "De novo sequences" have spaces and lowercase
    if ' ' in stripped.lower() and not any(c in stripped for c in '+:\t'):
        return True
    # Skip short PDB ID lines (4-6 chars, alphanumeric, often with digits)
    if len(stripped) <= 6 and stripped.isalnum() and any(c.isdigit() for c in stripped):
        return True
    return False


def clean_sequence(sequence: str) -> str:
    """
    Clean a sequence by uppercasing, removing whitespace, and stripping annotations.
    
    Handles special markers and formats:
        - '+' marks ligand-binding residues (stripped, keeping the amino acid)
        - ':' denotes chain boundaries (stripped, chains concatenated)
        - Tab-separated format: "index\\tsequence" -> extracts sequence part
        - Terminal 'X' markers (stripped)
    
    Args:
        sequence: Raw sequence string.
    
    Returns:
        Cleaned sequence string with only amino acid characters.
    """
    cleaned = sequence.strip()
    
    # Handle tab-separated format (e.g., "0\tSEQUENCE")
    if '\t' in cleaned:
        parts = cleaned.split('\t')
        # Take the last part that looks like a sequence (not just a number)
        for part in reversed(parts):
            if part and not part.isdigit():
                cleaned = part
                break
    
    cleaned = cleaned.upper().replace(" ", "")
    # Remove binding site markers (+) and chain separators (:)
    cleaned = cleaned.replace("+", "").replace(":", "")
    # Truncate at terminator X (consistent with esm_loss.py and gen_func.py)
    if 'X' in cleaned:
        cleaned = cleaned[:cleaned.index('X')]
    return cleaned


def load_sequences_from_file(
    filepath: str,
    allow_extended: bool = False
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load and validate sequences from a single file.
    
    Args:
        filepath: Path to a .txt file with one sequence per line.
        allow_extended: If True, allows extended amino acid characters.
    
    Returns:
        Tuple of (sequences, sequence_ids, source_files) where:
            - sequences: List of cleaned, validated sequences
            - sequence_ids: List of sequence identifiers
            - source_files: List of source file basenames (same for all)
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no valid sequences are found.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Sequence file not found: {filepath}")
    
    source_file = filepath.name
    sequences = []
    sequence_ids = []
    source_files = []
    
    invalid_count = 0
    
    seq_num = 0
    with open(filepath, "r") as f:
        for line_num, line in enumerate(f, start=1):
            # Skip headers, separators, and empty lines
            if is_skippable_line(line):
                continue
            
            sequence = clean_sequence(line)
            
            if not sequence:
                continue
            
            if validate_sequence(sequence, allow_extended=allow_extended):
                seq_num += 1
                seq_id = f"{filepath.stem}_{seq_num}"
                sequences.append(sequence)
                sequence_ids.append(seq_id)
                source_files.append(source_file)
            else:
                invalid_count += 1
    
    if invalid_count > 0:
        print(f"  Warning: Skipped {invalid_count} invalid sequences in {source_file}")
    
    if not sequences:
        raise ValueError(f"No valid sequences found in {filepath}")
    
    return sequences, sequence_ids, source_files


def load_sequences(
    input_path: str,
    allow_extended: bool = False
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load sequences from a file or directory.
    
    Args:
        input_path: Path to a .txt file or directory containing .txt files.
        allow_extended: If True, allows extended amino acid characters.
    
    Returns:
        Tuple of (sequences, sequence_ids, source_files) where:
            - sequences: List of all cleaned, validated sequences
            - sequence_ids: List of unique sequence identifiers
            - source_files: List of source file basenames
    
    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If no .txt files are found or no valid sequences.
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    all_sequences = []
    all_sequence_ids = []
    all_source_files = []
    
    if input_path.is_file():
        # Single file
        if not input_path.suffix == ".txt":
            raise ValueError(f"Expected .txt file, got: {input_path}")
        
        print(f"Loading sequences from: {input_path.name}")
        seqs, ids, sources = load_sequences_from_file(
            str(input_path), allow_extended=allow_extended
        )
        all_sequences.extend(seqs)
        all_sequence_ids.extend(ids)
        all_source_files.extend(sources)
        print(f"  Loaded {len(seqs)} sequences")
    
    elif input_path.is_dir():
        # Directory - load all .txt files
        txt_files = sorted(input_path.glob("*.txt"))
        
        if not txt_files:
            raise ValueError(f"No .txt files found in: {input_path}")
        
        print(f"Loading sequences from directory: {input_path}")
        print(f"Found {len(txt_files)} .txt files")
        
        for txt_file in txt_files:
            print(f"  Processing: {txt_file.name}")
            seqs, ids, sources = load_sequences_from_file(
                str(txt_file), allow_extended=allow_extended
            )
            all_sequences.extend(seqs)
            all_sequence_ids.extend(ids)
            all_source_files.extend(sources)
            print(f"    Loaded {len(seqs)} sequences")
    
    else:
        raise ValueError(f"Input path is neither a file nor directory: {input_path}")
    
    print(f"\nTotal: {len(all_sequences)} sequences from {len(set(all_source_files))} file(s)")
    
    return all_sequences, all_sequence_ids, all_source_files


def get_sequence_stats(sequences: List[str]) -> Dict:
    """
    Compute basic statistics for a list of sequences.
    
    Args:
        sequences: List of protein sequences.
    
    Returns:
        Dictionary with sequence statistics.
    """
    lengths = [len(s) for s in sequences]
    return {
        "count": len(sequences),
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "mean_length": sum(lengths) / len(lengths) if lengths else 0,
    }

