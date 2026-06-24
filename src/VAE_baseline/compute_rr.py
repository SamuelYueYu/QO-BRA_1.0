#!/usr/bin/env python3
"""
Standalone script to compute RR (Relative Ratio) metrics for a sequence file.

Usage:
    # Compute RR of denovo-random-Zn9.txt relative to Zn training set (6000 seqs):
    python compute_rr.py --metal Zn "Training data/denovo-random-Zn9.txt"
    
    # Compare two files directly (generated vs training):
    python compute_rr.py --generated generated_seqs.txt --training training_seqs.txt
"""

import argparse
import sys
import os
import json
from collections import Counter

from metrics import (
    compute_all_relative_ratios,
    compute_aa_frequencies,
    compute_sequence_lengths,
    compute_binding_site_counts,
    compute_chain_counts,
    compute_nuv,
    compute_uniqueness,
    compute_validity,
    get_valid_sequences,
)
from data import (
    read_sequences_from_folder,
    remove_similar_sequences,
    crop,
)
import numpy as np


def load_sequences_from_file(filepath: str) -> list:
    """
    Load sequences from file, skipping asterisk lines.
    """
    sequences = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('*'):
                sequences.append(line)
    return sequences


def build_training_set(data_dir: str, metal: str, max_len: int = 512, cap: int = 6000) -> list:
    """
    Build the training set using the same logic as load_data in data.py.
    
    This replicates QOBRA's preprocessing:
    1. Load sequences from {metal}_bind folder
    2. Remove similar sequences (deduplication)
    3. 80/20 split using modulo 5
    4. Filter for sequences with binding sites
    5. Cap at 6000 training sequences
    
    Returns list of training sequence strings.
    """
    folder_path = os.path.join(data_dir, f"{metal}_bind")
    all_sequences, filenames = read_sequences_from_folder(folder_path)
    
    if len(all_sequences) == 0:
        raise ValueError(f"No sequences found in {folder_path}")
    
    print(f"Loaded {len(all_sequences)} total sequences from {folder_path}")
    
    # Remove highly similar sequences (matching QOBRA's prep() function)
    all_sequences = remove_similar_sequences(all_sequences)
    print(f"After deduplication: {len(all_sequences)} unique sequences")
    
    # Split sequences into training (80%) using modulo 5
    # No shuffling for reproducibility (matching QOBRA)
    all_input_seqs = []  # 80% for training candidates
    
    for i, seq in enumerate(all_sequences):
        # Normalize: replace newlines with chain separator (matching QOBRA)
        s = seq.replace("\n", ":")
        
        if i % 5 > 0:
            all_input_seqs.append(s)  # 80% for training
    
    print(f"After 80/20 split: {len(all_input_seqs)} training candidates")
    
    # Filter and prepare final training set
    train_seqs = []
    for s in all_input_seqs:
        seg = crop(s, max_len)  # Crop to max_len (matching QOBRA)
        char_cnts = Counter(seg)
        
        # Only include sequences with binding sites and within capacity limit
        if char_cnts['+'] > 0 and len(train_seqs) < cap:
            train_seqs.append(seg)
    
    print(f"Final training set: {len(train_seqs)} sequences (cap={cap}, filtered for binding sites)")
    
    return train_seqs


def compute_distribution_stats(values: list) -> dict:
    """Compute statistics for a distribution and return as dict."""
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
    
    arr = np.array(values)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "median": float(np.median(arr))
    }


def print_distribution_stats(values: list, name: str):
    """Print statistics for a distribution."""
    if not values:
        print(f"  {name}: No data")
        return
    
    arr = np.array(values)
    print(f"  {name}:")
    print(f"    Mean: {arr.mean():.2f}")
    print(f"    Std:  {arr.std():.2f}")
    print(f"    Min:  {arr.min()}")
    print(f"    Max:  {arr.max()}")
    print(f"    Median: {np.median(arr):.2f}")


def print_rr_table(rr: dict, title: str):
    """Print RR values in a clear table format."""
    print("\n" + "=" * 60)
    print("RELATIVE RATIO (RR) VALUES")
    print(title)
    print("=" * 60)
    
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  METRIC                    RR VALUE         RR STD      │")
    print("├─────────────────────────────────────────────────────────┤")
    print(f"│  RR_aa (amino acids)       {rr['RR_aa']['mean']:8.4f}         {rr['RR_aa']['std']:8.4f}     │")
    print(f"│  RR_length                 {rr['RR_length']['mean']:8.4f}         {rr['RR_length']['std']:8.4f}     │")
    print(f"│  RR_binding_sites          {rr['RR_binding_sites']['mean']:8.4f}         {rr['RR_binding_sites']['std']:8.4f}     │")
    print(f"│  RR_chains                 {rr['RR_chains']['mean']:8.4f}         {rr['RR_chains']['std']:8.4f}     │")
    print("└─────────────────────────────────────────────────────────┘")


def print_nuv_table(nuv: dict, title: str):
    """Print NUV values in a clear table format."""
    print("\n" + "=" * 60)
    print("NOVELTY, UNIQUENESS, VALIDITY (NUV) VALUES")
    print(title)
    print("=" * 60)
    
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  METRIC                              VALUE              │")
    print("├─────────────────────────────────────────────────────────┤")
    print(f"│  N (Novelty)                         {nuv['N']:8.4f}             │")
    print(f"│  U (Uniqueness)                      {nuv['U']:8.4f}             │")
    print(f"│  V (Validity)                        {nuv['V']:8.4f}             │")
    print("└─────────────────────────────────────────────────────────┘")


def save_results(output_path: str, results: dict):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute RR metrics for sequence files."
    )
    parser.add_argument(
        "file", nargs="?", 
        help="Sequence file to evaluate (e.g., denovo-random-Zn9.txt)"
    )
    parser.add_argument(
        "--metal", "-m",
        choices=["Ca", "Mg", "Zn"],
        help="Metal type to build training set from (uses Training data/{metal}_bind/)"
    )
    parser.add_argument(
        "--data-dir", "-d",
        default="Training data",
        help="Path to Training data directory (default: 'Training data')"
    )
    parser.add_argument(
        "--cap", "-c",
        type=int, default=6000,
        help="Maximum number of training sequences (default: 6000)"
    )
    parser.add_argument(
        "--generated", "-g",
        help="Generated sequences file (for direct comparison)"
    )
    parser.add_argument(
        "--training", "-t",
        help="Training sequences file (for direct comparison)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file to save results (default: auto-generated based on input)"
    )
    
    args = parser.parse_args()
    
    # Mode 1: Compare generated vs training files directly
    if args.generated and args.training:
        print(f"Loading generated sequences from: {args.generated}")
        gen_seqs = load_sequences_from_file(args.generated)
        print(f"  Loaded {len(gen_seqs)} sequences")
        
        print(f"\nLoading training sequences from: {args.training}")
        train_seqs = load_sequences_from_file(args.training)
        print(f"  Loaded {len(train_seqs)} sequences")
        
        # Compute NUV
        print("\nComputing NUV metrics (this may take a moment for novelty)...")
        nuv = compute_nuv(gen_seqs, train_seqs)
        print_nuv_table(nuv, "Generated vs Training")
        
        # Compute RR
        rr = compute_all_relative_ratios(train_seqs, gen_seqs)
        print_rr_table(rr, "Generated vs Training")
        
        # Save results
        results = {
            "mode": "generated_vs_training",
            "generated_file": args.generated,
            "training_file": args.training,
            "num_generated": len(gen_seqs),
            "num_training": len(train_seqs),
            "N": nuv["N"],
            "U": nuv["U"],
            "V": nuv["V"],
            "RR_aa": rr["RR_aa"],
            "RR_length": rr["RR_length"],
            "RR_binding_sites": rr["RR_binding_sites"],
            "RR_chains": rr["RR_chains"],
        }
        output_path = args.output or f"rr_results_{os.path.basename(args.generated).replace('.txt', '')}.json"
        save_results(output_path, results)
        
    # Mode 2: Compare file against metal training set (build from {metal}_bind folder)
    elif args.file and args.metal:
        print(f"Loading sequences from: {args.file}")
        eval_seqs = load_sequences_from_file(args.file)
        print(f"  Loaded {len(eval_seqs)} sequences")
        
        print(f"\nBuilding {args.metal} training set from {args.data_dir}/{args.metal}_bind/...")
        train_seqs = build_training_set(args.data_dir, args.metal, cap=args.cap)
        
        # Compute distribution statistics
        lengths = compute_sequence_lengths(eval_seqs)
        binding_sites = compute_binding_site_counts(eval_seqs)
        chains = compute_chain_counts(eval_seqs)
        aa_freq = compute_aa_frequencies(eval_seqs)
        
        # Print distribution statistics
        print("\n" + "=" * 60)
        print(f"Distribution Statistics: {args.file}")
        print("=" * 60)
        
        print_distribution_stats(lengths, "Sequence Length")
        print_distribution_stats(binding_sites, "Binding Sites (+)")
        print_distribution_stats(chains, "Chain Separators (:)")
        
        # Print AA frequencies
        print("\n  Amino Acid Frequencies (top 15):")
        sorted_aa = sorted(aa_freq.items(), key=lambda x: -x[1])
        for token, freq in sorted_aa[:15]:
            print(f"    {token!r:6s}: {freq:.6f}")
        
        # Compute NUV
        print("\nComputing NUV metrics (this may take a moment for novelty)...")
        nuv = compute_nuv(eval_seqs, train_seqs)
        print_nuv_table(nuv, f"{os.path.basename(args.file)} vs {args.metal} Training Set")
        
        # Compute RR
        rr = compute_all_relative_ratios(train_seqs, eval_seqs)
        print_rr_table(rr, f"{os.path.basename(args.file)} vs {args.metal} Training Set ({len(train_seqs)} seqs)")
        
        # Save results
        results = {
            "mode": "file_vs_metal_training",
            "eval_file": args.file,
            "metal": args.metal,
            "num_eval_seqs": len(eval_seqs),
            "num_training_seqs": len(train_seqs),
            "training_cap": args.cap,
            "distributions": {
                "sequence_length": compute_distribution_stats(lengths),
                "binding_sites": compute_distribution_stats(binding_sites),
                "chain_separators": compute_distribution_stats(chains),
            },
            "aa_frequencies": {k: float(v) for k, v in aa_freq.items()},
            "N": nuv["N"],
            "U": nuv["U"],
            "V": nuv["V"],
            "RR_aa": rr["RR_aa"],
            "RR_length": rr["RR_length"],
            "RR_binding_sites": rr["RR_binding_sites"],
            "RR_chains": rr["RR_chains"],
        }
        basename = os.path.basename(args.file).replace('.txt', '')
        output_path = args.output or f"rr_results_{basename}_vs_{args.metal}.json"
        save_results(output_path, results)
        
    # Mode 3: Single file analysis (self-RR)
    elif args.file:
        print(f"Loading sequences from: {args.file}")
        seqs = load_sequences_from_file(args.file)
        print(f"  Loaded {len(seqs)} sequences")
        
        # Compute distribution statistics
        lengths = compute_sequence_lengths(seqs)
        binding_sites = compute_binding_site_counts(seqs)
        chains = compute_chain_counts(seqs)
        aa_freq = compute_aa_frequencies(seqs)
        
        print("\n" + "=" * 60)
        print("Distribution Statistics")
        print("=" * 60)
        
        print_distribution_stats(lengths, "Sequence Length")
        print_distribution_stats(binding_sites, "Binding Sites (+)")
        print_distribution_stats(chains, "Chain Separators (:)")
        
        print("\n  Amino Acid Frequencies (top 15):")
        sorted_aa = sorted(aa_freq.items(), key=lambda x: -x[1])
        for token, freq in sorted_aa[:15]:
            print(f"    {token!r:6s}: {freq:.6f}")
        
        # Compute U and V (N doesn't make sense for self-comparison)
        valid_seqs = get_valid_sequences(seqs)
        U = compute_uniqueness(seqs, valid_seqs)
        V = compute_validity(seqs)
        
        print("\n" + "=" * 60)
        print("UNIQUENESS AND VALIDITY")
        print("(N/Novelty requires a separate training set for comparison)")
        print("=" * 60)
        print("\n┌─────────────────────────────────────────────────────────┐")
        print("│  METRIC                              VALUE              │")
        print("├─────────────────────────────────────────────────────────┤")
        print(f"│  U (Uniqueness)                      {U:8.4f}             │")
        print(f"│  V (Validity)                        {V:8.4f}             │")
        print("└─────────────────────────────────────────────────────────┘")
        
        # Self-RR
        rr = compute_all_relative_ratios(seqs, seqs)
        print_rr_table(rr, "Self-comparison (sanity check - should be ~1.0)")
        
        # Save results
        results = {
            "mode": "single_file_analysis",
            "file": args.file,
            "num_sequences": len(seqs),
            "distributions": {
                "sequence_length": compute_distribution_stats(lengths),
                "binding_sites": compute_distribution_stats(binding_sites),
                "chain_separators": compute_distribution_stats(chains),
            },
            "aa_frequencies": {k: float(v) for k, v in aa_freq.items()},
            "U": U,
            "V": V,
            "self_RR_aa": rr["RR_aa"],
            "self_RR_length": rr["RR_length"],
            "self_RR_binding_sites": rr["RR_binding_sites"],
            "self_RR_chains": rr["RR_chains"],
        }
        basename = os.path.basename(args.file).replace('.txt', '')
        output_path = args.output or f"rr_results_{basename}.json"
        save_results(output_path, results)
        
        print("\nNote: To compute RR against a metal training set, use:")
        print(f"  python compute_rr.py --metal Zn \"{args.file}\"")
        
    else:
        parser.print_help()
        print("\n\nExamples:")
        print("  # Compute RR of denovo-random-Zn9.txt vs Zn training set:")
        print('  python compute_rr.py --metal Zn "Training data/denovo-random-Zn9.txt"')
        print("")
        print("  # Compare two files directly:")
        print("  python compute_rr.py -g generated.txt -t training.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
