#!/usr/bin/env python3
"""
Select Best Denovo Script

Scans numbered subfolders (0, 1, 2, ...) inside an experiment directory,
reads denovo-*.txt files, computes ESM log-likelihood scores for each,
and selects the one with the highest overall score to replace
the denovo-qobra-{metal}.txt file outside the experiment directory.

Sequence Format Reminder:
    - A-Y: Standard amino acids
    - +: Ligand-binding residue marker (follows amino acid)
    - :: Chain boundary separator
    - X: Sequence terminator

Usage:
    python select_best_denovo.py --experiment_dir "Training data/Zn_2_0.0001_500_32_esm2_t30_150M_UR50D/"
    
    # With specific metal type (defaults to extracting from folder name)
    python select_best_denovo.py --experiment_dir "Training data/Zn_2_0.0001_500_32_esm2_t30_150M_UR50D/" --metal Zn
    
    # Dry run (show what would happen without modifying files)
    python select_best_denovo.py --experiment_dir "Training data/Zn_2_0.0001_500_32_esm2_t30_150M_UR50D/" --dry_run
"""

import argparse
import os
import shutil
import re
from glob import glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from load_sequences import load_sequences_from_file, get_sequence_stats
from esm_scoring import ESMScorer


def find_numbered_subfolders(experiment_dir: Path) -> List[Path]:
    """
    Find all numbered subfolders (0, 1, 2, ...) in the experiment directory.
    
    Args:
        experiment_dir: Path to the experiment directory.
    
    Returns:
        Sorted list of numbered subfolder paths.
    """
    subfolders = []
    for item in experiment_dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            subfolders.append(item)
    return sorted(subfolders, key=lambda x: int(x.name))


def find_denovo_file(folder: Path) -> Optional[Path]:
    """
    Find the denovo-*.txt file in a folder.
    
    Args:
        folder: Path to search in.
    
    Returns:
        Path to the denovo file, or None if not found.
    """
    pattern = folder / "denovo-*.txt"
    matches = list(folder.glob("denovo-*.txt"))
    if matches:
        return matches[0]  # Expect only one denovo file per folder
    return None


def extract_metal_from_folder_name(folder_name: str) -> str:
    """
    Extract metal type from experiment folder name.
    
    Example: "Zn_2_0.0001_500_32_esm2_t30_150M_UR50D" -> "Zn"
    
    Args:
        folder_name: Name of the experiment folder.
    
    Returns:
        Metal type string (e.g., "Zn", "Ca", "Mg").
    """
    # Pattern: Metal_layers_esmlambda_steps_K_model
    match = re.match(r"([A-Za-z]+)_", folder_name)
    if match:
        return match.group(1)
    return "Unknown"


def score_denovo_file(
    scorer: ESMScorer,
    filepath: Path,
    batch_size: int = 8
) -> Tuple[float, int, List[float]]:
    """
    Compute ESM scores for all sequences in a denovo file.
    
    Args:
        scorer: ESMScorer instance.
        filepath: Path to the denovo file.
        batch_size: Batch size for scoring.
    
    Returns:
        Tuple of (mean_score, sequence_count, all_scores).
    """
    sequences, sequence_ids, source_files = load_sequences_from_file(str(filepath))
    
    if not sequences:
        return float('-inf'), 0, []
    
    scores = scorer.compute_log_likelihood_batch(
        sequences=sequences,
        sequence_ids=sequence_ids,
        source_files=source_files,
        show_progress=False,
        batch_size=batch_size
    )
    
    mean_lls = [s.mean_log_likelihood for s in scores]
    overall_mean = sum(mean_lls) / len(mean_lls) if mean_lls else float('-inf')
    
    return overall_mean, len(sequences), mean_lls


def main():
    parser = argparse.ArgumentParser(
        description="Select best denovo file from numbered subfolders based on ESM scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate and select best from numbered folders
    python select_best_denovo.py --experiment_dir "../Training data/Zn_2_0.0001_500_32_esm2_t30_150M_UR50D/"
    
    # Dry run to see scores without copying
    python select_best_denovo.py --experiment_dir "../Training data/Zn_2_0.0001_500_32_esm2_t30_150M_UR50D/" --dry_run
    
    # Specify metal type explicitly
    python select_best_denovo.py --experiment_dir "../Training data/Zn_2_0.0001_500_32_esm2_t30_150M_UR50D/" --metal Zn
        """
    )
    
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to the experiment directory containing numbered subfolders"
    )
    parser.add_argument(
        "--metal",
        type=str,
        default=None,
        help="Metal type (e.g., Zn, Ca, Mg). If not provided, extracted from folder name."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="esm2_t33_650M_UR50D",
        help="ESM model to use for scoring (default: esm2_t33_650M_UR50D)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu, or auto-detect if not specified)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for GPU processing (default: 8)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show scores and best file without copying"
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="9",
        help="Suffix for output filename (default: 9, creates denovo-qobra-{metal}9.txt)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    experiment_dir = Path(args.experiment_dir).resolve()
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    print("=" * 70)
    print("SELECT BEST DENOVO")
    print("=" * 70)
    print(f"Experiment directory: {experiment_dir}")
    
    # Extract metal type
    metal = args.metal or extract_metal_from_folder_name(experiment_dir.name)
    print(f"Metal type: {metal}")
    
    # Find numbered subfolders
    subfolders = find_numbered_subfolders(experiment_dir)
    if not subfolders:
        raise ValueError(f"No numbered subfolders found in {experiment_dir}")
    
    print(f"Found {len(subfolders)} numbered subfolders: {[f.name for f in subfolders]}")
    
    # Find denovo files in each subfolder
    denovo_files = {}
    for folder in subfolders:
        denovo_file = find_denovo_file(folder)
        if denovo_file:
            denovo_files[folder.name] = denovo_file
        else:
            print(f"  Warning: No denovo-*.txt found in {folder.name}/")
    
    if not denovo_files:
        raise ValueError("No denovo files found in any numbered subfolder")
    
    print(f"\nFound {len(denovo_files)} denovo files to evaluate:")
    for name, path in denovo_files.items():
        print(f"  {name}: {path.name}")
    
    # Initialize ESM scorer
    print(f"\n[Step 1] Initializing ESM scorer...")
    print(f"  Model: {args.model}")
    scorer = ESMScorer(model_name=args.model, device=args.device)
    
    # Score each denovo file
    print(f"\n[Step 2] Scoring denovo files...")
    results = {}
    for name, filepath in denovo_files.items():
        print(f"\n  Scoring {name}/{filepath.name}...")
        mean_score, seq_count, all_scores = score_denovo_file(
            scorer, filepath, batch_size=args.batch_size
        )
        results[name] = {
            "path": filepath,
            "mean_score": mean_score,
            "seq_count": seq_count,
            "all_scores": all_scores,
        }
        print(f"    Sequences: {seq_count}")
        print(f"    Mean ESM log-likelihood: {mean_score:.4f}")
    
    # Find best file
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    # Sort by mean score (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["mean_score"], reverse=True)
    
    print("\nRanking (by mean ESM log-likelihood):")
    for rank, (name, data) in enumerate(sorted_results, 1):
        marker = " <-- BEST" if rank == 1 else ""
        print(f"  {rank}. Folder {name}: {data['mean_score']:.4f} ({data['seq_count']} seqs){marker}")
    
    best_name, best_data = sorted_results[0]
    best_path = best_data["path"]
    
    # Determine output path
    output_filename = f"denovo-qobra-{metal}{args.output_suffix}.txt"
    output_path = experiment_dir.parent / output_filename
    
    print(f"\nBest denovo file: {best_name}/{best_path.name}")
    print(f"  Mean ESM log-likelihood: {best_data['mean_score']:.4f}")
    print(f"  Sequence count: {best_data['seq_count']}")
    print(f"\nTarget output: {output_path}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would copy:")
        print(f"  From: {best_path}")
        print(f"  To:   {output_path}")
        print("\nNo files were modified. Remove --dry_run to execute.")
    else:
        print(f"\n[Step 3] Copying best denovo file to output...")
        shutil.copy2(best_path, output_path)
        print(f"  Copied: {best_path.name} -> {output_path.name}")
        print("\nDone! Best denovo file has been selected and copied.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

