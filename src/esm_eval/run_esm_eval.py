#!/usr/bin/env python3
"""
ESM Evaluation Pipeline - Main Entry Point

Evaluates protein sequence plausibility using Meta's ESM models.
Computes log-likelihood scores and generates analysis outputs.

Usage:
    python run_esm_eval.py --input "Training data/" --output_csv results_all.csv --output_fig esm_ll_histograms.png
"""

import argparse
import csv
import os
from pathlib import Path
from typing import List, Dict

from load_sequences import load_sequences, get_sequence_stats
from esm_scoring import ESMScorer, SequenceScore
from plot_histograms import plot_log_likelihood_histograms


def write_csv(results: List[Dict], output_path: str):
    """
    Write results to a CSV file.
    
    Args:
        results: List of result dictionaries.
        output_path: Path to the output CSV file.
    """
    if not results:
        print(f"Warning: No results to write to {output_path}")
        return
    
    fieldnames = [
        "sequence_id",
        "source_file",
        "sequence",
        "length",
        "total_log_likelihood",
        "mean_log_likelihood",
    ]
    
    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Wrote {len(results)} results to: {output_path}")


def write_per_file_csvs(results: List[Dict], output_dir: str = "results"):
    """
    Write separate CSV files for each source file.
    
    Args:
        results: List of result dictionaries.
        output_dir: Directory to write per-file CSVs.
    """
    # Group by source file
    grouped = {}
    for r in results:
        source = r["source_file"]
        if source not in grouped:
            grouped[source] = []
        grouped[source].append(r)
    
    # Write each file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for source_file, file_results in grouped.items():
        csv_name = Path(source_file).stem + ".csv"
        csv_path = output_path / csv_name
        write_csv(file_results, str(csv_path))


def print_summary_stats(results: List[Dict]):
    """
    Print summary statistics for the results.
    
    Args:
        results: List of result dictionaries.
    """
    if not results:
        print("No results to summarize")
        return
    
    # Group by source file
    grouped = {}
    for r in results:
        source = r["source_file"]
        if source not in grouped:
            grouped[source] = []
        grouped[source].append(r)
    
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    for source_file in sorted(grouped.keys()):
        file_results = grouped[source_file]
        mean_lls = [r["mean_log_likelihood"] for r in file_results]
        
        mean_of_means = sum(mean_lls) / len(mean_lls)
        min_ll = min(mean_lls)
        max_ll = max(mean_lls)
        
        print(f"\n{source_file}:")
        print(f"  Count: {len(file_results)}")
        print(f"  Mean LL (mean): {mean_of_means:.4f}")
        print(f"  Mean LL (min):  {min_ll:.4f}")
        print(f"  Mean LL (max):  {max_ll:.4f}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate protein sequences using ESM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Score all files in Training data/
    python run_esm_eval.py --input "Training data/"
    
    # Score a specific file
    python run_esm_eval.py --input "Training data/denovo-train-Zn9.txt"
    
    # Use a different model
    python run_esm_eval.py --input "Training data/" --model esm2_t12_35M_UR50D
    
    # Custom output paths
    python run_esm_eval.py --input "Training data/" --output_csv my_results.csv --output_fig my_plot.png
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="Training data/",
        help="Path to input file or directory (default: Training data/)"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results_all.csv",
        help="Path for combined CSV output (default: results_all.csv)"
    )
    parser.add_argument(
        "--output_fig",
        type=str,
        default="esm_ll_histograms.png",
        help="Path for histogram figure output (default: esm_ll_histograms.png)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="esm2_t33_650M_UR50D",
        help="ESM model to use (default: esm2_t33_650M_UR50D)"
    )
    parser.add_argument(
        "--per_file_csv",
        action="store_true",
        help="Also write per-file CSVs to results/ directory"
    )
    parser.add_argument(
        "--skip_plot",
        action="store_true",
        help="Skip histogram generation"
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
        help="Batch size for GPU processing (default: 8, increase for more GPU memory)"
    )
    
    args = parser.parse_args()
    
    # Resolve input path - use as-is (relative to CWD or absolute)
    input_path = Path(args.input).resolve()
    
    print("=" * 60)
    print("ESM EVALUATION PIPELINE")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Model: {args.model}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Output Figure: {args.output_fig}")
    print("=" * 60)
    
    # Step 1: Load sequences
    print("\n[Step 1/4] Loading sequences...")
    sequences, sequence_ids, source_files = load_sequences(str(input_path))
    
    stats = get_sequence_stats(sequences)
    print(f"Sequence length stats: min={stats['min_length']}, max={stats['max_length']}, mean={stats['mean_length']:.1f}")
    
    # Step 2: Initialize scorer and compute scores
    print("\n[Step 2/4] Computing ESM log-likelihoods...")
    scorer = ESMScorer(model_name=args.model, device=args.device)
    
    scores = scorer.compute_log_likelihood_batch(
        sequences=sequences,
        sequence_ids=sequence_ids,
        source_files=source_files,
        show_progress=True,
        batch_size=args.batch_size
    )
    
    # Convert to dict format
    results = scorer.scores_to_dict_list(scores)
    
    # Step 3: Write CSV outputs
    print("\n[Step 3/4] Writing CSV outputs...")
    write_csv(results, args.output_csv)
    
    if args.per_file_csv:
        write_per_file_csvs(results)
    
    # Step 4: Generate histogram
    if not args.skip_plot:
        print("\n[Step 4/4] Generating histogram...")
        plot_log_likelihood_histograms(results, output_path=args.output_fig)
    else:
        print("\n[Step 4/4] Skipping histogram (--skip_plot)")
    
    # Print summary
    print_summary_stats(results)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

