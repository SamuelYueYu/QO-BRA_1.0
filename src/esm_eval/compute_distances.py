#!/usr/bin/env python3
"""
Compute distributional distances between ESM log-likelihood distributions.

This module provides post-hoc analysis of ESM evaluation results by computing
statistical distances between sequence sets identified by their source files.

Metrics computed:
    1. Kolmogorov-Smirnov (KS) statistic and p-value
       - Measures the maximum difference between cumulative distribution functions (CDFs)
       - Sensitive to differences in shape, location, and scale of distributions
       - KS statistic D ranges from 0 (identical) to 1 (completely non-overlapping)
       - p-value tests the null hypothesis that both samples come from the same distribution
    
    2. Wasserstein distance (Earth Mover's Distance)
       - Measures the minimum "work" needed to transform one distribution into another
       - Quantifies the distance in value space between distributions
       - Sensitive to the actual values, not just ranks
       - Units are the same as the input data (log-likelihood units)

Interpretation notes:
    - These metrics quantify distributional distinctness but do NOT imply biological validity
    - A low distance suggests similar ESM log-likelihood profiles
    - A high distance indicates the model assigns different plausibility to the sequence sets
    - Neither metric accounts for sequence diversity, length distribution, or functional properties

Usage:
    python compute_distances.py --input_csv results_all.csv --output_csv esm_distribution_distances.csv
"""

import argparse
import csv
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


def load_distributions_from_csv(csv_path: str) -> Dict[str, np.ndarray]:
    """
    Load mean log-likelihood distributions grouped by source file.
    
    Reads an ESM evaluation CSV and extracts mean_log_likelihood values
    for each distinct source_file, returning them as numpy arrays for
    efficient statistical computation.
    
    Args:
        csv_path: Path to the ESM evaluation results CSV file.
                  Must contain columns: source_file, mean_log_likelihood
    
    Returns:
        Dictionary mapping source_file names to numpy arrays of
        mean_log_likelihood values.
    
    Raises:
        FileNotFoundError: If the CSV file does not exist.
        KeyError: If required columns are missing from the CSV.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    
    distributions: Dict[str, List[float]] = {}
    
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        
        # Validate required columns
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV file: {csv_path}")
        
        required_cols = {"source_file", "mean_log_likelihood"}
        missing_cols = required_cols - set(reader.fieldnames)
        if missing_cols:
            raise KeyError(f"Missing required columns in CSV: {missing_cols}")
        
        for row in reader:
            source = row["source_file"]
            value = float(row["mean_log_likelihood"])
            
            if source not in distributions:
                distributions[source] = []
            distributions[source].append(value)
    
    # Convert to numpy arrays
    return {source: np.array(values) for source, values in distributions.items()}


def compute_pairwise_distances(
    distributions: Dict[str, np.ndarray]
) -> List[Dict]:
    """
    Compute KS and Wasserstein distances for all pairs of distributions.
    
    For each unique pair of source files, computes:
        - Kolmogorov-Smirnov statistic (D) and p-value
        - Wasserstein distance
    
    The order of pairs is deterministic: sorted alphabetically by source file name.
    Gracefully handles unequal sample sizes between groups.
    
    Args:
        distributions: Dictionary mapping source_file names to numpy arrays
                       of mean_log_likelihood values.
    
    Returns:
        List of dictionaries, each containing:
            - group_1: First source file name
            - group_2: Second source file name
            - ks_statistic: KS statistic (D), maximum CDF difference
            - ks_pvalue: Two-sample KS test p-value
            - wasserstein_distance: Earth mover's distance between distributions
    """
    results = []
    
    # Get sorted list of source files for deterministic ordering
    source_files = sorted(distributions.keys())
    
    # Compute distances for all unique pairs
    for group_1, group_2 in combinations(source_files, 2):
        values_1 = distributions[group_1]
        values_2 = distributions[group_2]
        
        # Kolmogorov-Smirnov test
        # ks_2samp handles unequal sample sizes
        ks_stat, ks_pval = ks_2samp(values_1, values_2)
        
        # Wasserstein distance (Earth Mover's Distance)
        # Uses raw values without normalization
        w_dist = wasserstein_distance(values_1, values_2)
        
        results.append({
            "group_1": group_1,
            "group_2": group_2,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pval,
            "wasserstein_distance": w_dist,
        })
    
    return results


def build_distance_matrices(
    results: List[Dict],
    source_files: List[str]
) -> Dict[str, Dict[Tuple[str, str], float]]:
    """
    Build symmetric distance matrices from pairwise results.
    
    Args:
        results: List of pairwise distance dictionaries.
        source_files: Sorted list of source file names.
    
    Returns:
        Dictionary mapping metric name to a dict of (group1, group2) -> value.
    """
    metrics = ["ks_statistic", "ks_pvalue", "wasserstein_distance"]
    matrices = {m: {} for m in metrics}
    
    # Fill in pairwise values (symmetric)
    for r in results:
        g1, g2 = r["group_1"], r["group_2"]
        for metric in metrics:
            val = r[metric]
            matrices[metric][(g1, g2)] = val
            matrices[metric][(g2, g1)] = val  # Symmetric
    
    # Fill diagonal
    for metric in metrics:
        for g in source_files:
            if metric == "ks_pvalue":
                matrices[metric][(g, g)] = 1.0  # p-value of 1 for identical distributions
            else:
                matrices[metric][(g, g)] = 0.0  # Zero distance to self
    
    return matrices


def write_distances_csv(
    results: List[Dict],
    source_files: List[str],
    output_path: str
):
    """
    Write distance results to a CSV file in square matrix format.
    
    Output format has three stacked matrices (one per metric), where
    rows and columns represent source files and cells contain distances.
    
    Args:
        results: List of distance result dictionaries.
        source_files: Sorted list of source file names.
        output_path: Path for the output CSV file.
    """
    if not results:
        print(f"Warning: No results to write to {output_path}")
        return
    
    # Build symmetric matrices
    matrices = build_distance_matrices(results, source_files)
    
    # Clean names for display (remove .txt extension)
    clean_names = [Path(f).stem for f in source_files]
    
    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    metrics = ["ks_statistic", "ks_pvalue", "wasserstein_distance"]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        for metric_idx, metric in enumerate(metrics):
            # Write metric header row
            writer.writerow([metric] + clean_names)
            
            # Write each row of the matrix
            for i, source in enumerate(source_files):
                row_name = clean_names[i]
                row_values = []
                for other_source in source_files:
                    val = matrices[metric][(source, other_source)]
                    # Format based on metric type
                    if metric == "ks_pvalue" and val < 1e-10:
                        row_values.append(f"{val:.2e}")
                    elif metric == "ks_pvalue":
                        row_values.append(f"{val:.6f}")
                    else:
                        row_values.append(f"{val:.6f}")
                writer.writerow([row_name] + row_values)
            
            # Add blank line between matrices (except after last)
            if metric_idx < len(metrics) - 1:
                writer.writerow([])
    
    n_groups = len(source_files)
    print(f"Wrote {n_groups}x{n_groups} distance matrices (3 metrics) to: {output_path}")


def format_group_name(filename: str) -> str:
    """
    Format a source filename for display (remove extension, clean up).
    
    Args:
        filename: Source file name (e.g., "denovo-qobra-Zn9.txt")
    
    Returns:
        Cleaned display name (e.g., "denovo-qobra-Zn9")
    """
    return Path(filename).stem


def print_summary(results: List[Dict], distributions: Dict[str, np.ndarray]):
    """
    Print a human-readable summary of distance results to stdout.
    
    Args:
        results: List of distance result dictionaries.
        distributions: Original distributions for sample size info.
    """
    print("\n" + "=" * 60)
    print("ESM LOG-LIKELIHOOD DISTRIBUTION DISTANCES")
    print("=" * 60)
    
    # Print sample sizes
    print("\nSample sizes:")
    for source in sorted(distributions.keys()):
        name = format_group_name(source)
        n = len(distributions[source])
        print(f"  {name}: n={n}")
    
    # Print KS statistics
    print("\n" + "-" * 60)
    print("Kolmogorov-Smirnov Statistics (D, p-value)")
    print("-" * 60)
    print("  Measures maximum difference between CDFs (shape difference)")
    print()
    
    for r in results:
        name_1 = format_group_name(r["group_1"])
        name_2 = format_group_name(r["group_2"])
        ks_stat = r["ks_statistic"]
        ks_pval = r["ks_pvalue"]
        
        # Format p-value with scientific notation if very small
        if ks_pval < 1e-10:
            pval_str = f"{ks_pval:.2e}"
        else:
            pval_str = f"{ks_pval:.6f}"
        
        print(f"  KS({name_1}, {name_2}) = {ks_stat:.4f}  (p = {pval_str})")
    
    # Print Wasserstein distances
    print("\n" + "-" * 60)
    print("Wasserstein Distances (Earth Mover's Distance)")
    print("-" * 60)
    print("  Measures distance in value space (distribution shift)")
    print()
    
    for r in results:
        name_1 = format_group_name(r["group_1"])
        name_2 = format_group_name(r["group_2"])
        w_dist = r["wasserstein_distance"]
        
        print(f"  Wasserstein({name_1}, {name_2}) = {w_dist:.6f}")
    
    print("\n" + "=" * 60)
    print("Note: These metrics quantify distinctness but do NOT imply biological validity.")
    print("=" * 60 + "\n")


def main():
    """Main entry point for the compute_distances CLI."""
    parser = argparse.ArgumentParser(
        description="Compute distributional distances between ESM log-likelihood distributions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compute distances from ESM results
    python compute_distances.py --input_csv results_all.csv --output_csv esm_distribution_distances.csv
    
    # Use custom file paths
    python compute_distances.py --input_csv eval_data/results_all.csv --output_csv analysis/distances.csv

Metrics:
    KS statistic:   Maximum difference between CDFs (0 to 1)
    Wasserstein:    Earth mover's distance in log-likelihood units
        """
    )
    
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV with ESM evaluation results (must have source_file and mean_log_likelihood columns)"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="esm_distribution_distances.csv",
        help="Path for output CSV with pairwise distances (default: esm_distribution_distances.csv)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("COMPUTING DISTRIBUTION DISTANCES")
    print("=" * 60)
    print(f"Input CSV:  {args.input_csv}")
    print(f"Output CSV: {args.output_csv}")
    print("=" * 60)
    
    # Load distributions
    print("\n[Step 1/3] Loading distributions from CSV...")
    distributions = load_distributions_from_csv(args.input_csv)
    
    n_groups = len(distributions)
    n_sequences = sum(len(v) for v in distributions.values())
    print(f"  Loaded {n_sequences} sequences across {n_groups} groups")
    
    # Compute distances
    print("\n[Step 2/3] Computing pairwise distances...")
    n_pairs = n_groups * (n_groups - 1) // 2
    print(f"  Computing {n_pairs} pairwise comparisons")
    
    results = compute_pairwise_distances(distributions)
    
    # Write output
    print("\n[Step 3/3] Writing results...")
    source_files = sorted(distributions.keys())
    write_distances_csv(results, source_files, args.output_csv)
    
    # Print summary
    print_summary(results, distributions)
    
    print("Done!")


if __name__ == "__main__":
    main()

