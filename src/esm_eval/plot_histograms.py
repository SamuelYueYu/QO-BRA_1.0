"""
Histogram plotting for ESM evaluation results.

Creates publication-quality histograms of log-likelihood distributions.
"""

import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt


def load_results_from_csv(csv_path: str) -> List[Dict]:
    """
    Load results from a CSV file.
    
    Args:
        csv_path: Path to the results CSV file.
    
    Returns:
        List of dictionaries with sequence scoring results.
    """
    results = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "sequence_id": row["sequence_id"],
                "source_file": row["source_file"],
                "sequence": row["sequence"],
                "length": int(row["length"]),
                "total_log_likelihood": float(row["total_log_likelihood"]),
                "mean_log_likelihood": float(row["mean_log_likelihood"]),
            })
    return results


def group_by_source_file(results: List[Dict]) -> Dict[str, List[float]]:
    """
    Group mean log-likelihoods by source file.
    
    Args:
        results: List of result dictionaries.
    
    Returns:
        Dictionary mapping source_file -> list of mean_log_likelihood values.
    """
    grouped = {}
    for r in results:
        source = r["source_file"]
        if source not in grouped:
            grouped[source] = []
        grouped[source].append(r["mean_log_likelihood"])
    return grouped


def compute_common_bins(
    grouped_data: Dict[str, List[float]],
    n_bins: int = 50
) -> Tuple[float, float, int]:
    """
    Compute common bin parameters for all histograms.
    
    Args:
        grouped_data: Dictionary of source_file -> values.
        n_bins: Number of bins to use.
    
    Returns:
        Tuple of (min_value, max_value, n_bins).
    """
    all_values = []
    for values in grouped_data.values():
        all_values.extend(values)
    
    if not all_values:
        return -10, 0, n_bins
    
    min_val = min(all_values)
    max_val = max(all_values)
    
    # Add small padding
    padding = (max_val - min_val) * 0.05
    return min_val - padding, max_val + padding, n_bins


def plot_log_likelihood_histograms(
    results: List[Dict],
    output_path: str = "esm_ll_histograms.png",
    n_bins: int = 50,
    figsize: Tuple[float, float] = (5, 5),
    dpi: int = 500
):
    """
    Create overlaid histograms of mean log-likelihood distributions.
    
    Creates one step histogram per source file, all overlaid on the same axes
    with matching bins and x-axis limits.
    
    Args:
        results: List of result dictionaries from ESM scoring.
        output_path: Path to save the output figure.
        n_bins: Number of bins for the histograms.
        figsize: Figure size in inches (width, height).
        dpi: Resolution of the output figure.
    """
    # Group data by source file
    grouped = group_by_source_file(results)
    
    if not grouped:
        print("Warning: No data to plot")
        return
    
    # Compute common bins
    bin_min, bin_max, n_bins = compute_common_bins(grouped, n_bins)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Define colors for different source files
    colors = plt.cm.tab10.colors
    
    # Custom ordering: train -> random -> cnn -> qobra (QOBRA generated)
    def sort_key(filename):
        stem = Path(filename).stem.lower()
        if "train" in stem:
            return (0, stem)
        elif "random" in stem:
            return (1, stem)
        elif "cnn" in stem:
            return (2, stem)
        elif "qobra" in stem:
            return (3, stem)
        elif "model" in stem:
            return (4, stem)
        else:
            return (5, stem)
    
    source_files = sorted(grouped.keys(), key=sort_key)
    
    # Plot each source file
    for i, source_file in enumerate(source_files):
        values = grouped[source_file]
        color = colors[i % len(colors)]
        
        # Remove .txt extension for cleaner legend
        label = Path(source_file).stem
        
        ax.hist(
            values,
            bins=n_bins,
            range=(bin_min, bin_max),
            histtype="stepfilled",
            linewidth=1.5,
            label=label,
            color=color,
            alpha=0.5,
            edgecolor=color,
            density=True
        )
    
    # Labels and legend
    ax.set_xlabel("Mean log-likelihood per residue")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left", frameon=True, fontsize=8)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Saved histogram to: {output_path}")


def plot_from_csv(
    csv_path: str,
    output_path: str = "esm_ll_histograms.png",
    n_bins: int = 50,
    figsize: Tuple[float, float] = (5, 5),
    dpi: int = 500
):
    """
    Load results from CSV and create histogram plot.
    
    Args:
        csv_path: Path to the results CSV file.
        output_path: Path to save the output figure.
        n_bins: Number of bins for the histograms.
        figsize: Figure size in inches.
        dpi: Resolution of the output figure.
    """
    results = load_results_from_csv(csv_path)
    plot_log_likelihood_histograms(
        results,
        output_path=output_path,
        n_bins=n_bins,
        figsize=figsize,
        dpi=dpi,
    )

