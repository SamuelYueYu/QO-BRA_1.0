"""
ESM Evaluation Package

Evaluates protein sequence plausibility using Meta's ESM models.
Computes log-likelihood scores and generates analysis outputs.

Usage:
    from esm_eval import load_sequences, ESMScorer, plot_log_likelihood_histograms
    
    # Load sequences
    sequences, ids, sources = load_sequences("Training data/")
    
    # Score sequences
    scorer = ESMScorer(model_name="esm2_t33_650M_UR50D")
    scores = scorer.compute_log_likelihood_batch(sequences, ids, sources)
    
    # Convert to dict and plot
    results = scorer.scores_to_dict_list(scores)
    plot_log_likelihood_histograms(results, output_path="histogram.png")
"""

from .load_sequences import (
    load_sequences,
    load_sequences_from_file,
    get_sequence_stats,
    validate_sequence,
    clean_sequence,
)
from .esm_scoring import ESMScorer, SequenceScore
from .plot_histograms import (
    plot_log_likelihood_histograms,
    plot_from_csv,
    load_results_from_csv,
)
from .compute_distances import (
    load_distributions_from_csv,
    compute_pairwise_distances,
    write_distances_csv,
)

__all__ = [
    # Sequence loading
    "load_sequences",
    "load_sequences_from_file",
    "get_sequence_stats",
    "validate_sequence",
    "clean_sequence",
    # ESM scoring
    "ESMScorer",
    "SequenceScore",
    # Plotting
    "plot_log_likelihood_histograms",
    "plot_from_csv",
    "load_results_from_csv",
    # Distance computation
    "load_distributions_from_csv",
    "compute_pairwise_distances",
    "write_distances_csv",
]

