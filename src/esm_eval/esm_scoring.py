"""
ESM model scoring for protein sequences.

Computes log-likelihood scores using Meta's ESM (Evolutionary Scale Modeling) models.
"""

import torch
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class SequenceScore:
    """Container for sequence scoring results."""
    sequence_id: str
    source_file: str
    sequence: str
    length: int
    total_log_likelihood: float
    mean_log_likelihood: float


class ESMScorer:
    """
    ESM model wrapper for computing protein sequence log-likelihoods.
    
    Uses masked language modeling to compute pseudo-log-likelihood scores,
    which estimate how plausible each sequence is under the ESM model.
    
    Attributes:
        model_name: Name of the ESM model to use.
        device: PyTorch device (cuda or cpu).
        model: The loaded ESM model.
        alphabet: The ESM alphabet for tokenization.
        batch_converter: Converts sequences to model input format.
    """
    
    AVAILABLE_MODELS = [
        "esm2_t6_8M_UR50D",
        "esm2_t12_35M_UR50D",
        "esm2_t30_150M_UR50D",
        "esm2_t33_650M_UR50D",
        "esm2_t36_3B_UR50D",
        "esm2_t48_15B_UR50D",
    ]
    
    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        device: Optional[str] = None
    ):
        """
        Initialize the ESM scorer.
        
        Args:
            model_name: Name of the ESM model to load.
            device: Device to use ('cuda', 'cpu', or None for auto-detect).
        """
        self.model_name = model_name
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Initializing ESM scorer")
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the ESM model and alphabet."""
        import esm
        
        print("  Loading model...")
        
        # Get the model loader function
        model_loader = getattr(esm.pretrained, self.model_name, None)
        if model_loader is None:
            raise ValueError(
                f"Unknown model: {self.model_name}. "
                f"Available: {self.AVAILABLE_MODELS}"
            )
        
        self.model, self.alphabet = model_loader()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Get special token indices
        self.mask_idx = self.alphabet.mask_idx
        self.padding_idx = self.alphabet.padding_idx
        self.cls_idx = self.alphabet.cls_idx
        self.eos_idx = self.alphabet.eos_idx
        
        print("  Model loaded successfully")
    
    @torch.no_grad()
    def compute_log_likelihood(
        self,
        sequence: str,
        sequence_id: str = "seq",
        source_file: str = ""
    ) -> SequenceScore:
        """
        Compute the pseudo-log-likelihood for a single sequence.
        
        GPU-OPTIMIZED: Uses vectorized operations instead of Python loops.
        Uses single-pass inference: run one forward pass and compute the
        log probability of each true amino acid at its position.
        
        Args:
            sequence: Amino acid sequence string.
            sequence_id: Identifier for this sequence.
            source_file: Source file name.
        
        Returns:
            SequenceScore with computed log-likelihoods.
        """
        # Prepare input
        data = [(sequence_id, sequence)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)
        
        seq_len = len(sequence)
        
        # Single forward pass - no masking needed
        output = self.model(tokens)
        logits = output["logits"]
        
        # Compute log probabilities for all positions at once
        log_probs = torch.log_softmax(logits[0], dim=-1)
        
        if seq_len > 0:
            # VECTORIZED: Sum log probabilities of true tokens in one operation
            # (positions 1 to seq_len, skipping CLS)
            positions = torch.arange(1, seq_len + 1, device=self.device)
            true_tokens = tokens[0, 1:seq_len + 1]
            total_log_likelihood = log_probs[positions, true_tokens].sum().item()
            mean_log_likelihood = total_log_likelihood / seq_len
        else:
            total_log_likelihood = 0.0
            mean_log_likelihood = 0.0
        
        return SequenceScore(
            sequence_id=sequence_id,
            source_file=source_file,
            sequence=sequence,
            length=seq_len,
            total_log_likelihood=total_log_likelihood,
            mean_log_likelihood=mean_log_likelihood,
        )
    
    @torch.no_grad()
    def compute_log_likelihood_batch(
        self,
        sequences: List[str],
        sequence_ids: List[str],
        source_files: List[str],
        show_progress: bool = True,
        batch_size: int = 8
    ) -> List[SequenceScore]:
        """
        Compute log-likelihoods for multiple sequences with GPU batching.
        
        GPU-OPTIMIZED: Uses fully vectorized operations for log-likelihood
        computation instead of Python loops over positions.
        
        Args:
            sequences: List of amino acid sequences.
            sequence_ids: List of sequence identifiers.
            source_files: List of source file names.
            show_progress: Whether to show progress updates.
            batch_size: Number of sequences to process per GPU batch.
        
        Returns:
            List of SequenceScore objects.
        """
        n_sequences = len(sequences)
        results = []
        
        if show_progress:
            print(f"\nScoring {n_sequences} sequences (batch_size={batch_size})...")
        
        # Process in batches
        for batch_start in range(0, n_sequences, batch_size):
            batch_end = min(batch_start + batch_size, n_sequences)
            
            batch_seqs = sequences[batch_start:batch_end]
            batch_ids = sequence_ids[batch_start:batch_end]
            batch_sources = source_files[batch_start:batch_end]
            current_batch_size = len(batch_seqs)
            
            # Prepare batch data
            data = [(seq_id, seq) for seq_id, seq in zip(batch_ids, batch_seqs)]
            _, _, tokens = self.batch_converter(data)
            tokens = tokens.to(self.device)
            
            # Single forward pass for entire batch
            output = self.model(tokens)
            logits = output["logits"]
            
            # VECTORIZED: Compute log_softmax for all positions at once
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Compute scores for each sequence in batch using vectorized operations
            for i, (seq, seq_id, source) in enumerate(zip(batch_seqs, batch_ids, batch_sources)):
                seq_len = len(seq)
                
                if seq_len > 0:
                    # VECTORIZED: Extract log probs for true tokens in one operation
                    # tokens[i, 1:seq_len+1] are the true token indices (skip CLS at 0)
                    # Use advanced indexing to gather all log probs at once
                    positions = torch.arange(1, seq_len + 1, device=self.device)
                    true_tokens = tokens[i, 1:seq_len + 1]
                    
                    # Gather log probs for true tokens: log_probs[i, positions, true_tokens]
                    total_ll = log_probs[i, positions, true_tokens].sum().item()
                    mean_ll = total_ll / seq_len
                else:
                    total_ll = 0.0
                    mean_ll = 0.0
                
                results.append(SequenceScore(
                    sequence_id=seq_id,
                    source_file=source,
                    sequence=seq,
                    length=seq_len,
                    total_log_likelihood=total_ll,
                    mean_log_likelihood=mean_ll,
                ))
            
            if show_progress and (batch_end % 500 == 0 or batch_end == n_sequences):
                print(f"  Processed {batch_end}/{n_sequences} sequences")
        
        if show_progress:
            print(f"  Completed scoring {n_sequences} sequences")
        
        return results
    
    def scores_to_dict_list(self, scores: List[SequenceScore]) -> List[Dict]:
        """
        Convert list of SequenceScore objects to list of dictionaries.
        
        Args:
            scores: List of SequenceScore objects.
        
        Returns:
            List of dictionaries suitable for CSV export.
        """
        return [
            {
                "sequence_id": s.sequence_id,
                "source_file": s.source_file,
                "sequence": s.sequence,
                "length": s.length,
                "total_log_likelihood": s.total_log_likelihood,
                "mean_log_likelihood": s.mean_log_likelihood,
            }
            for s in scores
        ]

