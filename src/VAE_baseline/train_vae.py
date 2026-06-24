#!/usr/bin/env python3
"""
PyTorch Lightning VAE - Training Script

A toy VAE baseline using DISCRETE tokens with Cross-Entropy loss.
This is a proper sequence VAE for fair comparison with QOBRA.

Usage:
    python train_vae.py --metal Ca
    python train_vae.py --metal Mg --epochs 500 --batch_size 64
    python train_vae.py --metal Zn --latent_dim 128

Data layout:
    Training data/
        Ca_bind/
        Mg_bind/
        Zn_bind/
"""

import os
import argparse
import random
import numpy as np
import torch

# Enable Tensor Cores on A100/H100 GPUs for better performance
torch.set_float32_matmul_precision('high')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from data import (load_data, get_dataloader, ids_to_sequence,
                  get_effective_sequence, compare_sequences_effective)
from metrics import (compute_nuv, compute_reconstruction_rate,
                     compute_all_relative_ratios, compute_aa_frequencies,
                     compute_sequence_lengths, compute_binding_site_counts,
                     compute_chain_counts)
from visualize import save_architecture_diagrams
from models import (DiscreteSequenceVAE_MLP, DiscreteSequenceVAE_CNN, 
                    DiscreteSequenceVAE_Transformer)


# =============================================================================
# RR DISTRIBUTION PLOTTING
# =============================================================================

def load_reference_sequences(filepath: str) -> list:
    """
    Load reference/de novo sequences from a file.
    
    Format: sequences separated by lines of asterisks (*).
    First line is a header.
    
    Parameters:
    - filepath: Path to the de novo sequences file
    
    Returns:
    - sequences: List of sequence strings
    """
    if not os.path.exists(filepath):
        return []
    
    sequences = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip header, empty lines, and separator lines
            if not line or line.startswith('*') or line.lower().startswith('de novo'):
                continue
            # Append X to sequences ending with : to form proper :X terminator
            if line.endswith(':'):
                line = line + 'X'
            # This is a sequence line
            sequences.append(line)
    
    return sequences


def plot_rr_distributions(training_seqs: list, generated_seqs: list, save_path: str,
                          reference_seqs: list = None, reference_label: str = 'Reference',
                          random_baseline_seqs: list = None, random_baseline_label: str = 'Random Baseline',
                          max_len: int = 512):
    """
    Plot the four RR distribution comparisons (QOBRA-style).
    
    Parameters:
    - training_seqs: Training set sequences
    - generated_seqs: VAE-generated sequences
    - save_path: Path to save the plot
    - reference_seqs: Optional third dataset (e.g., QOBRA de novo)
    - reference_label: Label for the reference dataset
    - random_baseline_seqs: Optional fourth dataset (e.g., random token sampling)
    - random_baseline_label: Label for the random baseline dataset
    - max_len: Maximum sequence length for binning
    """
    from collections import Counter
    
    has_reference = reference_seqs is not None and len(reference_seqs) > 0
    has_random = random_baseline_seqs is not None and len(random_baseline_seqs) > 0
    
    # (a) Token frequencies (QOBRA-style: normalized per-sequence, then averaged)
    train_token_freq_dict = compute_aa_frequencies(training_seqs)
    gen_token_freq_dict = compute_aa_frequencies(generated_seqs)
    ref_token_freq_dict = compute_aa_frequencies(reference_seqs) if has_reference else {}
    rand_token_freq_dict = compute_aa_frequencies(random_baseline_seqs) if has_random else {}
    
    # Sort tokens: regular AAs first, then AA+, then : and :X
    all_tokens = sorted(
        set(train_token_freq_dict.keys()) | set(gen_token_freq_dict.keys()) | 
        set(ref_token_freq_dict.keys()) | set(rand_token_freq_dict.keys()),
        key=lambda x: ('+' in x and ':' not in x, x == ':', x == ':X', x)
    )
    
    train_token_freq = [train_token_freq_dict.get(t, 0) for t in all_tokens]
    gen_token_freq = [gen_token_freq_dict.get(t, 0) for t in all_tokens]
    ref_token_freq = [ref_token_freq_dict.get(t, 0) for t in all_tokens]
    rand_token_freq = [rand_token_freq_dict.get(t, 0) for t in all_tokens]
    
    # (b) Chain counts
    train_chains = compute_chain_counts(training_seqs)
    gen_chains = compute_chain_counts(generated_seqs)
    ref_chains = compute_chain_counts(reference_seqs) if has_reference else []
    rand_chains = compute_chain_counts(random_baseline_seqs) if has_random else []
    
    # (c) Sequence lengths
    train_lengths = compute_sequence_lengths(training_seqs)
    gen_lengths = compute_sequence_lengths(generated_seqs)
    ref_lengths = compute_sequence_lengths(reference_seqs) if has_reference else []
    rand_lengths = compute_sequence_lengths(random_baseline_seqs) if has_random else []
    
    # (d) Binding site counts
    train_binding = compute_binding_site_counts(training_seqs)
    gen_binding = compute_binding_site_counts(generated_seqs)
    ref_binding = compute_binding_site_counts(reference_seqs) if has_reference else []
    rand_binding = compute_binding_site_counts(random_baseline_seqs) if has_random else []
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    train_color = '#1f77b4'   # Blue
    gen_color = '#ff7f0e'     # Orange
    ref_color = '#2ca02c'     # Green
    rand_color = '#9467bd'    # Purple
    
    # (a) Token frequency
    ax_a = fig.add_subplot(2, 1, 1)
    x = np.arange(len(all_tokens))
    n_bars = 2 + (1 if has_reference else 0) + (1 if has_random else 0)
    width = 0.8 / n_bars
    
    # Calculate bar positions based on how many datasets we have
    positions = []
    labels_colors = [('Training', train_color, train_token_freq)]
    positions.append(-width * (n_bars - 1) / 2)
    
    labels_colors.append(('VAE De novo', gen_color, gen_token_freq))
    positions.append(positions[-1] + width)
    
    if has_reference:
        labels_colors.append((reference_label, ref_color, ref_token_freq))
        positions.append(positions[-1] + width)
    
    if has_random:
        labels_colors.append((random_baseline_label, rand_color, rand_token_freq))
        positions.append(positions[-1] + width)
    
    for i, (label, color, freqs) in enumerate(labels_colors):
        ax_a.bar(x + positions[i], freqs, width, label=label, color=color, alpha=0.8)
    
    ax_a.set_xlabel('AA/AA+', fontsize=11)
    ax_a.set_ylabel('Frequency', fontsize=11)
    ax_a.set_title('(a) Frequency of occurrence of each AA', fontsize=12, fontweight='bold')
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(all_tokens, rotation=45, ha='right', fontsize=8)
    ax_a.legend(loc='upper right', fontsize=10)
    ax_a.grid(True, alpha=0.3, axis='y')
    
    # (b) Chain numbers
    ax_b = fig.add_subplot(2, 3, 4)
    all_chain_lists = [train_chains, gen_chains]
    if has_reference:
        all_chain_lists.append(ref_chains)
    if has_random:
        all_chain_lists.append(rand_chains)
    max_chains = max(max(c) if c else 0 for c in all_chain_lists) + 1
    bins_chains = np.arange(-0.5, max_chains + 1.5, 1)
    ax_b.hist(train_chains, bins=bins_chains, density=True, alpha=0.7, label='Training', color=train_color)
    ax_b.hist(gen_chains, bins=bins_chains, density=True, alpha=0.7, label='VAE De novo', color=gen_color)
    if has_reference:
        ax_b.hist(ref_chains, bins=bins_chains, density=True, alpha=0.7, label=reference_label, color=ref_color)
    if has_random:
        ax_b.hist(rand_chains, bins=bins_chains, density=True, alpha=0.7, label=random_baseline_label, color=rand_color)
    ax_b.set_xlabel('Count', fontsize=11)
    ax_b.set_ylabel('Frequency', fontsize=11)
    ax_b.set_title('(b) Distribution of chain numbers', fontsize=12, fontweight='bold')
    ax_b.legend(fontsize=9)
    ax_b.set_yscale('log')
    ax_b.grid(True, alpha=0.3, axis='y')
    
    # (c) Length - QOBRA uses 8-amino-acid bins
    ax_c = fig.add_subplot(2, 3, 5)
    bin_size = 8
    bins_len = [i * bin_size for i in range(max_len // bin_size + 2)]  # QOBRA style
    
    # Bin the data manually for bar plot (QOBRA style)
    train_len_counts = Counter(train_lengths)
    gen_len_counts = Counter(gen_lengths)
    ref_len_counts = Counter(ref_lengths) if has_reference else Counter()
    rand_len_counts = Counter(rand_lengths) if has_random else Counter()
    
    # Compute binned counts
    train_binned = [0] * (len(bins_len) - 1)
    gen_binned = [0] * (len(bins_len) - 1)
    ref_binned = [0] * (len(bins_len) - 1) if has_reference else []
    rand_binned = [0] * (len(bins_len) - 1) if has_random else []
    
    for i in range(len(bins_len) - 1):
        for k in train_len_counts.keys():
            if bins_len[i] < k <= bins_len[i + 1]:
                train_binned[i] += train_len_counts[k]
        for k in gen_len_counts.keys():
            if bins_len[i] < k <= bins_len[i + 1]:
                gen_binned[i] += gen_len_counts[k]
        if has_reference:
            for k in ref_len_counts.keys():
                if bins_len[i] < k <= bins_len[i + 1]:
                    ref_binned[i] += ref_len_counts[k]
        if has_random:
            for k in rand_len_counts.keys():
                if bins_len[i] < k <= bins_len[i + 1]:
                    rand_binned[i] += rand_len_counts[k]
    
    # Normalize to frequencies
    train_binned_freq = [c / len(train_lengths) if train_lengths else 0 for c in train_binned]
    gen_binned_freq = [c / len(gen_lengths) if gen_lengths else 0 for c in gen_binned]
    ref_binned_freq = [c / len(ref_lengths) if ref_lengths else 0 for c in ref_binned] if has_reference else []
    rand_binned_freq = [c / len(rand_lengths) if rand_lengths else 0 for c in rand_binned] if has_random else []
    
    bin_widths = np.diff(bins_len)
    ax_c.bar(bins_len[:-1], train_binned_freq, width=bin_widths, align='edge', 
             alpha=1.0, label='Training', color=train_color)
    ax_c.bar(bins_len[:-1], gen_binned_freq, width=bin_widths, align='edge', 
             alpha=0.6, label='VAE De novo', color=gen_color)
    if has_reference:
        ax_c.bar(bins_len[:-1], ref_binned_freq, width=bin_widths, align='edge', 
                 alpha=0.6, label=reference_label, color=ref_color)
    if has_random:
        ax_c.bar(bins_len[:-1], rand_binned_freq, width=bin_widths, align='edge', 
                 alpha=0.6, label=random_baseline_label, color=rand_color)
    ax_c.set_xlabel('Length', fontsize=11)
    ax_c.set_ylabel('Frequency', fontsize=11)
    ax_c.set_title('(c) Distribution of length', fontsize=12, fontweight='bold')
    ax_c.legend(fontsize=9)
    ax_c.set_yscale('log')
    ax_c.grid(True, alpha=0.3, axis='y')
    
    # (d) Binding sites
    ax_d = fig.add_subplot(2, 3, 6)
    all_bind_lists = [train_binding, gen_binding]
    if has_reference:
        all_bind_lists.append(ref_binding)
    if has_random:
        all_bind_lists.append(rand_binding)
    max_binding = max(max(b) if b else 0 for b in all_bind_lists) + 1
    bins_binding = np.arange(-0.5, max_binding + 1.5, 1)
    ax_d.hist(train_binding, bins=bins_binding, density=True, alpha=0.7, label='Training', color=train_color)
    ax_d.hist(gen_binding, bins=bins_binding, density=True, alpha=0.7, label='VAE De novo', color=gen_color)
    if has_reference:
        ax_d.hist(ref_binding, bins=bins_binding, density=True, alpha=0.7, label=reference_label, color=ref_color)
    if has_random:
        ax_d.hist(rand_binding, bins=bins_binding, density=True, alpha=0.7, label=random_baseline_label, color=rand_color)
    ax_d.set_xlabel('Count', fontsize=11)
    ax_d.set_ylabel('Frequency', fontsize=11)
    ax_d.set_title('(d) Distribution of binding sites', fontsize=12, fontweight='bold')
    ax_d.legend(fontsize=9)
    ax_d.set_yscale('log')
    ax_d.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"RR distributions plot saved to: {save_path}")


# =============================================================================
# REAL-TIME LOSS PLOTTING CALLBACK
# =============================================================================

class RealTimePlotCallback(pl.Callback):
    """Callback to plot training loss vs epochs in real time."""
    
    def __init__(self, save_path: str, update_every: int = 1):
        super().__init__()
        self.save_path = save_path
        self.update_every = update_every
        self.losses = []
        self.recons = []
        self.kls = []
        self.betas = []
    
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if 'train_loss' in metrics:
            self.losses.append(float(metrics['train_loss']))
        if 'train_recon' in metrics:
            self.recons.append(float(metrics['train_recon']))
        if 'train_kl' in metrics:
            self.kls.append(float(metrics['train_kl']))
        if 'beta' in metrics:
            self.betas.append(float(metrics['beta']))
        
        if trainer.current_epoch % self.update_every == 0 and len(self.losses) > 0:
            self._update_plot()
    
    def _update_plot(self):
        epochs = range(1, len(self.losses) + 1)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        axes[0, 0].plot(epochs, self.losses, 'b-', linewidth=1)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        if self.recons:
            axes[0, 1].plot(epochs, self.recons, 'g-', linewidth=1)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('CE Loss')
            axes[0, 1].set_title('Reconstruction Loss (Cross-Entropy)')
            axes[0, 1].grid(True, alpha=0.3)
        
        if self.kls:
            axes[1, 0].plot(epochs, self.kls, 'r-', linewidth=1)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('KL Loss')
            axes[1, 0].set_title('KL Divergence')
            axes[1, 0].grid(True, alpha=0.3)
        
        if self.betas:
            axes[1, 1].plot(epochs, self.betas, 'm-', linewidth=1)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Beta')
            axes[1, 1].set_title('Beta Schedule')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a discrete sequence VAE baseline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required
    parser.add_argument('--metal', type=str, required=True, 
                        choices=['Ca', 'Mg', 'Zn'],
                        help='Metal type: Ca, Mg, or Zn')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='Training data')
    parser.add_argument('--max_len', type=int, default=512,
                        help='Sequence length (fixed at 512)')
    parser.add_argument('--cap', type=int, default=6000,
                        help='Max training sequences')
    
    # Model
    parser.add_argument('--model_type', type=str, default='mlp',
                        choices=['mlp', 'cnn', 'transformer'],
                        help='Model architecture: mlp, cnn, or transformer')
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='Token embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden layer dimension (MLP) or channels (CNN)')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent space dimension')
    
    # Transformer-specific
    parser.add_argument('--n_heads', type=int, default=4,
                        help='Number of attention heads (transformer)')
    parser.add_argument('--n_encoder_layers', type=int, default=4,
                        help='Number of transformer encoder layers')
    parser.add_argument('--n_decoder_layers', type=int, default=4,
                        help='Number of transformer decoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=512,
                        help='Feedforward dimension in transformer')
    
    # Training
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=0.1,
                        help='Beta weight for KL term')
    parser.add_argument('--beta_warmup', type=int, default=250,
                        help='Epochs to anneal beta')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_samples', type=int, default=6000)
    
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    
    # Output directory includes model type to avoid overwriting results
    if args.output_dir is None:
        args.output_dir = f'./outputs/{args.metal}_{args.model_type}'
    os.makedirs(args.output_dir, exist_ok=True)
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Loading data: {args.metal}_bind")
    print(f"{'='*60}")
    
    train_dataset, test_dataset, token_to_id, id_to_token, vocab_size = load_data(
        args.data_dir, args.metal, args.max_len, args.cap, args.seed
    )
    
    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = get_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Training: {len(train_dataset)}, Test: {len(test_dataset)}")
    print(f"Vocab size: {vocab_size}")
    
    # =========================================================================
    # CREATE MODEL
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Creating Discrete Sequence VAE ({args.model_type.upper()})")
    print(f"{'='*60}")
    
    if args.model_type == 'mlp':
        model = DiscreteSequenceVAE_MLP(
            vocab_size=vocab_size,
            seq_len=args.max_len,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            lr=args.lr,
            beta=args.beta,
            beta_warmup=args.beta_warmup,
        )
    elif args.model_type == 'cnn':
        model = DiscreteSequenceVAE_CNN(
            vocab_size=vocab_size,
            seq_len=args.max_len,
            embed_dim=args.embed_dim,
            hidden_channels=args.hidden_dim // 4,  # Scale down for CNN channels
            latent_dim=args.latent_dim,
            lr=args.lr,
            beta=args.beta,
            beta_warmup=args.beta_warmup,
        )
    elif args.model_type == 'transformer':
        model = DiscreteSequenceVAE_Transformer(
            vocab_size=vocab_size,
            seq_len=args.max_len,
            embed_dim=args.embed_dim,
            n_heads=args.n_heads,
            n_encoder_layers=args.n_encoder_layers,
            n_decoder_layers=args.n_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            latent_dim=args.latent_dim,
            dropout=0.1,
            lr=args.lr,
            beta=args.beta,
            beta_warmup=args.beta_warmup,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model type: {args.model_type.upper()}")
    print(f"Embed dim: {args.embed_dim}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Beta: {args.beta} (warmup: {args.beta_warmup} epochs)")
    print(f"Parameters: {n_params:,}")
    
    # =========================================================================
    # SAVE ARCHITECTURE DIAGRAMS
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Saving architecture diagrams")
    print(f"{'='*60}")
    save_architecture_diagrams(model, args.output_dir, vocab_size, args.max_len, args.model_type)
    
    # =========================================================================
    # TRAIN
    # =========================================================================
    plot_path = os.path.join(args.output_dir, 'training_losses.png')
    plot_callback = RealTimePlotCallback(save_path=plot_path)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='best_model',
        save_top_k=1,
        monitor='train_loss',
        mode='min',
    )
    
    print(f"\n{'='*60}")
    print(f"Training for {args.epochs} epochs")
    print(f"{'='*60}")
    
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator='auto',
        devices=1,
        callbacks=[plot_callback, checkpoint_callback],
        enable_progress_bar=True,
        log_every_n_steps=1,
    )
    
    trainer.fit(model, train_loader)
    print(f"\nTraining complete!")
    
    # =========================================================================
    # EVALUATION
    # =========================================================================
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")
    
    model.eval()
    model.to(device)
    
    # Reconstruct training set
    print("\nReconstructing training set...")
    train_orig_seqs = []
    train_recon_seqs = []
    
    with torch.no_grad():
        for batch in train_loader:
            x, seq_lens = batch
            x = x.to(device)
            recon_ids = model.reconstruct(x)
            
            for i in range(x.size(0)):
                slen = int(seq_lens[i])
                orig_seq = ids_to_sequence(x[i].cpu().numpy(), id_to_token, slen)
                recon_seq = ids_to_sequence(recon_ids[i].cpu().numpy(), id_to_token, slen)
                train_orig_seqs.append(orig_seq)
                train_recon_seqs.append(recon_seq)
    
    # Reconstruct test set
    print("Reconstructing test set...")
    test_orig_seqs = []
    test_recon_seqs = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, seq_lens = batch
            x = x.to(device)
            recon_ids = model.reconstruct(x)
            
            for i in range(x.size(0)):
                slen = int(seq_lens[i])
                orig_seq = ids_to_sequence(x[i].cpu().numpy(), id_to_token, slen)
                recon_seq = ids_to_sequence(recon_ids[i].cpu().numpy(), id_to_token, slen)
                test_orig_seqs.append(orig_seq)
                test_recon_seqs.append(recon_seq)
    
    # Generate samples from prior
    print(f"Generating {args.n_samples} samples from prior...")
    with torch.no_grad():
        gen_ids = model.sample(args.n_samples, device=device)
    
    generated_seqs = []
    for i in range(args.n_samples):
        seq = ids_to_sequence(gen_ids[i].cpu().numpy(), id_to_token)
        generated_seqs.append(get_effective_sequence(seq))
    
    training_seqs = train_dataset.sequences
    
    # =========================================================================
    # COMPUTE METRICS
    # =========================================================================
    print(f"\n{'='*60}")
    print("Computing metrics (N, U, V, R, RR)")
    print(f"{'='*60}")
    
    def compute_effective_R(orig_seqs, recon_seqs):
        if not orig_seqs:
            return 0.0
        matches = sum(1 for o, r in zip(orig_seqs, recon_seqs) 
                     if compare_sequences_effective(o, r)[0])
        return matches / len(orig_seqs)
    
    # Training metrics
    train_R = compute_effective_R(train_orig_seqs, train_recon_seqs)
    print(f"\n[Training Set] R (effective): {train_R:.4f}")
    
    # Test metrics
    test_R = compute_effective_R(test_orig_seqs, test_recon_seqs)
    print(f"[Test Set] R (effective): {test_R:.4f}")
    
    # NUV metrics (parallelized for speed)
    print(f"\nComputing N,U,V metrics (comparing {len(generated_seqs)} generated vs {len(training_seqs)} training)...")
    import time
    nuv_start = time.time()
    nuv = compute_nuv(generated_seqs, training_seqs)
    nuv_time = time.time() - nuv_start
    print(f"N,U,V computed in {nuv_time:.1f}s")
    print(f"\nN (novelty): {nuv['N']:.4f}")
    print(f"U (uniqueness): {nuv['U']:.4f}")
    print(f"V (validity): {nuv['V']:.4f}")
    
    # NUVR composite metrics
    N, U, V = nuv['N'], nuv['U'], nuv['V']
    NUVR_train = N * U * V * train_R
    NUVR_test = N * U * V * test_R
    print(f"\nNUVR (with R_train): {NUVR_train:.4f}")
    print(f"NUVR (with R_test):  {NUVR_test:.4f}")
    
    # RR metrics
    rr = compute_all_relative_ratios(training_seqs, generated_seqs, max_len=args.max_len)
    print(f"\nRR_aa: mean={rr['RR_aa']['mean']:.4f}, std={rr['RR_aa']['std']:.4f}")
    print(f"RR_length: mean={rr['RR_length']['mean']:.4f}, std={rr['RR_length']['std']:.4f}")
    print(f"RR_binding: mean={rr['RR_binding_sites']['mean']:.4f}, std={rr['RR_binding_sites']['std']:.4f}")
    print(f"RR_chains: mean={rr['RR_chains']['mean']:.4f}, std={rr['RR_chains']['std']:.4f}")
    
    # RR plot - load reference sequences for Zn if available
    reference_seqs = None
    reference_label = 'Reference'
    random_baseline_seqs = None
    random_baseline_label = 'Random Baseline'
    if args.metal == 'Zn':
        # Load QOBRA de novo sequences
        ref_file = os.path.join(args.data_dir, 'denovo-model-Zn9.txt')
        reference_seqs = load_reference_sequences(ref_file)
        if reference_seqs:
            print(f"Loaded {len(reference_seqs)} QOBRA reference sequences from {ref_file}")
            reference_label = 'QOBRA De novo'
        
        # Load random baseline sequences
        rand_file = os.path.join(args.data_dir, 'denovo-random-Zn9.txt')
        random_baseline_seqs = load_reference_sequences(rand_file)
        if random_baseline_seqs:
            print(f"Loaded {len(random_baseline_seqs)} random baseline sequences from {rand_file}")
            random_baseline_label = 'Random'
    
    rr_plot_path = os.path.join(args.output_dir, 'rr_distributions.png')
    plot_rr_distributions(training_seqs, generated_seqs, rr_plot_path,
                          reference_seqs=reference_seqs, reference_label=reference_label,
                          random_baseline_seqs=random_baseline_seqs, random_baseline_label=random_baseline_label,
                          max_len=args.max_len)
    
    # Example samples
    print("\nExample generated samples:")
    for i in range(min(5, args.n_samples)):
        print(f"  {i+1}: {generated_seqs[i][:80]}...")
    
    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    
    # Metrics file
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Discrete Sequence VAE ({args.model_type.upper()}) - Evaluation Metrics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model type: {args.model_type.upper()}\n")
        f.write(f"Metal: {args.metal}\n")
        f.write(f"Vocab size: {vocab_size}\n")
        f.write(f"Embed dim: {args.embed_dim}\n")
        f.write(f"Hidden dim: {args.hidden_dim}\n")
        f.write(f"Latent dim: {args.latent_dim}\n")
        f.write(f"Beta: {args.beta}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Parameters: {n_params:,}\n")
        f.write(f"Training: {len(train_dataset)}, Test: {len(test_dataset)}\n\n")
        f.write("--- Reconstruction ---\n")
        f.write(f"  Train R: {train_R:.6f}\n")
        f.write(f"  Test R: {test_R:.6f}\n\n")
        f.write("--- Generation (NUV) ---\n")
        f.write(f"  N: {nuv['N']:.6f}\n")
        f.write(f"  U: {nuv['U']:.6f}\n")
        f.write(f"  V: {nuv['V']:.6f}\n\n")
        f.write("--- Composite (NUVR) ---\n")
        f.write(f"  NUVR (R_train): {NUVR_train:.6f}\n")
        f.write(f"  NUVR (R_test):  {NUVR_test:.6f}\n\n")
        f.write("--- Relative Ratios ---\n")
        f.write(f"  RR_aa: {rr['RR_aa']['mean']:.6f} ± {rr['RR_aa']['std']:.6f}\n")
        f.write(f"  RR_length: {rr['RR_length']['mean']:.6f} ± {rr['RR_length']['std']:.6f}\n")
        f.write(f"  RR_binding: {rr['RR_binding_sites']['mean']:.6f} ± {rr['RR_binding_sites']['std']:.6f}\n")
        f.write(f"  RR_chains: {rr['RR_chains']['mean']:.6f} ± {rr['RR_chains']['std']:.6f}\n")
    
    # Reconstructions
    for name, orig_seqs, recon_seqs, R in [
        ('train', train_orig_seqs, train_recon_seqs, train_R),
        ('test', test_orig_seqs, test_recon_seqs, test_R),
    ]:
        path = os.path.join(args.output_dir, f'reconstructions_{name}.txt')
        with open(path, 'w') as f:
            f.write(f"# {name.title()} Set Reconstructions - R: {R:.4f}\n")
            f.write("=" * 100 + "\n\n")
            for i, (orig, recon) in enumerate(zip(orig_seqs, recon_seqs)):
                match, o_eff, r_eff = compare_sequences_effective(orig, recon)
                f.write(f"[{i:4d}] {'✓' if match else '✗'} (len={len(o_eff)})\n")
                f.write(f"  IN:  {o_eff}\n")
                f.write(f"  OUT: {r_eff}\n")
                if not match:
                    diff = ''.join('^' if o_eff[j] != r_eff[j] else ' ' for j in range(min(len(o_eff), len(r_eff))))
                    f.write(f"  DIFF:{diff}\n")
                f.write("\n")
        print(f"Saved: {path}")
    
    # Generated sequences
    with open(os.path.join(args.output_dir, 'generated_sequences.txt'), 'w') as f:
        f.write(f"# Generated sequences - N={nuv['N']:.4f}, U={nuv['U']:.4f}, V={nuv['V']:.4f}\n\n")
        for i, seq in enumerate(generated_seqs):
            f.write(f"{i}\t{seq}\n")
    
    # Training log
    with open(os.path.join(args.output_dir, 'training_log.txt'), 'w') as f:
        f.write('epoch\tloss\trecon\tkl\tbeta\n')
        for i in range(len(plot_callback.losses)):
            f.write(f"{i+1}\t{plot_callback.losses[i]:.6f}\t"
                   f"{plot_callback.recons[i] if i < len(plot_callback.recons) else 0:.6f}\t"
                   f"{plot_callback.kls[i] if i < len(plot_callback.kls) else 0:.6f}\t"
                   f"{plot_callback.betas[i] if i < len(plot_callback.betas) else args.beta:.6f}\n")
    
    print(f"\n{'='*60}")
    print(f"All outputs saved to {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
