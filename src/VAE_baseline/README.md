# PyTorch Lightning VAE Baseline

A Variational Autoencoder (VAE) baseline for protein sequence generation using PyTorch Lightning. Designed for fair comparison with QOBRA using identical data preprocessing.

## Features

- **Three model architectures**: MLP, CNN, and Transformer VAEs
- **Discrete token representation**: Uses integer token IDs with Cross-Entropy loss
- **QOBRA-compatible preprocessing**: 
  - LCS-based sequence deduplication
  - 80/20 train/test split (every 5th sequence for test)
  - Cyclic padding to fixed length (512 tokens)
  - Filtering for sequences with binding sites
- **Comprehensive metrics**: N, U, V, R, RR, and NUVR composite
- **Architecture visualization**: Encoder/decoder structure diagrams

## Requirements

```bash
pip install pytorch-lightning torch numpy matplotlib
```

## Usage

```bash
# Train MLP VAE on Calcium-binding proteins
python train_vae.py --metal Ca --model_type mlp

# Train CNN VAE on Magnesium
python train_vae.py --metal Mg --model_type cnn --epochs 200

# Train Transformer VAE on Zinc
python train_vae.py --metal Zn --model_type transformer --epochs 100
```

### Command-line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--metal` | Required | Metal type: `Ca`, `Mg`, or `Zn` |
| `--model_type` | `mlp` | Model architecture: `mlp`, `cnn`, or `transformer` |
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 256 | Training batch size |
| `--max_len` | 512 | Maximum sequence length |
| `--embed_dim` | 64 | Token embedding dimension |
| `--hidden_dim` | 1024 | Hidden layer dimension (MLP/CNN) |
| `--latent_dim` | 128 | Latent space dimension |
| `--lr` | 1e-3 | Learning rate |
| `--beta` | 1.0 | KL divergence weight |
| `--beta_warmup` | 500 | Beta warmup steps |
| `--n_samples` | 100 | Number of sequences to generate |

#### Transformer-specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_heads` | 4 | Number of attention heads |
| `--n_encoder_layers` | 2 | Number of encoder layers |
| `--n_decoder_layers` | 2 | Number of decoder layers |
| `--dim_feedforward` | 512 | Feedforward dimension |

## Data Layout

```
Training data/
    Ca_bind/
    Mg_bind/
    Zn_bind/
    denovo-model-Zn9.txt  (optional: reference sequences for comparison)
```

Each subfolder contains `.txt` files where:
- Each file represents one protein complex (PDB code as filename)
- Each line in a file is a separate chain
- Binding residues are marked with `+` (e.g., `D+` for binding aspartate)

## Data Preprocessing

Matching QOBRA's preprocessing pipeline:

1. **Load sequences**: Read all chains from each file, preserving newlines as chain separators
2. **Deduplicate**: Remove highly similar sequences using LCS (Longest Common Subsequence)
3. **Split**: 80% training, 20% testing (deterministic, every 5th sequence to test)
4. **Filter**: Only keep sequences containing binding sites (`+`)
5. **Cap**: Maximum 6000 training sequences
6. **Tokenize**: Convert to discrete token IDs (42 tokens: 20 AAs, 20 AAs+, `:`, `:X`)
7. **Pad**: Cyclic padding to 512 tokens with `:X` terminator

## Outputs

All outputs are saved to `./outputs/{metal}_{model_type}/`:

| File | Description |
|------|-------------|
| `best_model.ckpt` | Best model checkpoint |
| `training_losses.png` | Loss curves (total, reconstruction, KL) |
| `training_log.txt` | Loss values per epoch |
| `metrics.txt` | All evaluation metrics |
| `generated_sequences.txt` | Generated sequence samples |
| `reconstruction_comparison.txt` | Input vs reconstructed sequences |
| `rr_distributions.png` | RR distribution plots |
| `encoder_architecture.png` | Encoder structure diagram |
| `decoder_architecture.png` | Decoder structure diagram |
| `full_vae_architecture.png` | Complete VAE diagram |
| `model_summary.txt` | Text summary of model architecture |

## Metrics

### Core Metrics

- **N (Novelty)**: Fraction of generated sequences not similar to training set
- **U (Uniqueness)**: Fraction of unique sequences among generated
- **V (Validity)**: Fraction passing biological validity checks (has binding sites, valid AAs)
- **R (Reconstruction)**: Fraction of perfectly reconstructed sequences
  - `R_train`: Reconstruction rate on training set
  - `R_test`: Reconstruction rate on test set

### Composite Metric

- **NUVR**: Product of N, U, V, and R (computed with both R_train and R_test)

### Distribution Metrics (RR)

- **Token frequency**: Amino acid usage distribution
- **Chain numbers**: Number of chains per sequence
- **Sequence length**: Length distribution
- **Binding sites**: Number of binding residues per sequence

### Reference Comparison (Zn only)

When training on Zn, if `denovo-model-Zn9.txt` exists in the Training data folder, the RR plots will include a third histogram (green) showing the QOBRA de novo sequences for comparison. This allows direct visual comparison between:
- Training set (blue)
- VAE-generated sequences (orange)
- QOBRA de novo sequences (green)

## Model Architectures

### MLP VAE
Fully connected encoder/decoder with embedding layer for discrete tokens.

### CNN VAE  
1D convolutional encoder/decoder, better at capturing local sequence patterns.

### Transformer VAE
Self-attention based architecture with positional encoding, captures long-range dependencies.

## Example Output

```
=== FINAL METRICS ===
--- Reconstruction (R) ---
  R_train: 0.3275
  R_test:  0.2492

--- Generation Quality ---
  N (Novelty):    0.9800
  U (Uniqueness): 0.9500
  V (Validity):   0.8700

--- Composite (NUVR) ---
  NUVR (R_train): 0.2649
  NUVR (R_test):  0.2016
```
