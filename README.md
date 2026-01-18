# QOBRA - Quantum Operator-Based Real-Amplitude Autoencoder

A quantum machine learning framework for generating biologically plausible protein sequences. QOBRA combines variational quantum circuits with ESM (Evolutionary Scale Modeling) loss to generate novel protein sequences that match the statistical properties of natural proteins.

## Overview

QOBRA trains a quantum autoencoder with **two-phase training**:

| Phase | Loss Function | What's Trained | Goal |
|-------|---------------|----------------|------|
| **Phase 1** | MMD (Maximum Mean Discrepancy) | Encoder | Learn latent space matching Gaussian distribution |
| **Phase 2** | Fidelity + ESM | Decoder (encoder frozen) | Reconstruct biologically plausible sequences |

### Architecture

```
Input Sequence → [Encoder] → Latent State → [Decoder] → Output Sequence
     ↓                            ↓                          ↓
  Feature Map           Gaussian-like space           Reconstruction
  (RawFeatureVector)    (trained via MMD)             (trained via Fidelity + ESM)
```

## Installation

### Dependencies

```bash
# Core quantum computing
pip install qiskit qiskit-machine-learning qiskit-algorithms

# Deep learning & ESM
pip install torch fair-esm

# Scientific computing
pip install numpy scipy matplotlib tqdm

# For ESM evaluation (specific versions)
pip install -r esm_eval/requirements.txt
```

**ESM Evaluation Dependencies:**
- `torch>=1.12.0`
- `fair-esm>=2.0.0`
- `matplotlib>=3.5.0`
- `scipy>=1.7.0`
- `numpy>=1.20.0`

## Quick Start

### Training

```bash
cd src

# Train on Zinc-binding proteins with default settings
python train.py Zn --num-qubits 9 --reps 2

# Train with ESM loss enabled (recommended)
python train.py Zn --num-qubits 9 --reps 2 --lambda-esm 0.001

# Train with ESM warmup (linear annealing)
python train.py Zn --num-qubits 9 --reps 2 --lambda-esm 0.001 --warmup-esm 500

# Multiple metals
python train.py Ca Mg Zn --num-qubits 9 --reps 2
```

### Generation

```bash
# Generate sequences using trained model
python gen.py Zn --num-qubits 9 --reps 2 --mode 1
```

## Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `metals` | - | (required) | Metal types to analyze (e.g., `Zn`, `Ca Mg`) |
| `--num-qubits` | `-q` | 9 | Total number of qubits in the circuit |
| `--reps` | `-r` | 2 | Number of repetitions/layers in the ansatz |
| `--lambda-mmd` | - | 1.0 | Weight for MMD loss (phase 1) |
| `--lambda-fidelity` | - | 1.0 | Weight for Fidelity loss (phase 2) |
| `--lambda-esm` | - | 1.0 | Weight for ESM loss (phase 2) |
| `--esm-k` | - | 32 | Number of positions to mask per sequence for ESM |
| `--esm-model` | - | `esm2_t6_8M_UR50D` | ESM model name |
| `--warmup-esm` | - | None | Iterations for ESM lambda linear warmup |
| `--mode` | `-m` | 0 | 0 = training, 1 = generation |

## Project Structure

```
src/
├── train.py              # Main training script
├── gen.py                # Sequence generation script
├── ansatz.py             # Quantum circuit definitions & CLI parsing
├── model.py              # Encoder/decoder quantum circuit models
├── config.py             # Configuration & hyperparameters
├── cost.py               # Two-phase loss functions
├── losses.py             # Individual loss computations (MMD, Fidelity)
├── esm_loss.py           # ESM pseudo-log-likelihood loss
├── encoding.py           # Sequence encoding & unitary operations
├── coding.py             # Token encoding/decoding utilities
├── inputs.py             # Data loading & preprocessing
├── count.py              # Counting utilities
├── gen_func.py           # Generation helper functions
├── utils.py              # Device management, checkpointing, progress
├── plotting.py           # Visualization functions
├── QOBRA_demo.ipynb      # Interactive demo notebook
├── run-Zn.sh             # Example run script for Zinc
├── run-Ca.sh             # Example run script for Calcium
├── run-Mg.sh             # Example run script for Magnesium
├── Training data/        # Protein sequence datasets
│   ├── Zn_bind/          # Zinc-binding protein sequences
│   ├── Ca_bind/          # Calcium-binding protein sequences
│   ├── Mg_bind/          # Magnesium-binding protein sequences
│   ├── denovo-train-Zn9.txt    # Training sequences
│   ├── denovo-random-Zn9.txt   # Random baseline sequences
│   ├── denovo-cnn-Zn9.txt      # CNN-VAE generated sequences
│   └── denovo-qobra-Zn9.txt    # QOBRA generated sequences
└── esm_eval/             # ESM evaluation pipeline (see below)
    ├── run_esm_eval.py       # Main entry point
    ├── esm_scoring.py        # ESM model wrapper
    ├── load_sequences.py     # Sequence loading utilities
    ├── plot_histograms.py    # Histogram visualization
    ├── compute_distances.py  # Distribution distance metrics
    ├── select_best_denovo.py # Select best from experiments
    ├── __init__.py           # Package exports
    └── requirements.txt      # Python dependencies
```

## Module Descriptions

### Core Training Modules

| Module | Purpose |
|--------|---------|
| `ansatz.py` | Defines quantum circuit structure, parses CLI arguments, creates RealAmplitudes ansatz |
| `model.py` | Builds encoder (`qc_e`), decoder (`qc_d`), and full autoencoder (`qc_ed`) circuits |
| `config.py` | Training configuration: loss weights, batch sizes, frequencies, epoch-based batching |
| `cost.py` | Two-phase loss functions: `encoder_loss()` (MMD) and `decoder_loss()` (Fidelity + ESM) |
| `train.py` | Main training loop with COBYLA optimizer, checkpoint saving, progress tracking |

### Loss Functions

| Module | Function | Description |
|--------|----------|-------------|
| `losses.py` | `compute_kernel_loss()` | MMD loss using RBF kernel for latent space regularization |
| `losses.py` | `compute_fidelity_loss()` | Reconstruction fidelity between input and output states |
| `esm_loss.py` | `esm_loss()` | ESM pseudo-log-likelihood with random mask subsampling |

### ESM Loss Details

The ESM loss uses **pseudo-log-likelihood with random mask subsampling** for efficiency:

```python
def compute_esm_pll(sequences, K=32, model_name="esm2_t6_8M_UR50D"):
    # For each sequence:
    # 1. Randomly select K positions (default: 32)
    # 2. Mask those positions with ESM's mask token
    # 3. Forward pass through ESM
    # 4. Compute cross-entropy only at masked positions
    # 5. Average over all masked positions
```

This is ~100-1000x faster than full masked LM scoring while maintaining similar accuracy.

### Generation Module

| Module | Purpose |
|--------|---------|
| `gen.py` | Main generation loop: samples latent space, decodes, validates sequences |
| `gen_func.py` | Helper functions: batch decoding, k-mer novelty checking, PyMOL script generation |

## Sequence Format

Sequences use a specialized annotation format:

| Symbol | Meaning |
|--------|---------|
| `A-Y` | Standard amino acids |
| `+` | Ligand-binding residue marker (follows the amino acid) |
| `:` | Chain boundary separator |
| `X` | Sequence terminator |

**Example:**
```
MKFL+VLC+GKDF:PEPTIDE+CHAIN:
```
- `L+` and `C+` are binding residues
- Two chains separated by `:`

## Two-Phase Training

### Phase 1: Encoder Training

**Objective:** Learn an encoder that maps protein sequences to a well-structured latent space (Gaussian distribution).

```
Loss = λ_mmd × L_MMD
```

- Only encoder parameters are optimized
- MMD (Maximum Mean Discrepancy) matches encoded distribution to target Gaussian
- Uses RBF kernel for distribution comparison

### Phase 2: Decoder Training

**Objective:** Learn a decoder that reconstructs biologically plausible sequences from latent representations.

```
Loss = λ_fidelity × L_Fidelity + λ_esm × L_ESM
```

- Encoder is **frozen** (uses optimized parameters from Phase 1)
- Only decoder parameters are optimized
- Fidelity loss ensures accurate reconstruction
- ESM loss ensures biological plausibility

### ESM Warmup

Optional linear warmup for ESM loss to stabilize early training:

```python
# With --warmup-esm 500
λ_esm(t) = (t / 500) × λ_esm_max  # for t < 500
λ_esm(t) = λ_esm_max              # for t ≥ 500
```

## Output Files

Training produces these outputs in `Training data/{experiment_name}/`:

| File | Description |
|------|-------------|
| `opt-e-{S}.pkl` | Optimized encoder parameters |
| `opt-d-{S}.pkl` | Optimized decoder parameters |
| `Results-{S}.txt` | Training/test reconstruction accuracy |
| `R-{S}.txt` | Detailed per-sequence results |
| `qc_e.png` | Encoder circuit visualization |
| `qc_ed.png` | Full autoencoder circuit visualization |
| `qc_d.png` | Decoder circuit visualization |
| `checkpoints/` | Training checkpoints |

Generation produces:

| File | Description |
|------|-------------|
| `{seed}/denovo-{seed}.txt` | Generated sequences |
| `{seed}/Samples/{i}/` | Per-sequence outputs with PyMOL scripts |
| `{seed}/Bar-{seed}-{S}.png` | Amino acid frequency comparison |
| `{seed}/RR-*.pkl` | Statistical analysis results |

---

# ESM Evaluation Pipeline

A standalone module for evaluating protein sequence quality using ESM log-likelihood scores. Located in `esm_eval/`.

## Purpose

Computes **pseudo-log-likelihood scores** for protein sequences using ESM models, enabling:
- Quality assessment of generated sequences
- Distribution comparison between sequence sets
- Statistical distance metrics (KS, Wasserstein)

## Installation

```bash
cd esm_eval
pip install -r requirements.txt
```

## Usage

### Step 1: Compute ESM Log-Likelihoods

```bash
cd src/esm_eval

# Score all sequence files in Training data/
python run_esm_eval.py --input "../Training data/"

# Use a specific model
python run_esm_eval.py --input "../Training data/" --model esm2_t33_650M_UR50D

# Score a single file
python run_esm_eval.py --input "../Training data/denovo-qobra-Zn9.txt"

# Full options
python run_esm_eval.py --input "../Training data/" \
    --output_csv results_all.csv \
    --output_fig esm_ll_histograms.png \
    --model esm2_t33_650M_UR50D \
    --batch_size 8
```

**Command Line Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--input` | Path to input file or directory | `Training data/` |
| `--output_csv` | Path for combined CSV output | `results_all.csv` |
| `--output_fig` | Path for histogram figure | `esm_ll_histograms.png` |
| `--model` | ESM model to use | `esm2_t33_650M_UR50D` |
| `--batch_size` | Batch size for GPU processing | `8` |
| `--device` | Device (`cuda`, `cpu`, or auto-detect) | auto |
| `--per_file_csv` | Also write per-file CSVs | `False` |
| `--skip_plot` | Skip histogram generation | `False` |

**Outputs:**
- `results_all.csv` — Per-sequence scores
- `esm_ll_histograms.png` — Distribution comparison plot

### Step 2: Compute Distribution Distances

```bash
python compute_distances.py --input_csv results_all.csv --output_csv esm_distribution_distances.csv
```

**Metrics computed:**
- **KS Statistic (D)**: Maximum difference between CDFs (0-1 scale)
- **KS p-value**: Statistical significance of distribution difference
- **Wasserstein Distance**: Earth mover's distance in log-likelihood units

### Step 3: Select Best Denovo File (Optional)

If you have multiple experiment runs in numbered subfolders (0/, 1/, 2/, ...):

```bash
# Dry run to see scores without copying
python select_best_denovo.py --experiment_dir "Training data/Zn_2_0.0001_500_32_esm2_t30_150M_UR50D/" --dry_run

# Execute selection
python select_best_denovo.py --experiment_dir "Training data/Zn_2_0.0001_500_32_esm2_t30_150M_UR50D/"
```

## Input Format

Sequences should be in `.txt` files with one sequence per line:

```
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAA
MKFLILLFNILCLFPVLAADNHGVGPQGASGVDPITFDINSNQTGVQLTLFRPGQSGQTVNVTGSGDVSTAHITVGSQQ
```

**Special markers (automatically handled):**
- `+` : Ligand-binding residue marker (stripped)
- `:` : Chain boundary separator (stripped)
- `X` : Sequence terminator (stripped)

## Output Format

### CSV Output (`results_all.csv`)

| Column | Description |
|--------|-------------|
| `sequence_id` | Unique identifier |
| `source_file` | Original filename |
| `sequence` | Cleaned sequence |
| `length` | Sequence length |
| `total_log_likelihood` | Sum of per-residue log-likelihoods |
| `mean_log_likelihood` | Average log-likelihood per residue |

### Histogram Output

Overlaid density histograms showing log-likelihood distributions for each input file, useful for comparing generated vs. training sequences.

## Programmatic Usage

```python
from esm_eval import load_sequences, ESMScorer, plot_log_likelihood_histograms

# Load sequences
sequences, ids, sources = load_sequences("../Training data/")

# Score with ESM
scorer = ESMScorer(model_name="esm2_t33_650M_UR50D")
scores = scorer.compute_log_likelihood_batch(sequences, ids, sources)

# Plot distributions
results = scorer.scores_to_dict_list(scores)
plot_log_likelihood_histograms(results, output_path="histogram.png")
```

## Interpreting Log-Likelihood Scores

| Range | Interpretation |
|-------|----------------|
| -1.5 to -0.5 | Natural proteins (high quality) |
| -2.5 to -1.0 | Good generated sequences |
| -4.0 to -2.5 | Poor/random sequences |

**Key insight:** Compare distributions, not individual values. The histogram plot shows how generated sequences compare to training data.

## Distance Metrics

| Metric | What it Measures | Range |
|--------|------------------|-------|
| **KS Statistic (D)** | Maximum CDF difference (shape) | 0 to 1 |
| **KS p-value** | Statistical significance | 0 to 1 |
| **Wasserstein** | Earth mover's distance (shift) | ≥ 0 |

---

## Available ESM Models

| Model | Parameters | Speed | Quality |
|-------|------------|-------|---------|
| `esm2_t6_8M_UR50D` | 8M | Fastest | Lower |
| `esm2_t12_35M_UR50D` | 35M | Fast | Medium |
| `esm2_t30_150M_UR50D` | 150M | Medium | Good |
| `esm2_t33_650M_UR50D` | 650M | Slow | High |
| `esm2_t36_3B_UR50D` | 3B | Very slow | Very high |
| `esm2_t48_15B_UR50D` | 15B | Slowest | Highest |

**Recommendations:**
- For training, use smaller models (`esm2_t6_8M_UR50D`) for speed.
- For evaluation, use larger models (`esm2_t33_650M_UR50D`) for accuracy.

---

## Example Workflow

```bash
# 1. Train QOBRA on Zinc-binding proteins
cd src
python train.py Zn -q 9 -r 2 --lambda-esm 0.001 --warmup-esm 500

# 2. Generate new sequences
python gen.py Zn -q 9 -r 2 -m 1

# 3. Evaluate generated sequences with ESM
cd esm_eval
python run_esm_eval.py --input "../Training data/" --model esm2_t33_650M_UR50D
python compute_distances.py --input_csv results_all.csv

# 4. Compare distributions
# View esm_ll_histograms.png and esm_distribution_distances.csv
```

## Citation

If you use ESM models, please cite:

```bibtex
@article{lin2023evolutionary,
  title={Evolutionary-scale prediction of atomic-level protein structure with a language model},
  author={Lin, Zeming and others},
  journal={Science},
  volume={379},
  number={6637},
  pages={1123--1130},
  year={2023}
}
```
