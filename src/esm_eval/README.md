# ESM Evaluation Pipeline

Evaluate protein sequence plausibility using Meta's ESM (Evolutionary Scale Modeling) models. Computes log-likelihood scores and generates analysis outputs.

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `torch>=1.12.0`
- `fair-esm>=2.0.0`
- `matplotlib>=3.5.0`
- `scipy>=1.7.0`
- `numpy>=1.20.0`

## Workflow

### Step 1: Score Sequences

Run the main evaluation script to compute ESM log-likelihood scores:

```bash
python run_esm_eval.py --input "..Training data/" --output_csv results_all.csv --output_fig esm_ll_histograms.png
```

**Options:**
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

**Available Models:**
- `esm2_t6_8M_UR50D` (smallest, fastest)
- `esm2_t12_35M_UR50D`
- `esm2_t30_150M_UR50D`
- `esm2_t33_650M_UR50D` (default, good balance)
- `esm2_t36_3B_UR50D`
- `esm2_t48_15B_UR50D` (largest, most accurate)

### Step 2: Compute Distribution Distances (Optional)

Compare log-likelihood distributions between sequence groups:

```bash
python compute_distances.py --input_csv results_all.csv --output_csv esm_distribution_distances.csv
```

**Metrics computed:**
- **KS Statistic**: Maximum difference between CDFs (0-1 scale)
- **KS p-value**: Statistical significance of distribution difference
- **Wasserstein Distance**: Earth mover's distance in log-likelihood units

### Alternative: Select Best Denovo File

If you have multiple experiment runs in numbered subfolders (0/, 1/, 2/, ...):

```bash
# Dry run to see scores without copying
python select_best_denovo.py --experiment_dir "Training data/Zn_2_0.0001_500_32_esm2_t30_150M_UR50D/" --dry_run

# Execute selection
python select_best_denovo.py --experiment_dir "Training data/Zn_2_0.0001_500_32_esm2_t30_150M_UR50D/"
```

## File Structure

| File | Description |
|------|-------------|
| `run_esm_eval.py` | Main entry point - scores sequences and generates outputs |
| `compute_distances.py` | Post-hoc analysis of distribution distances |
| `select_best_denovo.py` | Select best denovo file from multiple experiments |
| `load_sequences.py` | Sequence loading and validation utilities |
| `esm_scoring.py` | ESM model wrapper for computing log-likelihoods |
| `plot_histograms.py` | Histogram plotting utilities |

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

## Examples

```bash
# Score a single file
python run_esm_eval.py --input "Training data/denovo-train-Zn9.txt"

# Use a smaller/faster model
python run_esm_eval.py --input "Training data/" --model esm2_t12_35M_UR50D

# Increase batch size for more GPU memory
python run_esm_eval.py --input "Training data/" --batch_size 16

# Full pipeline
python run_esm_eval.py --input "Training data/" --output_csv results_all.csv
python compute_distances.py --input_csv results_all.csv --output_csv distances.csv
```