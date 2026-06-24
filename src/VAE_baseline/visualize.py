"""
Discrete Sequence VAE - Visualization Module

Creates architecture diagrams for both MLP and CNN VAE models.
"""

import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def draw_block(ax, x, y, width, height, label, color='lightblue', fontsize=8):
    """Draw a single layer block."""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color, edgecolor='black', linewidth=1.5
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, fontweight='bold')


def draw_arrow(ax, start, end, color='black'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))


# =============================================================================
# MLP Architecture Visualization
# =============================================================================

def visualize_mlp_encoder(model, save_path, vocab_size, seq_len):
    """Visualize MLP encoder architecture."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-2, 4)
    ax.axis('off')
    ax.set_title('MLP VAE Encoder Architecture', fontsize=14, fontweight='bold')
    
    latent = model.latent_dim
    embed = model.embed_dim
    
    # Positions
    y = 1.5
    positions = [(1, y), (3, y), (5, y), (7, y), (9, y), (11, y), (13, 2.2), (13, 0.8)]
    
    # Blocks
    draw_block(ax, 1, y, 1.4, 1.2, f'Input\n[{seq_len}]\nIDs', color='#90EE90')
    draw_block(ax, 3, y, 1.4, 1.2, f'Embed\n{vocab_size}→{embed}', color='#FFD700')
    draw_block(ax, 5, y, 1.4, 1.2, f'Flatten\n{seq_len}×{embed}', color='#87CEEB')
    draw_block(ax, 7, y, 1.4, 1.2, f'Linear\n+LN+ReLU', color='#87CEEB')
    draw_block(ax, 9, y, 1.4, 1.2, f'Linear\n+LN+ReLU', color='#87CEEB')
    draw_block(ax, 11, y, 1.4, 1.2, f'Hidden\n[{model.hparams.hidden_dim//2}]', color='#87CEEB')
    draw_block(ax, 13, 2.2, 1.2, 0.8, f'μ [{latent}]', color='#DDA0DD')
    draw_block(ax, 13, 0.8, 1.2, 0.8, f'logσ² [{latent}]', color='#DDA0DD')
    
    # Arrows
    for i in range(5):
        draw_arrow(ax, (positions[i][0]+0.7, y), (positions[i+1][0]-0.7, y))
    draw_arrow(ax, (11.7, y+0.2), (12.4, 2.2))
    draw_arrow(ax, (11.7, y-0.2), (12.4, 0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"MLP Encoder diagram saved to: {save_path}")


def visualize_mlp_decoder(model, save_path, vocab_size, seq_len):
    """Visualize MLP decoder architecture."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 3)
    ax.axis('off')
    ax.set_title('MLP VAE Decoder Architecture', fontsize=14, fontweight='bold')
    
    latent = model.latent_dim
    
    y = 1
    draw_block(ax, 1, y, 1.2, 1.0, f'z\n[{latent}]', color='#DDA0DD')
    draw_block(ax, 3.5, y, 1.4, 1.0, f'Linear\n+LN+ReLU', color='#87CEEB')
    draw_block(ax, 6, y, 1.4, 1.0, f'Linear\n+LN+ReLU', color='#87CEEB')
    draw_block(ax, 8.5, y, 1.4, 1.0, f'Linear\n{seq_len}×{vocab_size}', color='#87CEEB')
    draw_block(ax, 11, y, 1.4, 1.0, f'Reshape\n[{seq_len},{vocab_size}]', color='#FFD700')
    draw_block(ax, 13.5, y, 1.2, 1.0, f'Logits\n→Softmax', color='#90EE90')
    
    for start, end in [(1.6, 2.8), (4.2, 5.3), (6.7, 7.8), (9.2, 10.3), (11.7, 12.8)]:
        draw_arrow(ax, (start, y), (end, y))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"MLP Decoder diagram saved to: {save_path}")


# =============================================================================
# CNN Architecture Visualization
# =============================================================================

def visualize_cnn_encoder(model, save_path, vocab_size, seq_len):
    """Visualize CNN encoder architecture."""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(-1, 19)
    ax.set_ylim(-2, 5)
    ax.axis('off')
    ax.set_title('CNN VAE Encoder Architecture', fontsize=14, fontweight='bold')
    
    latent = model.latent_dim
    embed = model.embed_dim
    ch = model.hparams.hidden_channels
    
    y = 2
    # Input -> Embed -> Permute
    draw_block(ax, 0.5, y, 1.2, 1.0, f'Input\n[{seq_len}]', color='#90EE90')
    draw_block(ax, 2.2, y, 1.2, 1.0, f'Embed\n[{seq_len},{embed}]', color='#FFD700')
    draw_block(ax, 4, y, 1.2, 1.0, f'Permute\n[{embed},{seq_len}]', color='#FFD700')
    
    # Conv layers (stride 2)
    dims = [seq_len, seq_len//2, seq_len//4, seq_len//8, seq_len//16, seq_len//32]
    channels = [embed, ch, ch*2, ch*4, ch*4, ch*4]
    
    x_pos = 6
    for i in range(5):
        draw_block(ax, x_pos, y, 1.3, 1.0, 
                  f'Conv1d\nk=5,s=2\n{channels[i]}→{channels[i+1]}\n[{dims[i+1]}]', 
                  color='#87CEEB', fontsize=7)
        x_pos += 1.8
    
    # Flatten -> FC -> mu/logvar
    draw_block(ax, x_pos, y, 1.2, 1.0, f'Flatten\n[{ch*4}×{dims[-1]}]', color='#FFD700')
    draw_block(ax, x_pos+2, 2.8, 1.0, 0.8, f'μ [{latent}]', color='#DDA0DD')
    draw_block(ax, x_pos+2, 1.2, 1.0, 0.8, f'logσ²\n[{latent}]', color='#DDA0DD')
    
    # Arrows
    for start, end in [(1.1, 1.6), (2.8, 3.4), (4.6, 5.3)]:
        draw_arrow(ax, (start, y), (end, y))
    
    x_a = 6
    for _ in range(4):
        draw_arrow(ax, (x_a+0.65, y), (x_a+1.15, y))
        x_a += 1.8
    
    draw_arrow(ax, (x_a+0.65, y), (x_a+1.1, y))
    draw_arrow(ax, (x_pos+0.6, y+0.2), (x_pos+1.5, 2.8))
    draw_arrow(ax, (x_pos+0.6, y-0.2), (x_pos+1.5, 1.2))
    
    # Annotation
    ax.text(9, -0.8, 'Progressive downsampling: 512 → 256 → 128 → 64 → 32 → 16', 
            ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"CNN Encoder diagram saved to: {save_path}")


def visualize_cnn_decoder(model, save_path, vocab_size, seq_len):
    """Visualize CNN decoder architecture."""
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(-1, 19)
    ax.set_ylim(-1, 4)
    ax.axis('off')
    ax.set_title('CNN VAE Decoder Architecture', fontsize=14, fontweight='bold')
    
    latent = model.latent_dim
    ch = model.hparams.hidden_channels
    enc_len = seq_len // 32
    
    y = 1.5
    # z -> FC -> Reshape
    draw_block(ax, 0.5, y, 1.0, 0.9, f'z\n[{latent}]', color='#DDA0DD')
    draw_block(ax, 2.2, y, 1.2, 0.9, f'Linear\n[{ch*4}×{enc_len}]', color='#87CEEB')
    draw_block(ax, 4, y, 1.2, 0.9, f'Reshape\n[{ch*4},{enc_len}]', color='#FFD700')
    
    # ConvTranspose layers (stride 2)
    dims = [enc_len, enc_len*2, enc_len*4, enc_len*8, enc_len*16, enc_len*32]
    channels = [ch*4, ch*4, ch*4, ch*2, ch, vocab_size]
    
    x_pos = 6
    for i in range(5):
        draw_block(ax, x_pos, y, 1.3, 1.0, 
                  f'ConvT1d\nk=4,s=2\n{channels[i]}→{channels[i+1]}\n[{dims[i+1]}]', 
                  color='#87CEEB', fontsize=7)
        x_pos += 1.8
    
    # Output
    draw_block(ax, x_pos, y, 1.2, 0.9, f'Permute\n[{seq_len},{vocab_size}]', color='#FFD700')
    draw_block(ax, x_pos+1.8, y, 1.0, 0.9, f'Logits\n→Softmax', color='#90EE90')
    
    # Arrows
    for start, end in [(1.0, 1.6), (2.8, 3.4), (4.6, 5.3)]:
        draw_arrow(ax, (start, y), (end, y))
    
    x_a = 6
    for _ in range(5):
        draw_arrow(ax, (x_a+0.65, y), (x_a+1.15, y))
        x_a += 1.8
    draw_arrow(ax, (x_pos+0.6, y), (x_pos+1.2, y))
    
    # Annotation
    ax.text(9, -0.3, 'Progressive upsampling: 16 → 32 → 64 → 128 → 256 → 512', 
            ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"CNN Decoder diagram saved to: {save_path}")


# =============================================================================
# Full VAE Visualization
# =============================================================================

def visualize_full_vae(model, save_path, vocab_size, seq_len, model_type='mlp'):
    """Visualize full VAE architecture."""
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.set_xlim(-1, 19)
    ax.set_ylim(-3, 6)
    ax.axis('off')
    ax.set_title(f'Discrete Sequence VAE ({model_type.upper()}) Architecture', 
                fontsize=16, fontweight='bold')
    
    latent = model.latent_dim
    embed = model.embed_dim
    
    y = 2.5
    
    # === ENCODER SECTION ===
    ax.text(3, 5.2, 'ENCODER', ha='center', fontsize=12, fontweight='bold', color='#2E86AB')
    
    draw_block(ax, 0.5, y, 1.0, 0.9, f'x\n[{seq_len}]', color='#90EE90')
    draw_block(ax, 2, y, 1.0, 0.9, f'Embed\n[{embed}]', color='#FFD700')
    
    if model_type == 'cnn':
        draw_block(ax, 3.5, y, 1.2, 0.9, 'Conv1d\n×5', color='#87CEEB')
        draw_block(ax, 5.2, y, 1.0, 0.9, 'Flatten', color='#FFD700')
    elif model_type == 'transformer':
        n_enc = model.hparams.n_encoder_layers
        draw_block(ax, 3.5, y, 1.0, 0.9, '+PosEnc', color='#FFD700')
        draw_block(ax, 5.2, y, 1.2, 0.9, f'TF Enc\n×{n_enc}', color='#87CEEB')
    else:
        draw_block(ax, 3.5, y, 1.0, 0.9, 'Flatten', color='#FFD700')
        draw_block(ax, 5.2, y, 1.2, 0.9, 'Linear\n×2', color='#87CEEB')
    
    # mu/logvar
    draw_block(ax, 7, y+0.9, 0.9, 0.7, f'μ [{latent}]', color='#DDA0DD', fontsize=8)
    draw_block(ax, 7, y-0.9, 0.9, 0.7, f'logσ²', color='#DDA0DD', fontsize=8)
    
    # === LATENT SECTION ===
    ax.text(8.5, 5.2, 'LATENT', ha='center', fontsize=12, fontweight='bold', color='#A23B72')
    draw_block(ax, 8.5, y, 1.4, 1.4, f'z = μ+σ·ε\n[{latent}]', color='#F4D35E')
    
    # === DECODER SECTION ===
    ax.text(14, 5.2, 'DECODER', ha='center', fontsize=12, fontweight='bold', color='#F18F01')
    
    if model_type == 'cnn':
        draw_block(ax, 10.5, y, 1.0, 0.9, 'Linear', color='#87CEEB')
        draw_block(ax, 12, y, 1.0, 0.9, 'Reshape', color='#FFD700')
        draw_block(ax, 13.7, y, 1.2, 0.9, 'ConvT1d\n×5', color='#87CEEB')
    elif model_type == 'transformer':
        n_dec = model.hparams.n_decoder_layers
        draw_block(ax, 10.5, y, 1.0, 0.9, 'Linear', color='#87CEEB')
        draw_block(ax, 12, y, 1.0, 0.9, '+PosEnc', color='#FFD700')
        draw_block(ax, 13.7, y, 1.2, 0.9, f'TF Dec\n×{n_dec}', color='#87CEEB')
    else:
        draw_block(ax, 10.5, y, 1.2, 0.9, 'Linear\n×3', color='#87CEEB')
        draw_block(ax, 12.3, y, 1.0, 0.9, 'Reshape', color='#FFD700')
        draw_block(ax, 13.7, y, 1.0, 0.9, '', color='white')  # placeholder
    
    draw_block(ax, 15.5, y, 1.2, 0.9, f'Logits\n[{seq_len},{vocab_size}]', color='#90EE90')
    draw_block(ax, 17.3, y, 1.0, 0.9, 'argmax\n→tokens', color='#90EE90')
    
    # Arrows
    draw_arrow(ax, (1.0, y), (1.5, y))
    draw_arrow(ax, (2.5, y), (3.0, y))
    draw_arrow(ax, (4.1, y), (4.7, y))
    draw_arrow(ax, (5.8, y+0.2), (6.55, y+0.9))
    draw_arrow(ax, (5.8, y-0.2), (6.55, y-0.9))
    draw_arrow(ax, (7.45, y+0.9), (7.8, y+0.4))
    draw_arrow(ax, (7.45, y-0.9), (7.8, y-0.4))
    draw_arrow(ax, (9.2, y), (10.0, y))
    draw_arrow(ax, (11.1, y), (11.5, y))
    draw_arrow(ax, (12.8, y), (13.2, y))
    draw_arrow(ax, (14.3, y), (14.9, y))
    draw_arrow(ax, (16.1, y), (16.8, y))
    
    # Loss annotation
    ax.text(8.5, -1.0, 'Loss = CrossEntropy(x, logits) + β · KL(q(z|x) || p(z))', 
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Legend
    legend_y = -2
    legend_items = [
        (2, legend_y, '#90EE90', 'Input/Output'),
        (5, legend_y, '#FFD700', 'Embedding/Reshape'),
        (8, legend_y, '#87CEEB', 'Linear/Conv'),
        (11, legend_y, '#DDA0DD', 'Latent Params'),
        (14, legend_y, '#F4D35E', 'Reparameterization'),
    ]
    for x, ly, color, label in legend_items:
        ax.add_patch(plt.Rectangle((x-0.3, ly-0.15), 0.6, 0.3, facecolor=color, edgecolor='black'))
        ax.text(x+0.5, ly, label, va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Full VAE diagram saved to: {save_path}")


# =============================================================================
# Model Summary
# =============================================================================

def get_model_summary(model, vocab_size, seq_len, model_type='mlp'):
    """Generate text summary of model architecture."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    lines = [
        "=" * 70,
        f"DISCRETE SEQUENCE VAE ({model_type.upper()}) - ARCHITECTURE SUMMARY",
        "=" * 70,
        "",
        f"Model Type: {model_type.upper()}",
        f"Vocabulary Size: {vocab_size}",
        f"Sequence Length: {seq_len}",
        f"Embedding Dimension: {model.embed_dim}",
        f"Latent Dimension: {model.latent_dim}",
        "",
    ]
    
    if model_type == 'cnn':
        ch = model.hparams.hidden_channels
        lines.extend([
            "--- ENCODER (CNN) ---",
            f"  Embedding: {vocab_size} -> {model.embed_dim}",
            f"  Conv1d: {model.embed_dim} -> {ch}, kernel=5, stride=2  (512->256)",
            f"  Conv1d: {ch} -> {ch*2}, kernel=5, stride=2  (256->128)",
            f"  Conv1d: {ch*2} -> {ch*4}, kernel=5, stride=2  (128->64)",
            f"  Conv1d: {ch*4} -> {ch*4}, kernel=5, stride=2  (64->32)",
            f"  Conv1d: {ch*4} -> {ch*4}, kernel=5, stride=2  (32->16)",
            f"  Flatten: {ch*4} × 16 = {ch*4*16}",
            f"  Linear (mu): {ch*4*16} -> {model.latent_dim}",
            f"  Linear (logvar): {ch*4*16} -> {model.latent_dim}",
            "",
            "--- DECODER (CNN) ---",
            f"  Linear: {model.latent_dim} -> {ch*4*16}",
            f"  Reshape: {ch*4} × 16",
            f"  ConvT1d: {ch*4} -> {ch*4}, kernel=4, stride=2  (16->32)",
            f"  ConvT1d: {ch*4} -> {ch*4}, kernel=4, stride=2  (32->64)",
            f"  ConvT1d: {ch*4} -> {ch*2}, kernel=4, stride=2  (64->128)",
            f"  ConvT1d: {ch*2} -> {ch}, kernel=4, stride=2  (128->256)",
            f"  ConvT1d: {ch} -> {vocab_size}, kernel=4, stride=2  (256->512)",
        ])
    elif model_type == 'transformer':
        n_enc = model.hparams.n_encoder_layers
        n_dec = model.hparams.n_decoder_layers
        n_heads = model.hparams.n_heads
        ff_dim = model.hparams.dim_feedforward
        lines.extend([
            "--- ENCODER (TRANSFORMER) ---",
            f"  Embedding: {vocab_size} -> {model.embed_dim}",
            f"  Positional Encoding: {seq_len} positions",
            f"  Transformer Encoder: {n_enc} layers",
            f"    - {n_heads} attention heads",
            f"    - feedforward dim: {ff_dim}",
            f"    - Pre-LN normalization",
            f"  Mean Pooling: ({seq_len}, {model.embed_dim}) -> ({model.embed_dim})",
            f"  Linear (mu): {model.embed_dim} -> {model.latent_dim}",
            f"  Linear (logvar): {model.embed_dim} -> {model.latent_dim}",
            "",
            "--- DECODER (TRANSFORMER) ---",
            f"  Linear: {model.latent_dim} -> {seq_len * model.embed_dim}",
            f"  Reshape: ({seq_len}, {model.embed_dim})",
            f"  Positional Encoding: {seq_len} positions",
            f"  Transformer Decoder: {n_dec} layers",
            f"    - {n_heads} attention heads",
            f"    - feedforward dim: {ff_dim}",
            f"    - Pre-LN normalization",
            f"  Output Projection: {model.embed_dim} -> {vocab_size}",
        ])
    else:
        hd = model.hparams.hidden_dim
        lines.extend([
            "--- ENCODER (MLP) ---",
            f"  Embedding: {vocab_size} -> {model.embed_dim}",
            f"  Flatten: {seq_len} × {model.embed_dim} = {seq_len * model.embed_dim}",
            f"  Linear: {seq_len * model.embed_dim} -> {hd}",
            f"  LayerNorm + ReLU + Dropout",
            f"  Linear: {hd} -> {hd//2}",
            f"  LayerNorm + ReLU + Dropout",
            f"  Linear (mu): {hd//2} -> {model.latent_dim}",
            f"  Linear (logvar): {hd//2} -> {model.latent_dim}",
            "",
            "--- DECODER (MLP) ---",
            f"  Linear: {model.latent_dim} -> {hd//2}",
            f"  LayerNorm + ReLU + Dropout",
            f"  Linear: {hd//2} -> {hd}",
            f"  LayerNorm + ReLU + Dropout",
            f"  Linear: {hd} -> {seq_len * vocab_size}",
            f"  Reshape: [{seq_len}, {vocab_size}]",
        ])
    
    lines.extend([
        "",
        "--- PARAMETERS ---",
        f"  Total: {total_params:,}",
        f"  Trainable: {trainable:,}",
        "",
        "--- LAYER DETAILS ---",
    ])
    
    for name, param in model.named_parameters():
        lines.append(f"  {name}: {list(param.shape)} = {param.numel():,}")
    
    lines.append("=" * 70)
    return "\n".join(lines)


# =============================================================================
# Transformer Architecture Visualization
# =============================================================================

def visualize_transformer_encoder(model, save_path, vocab_size, seq_len):
    """Visualize Transformer encoder architecture."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-2, 5)
    ax.axis('off')
    ax.set_title('Transformer VAE Encoder Architecture', fontsize=14, fontweight='bold')
    
    latent = model.latent_dim
    embed = model.embed_dim
    n_layers = model.hparams.n_encoder_layers
    n_heads = model.hparams.n_heads
    
    y = 2
    draw_block(ax, 1, y, 1.2, 1.0, f'Input\n[{seq_len}]', color='#90EE90')
    draw_block(ax, 3, y, 1.2, 1.0, f'Embed\n[{seq_len},{embed}]', color='#FFD700')
    draw_block(ax, 5, y, 1.2, 1.0, f'+ PosEnc\n[{seq_len},{embed}]', color='#FFD700')
    draw_block(ax, 7.5, y, 2.0, 1.2, f'Transformer\nEncoder\n×{n_layers} layers\n{n_heads} heads', color='#87CEEB')
    draw_block(ax, 10, y, 1.2, 1.0, f'Mean\nPool\n[{embed}]', color='#FFD700')
    draw_block(ax, 12, 2.6, 1.0, 0.8, f'μ [{latent}]', color='#DDA0DD')
    draw_block(ax, 12, 1.4, 1.0, 0.8, f'logσ²', color='#DDA0DD')
    
    for start, end in [(1.6, 2.4), (3.6, 4.4), (5.6, 6.5), (8.5, 9.4)]:
        draw_arrow(ax, (start, y), (end, y))
    draw_arrow(ax, (10.6, y+0.2), (11.5, 2.6))
    draw_arrow(ax, (10.6, y-0.2), (11.5, 1.4))
    
    ax.text(7.5, 0.3, 'Self-Attention + FFN + LayerNorm', ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Transformer Encoder diagram saved to: {save_path}")


def visualize_transformer_decoder(model, save_path, vocab_size, seq_len):
    """Visualize Transformer decoder architecture."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 4)
    ax.axis('off')
    ax.set_title('Transformer VAE Decoder Architecture', fontsize=14, fontweight='bold')
    
    latent = model.latent_dim
    embed = model.embed_dim
    n_layers = model.hparams.n_decoder_layers
    n_heads = model.hparams.n_heads
    
    y = 1.5
    draw_block(ax, 1, y, 1.0, 0.9, f'z\n[{latent}]', color='#DDA0DD')
    draw_block(ax, 3, y, 1.2, 0.9, f'Linear\n[{seq_len}×{embed}]', color='#87CEEB')
    draw_block(ax, 5, y, 1.2, 0.9, f'Reshape\n[{seq_len},{embed}]', color='#FFD700')
    draw_block(ax, 7, y, 1.2, 0.9, f'+ PosEnc', color='#FFD700')
    draw_block(ax, 9.5, y, 2.0, 1.0, f'Transformer\nDecoder\n×{n_layers} layers', color='#87CEEB')
    draw_block(ax, 12, y, 1.2, 0.9, f'Linear\n[{vocab_size}]', color='#87CEEB')
    draw_block(ax, 14, y, 1.0, 0.9, f'Logits', color='#90EE90')
    
    for start, end in [(1.5, 2.4), (3.6, 4.4), (5.6, 6.4), (7.6, 8.5), (10.5, 11.4), (12.6, 13.5)]:
        draw_arrow(ax, (start, y), (end, y))
    
    ax.text(9.5, 0.3, 'Self-Attention + FFN + LayerNorm', ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Transformer Decoder diagram saved to: {save_path}")


def save_architecture_diagrams(model, output_dir, vocab_size, seq_len, model_type='mlp'):
    """
    Save all architecture diagrams and model summary.
    
    Parameters:
    - model: VAE model (MLP, CNN, or Transformer)
    - output_dir: Directory to save files
    - vocab_size: Vocabulary size
    - seq_len: Sequence length
    - model_type: 'mlp', 'cnn', or 'transformer'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save text summary
    summary = get_model_summary(model, vocab_size, seq_len, model_type)
    summary_path = os.path.join(output_dir, 'architecture_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"Architecture summary saved to: {summary_path}")
    
    # Save diagrams based on model type
    if model_type == 'cnn':
        visualize_cnn_encoder(model, os.path.join(output_dir, 'architecture_encoder.png'), 
                             vocab_size, seq_len)
        visualize_cnn_decoder(model, os.path.join(output_dir, 'architecture_decoder.png'), 
                             vocab_size, seq_len)
    elif model_type == 'transformer':
        visualize_transformer_encoder(model, os.path.join(output_dir, 'architecture_encoder.png'), 
                                      vocab_size, seq_len)
        visualize_transformer_decoder(model, os.path.join(output_dir, 'architecture_decoder.png'), 
                                      vocab_size, seq_len)
    else:
        visualize_mlp_encoder(model, os.path.join(output_dir, 'architecture_encoder.png'), 
                             vocab_size, seq_len)
        visualize_mlp_decoder(model, os.path.join(output_dir, 'architecture_decoder.png'), 
                             vocab_size, seq_len)
    
    # Full VAE diagram
    visualize_full_vae(model, os.path.join(output_dir, 'architecture_full.png'), 
                      vocab_size, seq_len, model_type)
    
    print(f"All architecture files saved to: {output_dir}/")

