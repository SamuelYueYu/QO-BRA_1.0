"""
VAE Models for Discrete Protein Sequences

Available architectures:
- DiscreteSequenceVAE_MLP: Fully connected MLP-based VAE
- DiscreteSequenceVAE_CNN: 1D Convolutional VAE
- DiscreteSequenceVAE_Transformer: Transformer-based VAE
"""

from .mlp_vae import DiscreteSequenceVAE_MLP
from .cnn_vae import DiscreteSequenceVAE_CNN
from .transformer_vae import DiscreteSequenceVAE_Transformer, PositionalEncoding

__all__ = [
    'DiscreteSequenceVAE_MLP',
    'DiscreteSequenceVAE_CNN',
    'DiscreteSequenceVAE_Transformer',
    'PositionalEncoding',
]

