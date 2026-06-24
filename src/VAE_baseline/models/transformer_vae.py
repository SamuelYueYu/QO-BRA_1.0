"""
Transformer-based Variational Autoencoder for Discrete Protein Sequences
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input."""
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DiscreteSequenceVAE_Transformer(pl.LightningModule):
    """
    Variational Autoencoder for discrete protein sequences using Transformers.
    
    Uses:
    - Embedding layer for input tokens
    - Transformer encoder to capture global sequence patterns
    - Transformer decoder for autoregressive reconstruction
    - Cross-Entropy loss for reconstruction
    """
    
    def __init__(
        self,
        vocab_size: int,
        seq_len: int = 512,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        latent_dim: int = 128,
        dropout: float = 0.1,
        lr: float = 1e-3,
        beta: float = 0.1,
        beta_warmup: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.beta = beta
        self.beta_warmup = beta_warmup
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, seq_len, dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_encoder_layers,
            norm=nn.LayerNorm(embed_dim)
        )
        
        # Latent projections (from CLS token or mean pooling)
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)
        
        # Decoder: project latent to sequence of embeddings
        self.latent_to_seq = nn.Linear(latent_dim, seq_len * embed_dim)
        
        # Transformer Decoder (self-attention only, no cross-attention)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=n_decoder_layers,
            norm=nn.LayerNorm(embed_dim)
        )
        
        # Output projection to vocabulary
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
        # Track losses
        self.train_losses = []
        self.train_recons = []
        self.train_kls = []
        self.betas = []
    
    def encode(self, x):
        """Encode input token IDs to latent distribution parameters."""
        # x: (batch, seq_len) integer token IDs
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        embedded = self.pos_encoder(embedded)
        
        # Transformer encoding
        h = self.transformer_encoder(embedded)  # (batch, seq_len, embed_dim)
        
        # Mean pooling over sequence
        h_pooled = h.mean(dim=1)  # (batch, embed_dim)
        
        mu = self.fc_mu(h_pooled)
        logvar = self.fc_logvar(h_pooled)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to logits over vocabulary."""
        # z: (batch, latent_dim)
        batch_size = z.size(0)
        
        # Project latent to sequence of embeddings
        h = self.latent_to_seq(z)  # (batch, seq_len * embed_dim)
        h = h.view(batch_size, self.seq_len, self.embed_dim)  # (batch, seq_len, embed_dim)
        
        # Add positional encoding
        h = self.pos_encoder(h)
        
        # Transformer decoding
        h = self.transformer_decoder(h)  # (batch, seq_len, embed_dim)
        
        # Project to vocabulary
        logits = self.output_proj(h)  # (batch, seq_len, vocab_size)
        return logits
    
    def forward(self, x):
        """Forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar
    
    def get_current_beta(self):
        """Get current beta value with warmup annealing."""
        if self.beta_warmup > 0 and self.current_epoch < self.beta_warmup:
            return self.beta * (self.current_epoch + 1) / self.beta_warmup
        return self.beta
    
    def training_step(self, batch, batch_idx):
        """Single training step."""
        x, seq_lens = batch
        logits, mu, logvar = self(x)
        
        # Cross-entropy loss
        recon_loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            x.reshape(-1),
            reduction='mean'
        )
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        current_beta = self.get_current_beta()
        loss = recon_loss + current_beta * kl_loss
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_recon', recon_loss, prog_bar=True)
        self.log('train_kl', kl_loss, prog_bar=False)
        self.log('beta', current_beta, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """Track losses."""
        metrics = self.trainer.callback_metrics
        if 'train_loss' in metrics:
            self.train_losses.append(float(metrics['train_loss']))
        if 'train_recon' in metrics:
            self.train_recons.append(float(metrics['train_recon']))
        if 'train_kl' in metrics:
            self.train_kls.append(float(metrics['train_kl']))
        if 'beta' in metrics:
            self.betas.append(float(metrics['beta']))
    
    def configure_optimizers(self):
        """Configure optimizer with warmup."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
    
    def sample(self, n_samples: int, device=None):
        """Sample new sequences from the prior."""
        if device is None:
            device = self.device
        z = torch.randn(n_samples, self.latent_dim, device=device)
        logits = self.decode(z)
        tokens = torch.argmax(logits, dim=-1)
        return tokens
    
    def reconstruct(self, x):
        """Reconstruct input sequences."""
        mu, logvar = self.encode(x)
        z = mu
        logits = self.decode(z)
        tokens = torch.argmax(logits, dim=-1)
        return tokens

