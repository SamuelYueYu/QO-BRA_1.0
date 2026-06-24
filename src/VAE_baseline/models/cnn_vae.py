"""
1D CNN-based Variational Autoencoder for Discrete Protein Sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class DiscreteSequenceVAE_CNN(pl.LightningModule):
    """
    Variational Autoencoder for discrete protein sequences using 1D CNNs.
    
    Uses:
    - Embedding layer for input tokens
    - 1D Convolutional encoder to capture local sequence patterns
    - 1D Transposed Convolutional decoder
    - Cross-Entropy loss for reconstruction
    """
    
    def __init__(
        self,
        vocab_size: int,
        seq_len: int = 512,
        embed_dim: int = 64,
        hidden_channels: int = 128,
        latent_dim: int = 128,
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
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # CNN Encoder: (batch, embed_dim, seq_len) -> (batch, latent_dim)
        # Progressive downsampling with strided convolutions
        self.encoder_cnn = nn.Sequential(
            # Layer 1: 512 -> 256
            nn.Conv1d(embed_dim, hidden_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            
            # Layer 2: 256 -> 128
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            
            # Layer 3: 128 -> 64
            nn.Conv1d(hidden_channels * 2, hidden_channels * 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_channels * 4),
            nn.ReLU(),
            
            # Layer 4: 64 -> 32
            nn.Conv1d(hidden_channels * 4, hidden_channels * 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_channels * 4),
            nn.ReLU(),
            
            # Layer 5: 32 -> 16
            nn.Conv1d(hidden_channels * 4, hidden_channels * 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_channels * 4),
            nn.ReLU(),
        )
        
        # After 5 stride-2 convs: 512 -> 256 -> 128 -> 64 -> 32 -> 16
        self.encoded_len = seq_len // 32  # = 16
        self.encoded_channels = hidden_channels * 4  # = 512
        
        # Latent projections
        self.fc_mu = nn.Linear(self.encoded_channels * self.encoded_len, latent_dim)
        self.fc_logvar = nn.Linear(self.encoded_channels * self.encoded_len, latent_dim)
        
        # Decoder: latent -> sequence logits
        self.fc_decode = nn.Linear(latent_dim, self.encoded_channels * self.encoded_len)
        
        # CNN Decoder: (batch, encoded_channels, encoded_len) -> (batch, vocab_size, seq_len)
        self.decoder_cnn = nn.Sequential(
            # Layer 1: 16 -> 32
            nn.ConvTranspose1d(hidden_channels * 4, hidden_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_channels * 4),
            nn.ReLU(),
            
            # Layer 2: 32 -> 64
            nn.ConvTranspose1d(hidden_channels * 4, hidden_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_channels * 4),
            nn.ReLU(),
            
            # Layer 3: 64 -> 128
            nn.ConvTranspose1d(hidden_channels * 4, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            
            # Layer 4: 128 -> 256
            nn.ConvTranspose1d(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            
            # Layer 5: 256 -> 512
            nn.ConvTranspose1d(hidden_channels, vocab_size, kernel_size=4, stride=2, padding=1),
        )
        
        # Track losses
        self.train_losses = []
        self.train_recons = []
        self.train_kls = []
        self.betas = []
    
    def encode(self, x):
        """Encode input token IDs to latent distribution parameters."""
        # x: (batch, seq_len) integer token IDs
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        embedded = embedded.permute(0, 2, 1)  # (batch, embed_dim, seq_len) for Conv1d
        
        h = self.encoder_cnn(embedded)  # (batch, encoded_channels, encoded_len)
        h = h.view(h.size(0), -1)  # (batch, encoded_channels * encoded_len)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to logits over vocabulary."""
        # z: (batch, latent_dim)
        h = self.fc_decode(z)  # (batch, encoded_channels * encoded_len)
        h = h.view(-1, self.encoded_channels, self.encoded_len)  # (batch, channels, len)
        
        logits = self.decoder_cnn(h)  # (batch, vocab_size, seq_len)
        logits = logits.permute(0, 2, 1)  # (batch, seq_len, vocab_size)
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
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
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

