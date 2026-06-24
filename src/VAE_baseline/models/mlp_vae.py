"""
MLP-based Variational Autoencoder for Discrete Protein Sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class DiscreteSequenceVAE_MLP(pl.LightningModule):
    """
    Variational Autoencoder for discrete protein sequences.
    
    Uses:
    - Embedding layer for input tokens
    - Cross-Entropy loss for reconstruction (proper for discrete data)
    - Softmax output over vocabulary
    """
    
    def __init__(
        self,
        vocab_size: int,
        seq_len: int = 512,
        embed_dim: int = 64,
        hidden_dim: int = 512,
        latent_dim: int = 128,
        lr: float = 1e-3,
        beta: float = 0.1,
        beta_warmup: int = 100,
    ):
        """
        Initialize the VAE.
        
        Parameters:
        - vocab_size: Number of unique tokens
        - seq_len: Sequence length (512)
        - embed_dim: Embedding dimension for tokens
        - hidden_dim: Hidden layer dimension
        - latent_dim: Latent space dimension
        - lr: Learning rate
        - beta: Weight for KL term
        - beta_warmup: Epochs to anneal beta from 0 to target
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.beta = beta
        self.beta_warmup = beta_warmup
        
        # Embedding layer: token IDs -> dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Encoder: embedded sequence -> latent distribution
        self.encoder = nn.Sequential(
            nn.Linear(seq_len * embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder: latent -> logits over vocabulary for each position
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, seq_len * vocab_size),  # Output logits for each position
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
        embedded = embedded.view(embedded.size(0), -1)  # (batch, seq_len * embed_dim)
        h = self.encoder(embedded)
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
        logits = self.decoder(z)  # (batch, seq_len * vocab_size)
        logits = logits.view(-1, self.seq_len, self.vocab_size)  # (batch, seq_len, vocab_size)
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
        x, seq_lens = batch  # x: (batch, seq_len), seq_lens: (batch,)
        logits, mu, logvar = self(x)
        
        # Cross-entropy loss for reconstruction
        # logits: (batch, seq_len, vocab_size), x: (batch, seq_len)
        recon_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),  # (batch*seq_len, vocab_size)
            x.view(-1),  # (batch*seq_len,)
            reduction='mean'
        )
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
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
        logits = self.decode(z)  # (n_samples, seq_len, vocab_size)
        # Get most likely token at each position
        tokens = torch.argmax(logits, dim=-1)  # (n_samples, seq_len)
        return tokens
    
    def reconstruct(self, x):
        """Reconstruct input sequences."""
        mu, logvar = self.encode(x)
        z = mu  # Use mean for deterministic reconstruction
        logits = self.decode(z)
        tokens = torch.argmax(logits, dim=-1)
        return tokens

