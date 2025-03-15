import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encoder for SMILES sequence"""

    def __init__(self, vocab_size, latent_dim=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim

        # GRU layers with increasing sizes
        self.gru1 = nn.GRU(vocab_size, 512, batch_first=True)
        self.gru2 = nn.GRU(512, 1024, batch_first=True)
        self.gru3 = nn.GRU(1024, 2048, batch_first=True)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.15)

        # Latent representation
        self.latent_projection = nn.Linear(512 + 1024 + 2048, latent_dim)

        # Gaussian noise
        self.gaussian_noise = GaussianNoise(0.05)

    def forward(self, x):
        # One-hot encode the input
        x_one_hot = F.one_hot(x, self.vocab_size).float()

        # Apply dropout
        x_one_hot = self.dropout(x_one_hot)

        # Pass through GRU layers
        _, h1 = self.gru1(x_one_hot)
        # Make h1 contiguous before transposing
        h1_transposed = h1.contiguous().transpose(0, 1)
        x2, h2 = self.gru2(h1_transposed)
        # Make h2 contiguous before transposing
        h2_transposed = h2.contiguous().transpose(0, 1)
        x3, h3 = self.gru3(h2_transposed)

        # Concatenate hidden states
        h1 = h1.squeeze(0)
        h2 = h2.squeeze(0)
        h3 = h3.squeeze(0)
        hidden_concat = torch.cat([h1, h2, h3], dim=1)

        # Project to latent space
        latent = torch.tanh(self.latent_projection(hidden_concat))

        # Add Gaussian noise
        latent = self.gaussian_noise(latent)

        return latent


class Decoder(nn.Module):
    """Decoder for SMILES sequence"""

    def __init__(self, vocab_size, latent_dim=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim

        # Project latent to GRU initial states
        self.latent_to_states = nn.Linear(latent_dim, 512 + 1024 + 2048)

        # GRU layers
        self.gru1 = nn.GRU(vocab_size, 512, batch_first=True)
        self.gru2 = nn.GRU(512, 1024, batch_first=True)
        self.gru3 = nn.GRU(1024, 2048, batch_first=True)

        # Output projection
        self.output = nn.Linear(2048, vocab_size)

    def forward(self, x, latent):
        batch_size = x.size(0)

        # One-hot encode the input
        x_one_hot = F.one_hot(x, self.vocab_size).float()

        # Project latent to initial states
        states = F.relu(self.latent_to_states(latent))

        # Split the states for each GRU layer and make them contiguous
        h1 = states[:, :512].contiguous().unsqueeze(0)  # Add contiguous() here
        h2 = states[:, 512:512 + 1024].contiguous().unsqueeze(0)  # Add contiguous() here
        h3 = states[:, 512 + 1024:].contiguous().unsqueeze(0)  # Add contiguous() here

        # Pass through GRU layers
        x1, _ = self.gru1(x_one_hot, h1)
        x2, _ = self.gru2(x1, h2)
        x3, _ = self.gru3(x2, h3)

        # Project to vocabulary
        logits = self.output(x3)

        return logits


class Classifier(nn.Module):
    """Classification model for molecular properties"""

    def __init__(self, latent_dim=512, output_dim=7):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # MLP for property prediction
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, latent):
        return self.mlp(latent)


class GaussianNoise(nn.Module):
    """Add Gaussian noise for regularization"""

    def __init__(self, std=0.05):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


class MiniCDDD(nn.Module):
    """Complete miniCDDD model combining encoder, decoder, and classifier"""

    def __init__(self, vocab_size, latent_dim=512, prop_dims=7):
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim

        self.encoder = Encoder(vocab_size, latent_dim)
        self.decoder = Decoder(vocab_size, latent_dim)
        self.classifier = Classifier(latent_dim, prop_dims)

    def forward(self, encoder_inputs, decoder_inputs):
        # Encode input sequence
        latent = self.encoder(encoder_inputs)

        # Get reconstruction and property predictions
        reconstruction_logits = self.decoder(decoder_inputs, latent)
        classification_output = self.classifier(latent)

        return reconstruction_logits, classification_output

    def encode(self, encoder_inputs):
        """Encode a batch of inputs to the latent space"""
        return self.encoder(encoder_inputs)

    def decode(self, decoder_inputs, latent):
        """Decode from latent space to output sequence"""
        return self.decoder(decoder_inputs, latent)

    def predict_properties(self, latent):
        """Predict properties from latent space"""
        return self.classifier(latent)