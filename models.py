import re
from typing import Union, List

import numpy as np
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


class ScalerTransformLayer(torch.nn.Module):
    """Layer that applies StandardScaler inverse transform"""

    def __init__(self, scaler):
        super().__init__()
        self.register_buffer("mean", torch.tensor(scaler.mean_, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scaler.scale_, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale + self.mean


class ClassifierWithScaler(torch.nn.Module):
    def __init__(self, encoder, classifier, scaler_layer):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.scaler_layer = scaler_layer

    def forward(self, x):
        latent = self.encoder(x)
        props = self.classifier(latent)
        return self.scaler_layer(props)


class Seq2SeqModel(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_inputs, decoder_inputs):
        latent = self.encoder(encoder_inputs)
        return self.decoder(decoder_inputs, latent)


class CDDDEncoder(torch.nn.Module):
    """
    A self-contained SMILES encoder that packages the lookup table,
    tokenization logic, and encoder model into a single exportable unit.
    """

    def __init__(self, encoder_model, lookup_table, max_input_length):
        """Initialize with the necessary components"""
        super().__init__()
        self.encoder = encoder_model
        self.lookup_table = lookup_table
        self.max_input_length = max_input_length
        self.inv_lookup_table = {v: k for k, v in self.lookup_table.items()}

        # Extract special token IDs
        try:
            self.pad_id = self.lookup_table['<PAD>']
            self.sos_id = self.lookup_table['<SOS>']
            self.eos_id = self.lookup_table['<EOS>']
            self.unk_id = self.lookup_table['<UNK>']
        except KeyError as e:
            raise ValueError(f"Invalid lookup table, special token missing: {e}")

        # Compile regex pattern for tokenization
        self.smiles_pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.smiles_regex = re.compile(self.smiles_pattern)

    def extract_tokens_from_smiles(self, smi):
        """Extract tokens from a SMILES string using regex pattern"""
        return self.smiles_regex.findall(smi)

    def tokenize(self, smiles_string):
        """Tokenize a SMILES string and convert to padded token IDs"""
        tokens = self.extract_tokens_from_smiles(smiles_string)
        token_ids = [self.lookup_table.get(token, self.unk_id) for token in tokens]

        # Add SOS, EOS tokens and pad as needed
        if len(token_ids) <= self.max_input_length - 2:
            padded_tokens = [self.sos_id] + token_ids + [self.eos_id] + [self.pad_id] * (
                        self.max_input_length - len(token_ids) - 2)
        else:
            padded_tokens = [self.sos_id] + token_ids[: self.max_input_length - 2] + [self.eos_id]

        return np.array(padded_tokens)

    def forward(self, smiles_list: Union[str, List[str]], batch_size: int = 64) -> np.ndarray:
        """
        Encode SMILES strings directly to latent representations

        Args:
            smiles_list: A single SMILES string or a list of SMILES strings
            batch_size: Size of batches for processing (for memory management)

        Returns:
            Numpy array of latent representations
        """
        device = next(self.parameters()).device

        # Handle single SMILES string
        single_input = False
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
            single_input = True

        # Tokenize all SMILES
        tokenized = np.stack([self.tokenize(smi) for smi in smiles_list])

        # Convert to tensor
        token_tensor = torch.tensor(tokenized, device=device, dtype=torch.long)

        # Process in batches
        batches = torch.split(token_tensor, batch_size)
        latent_vectors = []

        self.encoder.eval()  # Set to evaluation mode
        with torch.no_grad():
            for batch in batches:
                batch_vector = self.encoder(batch).cpu().numpy()
                latent_vectors.append(batch_vector)

        # Combine results
        result = np.concatenate(latent_vectors, axis=0)

        # Return single vector for single input
        if single_input:
            return result[0]

        return result