import torch
import json
import os
import numpy as np
from models import MiniCDDD, Encoder, Decoder, Classifier


def save_lookup_table(lookup_table, filename):
    """Save the lookup table to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(lookup_table, f, indent=2)


def load_lookup_table(filename):
    """Load the lookup table from a JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)


def save_models(lightning_module, save_dir="./models", mean=None, std=None):
    """
    Save all models and necessary files

    Args:
        lightning_module: Trained PyTorch Lightning module
        save_dir: Directory to save models
        mean: Mean values for denormalization (optional)
        std: Standard deviation values for denormalization (optional)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save the full model
    torch.save(lightning_module.model.state_dict(), os.path.join(save_dir, "miniCDDD_model.pt"))

    # Save the encoder
    torch.save(lightning_module.get_encoder().state_dict(), os.path.join(save_dir, "encoder_model.pt"))

    # Save the decoder
    torch.save(lightning_module.get_decoder().state_dict(), os.path.join(save_dir, "decoder_model.pt"))

    # Save the classifier
    torch.save(lightning_module.get_classifier().state_dict(), os.path.join(save_dir, "classifier_model.pt"))

    # Save normalization parameters if provided
    if mean is not None and std is not None:
        np.savez(
            os.path.join(save_dir, "normalization_params.npz"),
            mean=mean,
            std=std
        )


def load_model(model_path, vocab_size, latent_dim=512, prop_dims=7):
    """
    Load a complete MiniCDDD model

    Args:
        model_path: Path to the saved model
        vocab_size: Size of the vocabulary
        latent_dim: Dimension of the latent space
        prop_dims: Dimension of the property output

    Returns:
        Loaded MiniCDDD model
    """
    model = MiniCDDD(vocab_size, latent_dim, prop_dims)
    model.load_state_dict(torch.load(model_path))
    return model


def load_encoder(model_path, vocab_size, latent_dim=512):
    """Load just the encoder component"""
    encoder = Encoder(vocab_size, latent_dim)
    encoder.load_state_dict(torch.load(model_path))
    return encoder


def load_decoder(model_path, vocab_size, latent_dim=512):
    """Load just the decoder component"""
    decoder = Decoder(vocab_size, latent_dim)
    decoder.load_state_dict(torch.load(model_path))
    return decoder


def load_classifier(model_path, latent_dim=512, output_dim=7):
    """Load just the classifier component"""
    classifier = Classifier(latent_dim, output_dim)
    classifier.load_state_dict(torch.load(model_path))
    return classifier


def load_normalization_params(params_path):
    """Load normalization parameters"""
    params = np.load(params_path)
    return params['mean'], params['std']


class DenormalizationLayer(torch.nn.Module):
    """Layer to denormalize the output of the classifier"""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x):
        return x * self.std + self.mean


def build_classifier_with_denormalization(encoder, classifier, mean, std):
    """Build a classifier model with denormalization"""
    denorm_layer = DenormalizationLayer(mean, std)

    class ClassifierWithDenorm(torch.nn.Module):
        def __init__(self, encoder, classifier, denorm_layer):
            super().__init__()
            self.encoder = encoder
            self.classifier = classifier
            self.denorm_layer = denorm_layer

        def forward(self, x):
            latent = self.encoder(x)
            props = self.classifier(latent)
            return self.denorm_layer(props)

    return ClassifierWithDenorm(encoder, classifier, denorm_layer)


def build_seq2seq_model(encoder, decoder):
    """Build a sequence-to-sequence model from encoder and decoder"""

    class Seq2SeqModel(torch.nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, encoder_inputs, decoder_inputs):
            latent = self.encoder(encoder_inputs)
            return self.decoder(decoder_inputs, latent)

    return Seq2SeqModel(encoder, decoder)