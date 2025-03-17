from pathlib import Path

import torch
import json
import os
import joblib
from models import MiniCDDD, Encoder, Decoder, Classifier, ScalerTransformLayer, ClassifierWithScaler, Seq2SeqModel


def save_lookup_table(lookup_table, filename):
    """Save the lookup table to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(lookup_table, f, indent=2)


def load_lookup_table(filename):
    """Load the lookup table from a JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)


def save_models(lightning_module, save_dir="./models", scaler=None, max_input_length=None):
    """
    Save all models and necessary files

    Args:
        lightning_module: Trained PyTorch Lightning module
        save_dir: Directory to save models
        scaler: StandardScaler instance for denormalization (optional)
        max_input_length: Maximum input sequence length (optional)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    models_dir = save_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save the full model
    torch.save(lightning_module.model.state_dict(), models_dir /"miniCDDD_model.pt")

    # Save the encoder
    torch.save(lightning_module.get_encoder().state_dict(), models_dir /"encoder_model.pt")

    # Save the decoder
    torch.save(lightning_module.get_decoder().state_dict(), models_dir /"decoder_model.pt")

    # Save the classifier
    torch.save(lightning_module.get_classifier().state_dict(), models_dir /"classifier_model.pt")

    # Save the scaler if provided
    if scaler is not None:
        joblib.dump(scaler, save_dir /"scaler.joblib")

    # Save max_input_length if provided
    if max_input_length is not None:
        with open(save_dir /"max_input_length.json", 'w') as f:
            json.dump({"max_input_length": int(max_input_length)}, f)


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
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model


def load_encoder(model_path, vocab_size, latent_dim=512):
    """Load just the encoder component"""
    encoder = Encoder(vocab_size, latent_dim)
    encoder.load_state_dict(torch.load(model_path, weights_only=True))
    return encoder


def load_decoder(model_path, vocab_size, latent_dim=512):
    """Load just the decoder component"""
    decoder = Decoder(vocab_size, latent_dim)
    decoder.load_state_dict(torch.load(model_path, weights_only=True))
    return decoder


def load_classifier(model_path, latent_dim=512, output_dim=7):
    """Load just the classifier component"""
    classifier = Classifier(latent_dim, output_dim)
    classifier.load_state_dict(torch.load(model_path, weights_only=True))
    return classifier


def load_scaler(scaler_path):
    """Load the StandardScaler"""
    return joblib.load(scaler_path)


def load_max_input_length(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        return data.get('max_input_length')


def build_classifier(encoder, classifier, scaler):
    """Build a classifier model with scaler transformation"""
    scaler_layer = ScalerTransformLayer(scaler)
    return ClassifierWithScaler(encoder, classifier, scaler_layer)
