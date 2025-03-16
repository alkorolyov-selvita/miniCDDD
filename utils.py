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


def save_models(lightning_module, save_dir="./models", scaler=None):
    """
    Save all models and necessary files

    Args:
        lightning_module: Trained PyTorch Lightning module
        save_dir: Directory to save models
        scaler: StandardScaler instance for denormalization (optional)
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

    # Save the scaler if provided
    if scaler is not None:
        joblib.dump(scaler, os.path.join(save_dir, "scaler.joblib"))


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


def load_scaler(scaler_path):
    """Load the StandardScaler"""
    return joblib.load(scaler_path)


def build_classifier(encoder, classifier, scaler):
    """Build a classifier model with scaler transformation"""
    scaler_layer = ScalerTransformLayer(scaler)
    return ClassifierWithScaler(encoder, classifier, scaler_layer)
