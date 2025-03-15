import torch
import numpy as np
from utils import (
    load_lookup_table,
    load_model,
    load_encoder,
    load_decoder,
    load_classifier,
    load_normalization_params,
    build_classifier_with_denormalization,
    build_seq2seq_model
)
from tokens import extract_tokens_from_smiles


class MiniCDDDInference:
    """Inference class for using trained miniCDDD models for molecule property prediction and canonicalization"""

    def __init__(self, model_dir="./output/models", lookup_table_path="./output/lookup_table.json"):
        """
        Initialize inference with trained models

        Args:
            model_dir: Directory containing saved models
            lookup_table_path: Path to the lookup table JSON file
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load lookup table
        self.lookup_table = load_lookup_table(lookup_table_path)
        self.vocab_size = len(self.lookup_table)
        self.inv_lookup_table = {v: k for k, v in self.lookup_table.items()}

        # Load special token IDs
        self.pad_id = self.lookup_table.get('<PAD>', 0)
        self.sos_id = self.lookup_table.get('< SOS >', 1)
        self.eos_id = self.lookup_table.get('<EOS>', 2)
        self.unk_id = self.lookup_table.get('<UNK>', 3)

        # Load model components
        self.encoder = load_encoder(f"{model_dir}/encoder_model.pt", self.vocab_size)
        self.decoder = load_decoder(f"{model_dir}/decoder_model.pt", self.vocab_size)
        self.classifier = load_classifier(f"{model_dir}/classifier_model.pt")

        # Load normalization parameters if available
        try:
            self.mean, self.std = load_normalization_params(f"{model_dir}/normalization_params.npz")
        except FileNotFoundError:
            self.mean, self.std = None, None

        # Create combined models
        if self.mean is not None and self.std is not None:
            self.property_model = build_classifier_with_denormalization(
                self.encoder, self.classifier, self.mean, self.std
            )
        else:
            self.property_model = None

        self.seq2seq_model = build_seq2seq_model(self.encoder, self.decoder)

        # Move models to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.classifier.to(self.device)
        if self.property_model:
            self.property_model.to(self.device)
        self.seq2seq_model.to(self.device)

        # Set models to evaluation mode
        self.encoder.eval()
        self.decoder.eval()
        self.classifier.eval()
        if self.property_model:
            self.property_model.eval()
        self.seq2seq_model.eval()

    def tokenize_smiles(self, smiles, max_length=None):
        """
        Convert a SMILES string to token IDs

        Args:
            smiles: SMILES string to tokenize
            max_length: Maximum sequence length (optional)

        Returns:
            torch.Tensor of token IDs
        """
        # Extract tokens from SMILES
        tokens = extract_tokens_from_smiles(smiles)

        # Convert tokens to IDs, handling unknown tokens
        token_ids = [self.lookup_table.get(token, self.unk_id) for token in tokens]

        # Add SOS token at beginning
        token_ids = [self.sos_id] + token_ids

        # Add EOS token at end if not already present
        if token_ids[-1] != self.eos_id:
            token_ids.append(self.eos_id)

        # Pad to max_length if specified
        if max_length is not None and len(token_ids) < max_length:
            token_ids = token_ids + [self.pad_id] * (max_length - len(token_ids))

        return torch.tensor(token_ids, device=self.device, dtype=torch.long).unsqueeze(0)

    def decode_tokens(self, token_ids):
        """
        Convert token IDs back to a SMILES string

        Args:
            token_ids: Tensor of token IDs

        Returns:
            SMILES string
        """
        # Remove batch dimension if present
        if token_ids.dim() > 1:
            token_ids = token_ids[0]

        # Convert to list and stop at EOS token
        tokens = []
        for idx in token_ids.cpu().numpy():
            if idx == self.eos_id:
                break
            elif idx != self.sos_id and idx != self.pad_id:
                tokens.append(self.inv_lookup_table.get(idx, '<UNK>'))

        # Join tokens to form SMILES string
        return ''.join(tokens)

    def predict_properties(self, smiles):
        """
        Predict molecular properties from a SMILES string

        Args:
            smiles: SMILES string

        Returns:
            Numpy array of predicted properties
        """
        if self.property_model is None:
            raise ValueError("Property model not available. Normalization parameters might be missing.")

        # Tokenize the SMILES
        token_ids = self.tokenize_smiles(smiles)

        # Predict properties
        with torch.no_grad():
            predictions = self.property_model(token_ids)

        return predictions.cpu().numpy()[0]

    def canonicalize(self, smiles, max_length=100, temperature=0.0):
        """
        Canonicalize a SMILES string using the sequence-to-sequence model

        Args:
            smiles: Input SMILES string
            max_length: Maximum output sequence length
            temperature: Sampling temperature (0 for greedy decoding)

        Returns:
            Canonicalized SMILES string
        """
        # Tokenize the input
        token_ids = self.tokenize_smiles(smiles)

        # Get the latent representation
        with torch.no_grad():
            latent = self.encoder(token_ids)

            # Start with SOS token
            current_token = torch.tensor([[self.sos_id]], device=self.device)
            output_tokens = [self.sos_id]

            # Decode step by step
            for _ in range(max_length):
                # Predict next token
                with torch.no_grad():
                    logits = self.decoder(current_token, latent)
                    logits = logits[0, -1, :]

                    # Apply temperature if non-zero
                    if temperature > 0:
                        probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, 1).item()
                    else:
                        next_token = torch.argmax(logits).item()

                # Stop if EOS token or exceeded max length
                if next_token == self.eos_id:
                    output_tokens.append(next_token)
                    break

                output_tokens.append(next_token)
                current_token = torch.tensor([[next_token]], device=self.device)

        # Convert token IDs back to SMILES
        return self.decode_tokens(torch.tensor(output_tokens))