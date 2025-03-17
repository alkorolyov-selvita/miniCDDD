from pathlib import Path

import numpy as np
import torch

from models import Seq2SeqModel
from utils import (
    load_lookup_table,
    load_encoder,
    load_decoder,
    load_classifier,
    load_scaler,
    build_classifier,
    load_max_input_length,
)
from tokens import SmilesTokenizer


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

        # Load max_input_length if available
        try:
            self.max_input_length = load_max_input_length(f"{model_dir}/max_input_length.json")
            if self.max_input_length is None:
                self.max_input_length = 100  # Default fallback
                print("Warning: max_input_length not specified in JSON, using default value of 100.")
        except:
            self.max_input_length = 100  # Default fallback
            print("Warning: max_input_length.json not found, using default max length of 100.")

        # Initialize the tokenizer
        self.tokenizer = SmilesTokenizer(lookup_table=self.lookup_table, max_length=self.max_input_length)

        # Load model components
        self.encoder = load_encoder(f"{model_dir}/encoder_model.pt", self.vocab_size)
        self.decoder = load_decoder(f"{model_dir}/decoder_model.pt", self.vocab_size)
        self.classifier = load_classifier(f"{model_dir}/classifier_model.pt")

        self.scaler = load_scaler(f"{model_dir}/scaler.joblib")

        self.property_model = build_classifier(
            self.encoder, self.classifier, self.scaler
        )

        self.seq2seq_model = Seq2SeqModel(self.encoder, self.decoder)

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

    def tokenize_smiles(self, smiles):
        """
        Convert a SMILES string to token IDs using the SmilesTokenizer

        Args:
            smiles: SMILES string to tokenize

        Returns:
            torch.Tensor of token IDs
        """
        # Use the SmilesTokenizer to tokenize the SMILES string
        token_ids = self.tokenizer.tokenize(smiles)

        # Convert to tensor and add batch dimension
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
            if idx == self.tokenizer.eos_id:
                break
            elif idx != self.tokenizer.sos_id and idx != self.tokenizer.padding_id:
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
            raise ValueError("Property model not available. Scaler might be missing.")

        # Tokenize the SMILES
        token_ids = self.tokenize_smiles(smiles)

        # Predict properties
        with torch.no_grad():
            predictions = self.property_model(token_ids)

        return predictions.cpu().numpy()[0]

    def canonicalize(self, smiles, max_length=None, temperature=0.0):
        """
        Canonicalize a SMILES string using the sequence-to-sequence model

        Args:
            smiles: Input SMILES string
            max_length: Maximum output sequence length (optional)
            temperature: Sampling temperature (0 for greedy decoding)

        Returns:
            Canonicalized SMILES string
        """
        # Use class max_length if not specified
        if max_length is None:
            max_length = self.max_input_length

        # Tokenize the input
        token_ids = self.tokenize_smiles(smiles)

        # Get the latent representation
        with torch.no_grad():
            latent = self.encoder(token_ids)

            # Start with SOS token
            current_token = torch.tensor([[self.tokenizer.sos_id]], device=self.device)
            output_tokens = [self.tokenizer.sos_id]

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
                if next_token == self.tokenizer.eos_id:
                    output_tokens.append(next_token)
                    break

                output_tokens.append(next_token)
                current_token = torch.tensor([[next_token]], device=self.device)

        # Convert token IDs back to SMILES
        return self.decode_tokens(torch.tensor(output_tokens))


class SMILESEncoder:
    """
    A lightweight class for encoding SMILES strings to latent representations
    using a pre-trained encoder model.
    """

    def __init__(self, save_dir="./output"):
        """
        Initialize the SMILES encoder with a trained encoder model

        Args:
            save_dir: Directory containing saved data
        """

        save_dir = Path(save_dir)
        model_dir = save_dir / "models"
        lookup_table_path = save_dir / "lookup_table.json"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load lookup table
        self.lookup_table = load_lookup_table(lookup_table_path)
        self.vocab_size = len(self.lookup_table)

        self.max_input_length = load_max_input_length(save_dir / 'max_input_length.json')

        # Initialize the tokenizer
        self.tokenizer = SmilesTokenizer(lookup_table=self.lookup_table, max_length=self.max_input_length)

        # Load just the encoder model
        self.encoder = load_encoder(model_dir / "encoder_model.pt", self.vocab_size)

        # Move encoder to device and set to evaluation mode
        self.encoder.to(self.device)
        self.encoder.eval()

    def encode_batch(self, smiles_list, batch_size=64):
        """
        Encode SMILES strings to latent representations

        Works with both a single SMILES string or a list of SMILES strings.
        Processes them in smaller batches to manage memory usage.

        Args:
            smiles_list: Single SMILES string or list of SMILES strings
            batch_size: Size of mini-batches for processing (adjust based on available GPU memory)

        Returns:
            - For a single SMILES: numpy array of shape (latent_dim,)
            - For a list of SMILES: numpy array of shape (n_smiles, latent_dim)
        """
        from tqdm import tqdm

        # Handle single SMILES string
        single_input = False
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
            single_input = True

        # Tokenize all SMILES at once
        tokenized = np.stack([self.tokenizer.tokenize(smi) for smi in smiles_list])

        # Convert to tensor
        token_tensor = torch.tensor(tokenized, device=self.device, dtype=torch.long)

        # Split into batches using torch.split
        batches = torch.split(token_tensor, batch_size)

        # List to store output vectors
        cddd_vec_list = []

        # Iterate over batches
        with torch.no_grad():  # Disable gradient computation for efficiency
            for batch in tqdm(batches, desc="Encoding SMILES"):
                batch_vec = self.encoder(batch).detach().cpu().numpy().astype(np.float32)  # Convert to float32
                cddd_vec_list.append(batch_vec)

        # Concatenate results
        cddd_vec = np.concatenate(cddd_vec_list, axis=0)

        # For single SMILES input, return the vector directly (not in a batch)
        if single_input:
            cddd_vec = cddd_vec[0]

        return cddd_vec

    def __call__(self, smiles, batch_size=64):
        """
        Allow using the class instance directly as a function.
        Forwards to encode_batch for both single SMILES and lists.

        Args:
            smiles: Input SMILES string or list of SMILES strings
            batch_size: Size of mini-batches for processing

        Returns:
            Same as encode_batch
        """
        return self.encode_batch(smiles, batch_size=batch_size)