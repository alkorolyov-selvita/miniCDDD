import os
from pathlib import Path

import torch
import tqdm
from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger

from models import MiniCDDD
from lightning_module import LitModel
from dataset import create_dataloaders
from utils import save_lookup_table, save_models


class TrainOnlyBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = tqdm.tqdm(
            disable=True,
        )
        return bar


def train_minicddd(
        df,
        lookup_table,
        feature_columns,
        max_input_length,
        scaler=None,
        latent_dim=512,
        batch_size=64,
        epochs=100,
        learning_rate=5e-4,
        scheduler_decay=0.9,
        scheduler_epochs=10,
        output_dir='./output',
        seed=42,
):
    """
    Train the miniCDDD model

    Args:
        df: Preprocessed DataFrame with tokenized SMILES
        lookup_table: Token to index mapping
        feature_columns: List of column names for property prediction
        max_input_length: Max length of token input array
        scaler: StandardScaler instance used to normalize features
        latent_dim: Dimension of the latent space
        batch_size: Batch size for training
        epochs: Maximum number of training epochs
        learning_rate: Initial learning rate
        scheduler_decay: Decay rate for learning rate scheduler
        scheduler_epochs: Number of steps between learning rate updates
        output_dir: Directory to save outputs
        seed: Random seed for reproducibility

    Returns:
        Trained LitModel model
    """
    # Set random seeds for reproducibility
    seed_everything(seed)

    # Create directory for outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        df,
        feature_columns,
        batch_size=batch_size,
        seed=seed
    )

    # Create model
    vocab_size = len(lookup_table)
    model = MiniCDDD(
        vocab_size=vocab_size,
        latent_dim=latent_dim,
        prop_dims=len(feature_columns)
    )

    # Create Lightning module
    lit_model = LitModel(
        model=model,
        learning_rate=learning_rate,
        scheduler_decay=scheduler_decay,
        scheduler_epochs=scheduler_epochs
    )

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename='minicddd-{epoch:02d}-{val_total_loss:.4f}',
        monitor='val_total_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )

    early_stopping = EarlyStopping(
        monitor='val_total_loss',
        patience=5,
        mode='min'
    )

    tqdm_train_bar = TrainOnlyBar()

    csv_logger = CSVLogger(
        save_dir=os.path.join(output_dir, 'logs'),
        name=None,
    )

    # Create trainer
    trainer = Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback, early_stopping, tqdm_train_bar],
        logger=[csv_logger],
        precision='bf16-true',
        log_every_n_steps=50,
        deterministic=True,
    )

    # Train the model
    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    # Save models
    save_models(
        lightning_module=lit_model,
        save_dir=output_dir,
        lookup_table=lookup_table,
        scaler=scaler,
        max_input_length=max_input_length,
    )

    return lit_model