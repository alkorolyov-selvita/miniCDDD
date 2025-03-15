import os
import torch
from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from models import MiniCDDD
from lightning_module import MiniCDDDLightning
from dataset import create_dataloaders
from utils import save_lookup_table, save_models


def train_minicddd(
        df,
        lookup_table,
        feature_columns,
        latent_dim=512,
        batch_size=64,
        epochs=100,
        learning_rate=5e-4,
        scheduler_decay=0.9,
        scheduler_steps=50000,
        output_dir='./output',
        seed=42
):
    """
    Train the miniCDDD model

    Args:
        df: Preprocessed DataFrame with tokenized SMILES
        lookup_table: Token to index mapping
        feature_columns: List of column names for property prediction
        latent_dim: Dimension of the latent space
        batch_size: Batch size for training
        epochs: Maximum number of training epochs
        learning_rate: Initial learning rate
        scheduler_decay: Decay rate for learning rate scheduler
        scheduler_steps: Number of steps between learning rate updates
        output_dir: Directory to save outputs
        seed: Random seed for reproducibility

    Returns:
        Trained MiniCDDDLightning model
    """
    # Set random seeds for reproducibility
    seed_everything(seed)

    # Create directory for outputs
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

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
    lightning_module = MiniCDDDLightning(
        model=model,
        learning_rate=learning_rate,
        scheduler_decay=scheduler_decay,
        scheduler_steps=scheduler_steps
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

    # Configure loggers
    tensorboard_logger = TensorBoardLogger(
        save_dir=os.path.join(output_dir, 'logs'),
        name='minicddd'
    )

    csv_logger = CSVLogger(
        save_dir=os.path.join(output_dir, 'logs'),
        name='minicddd_csv'
    )

    # Create trainer
    trainer = Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=[tensorboard_logger, csv_logger],
        log_every_n_steps=50,
        deterministic=True
    )

    # Train the model
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    # Save the models
    save_lookup_table(lookup_table, os.path.join(output_dir, 'lookup_table.json'))

    # Get mean and std from the original data for denormalization
    mean = df[feature_columns].mean().values
    std = df[feature_columns].std().values

    save_models(
        lightning_module=lightning_module,
        save_dir=models_dir,
        mean=mean,
        std=std
    )

    return lightning_module