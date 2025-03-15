import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from lightning import seed_everything

class TeacherForcingDataset(Dataset):
    """Dataset for teacher forcing training of SMILES sequences"""

    def __init__(self, dataframe, feature_columns):
        """
        Initialize the dataset with a pandas DataFrame

        Args:
            dataframe (pd.DataFrame): DataFrame containing tokenized SMILES and features
            feature_columns (list): List of column names for feature prediction
        """
        self.df = dataframe
        self.feature_columns = feature_columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get tokens for encoder and decoder
        encoder_inputs = torch.tensor(self.df['input_tokens'].iloc[idx], dtype=torch.long)
        decoder_inputs = torch.tensor(self.df['output_tokens'].iloc[idx][:-1], dtype=torch.long)  # all except last
        decoder_outputs = torch.tensor(self.df['output_tokens'].iloc[idx][1:], dtype=torch.long)  # all except first

        # Get molecular features
        classifier_outputs = torch.tensor(
            self.df[self.feature_columns].iloc[idx].values,
            dtype=torch.float32
        )

        return {
            'encoder_inputs': encoder_inputs,
            'decoder_inputs': decoder_inputs,
            'decoder_outputs': decoder_outputs,
            'classifier_outputs': classifier_outputs
        }


def create_dataloaders(df, feature_columns, batch_size=64, val_split=0.2, num_workers=4, seed=42):
    """
    Create train and validation dataloaders from a dataframe

    Args:
        df (pd.DataFrame): DataFrame with tokenized SMILES
        feature_columns (list): List of column names for feature prediction
        batch_size (int): Batch size for training
        val_split (float): Fraction of data to use for validation
        num_workers (int): Number of workers for data loading
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Set random seed for reproducibility
    seed_everything(seed)

    train_df, val_df = train_test_split(df, test_size=val_split, random_state=seed)

    # Create datasets
    train_dataset = TeacherForcingDataset(train_df, feature_columns)
    val_dataset = TeacherForcingDataset(val_df, feature_columns)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader