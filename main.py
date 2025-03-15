# # miniCDDD

import torch
import pandas as pd

from preprocess import preprocess_dataset
from tokens import tokenize_dataset
from train import train_minicddd

KEEP_STEREO = False
DESCRIPTORS = [
    'MolLogP', 'MolMR', 'BalabanJ', 'NumHAcceptors',
    'NumHDonors', 'NumValenceElectrons', 'TPSA'
]


def prepare_dataset(filename, smiles_col=None):
    if smiles_col is None:
        df = pd.read_csv(filename, header=None, names=['smiles'])
    else:
        df = pd.read_csv(filename).rename(columns={smiles_col: 'smiles'})[['smiles']]

    df, scaler = preprocess_dataset(df, KEEP_STEREO, DESCRIPTORS)
    df, lookup_table, max_length = tokenize_dataset(df)

    return df, scaler, lookup_table, max_length


if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)

    df, scaler, lookup_table, max_length = prepare_dataset('data/250k_rndm_zinc_drugs_clean_3.csv', smiles_col='smiles')
    print('max_length', max_length)
    print(df.info())
    print(lookup_table)

    # Train the model
    model = train_minicddd(
        df=df,
        lookup_table=lookup_table,
        feature_columns=DESCRIPTORS,
        batch_size=64,
        epochs=100,
        output_dir='./output'
    )

    print("Training complete!")




