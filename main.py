# # miniCDDD

import torch
import pandas as pd

from preprocess import preprocess_dataset, mem
from tokens import tokenize_dataset
from train import train_minicddd

KEEP_STEREO = False
DESCRIPTORS = [
    'MolLogP', 'MolMR', 'BalabanJ', 'NumHAcceptors',
    'NumHDonors', 'NumValenceElectrons', 'TPSA',
]


def prepare_dataset(filename, smiles_col=None, len_quantile=0.99):
    if smiles_col is None:
        df = pd.read_csv(filename, header=None, names=['smiles'])
    else:
        df = pd.read_csv(filename).rename(columns={smiles_col: 'smiles'})[['smiles']]

    df, scaler = preprocess_dataset(df, KEEP_STEREO, DESCRIPTORS)
    df, lookup_table, max_length = tokenize_dataset(df, len_quantile)

    return df, scaler, lookup_table, max_length


if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('medium')

    # df, scaler, lookup_table, max_length = mem.cache(prepare_dataset)('data/250k_rndm_zinc_drugs_clean_3.csv', smiles_col='smiles')
    # df, scaler, lookup_table, max_length = mem.cache(prepare_dataset)('data/600k_chembl_filtered.smi.zst')
    df, scaler, lookup_table, max_length = mem.cache(prepare_dataset)('data/1k_test.smi')

    print('max_length', max_length)
    print(df.info())
    print(lookup_table)


    # Train the model
    model = train_minicddd(
        df=df,
        lookup_table=lookup_table,
        feature_columns=DESCRIPTORS,
        max_input_length=max_length,
        scaler=scaler,  # Pass the scaler directly
        batch_size=128,
        epochs=1,
        output_dir='./1k_test',
    )

    print("Training complete!")