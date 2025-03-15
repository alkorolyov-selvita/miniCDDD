import re

# # Tokenization & Normalization

# For the tokenization process, we adopted the regular expression provided by [SmilesPE](https://github.com/XinhaoLi74/SmilesPE/blob/e5f27dfea0778966818ac0a9dd23ac646c62707d/SmilesPE/pretokenizer.py#L15-L18) for atom-wise tokenization. When it comes to padding and preparing both randomized and canonical smiles, our approach closely mirrors the original code. As for the normalization step, we strictly followed the methodology from the original source. This ensures a more stable training process for the classifier model. Without normalization, training this model would be challenging. Hence, we've saved the mean and standard deviation to ensure the model can revert predictions back to the original molecule descriptors values.

# Ref: https://github.com/XinhaoLi74/SmilesPE/blob/e5f27dfea0778966818ac0a9dd23ac646c62707d/SmilesPE/pretokenizer.py#L15-L18

def extract_atoms_from_smiles(smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    return [token for token in regex.findall(smi)]


def tokenization(df):
    # Extract unique atoms from the canonical_smiles and create lookup table
    all_atoms = set(atom for smiles in df['canonical_smiles'] for atom in extract_atoms_from_smiles(smiles))
    lookup_table = {atom: idx for idx, atom in enumerate(sorted(all_atoms))}

    # Tokenize random_smiles using the lookup table and create a new 'tokens' column
    df['input_tokens'] = df['random_smiles'].apply(lambda s: [lookup_table[atom] for atom in extract_atoms_from_smiles(s)])
    df['output_tokens'] = df['canonical_smiles'].apply(lambda s: [lookup_table[atom] for atom in extract_atoms_from_smiles(s)])
    return df, lookup_table


def pad_tokens_to_max_length_with_lookup(df, lookup_table):
    # Add a new ID for padding in the lookup table
    padding_id = max(lookup_table.values()) + 1
    lookup_table['<PAD>'] = padding_id
    lookup_table['<SOS>'] = padding_id + 1
    lookup_table['<EOS>'] = padding_id + 2

    # Find the maximum length from both the columns
    max_length_input = df['input_tokens'].apply(len).max() + 2 # +2 for <SOS> and <EOS>
    max_length_output = df['output_tokens'].apply(len).max() + 2 # +2 for <SOS> and <EOS>
    max_length = max(max_length_input, max_length_output)

    # Pad the tokens with the new padding ID
    df['input_tokens'] = df['input_tokens'].apply(lambda x: [lookup_table['<SOS>']] + x + [lookup_table['<EOS>']] + [padding_id]*(max_length - len(x) - 2))
    df['output_tokens'] = df['output_tokens'].apply(lambda x: [lookup_table['<SOS>']] + x + [lookup_table['<EOS>']] + [padding_id]*(max_length - len(x) - 2))

    return df, lookup_table, max_length


def normalize_dataframe(df):
    # Backup the 'canonical_smiles' and 'random_smiles' columns
    canonical_smiles_col = df['canonical_smiles'].copy()
    random_smiles_col = df['random_smiles'].copy()

    # Drop the 'canonical_smiles' and 'random_smiles' columns
    df = df.drop(columns=['canonical_smiles', 'random_smiles'])

    # Compute mean and standard deviation for normalization
    mean, std = df.mean(), df.std()

    # Normalize the dataframe
    df = (df - mean) / std

    # Add the 'canonical_smiles' and 'random_smiles' columns back to the dataframe
    df['canonical_smiles'] = canonical_smiles_col
    df['random_smiles'] = random_smiles_col

    return df, mean, std
