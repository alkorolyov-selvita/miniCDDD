import re
import json
import numpy as np

from tqdm import tqdm

# Tokenization & Normalization documentation preserved from original
# For the tokenization process, we adopted the regular expression provided by [SmilesPE]...

# Define the regex pattern once at module level for reuse
SMILES_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
SMILES_REGEX = re.compile(SMILES_PATTERN)


def extract_tokens_from_smiles(smi):
    """Extract tokens from a SMILES string using the defined pattern."""
    return SMILES_REGEX.findall(smi)


class SmilesTokenizer:
    """A tokenizer for SMILES strings based on a lookup table."""

    def __init__(self, lookup_filename=None, lookup_table=None, max_length=73):
        """Initialize the tokenizer with either a lookup table file or a dictionary."""
        if lookup_filename:
            self.lookup_table = self.load_lookup_table(lookup_filename)
        elif lookup_table:
            self.lookup_table = lookup_table
        else:
            raise ValueError("Either lookup_filename or lookup_table must be provided")

        self.max_length = max_length
        try:
            self.padding_id = self.lookup_table['<PAD>']
            self.sos_id = self.lookup_table['< SOS >']
            self.eos_id = self.lookup_table['<EOS>']
            self.unk_id = self.lookup_table['<UNK>']
        except KeyError as e:
            raise ValueError("Invalid lookup table, special tokens missing:" + str(e))


    def load_lookup_table(self, filename):
        """Load lookup table from a JSON file."""
        with open(filename, 'r') as f:
            return json.load(f)

    def save_lookup_table(self, filename):
        """Save the lookup table to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.lookup_table, f, indent=2)

    def _pad_tokens_to_max_length(self, tokens):
        """Pad token list to the maximum length with start and end tokens."""
        if len(tokens) <= self.max_length - 2:
            return [self.sos_id] + tokens + [self.eos_id] + [self.padding_id] * (self.max_length - len(tokens) - 2)

        return [np.nan] * self.max_length

    def tokenize(self, smiles_string):
        """Tokenize a SMILES string according to the lookup table, handling unknown tokens."""
        tokens = extract_tokens_from_smiles(smiles_string)
        token_ids = [self.lookup_table.get(token, self.unk_id) for token in tokens]
        padded_tokens = self._pad_tokens_to_max_length(token_ids)
        return np.array(padded_tokens)

    def batch_tokenize(self, smiles_list):
        """Tokenize a batch of SMILES strings."""
        return np.array([self.tokenize(smiles) for smiles in smiles_list])


def build_lookup_table(smiles_list):
    """Build a lookup table from a list of SMILES strings."""
    all_tokens = set()
    for smi in tqdm(smiles_list, desc="Building token set"):
        all_tokens |= set(extract_tokens_from_smiles(smi))

    lookup_table = {token: idx for idx, token in enumerate(sorted(all_tokens))}
    # Add special tokens
    next_idx = len(lookup_table)
    lookup_table['<UNK>'] = next_idx
    lookup_table['<PAD>'] = next_idx + 1
    lookup_table['< SOS >'] = next_idx + 2
    lookup_table['<EOS>'] = next_idx + 3

    return lookup_table


def encode_smiles_with_lookup(df, lookup_table, input_col='random_smiles', output_col='canonical_smiles'):
    """Encode SMILES strings using the lookup table."""
    df['input_tokens'] = df[input_col].apply(lambda s:
                                             [lookup_table.get(token, lookup_table['<UNK>']) for token in
                                              extract_tokens_from_smiles(s)])
    df['output_tokens'] = df[output_col].apply(lambda s:
                                               [lookup_table.get(token, lookup_table['<UNK>']) for token in
                                                extract_tokens_from_smiles(s)])
    return df


def filter_by_token_length(df, threshold_quantile=0.95):
    """Filter out molecules with tokens longer than the threshold."""
    token_lengths = df['input_tokens'].apply(len)
    threshold = token_lengths.quantile(threshold_quantile)
    return df[token_lengths < threshold].copy()


def pad_dataset_tokens(df, lookup_table):
    """Pad tokens in the dataset to a consistent length."""
    max_length_input = df['input_tokens'].apply(len).max() + 2  # +2 for SOS and EOS
    max_length_output = df['output_tokens'].apply(len).max() + 2
    max_length = max(max_length_input, max_length_output)

    padding_id = lookup_table['<PAD>']
    sos_id = lookup_table['< SOS >']
    eos_id = lookup_table['<EOS>']

    df['input_tokens'] = df['input_tokens'].apply(lambda x:
                                                  [sos_id] + x + [eos_id] + [padding_id] * (max_length - len(x) - 2))
    df['output_tokens'] = df['output_tokens'].apply(lambda x:
                                                    [sos_id] + x + [eos_id] + [padding_id] * (max_length - len(x) - 2))

    return df, max_length


def tokenize_dataset(df):
    """High-level function to tokenize a dataset of SMILES strings."""
    # 1. Build lookup table from canonical SMILES
    lookup_table = build_lookup_table(df['canonical_smiles'])

    # 2. Encode SMILES using the lookup table
    df = encode_smiles_with_lookup(df, lookup_table)

    # 3. Filter out molecules with overly long token sequences
    df = filter_by_token_length(df)

    # 4. Pad tokens to consistent length
    df, max_length = pad_dataset_tokens(df, lookup_table)

    return df, lookup_table, max_length