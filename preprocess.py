import logging
import warnings
from typing import Dict, Optional

import datamol as dm
import pandas as pd
import numpy as np
from rdkit.Chem.SaltRemover import SaltRemover
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pandas.api.types import is_integer_dtype, is_float_dtype
from joblib import Parallel, delayed, Memory
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
# from chembl_structure_pipeline import get_parent_mol, standardize_mol
from chembl_structure_pipeline.exclude_flag import exclude_flag
from chembl_structure_pipeline.standardizer import (update_mol_valences, remove_sgroups_from_mol, kekulize_mol,
                                                    remove_hs_from_mol, normalize_mol, uncharge_mol)

mem = Memory('.cache', verbose=False)

SALTS_SOLVENTS= """[Cl,Br,I]
[Li,Na,K,Ca,Mg]
[O,N]
[N](=O)(O)O
[P](=O)(O)(O)O
[P](F)(F)(F)(F)(F)F
[S](=O)(=O)(O)O
[CH3][S](=O)(=O)(O)
c1cc([CH3])ccc1[S](=O)(=O)(O)	p-Toluene sulfonate
[CH3]C(=O)O	  Acetic acid
FC(F)(F)C(=O)O	  TFA
OC(=O)C=CC(=O)O	  Fumarate/Maleate
OC(=O)C(=O)O	  Oxalate
OC(=O)C(O)C(O)C(=O)O	  Tartrate
C1CCCCC1[NH]C1CCCCC1	  Dicylcohexylammonium
F[B-](F)(F)F	Tetrafluoroboranuide
NC(CCCNC(=N)N)C(=O)O	Arginine
CN(C)CCO	Deanol
CCN(CC)CCO	2-(Diethylamino)ethanol
NCCO	Ethanolamine
CNCC(O)C(O)C(O)C(O)CO	DiMeglumine
CC(=O)O	Acetate
CC(=O)NCC(=O)O	Aceturate
CCCCCCCCCCCCCCCCCC(=O)O	Stearate
OC(=O)CCCCC(=O)O	Adipate
[Al]	Aluminium
N	Ammonium
OCC(O)C1OC(=O)C(=C1O)O	Ascorbate
NC(CC(=O)O)C(=O)O	Aspartate
[Ba]	Barium
C(Cc1ccccc1)NCc2ccccc2	Benethamine
C(CNCc1ccccc1)NCc2ccccc2	Benzathine
OC(=O)c1ccccc1	Benzoate
OS(=O)(=O)c1ccccc1	Besylate
[Bi]	Bismuth
Br	Bromide
CCCC=O	Butyraldehyde
CCCC(=O)OCC	Ethyl Butanoate
[Ca]	Calcium
CC1(C)C2CCC1(CS(=O)(=O)O)C(=O)C2	Camsylate
OC(=O)O	Carbonate
Cl	Chloride
[CH3][N+]([CH3])([CH3])CCO	Choline
OC(=O)CC(O)(CC(=O)O)C(=O)O	Citrate
OS(=O)(=O)c1ccc(Cl)cc1	Closylate
OS(=O)(=O)NC1CCCCC1	Cyclamate
OC(=O)C(Cl)Cl	Dichloroacetate
CCNCC	Diethylamine
CC(C)(N)CO	Dimethylethanolamine
OCCNCCO	Diolamine
NCCN	Edamine
OS(=O)(=O)CCS(=O)(=O)O	Edisylate
OCCN1CCCC1	Epolamine
CC(C)(C)N	Erbumine
CCCCCCCCCCCCOS(=O)(=O)O	Estolate
CCS(=O)(=O)O	Esylate
CCOS(=O)(=O)O	Ethylsulfate
F	Fluoride
OC=O	Formate
OCC(O)C(O)C(O)C(O)C(O)C(=O)O	Gluceptate
OCC(O)C(O)C(O)C(O)C(=O)O	Gluconate
OC1OC(C(O)C(O)C1O)C(=O)O	Glucuronate
NC(CCC(=O)O)C(=O)O	Glutamate
OCC(O)CO	Glycerate
OCC(O)COP(=O)(O)O	Glycerophosphate
F[P](F)(F)(F)(F)F	Hexafluorophosphate
OP=O	Hypophosphite
I	Iodide
OCCS(=O)(=O)O	Isethionate
[K]	Potassium
CC(O)C(=O)O	Lactate
OCC(O)C(OC1OC(CO)C(O)C(O)C1O)C(O)C(O)C(=O)O	Lactobionate
[Li]	Lithium
NCCCCC(N)C(=O)O	Lysine
OC(CC(=O)O)C(=O)O	Malate
OC(=O)C=CC(=O)O	Maleate and Fumarate
CS(=O)(=O)O	Mesylate
OP(=O)=O	Metaphosphate
COS(=O)(=O)O	Methosulfate
[Mg]	Magnesium
OP(=O)(O)F	Monofluorophosphate
[Na]	Sodium
OS(=O)(=O)c1cccc2c(cccc12)S(=O)(=O)O	Napadisilate
OS(=O)(=O)c1ccc2ccccc2c1	Napsylate
O[N](=O)O	Nitrate
OC(=O)C(=O)O	Oxalate
CCCCCCCCCCCCCCCC(=O)O	Palmitate
OC(=O)c1cc2ccccc2c(Cc3c(O)c(cc4ccccc34)C(=O)O)c1O	Pamoate
OCl(O)(O)O	Perchlorate
Nc1ccc(cc1)P(=O)(O)O	Phosphanilate
OP(=O)(O)O	Phosphate
OP(O)(=O)OP(O)(O)=O Pyrophosphate
Oc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]	Picrate
C1CNCCN1	Piperazine
CC(O)CO	Propylene Glycol
O=C1NS(=O)(=O)c2ccccc12	Saccharin
OC(=O)c1ccccc1O	Salicylate
[Ag]	Silver
[Sr]	Strontium
OC(=O)CCC(=O)O	Succinate
OS(=O)(=O)O	Sulfate
OC(=O)c1cccc(c1O)S(=O)(=O)O	Sulfosalicylate
[S-2]	Sulphide
OC(=O)c1ccc(cc1)C(=O)O	Terephthalate
Cc1ccc(cc1)S(=O)(=O)O	Tosylate
Oc1cc(Cl)c(Cl)cc1Cl	Triclofenate
CCN(CC)CC	Triethylamine
OC(=O)C(c1ccccc1)(c2ccccc2)c3ccccc3	Trifenatate
OC(=O)C(F)(F)F	Triflutate
NC(CO)(CO)CO	Tromethamine
CCCCC1CCC(CC1)C(=O)O	Buciclate
CCCC(=O)O	Butyrate
CCCCCC(=O)O	Caproate
CC12CCC(CC1)(C=C2)C(=O)O	Cyclotate
OC(=O)CCC1CCCC1	Cypionate
CN(C)CCC(=O)O	Daproate
OC(=O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O	EDTA
CCCCCCCCC=CCCCCCCCC(=O)O	Elaidate and oleate
CCCCCCC(=O)O	Enanthate
CCOC(=O)O	Etabonate
COCCO	Ethanediol
OC(=O)CNC(=O)c1ccccc1	Etiprate
CCC(CC)C(=O)OCO	Etzadroxil
CCCCCCCCCCCCCCOP(=O)(O)O	Fostedate
OC(=O)c1occc1	Furoate
OC(=O)c1ccccc1C(=O)c2ccc(O)cc2	Hybenzate
CCCCCCCCCCCC(=O)O	Laurate
CC=C(C)C(=O)O	Mebutate
COC(=O)CC(O)(CCCC(C)(C)O)C(=O)O	Mepesuccinate
OC(=O)c1cccc(c1)S(=O)(=O)O	Metazoate
CSCCC(N)C(=O)C	Methionil
OC(=O)c1cccnc1	Nicotinate
OO	Peroxide
OC(=O)CCc1ccccc1	Phenpropionate
OC(=O)Cc1ccccc1	Phenylacetate
CC(C)(C)C(=O)O	Pivalate
CCC(=O)O	Propionate
CC(C)(C)CC(=O)O	Tebutate
OCCN(CCO)CCO	Trolamine
CCCCCCCCCCC(=O)O	Undecylate
OC(=O)CCCCCCCCC=C	Undecylenate
CCCCC(=O)O	Valerate
O	Water
OC(=O)c1ccc2ccccc2c1O	Xinafoate
[Zn]	Zinc
c1c[nH]cn1	Imidazole
OCCN1CCOCC1	4-(2-Hydroxyethyl)morpholine
CC(=O)Nc1ccc(cc1)C(=O)O	4-Acetamidobenzoic acid
CC1(C)C(CCC1(C)C(=O)O)C(=O)O	Camphoric acid
CCCCCCCCCC(=O)O	Capric acid
CCCCCCCC(=O)O	Caprylic acid
OC(=O)C=Cc1ccccc1	Cinnamic acid
OC(C(O)C(O)C(=O)O)C(O)C(=O)O	Mucic acid
OC(=O)c1cc(O)ccc1O	Gentisic acid
OC(=O)CCCC(=O)O	Glutaric acid
OC(=O)CCC(=O)C(=O)O	2-Oxoglutaric acid
OCC(=O)O	Glycolic acid
CC(C)C(=O)O	Isobutyric acid
OC(C(=O)O)c1ccccc1	Mandelic acid
OC(=O)c1cc(=O)nc(=O)n1	Orotic acid
OC(=O)C1CCC(=O)N1	Pyroglutamic acid
OC(C(O)C(=O)O)C(=O)O	Tartrate
SC#N	Thiocyanic acid
CI	Methyl Iodide
OS(=O)O	Sulfurous Acid
C1CCC(CC1)NC2CCCCC2	Dicyclohexylamine
OS(=O)(=O)C(F)(F)F	Triflate
Cc1cc(C)c(c(C)c1)S(=O)(=O)O	Mesitylene sulfonate
OC(=O)CC(=O)O	Malonic acid
OS(=O)(=O)F	Fluorosulfuric acid
CC(=O)OS(=O)(=O)O	Acetylsulfate
[H]	Proton
[Rb]	Rubidium
[Cs]	Cesium
[Fr]	Francium
[Be]	Beryllium
[Ra]	Radium
C(=O)C(O)C(O)C(O)C(O)C(=O)O	Glucuronate open form
CC(O)CN(C)C	Dimepranol
[OH2]	WATER
ClCCl	DICHLOROMETHANE
ClC(Cl)Cl	TRICHLOROMETHANE
ClC(Cl)(Cl)Cl   TETRACHLOROROMETHANE
CCOC(=O)C	ETHYL ACETATE
CO	METHANOL
CC(C)O	PROPAN-2-OL
CC(=O)C	ACETONE
CS(=O)C DMSO
C[S+](C)[O-]    DMSO
CN(C)C=O    DMF
O1CCOCC1    DIOXANE
C1CCCCC1    CYCLOHEXANE
c1ccccc1    BENZENE
n1ccccc1    PYRIDINE
N1CCCCC1    PIPERIDINE
CCO	ETHANOL
"""
REMOVER = SaltRemover(defnData=SALTS_SOLVENTS)

def chunked_ser(series: pd.Series, size: int = 100):
    """
    Yield successive n-sized chunks from a Pandas Series.

    Args:
        series (pd.Series): Input Pandas Series.
        n (int): Size of each chunk.

    Yields:
        pd.Series: Chunk of the input Series.
    """
    for i in range(0, len(series), size):
        yield series.iloc[i:i + size]


def keep_largets_fragment(mol):
    try:
        return dm.keep_largets_fragment(mol)
    except:
        logging.warning('Largest Fragment Fail: %s', dm.to_smiles(mol))


def standardize_mol(m, sanitize=True):
    try:
        m = update_mol_valences(m)
        m = remove_sgroups_from_mol(m)
        m = kekulize_mol(m)
        m = remove_hs_from_mol(m)
        m = normalize_mol(m)
        m = uncharge_mol(m)
        if sanitize:
            Chem.SanitizeMol(m)
        return m
    except:
        return None


# def preprocess_smiles(smi: Optional[str], keep_stereo=True) -> Optional[str]:
#     if not smi:
#         return None
#
#     with dm.without_rdkit_log():
#         mol = dm.to_mol(smi)
#
#         if not mol:
#             return None
#
#         if exclude_flag(mol):
#             return None
#
#         mol = standardize_mol(mol)
#
#         if not mol:
#             return None
#
#         # mol, exclude = get_parent_mol(mol)
#         # if exclude:
#         #     return None
#
#         mol = dm.remove_salts_solvents(mol, defn_data=SALTS_SOLVENTS)
#
#         frags = Chem.GetMolFrags(mol, sanitizeFrags=True)
#         if len(frags) != 1:
#             return None
#
#         mol = rdMolStandardize.IsotopeParent(mol)
#         # mol = rdMolStandardize.FragmentParent(mol, skipStandardize=True)
#         mol = rdMolStandardize.ChargeParent(mol)
#         mol = rdMolStandardize.TautomerParent(mol)
#     return dm.to_smiles(mol, isomeric=keep_stereo)


def preprocess_smiles_ser(smiles: pd.Series, keep_stereo=True):
    # return smiles.apply(preprocess_smiles, args=(keep_stereo,))

    with dm.without_rdkit_log():
        mols = smiles.fillna('').apply(dm.to_mol).dropna()

        # Exclude metals and multi boron
        mask = mols.apply(exclude_flag)
        mols = mols[~mask]

        # Standardize
        mols = mols.apply(standardize_mol).dropna()

        # Remove salts, solvents
        # remover = SaltRemover(defnData=SALTS_SOLVENTS)
        mols = mols.apply(REMOVER.StripMol).dropna()

        # Keep only single mols
        mask = mols.apply(Chem.GetMolFrags, sanitizeFrags=False).apply(lambda x: len(x) != 1)
        mols = mols[~mask]

        # Neutralize
        mols = mols.apply(uncharge_mol).dropna()

        # Convert back to smiles
        smiles = mols.apply(dm.to_smiles, isomeric=keep_stereo).replace('', None).dropna()

        return smiles


def preprocess_smiles_parallel(smiles: pd.Series, keep_stereo=True, chunk_size=100) -> pd.Series:
    total = len(smiles) // chunk_size + 1

    res = Parallel(n_jobs=-4)(
        delayed(preprocess_smiles_ser)(smi, keep_stereo)
        for smi in tqdm(chunked_ser(smiles, chunk_size), total=total, desc='Preprocessing SMILES'))

    return pd.concat(res)


def preprocess_smiles_inplace(df, keep_stereo=True, smiles_col='smiles'):
    """
    Smart way to preprocess smiles, processing only unique `smiles`
    column values and then expanding results on the whole dataframe.

    Args:
        df (pd.Dataframe): dataframe containing `smiles` column
        keep_stereo (bool): whether to keep stereo information or not.
        smiles_col (str): name of the column containing smiles. Default: 'smiles'

    Returns:
        None: modifies in place
    """

    df[smiles_col] = mem.cache(preprocess_smiles_parallel)(df[smiles_col] , keep_stereo)
    df.dropna(subset=smiles_col, inplace=True)

    # smiles_set = set(df[smiles_col])
    # mapping = mem.cache(preprocess_smiles_mapping)(smiles_set, keep_stereo)
    #
    # df[smiles_col] = df[smiles_col].map(mapping)
    # df.dropna(subset=[smiles_col], inplace=True)


""" ================= FINGERPRINTS ================ """

def calc_fps(smiles: pd.Series, radius, size):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=size)
    with dm.without_rdkit_log():
        mols = smiles.fillna('').apply(dm.to_mol).dropna()
        fps = fpgen.GetFingerprints(mols.values)
        return pd.Series(fps, index=mols.index).dropna()


def calc_fps_parallel_mapping(smiles, radius=2, size=2048, chunk_size=100):
    if not isinstance(smiles, pd.Series):
        smiles = pd.Series(list(smiles))
    smiles = smiles.rename('smiles')

    results = Parallel(n_jobs=-4)(delayed(calc_fps)(smi, radius, size) for smi in tqdm(chunked_ser(smiles, chunk_size), total=len(smiles) // chunk_size))

    results = (pd.concat(results)
                 .rename('fps')
                 .dropna())

    # original smiles to processed fps
    mapping = pd.concat([smiles[results.index], results], axis=1)
    mapping = (mapping.set_index('smiles').to_dict())['fps']
    return mapping


def add_fingerprints_inplace(df, radius=2, size=2048):
    """
    Smart way to calculate fingerprints for unique `smiles`
    and then expanding results on the whole dataframe.

    Args:
        df (pd.Dataframe): dataframe containing `smiles` column
        radius (int): radius of fingerprint
        size (int): size of fingerprint

    Returns:
        None: modifies in place
    """
    smiles_set = set(df.smiles)
    mapping = mem.cache(calc_fps_parallel_mapping)(smiles_set, radius, size)
    df['fps'] = df.smiles.map(mapping)
    df.dropna(subset=['fps'], inplace=True)

    # df.set_index('smiles', drop=False, inplace=True)
    # res = mem.cache(calc_fps_parallel_mapping)(df.smiles.drop_duplicates(), radius=radius, size=size)
    # df['fps'] = None
    # df.loc[res.index, 'fps'] = res
    # df.reset_index(drop=True, inplace=True)
    # df.dropna(inplace=True)


""" ================= DESCRIPTORS ================ """

def calc_descriptor_by_name(mol, desc_name: str):
    try:
        func = getattr(Descriptors, desc_name)
        return func(mol)
    except:
        return None


def calc_descriptors(smi, descriptors):
    nan_arr = pd.Series([np.nan] * len(descriptors), index=descriptors)

    if pd.isna(smi):
        return nan_arr

    with dm.without_rdkit_log():
        mol = dm.to_mol(smi)

        if mol is None:
            return nan_arr

        res = []
        for desc_name in descriptors:
            if not hasattr(Descriptors, desc_name):
                raise ValueError(f'Invalid descriptor name {desc_name}: Missing')

            if not callable(getattr(Descriptors, desc_name)):
                raise ValueError(f'Invalid descriptor name {desc_name}: Not callable')

            res.append(calc_descriptor_by_name(mol, desc_name))

    return pd.Series(res, index=descriptors)


def calc_descriptor_ser(smi_ser, descriptors):
    return smi_ser.apply(calc_descriptors, args=(descriptors,))


def calc_descriptors_parallel(smi_ser, descriptors):
    chunk_size = 100
    total = len(smi_ser) // chunk_size + 1

    res = Parallel(n_jobs=-4)(
        delayed(calc_descriptor_ser)(smi, descriptors)
        for smi in tqdm(chunked_ser(smi_ser, chunk_size), total=total, desc='Descriptors'))
    return pd.concat(res)


def add_descriptors_parallel(df, descriptors, smiles_col='smiles', rename=None):
    desc_df = mem.cache(calc_descriptors_parallel)(df[smiles_col], descriptors)
    res = pd.concat([df, desc_df], axis=1)
    if rename:
        res.rename(columns=rename, inplace=True)
    return res

""" ================= MISC ================ """

def randomize_smiles(smiles: pd.Series):
    def _randomize(smi):
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi), doRandom=True)

    def _randomize_ser(smi_ser: pd.Series):
        return smi_ser.apply(_randomize)

    chunk_size = 100
    total = len(smiles) // chunk_size + 1
    res = Parallel(n_jobs=-4)(
        delayed(_randomize_ser)(chunk)
        for chunk in tqdm(chunked_ser(smiles, chunk_size), total=total, desc='Randomizing SMILES')
    )
    return pd.concat(res)




def convert_to_dtypes32(df, na_int_cols=None):
    for col in df.columns:
        if is_integer_dtype(df[col].dtype):
            df[col] = df[col].astype(np.int32)
        elif is_float_dtype(df[col].dtype):
            df[col] = df[col].astype(np.float32)

    if na_int_cols is not None:
        for col in na_int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col]).astype('Int32')


def duplicated_activity_to_median(chembl_df):
    median_activities = chembl_df.groupby(['uniprot_id', 'cmpd_id'])['pchembl_value'].median()
    chembl_df.drop(['pchembl_value', 'activity_type'], axis=1, inplace=True)
    chembl_df = chembl_df.drop_duplicates(subset=['uniprot_id', 'cmpd_id'])
    chembl_df = pd.merge(chembl_df, median_activities, on=['uniprot_id', 'cmpd_id'], how='inner')
    return chembl_df


""" ================= MAIN FUNC ================ """

def preprocess_dataset(df, keep_stereo, descriptors):
    preprocess_smiles_inplace(df, smiles_col='smiles', keep_stereo=keep_stereo)
    df = add_descriptors_parallel(df, descriptors)

    scaler = StandardScaler()
    df.loc[:, descriptors] = scaler.fit_transform(df[descriptors].values)

    df['random_smiles'] = mem.cache(randomize_smiles)(df['smiles'])
    df.rename(columns={'smiles': 'canonical_smiles'}, inplace=True)
    return df, scaler
