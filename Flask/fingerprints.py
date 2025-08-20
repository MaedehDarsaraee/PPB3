import os
import uuid
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, RDKFingerprint, Draw
from mhfp.encoder import MHFPEncoder

# Global Counters and Caches
ecfp4_counter = 0
rdkit_counter = 0
atompair_counter = 0
layered_counter = 0
mhfp6_counter = 0
ecfp6_counter = 0
fused_counter = 0

query_file_cache = {}
nn_file_cache = {}

# MHFP encoder (reusable object)
mhfp_encoder = MHFPEncoder()

# Fingerprint Calculation Functions

def calculate_ecfp4(smiles, radius=2, n_bits=4096):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits) if mol else None
    except Exception as e:
        print(f"Error calculating ECFP4 for {smiles}: {e}")
        return None


def calculate_atompair(smiles, n_bits=4096):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits) if mol else None
    except Exception as e:
        print(f"Error calculating AtomPair for {smiles}: {e}")
        return None


def calculate_layered(smiles, fp_size=4096, n_bits_per_hash=2, min_path=1, max_path=7):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.RDKFingerprint(mol, minPath=min_path, maxPath=max_path,
                                   fpSize=fp_size, nBitsPerHash=n_bits_per_hash, useHs=True) if mol else None
    except Exception as e:
        print(f"Error calculating Layered fingerprint for {smiles}: {e}")
        return None


def calculate_rdkit(smiles, fp_size=4096):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return RDKFingerprint(mol, fpSize=fp_size) if mol else None
    except Exception as e:
        print(f"Error calculating RDKit fingerprint for {smiles}: {e}")
        return None


def calculate_mhfp6(smiles, length=4096, radius=3):
    try:
        return mhfp_encoder.secfp_from_smiles(smiles, length=length, radius=radius,
                                              rings=True, kekulize=True, sanitize=False)
    except Exception as e:
        print(f"Error calculating MHFP6 for {smiles}: {e}")
        return None


def calculate_ecfp6(smiles, radius=3, n_bits=4096):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits) if mol else None
    except Exception as e:
        print(f"Error calculating ECFP6 for {smiles}: {e}")
        return None


def fuse_fingerprints(smiles):
    try:
        ecfp4_fp = calculate_ecfp4(smiles)
        mhfp6_fp = calculate_mhfp6(smiles)
        if ecfp4_fp is not None and mhfp6_fp is not None:
            return np.concatenate((ecfp4_fp, mhfp6_fp), axis=0).astype(np.int8)
        return None
    except Exception as e:
        print(f"Error fusing fingerprints for {smiles}: {e}")
        return None


# Map for easier selection in app.py
FINGERPRINT_FUNCTIONS = {
    "ECFP4": calculate_ecfp4,
    "AtomPair": calculate_atompair,
    "Layered": calculate_layered,
    "RDKit": calculate_rdkit,
    "MHFP6": calculate_mhfp6,
    "ECFP6": calculate_ecfp6,
    "Fused": fuse_fingerprints,
}

# Nearest Neighbor Search 
def find_nearest_neighbors_with_jar(smiles, target_id, fingerprint_data,
                                    db_file, num_neighbors=50,
                                    fp_type="ECfp4", scoring_method="TANIMOTO"):
    global ecfp4_counter, rdkit_counter, atompair_counter, layered_counter
    global mhfp6_counter, ecfp6_counter, fused_counter
    global query_file_cache, nn_file_cache

    jar_paths = {
        "ECfp4": "PPB3_ECFP4.jar",
        "RDKit": "PPB3_RDKIT.jar",
        "AtomPair": "PPB3_ATOMPAIR.jar",
        "Layered": "PPB3_LAYERED.jar",
        "MHFP6": "PPB3_MHFP6.jar",
        "ECfp6": "PPB3_ECFP6.jar",
        "Fused": "PPB3_FUSED.jar",
    }

    if fp_type not in jar_paths:
        raise ValueError(f"Unsupported fingerprint type: {fp_type}")
    jar_path = jar_paths[fp_type]

    # Cache key based on SMILES and fingerprint type
    cache_key = f"{smiles}_{fp_type}"
    if cache_key in nn_file_cache:
        return nn_file_cache[cache_key]

    # Counter selection
    counters = {
        "ECfp4": "ecfp4_counter",
        "RDKit": "rdkit_counter",
        "AtomPair": "atompair_counter",
        "Layered": "layered_counter",
        "MHFP6": "mhfp6_counter",
        "ECfp6": "ecfp6_counter",
        "Fused": "fused_counter",
    }
    globals()[counters[fp_type]] += 1
    query_index = globals()[counters[fp_type]]

    # Build query file
    query_fp_file_path = f"queries/{fp_type.lower()}_q{query_index}.txt"
    os.makedirs(os.path.dirname(query_fp_file_path), exist_ok=True)
    with open(query_fp_file_path, "w") as query_fp_file:
        query_fp_file.write(f"SMILES\tCPDid\tTARid\t{fp_type}\n")
        query_fp_file.write(f"{smiles}\tquery_{query_index}\t{target_id}\t{fingerprint_data}\n")

    query_file_cache[cache_key] = query_fp_file_path

    # Output file
    output_file_path = f"results/{fp_type.lower()}_q{uuid.uuid4().hex[:8]}_nn.txt"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    try:
        sub.run(
            [
                "java", "-Xmx4G", "-cp", jar_path, "aid.KNN1ss1",
                query_fp_file_path, db_file, fp_type, scoring_method,
                str(num_neighbors), output_file_path, fp_type
            ],
            check=True
        )
        nn_file_cache[cache_key] = output_file_path
        return output_file_path
    except sub.CalledProcessError as e:
        raise RuntimeError(f"Error running JAR: {e}")

# Molecule Image Generator (optional)
def generate_molecule_image(smiles, compound_id):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            image_path = f"static/molecules/{compound_id}.png"
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            Draw.MolToFile(mol, image_path)
            return image_path
    except Exception as e:
        print(f"Error generating image for SMILES {smiles}: {e}")
    return None
