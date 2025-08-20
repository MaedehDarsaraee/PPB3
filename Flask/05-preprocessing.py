from rdkit import Chem
from rdkit.Chem.MolStandardize import charge

#Preprocess a SMILES string
def preprocess_smiles(smi: str) -> str:
    try:
        # Step 1: Parse SMILES
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None

        # Step 2: converting the SMILES to the Non-isomeric format
        non_iso_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        if not non_iso_smiles:
            return None

        # Step 3: Remove counterions
        frags = Chem.GetMolFrags(Chem.MolFromSmiles(smi), asMols=True, sanitizeFrags=False)
        largest_frag = max(frags, key=lambda m: m.GetNumAtoms())
        cleaned_smiles = Chem.MolToSmiles(largest_frag, isomericSmiles=False)
        if not cleaned_smiles:
            return None

        # Step 4: Correct valence
        mol = Chem.MolFromSmiles(cleaned_smiles, sanitize=True)
        if mol is None:
            return None
        valence_corrected_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        if not valence_corrected_smiles:
            return None

        # Step 5: Neutralize
        uncharger = charge.Uncharger()
        mol = Chem.MolFromSmiles(valence_corrected_smiles)
        uncharged_mol = uncharger.uncharge(mol)
        final_smiles = Chem.MolToSmiles(uncharged_mol, isomericSmiles=False)

        return final_smiles

    except Exception as e:
        print(f"Error preprocessing {smi}: {e}")
        return None


def preprocess_smiles_list(smiles_list):
    valid = []
    invalid = []
    for smi in smiles_list:
        processed = preprocess_smiles(smi)
        if processed:
            valid.append(processed)
        else:
            invalid.append(smi)
    return valid, invalid
