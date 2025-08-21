import os
import re
import uuid
import subprocess as sub
from rdkit import Chem
from rdkit.Chem import Draw

# global counters and caches
ecfp4_counter = 0
rdkit_counter = 0
atompair_counter = 0
layered_counter = 0
mhfp6_counter = 0
ecfp6_counter = 0
fused_counter = 0
query_file_cache = {}
nn_file_cache = {}

def find_nearest_neighbors_with_jar(smiles, target_id, fingerprint_data, db_file,
                                    num_neighbors=50, fp_type="ECfp4", scoring_method="TANIMOTO"):
    global ecfp4_counter, rdkit_counter, atompair_counter, layered_counter
    global mhfp6_counter, ecfp6_counter, fused_counter, query_file_cache, nn_file_cache
    jar_files = {
        "ECfp4": "PPB3_ECFP4.jar",
        "RDKit": "PPB3_RDKIT.jar",
        "AtomPair": "PPB3_ATOMPAIR.jar",
        "Layered": "PPB3_LAYERED.jar",
        "MHFP6": "PPB3_MHFP6.jar",
        "ECfp6": "PPB3_ECFP6.jar",
        "Fused": "PPB3_FUSED.jar",
    }
    if fp_type not in jar_files:
        raise ValueError(f"unsupported fingerprint type: {fp_type}")

    jar_path = jar_files[fp_type]
    cache_key = f"{smiles}_{fp_type}"
    if cache_key in nn_file_cache:
        return nn_file_cache[cache_key]

    # increment query index based on fingerprint type
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

    # construct query file
    query_fp_file_path = f"queries/{fp_type.lower()}_q{query_index}.txt"
    os.makedirs(os.path.dirname(query_fp_file_path), exist_ok=True)

    # write fingerprint to query file
    with open(query_fp_file_path, "w") as f:
        f.write(f"smiles\tcpdid\ttarid\t{fp_type}\n")
        f.write(f"{smiles}\tquery_{query_index}\t{target_id}\t{fingerprint_data}\n")

    query_file_cache[cache_key] = query_fp_file_path
    output_file_path = f"results/{fp_type.lower()}_q{uuid.uuid4().hex[:8]}_nn.txt"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # run the java jar
    try:
        sub.run(
            [
                "java", "-Xmx4G", "-cp", jar_path, "aid.KNN1ss1",
                query_fp_file_path,
                db_file,
                fp_type,
                scoring_method,
                str(num_neighbors),
                output_file_path,
                fp_type], check=True)
        nn_file_cache[cache_key] = output_file_path
        return output_file_path
    except sub.CalledProcessError as e:
        raise RuntimeError(f"error running nearest neighbors jar: {e}")

# molecule image generator
def generate_molecule_image(smiles, compound_id):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            image_path = f"static/molecules/{compound_id}.png"
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            Draw.MolToFile(mol, image_path)
            return image_path
    except Exception as e:
        print(f"error generating image for {smiles}: {e}")
    return None
