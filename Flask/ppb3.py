from flask import Flask, request, jsonify, render_template, send_from_directory, session
import subprocess as sub
from flask import redirect, url_for
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, RDKFingerprint
from rdkit.Chem import Draw
from mhfp.encoder import MHFPEncoder
from tensorflow.keras.models import load_model
from rdkit.Chem.MolStandardize import charge
import pandas as pd
import os
import uuid
import tempfile
import re

# Initialize counters for each fingerprint type
ecfp4_counter = 0
rdkit_counter = 0
atompair_counter = 0 
layered_counter= 0
mhfp6_counter = 0
ecfp6_counter = 0
fused_counter = 0
query_file_cache = {} 
nn_file_cache = {} 

# Function to load the query counter from a file
def generate_unique_query_id():
    return str(uuid.uuid4())

# Initialize a global counter for prediction files
prediction_file_counter = 0

def get_next_prediction_file_path():
    """Generates a numeric prediction filename compatible with Java's naming system."""
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # Find existing prediction files and extract numbers
    existing_files = [f for f in os.listdir(results_dir) if f.startswith("preds_") and f.endswith(".txt")]
    numbers = []
    for filename in existing_files:
        match = re.search(r'preds_(\\d+)\\.txt', filename)
        if match:
            numbers.append(int(match.group(1)))

    # Determine the next available number
    next_number = max(numbers, default=0) + 1

    # Construct the new filename
    new_filename = f"preds_{next_number}.txt"
    new_path = os.path.join(results_dir, new_filename)

    return new_path

app = Flask(__name__)

# Limit TensorFlow resource usage for memory efficiency
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load DNN models
try:
    models = {
        'ECFP4': load_model('ecfp4_dnn_final_model.h5', compile=False),
        'AtomPair': load_model('atompair_dnn_model_full_data.h5', compile=False),
        'Layered': load_model('layered_dnn_model_full_data.h5', compile=False),
        'RDKit': load_model('rdkit_dnn_model_full_data.h5', compile=False),
        'MHFP6': load_model('mhfp6_dnn_model_full_data.h5', compile=False),
        'ECFP6': load_model('ecfp6_dnn_model_full_data.h5', compile=False),
        'Fused': load_model('fused11_dnn_model_full_data.h5', compile=False)
    }
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")

# Load target details and classification
try:
    target_details = pd.read_csv('TARGETSDETAILS_2nd.txt', sep="\t")
    target_classification = pd.read_csv('TARGETCLASSIFICATION_2nd.txt', sep="\t")

    # Convert target details to dictionaries for quick lookup
    target_id_to_name = target_details.set_index('CHEMBL_ID')['PREF_NAME'].to_dict()
    target_id_to_class = target_classification.set_index('CHEMBL_ID')['CLASS'].to_dict()
    target_id_to_organism = target_classification.set_index('CHEMBL_ID')['ORGANISM'].to_dict()
    target_id_to_type = target_classification.set_index('CHEMBL_ID')['TYPE'].to_dict()  

    # Load DNN target labels
    with open('DNNTARLABELS_2nd.txt', 'r') as f:
        target_labels = [line.strip() for idx, line in enumerate(f) if idx > 0 and line.strip()]
    print("Target details and DNN labels loaded successfully.")
except Exception as e:
    print(f"Error loading target details or labels: {e}")

# Initialize MHFPEncoder for MHFP6 calculation
mhfp_encoder = MHFPEncoder()

# Fingerprint calculation functions
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
        return Chem.RDKFingerprint(mol, minPath=min_path, maxPath=max_path, fpSize=fp_size, nBitsPerHash=n_bits_per_hash, useHs=True) if mol else None
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
        return mhfp_encoder.secfp_from_smiles(smiles, length=length, radius=radius, rings=True, kekulize=True, sanitize=False)
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
            fused_fp = np.concatenate((ecfp4_fp, mhfp6_fp), axis=0).astype(np.int8)
            return fused_fp
        else:
            return None
    except ValueError as e:
        print(e)
        return None


query_results_cache = {}
query_mapping = {}
query_counter = 0

def find_nearest_neighbors_with_jar(smiles, target_id, fingerprint_data, db_file, num_neighbors=50, fp_type="ECfp4", scoring_method="TANIMOTO"):
    global ecfp4_counter, rdkit_counter, atompair_counter, layered_counter, mhfp6_counter, ecfp6_counter, fused_counter, query_file_cache, nn_file_cache

    # Set the path to the appropriate JAR file
    if fp_type == "ECfp4":
        jar_path = "PPB3_ECFP4.jar"
    elif fp_type == "RDKit":
        jar_path = "PPB3_RDKIT.jar"
    elif fp_type == "AtomPair":
        jar_path = "PPB3_ATOMPAIR.jar"
    elif fp_type == "Layered" :
        jar_path = "PPB3_LAYERED.jar"
    elif fp_type == "MHFP6":
        jar_path = "PPB3_MHFP6.jar"
    elif fp_type == "ECfp6":
        jar_path = "PPB3_ECFP6.jar"
    elif fp_type == "Fused":
        jar_path = "PPB3_FUSED.jar"
    else:
        raise ValueError(f"Unsupported fingerprint type: {fp_type}")
    cache_key = f"{smiles}_{fp_type}"

    # Check if NN file already exists
    if cache_key in nn_file_cache:
        output_file_path = nn_file_cache[cache_key]
        print(f"Reusing cached NN file: {output_file_path}")
        return output_file_path
    else:
        if fp_type == "ECfp4":
            ecfp4_counter += 1
            query_index = ecfp4_counter
        elif fp_type == "RDKit":
            rdkit_counter += 1
            query_index = rdkit_counter
        elif fp_type == "AtomPair":
            atompair_counter += 1
            query_index = atompair_counter
        elif fp_type == "Layered" :
            layered_counter += 1
            query_index = layered_counter
        elif fp_type == "MHFP6":
            mhfp6_counter += 1
            query_index = mhfp6_counter
        elif fp_type == "ECfp6":
            ecfp6_counter += 1
            query_index = ecfp6_counter
        elif fp_type == "Fused":
            fused_counter += 1
            query_index = fused_counter
        else:
            raise ValueError(f"Unsupported fingerprint type: {fp_type}")

        # Construct query file path
        query_fp_file_path = f"queries/{fp_type.lower()}_q{query_index}.txt"

        # Ensure directories exist
        os.makedirs(os.path.dirname(query_fp_file_path), exist_ok=True)

        # Write the query fingerprint to a file
        with open(query_fp_file_path, "w") as query_fp_file:
            if fp_type == "ECfp4":
                query_fp_file.write("SMILES\tCPDid\tTARid\tECfp4\n")
            elif fp_type == "RDKit":
                query_fp_file.write("SMILES\tCPDid\tTARid\tRDKit\n")
            elif fp_type == "AtomPair":
                query_fp_file.write("SMILES\tCPDid\tTARid\tAtomPair\n")
            elif fp_type == "Layered":
                query_fp_file.write("SMILES\tCPDid\tTARid\tLayered\n")
            elif fp_type == "MHFP6":
                query_fp_file.write("SMILES\tCPDid\tTARid\tMHFP6\n")
            elif fp_type == "ECfp6":
                query_fp_file.write("SMILES\tCPDid\tTARid\tECfp6\n")
            elif fp_type == "Fused":
                query_fp_file.write("SMILES\tCPDid\tTARid\tFused\n")
            query_fp_file.write(f"{smiles}\tquery_{query_index}\t{target_id}\t{fingerprint_data}\n")

        # Store the query file path in the cache
        query_file_cache[cache_key] = query_fp_file_path

    # Construct output file path with query index
    output_file_path = f"results/{fp_type.lower()}_q{uuid.uuid4().hex[:8]}_nn.txt"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Run the JAR file to compute nearest neighbors
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
                fp_type              
            ],
            check=True)
        print(f"Nearest neighbors written to: {output_file_path}")

        # Cache the output file path
        nn_file_cache[cache_key] = output_file_path
        return output_file_path
    except sub.CalledProcessError as e:
        print(f"Error running JAR: {e}")
        raise RuntimeError(f"Error running JAR: {e}")

def generate_molecule_image(smiles, compound_id):
    """Generates an image of the molecule and saves it as a PNG file."""
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

app = Flask(__name__, static_folder="static")
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)
@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')
@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')
@app.route('/faq')
def faq():
    return render_template('faq.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/result', methods=['POST'])
def result():
    try:
        data = request.json
        smiles_list = data.get('smiles', [])
        if not smiles_list:
            return jsonify({'error': 'No SMILES provided.'}), 400

        # Define preprocessing functions
        def preprocess_smiles(smi):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None

            #Convert to non-isomeric SMILES
            non_iso_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            if non_iso_smiles is None:
                return None

            # Remove counterions
            mol = Chem.MolFromSmiles(smi)
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            largest_frag = max(frags, key=lambda m: m.GetNumAtoms())
            cleaned_smiles = Chem.MolToSmiles(largest_frag, isomericSmiles=False)
            if cleaned_smiles is None:
                return None

            # Correct valence
            mol = Chem.MolFromSmiles(cleaned_smiles, sanitize=True)
            if mol is None:
                return None
            valence_corrected_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            if valence_corrected_smiles is None:
                return None

            # Correct pH (uncharging)
            uncharger = charge.Uncharger()
            mol = Chem.MolFromSmiles(valence_corrected_smiles)
            uncharged_mol = uncharger.uncharge(mol)
            final_smiles = Chem.MolToSmiles(uncharged_mol, isomericSmiles=False)

            return final_smiles

        preprocessed_smiles_list = []
        invalid_smiles = []

        for smi in smiles_list:
            processed_smi = preprocess_smiles(smi)
            if processed_smi:
                preprocessed_smiles_list.append(processed_smi)
            else:
                invalid_smiles.append(smi)

        if not preprocessed_smiles_list:
            return jsonify({'error': 'All provided SMILES are invalid after preprocessing.'}), 400

        # Continue fingerprint calculation using preprocessed SMILES
        model_type = data.get('model_type', 'ECFP4')
        num_predictions = data.get('num_predictions', 20)

        fingerprint_function = {
            'ECFP4': calculate_ecfp4,
            'AtomPair': calculate_atompair,
            'Layered': calculate_layered,
            'RDKit': calculate_rdkit,
            'MHFP6': calculate_mhfp6,
            'ECFP6': calculate_ecfp6,
            'Fused': fuse_fingerprints
        }.get(model_type)

        model = models.get(model_type)
        if not fingerprint_function or not model:
            return jsonify({'error': f'Model type "{model_type}" not recognized.'}), 400
        results = []
        class_counts = {}
        organism_counts = {}
        type_counts = {}
        results_path = get_next_prediction_file_path()
        with open(results_path, "w") as results_file:
            results_file.write("SMILES\tTARid\n")

            for smi in preprocessed_smiles_list:
                fp = fingerprint_function(smi)
                if fp is None:
                    invalid_smiles.append(smi)
                    continue

                X_new = np.array([list(fp)])
                predictions = model.predict(X_new)

                top_indices = predictions[0].argsort()[-num_predictions:][::-1]
                top_targets = []

                for idx, target_idx in enumerate(top_indices):
                    target_id = target_labels[target_idx]
                    target_name = target_id_to_name.get(target_id, 'Unknown')
                    target_class = target_id_to_class.get(target_id, 'Unknown')
                    target_type = target_id_to_type.get(target_id, 'Unknown')
                    organism = target_id_to_organism.get(target_id, 'Unknown')
                    confidence = round(float(predictions[0][target_idx]), 2)

                    class_counts[target_class] = class_counts.get(target_class, 0) + 1
                    organism_counts[organism] = organism_counts.get(organism, 0) + 1
                    type_counts[target_type] = type_counts.get(target_type, 0) + 1

                    top_targets.append({
                        'rank': idx + 1,
                        'target_id': target_id,
                        'target_name': target_name,
                        'confidence': confidence,
                        'class': target_class,
                        'type': target_type,
                        'organism': organism,
                        'url': f"https://www.ebi.ac.uk/chembl/target_report_card/{target_id}"})

                    results_file.write(f"{smi}\t{target_id}\n")

                results.append({
                    'smiles': smi,
                    'predictions': top_targets})

        query_smiles_image = preprocessed_smiles_list[0]
        mol = Chem.MolFromSmiles(query_smiles_image)
        if mol:
            img_path = os.path.join('static', 'query_molecule.png')
            Draw.MolToFile(mol, img_path)
        else:
            img_path = os.path.join('static', 'placeholder_image.png')

        return render_template(
            'result.html',
            results=results,
            invalid_smiles=invalid_smiles,
            class_counts=class_counts,
            organism_counts=organism_counts,
            type_counts=type_counts,
            query_image_path=img_path,
            selected_model=model_type)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
print("DEBUG ERROR: No valid POST request received.")
@app.route('/fetch_nearest_neighbors', methods=['POST'])
def fetch_nearest_neighbors():
    global ecfp4_counter, rdkit_counter, atompair_counter, layered_counter

    try:
        # Log the incoming request data
        data = request.json
        print(f"Request Data: {data}")

        # Validate the input
        smiles = data.get('smiles')
        target_id = data.get('target_id')
        model_type = data.get('model_type', 'ECFP4')  

        if not smiles:
            return jsonify({'error': 'Invalid input. SMILES is required.'}), 400
        if not target_id:
            return jsonify({'error': 'Invalid input. Target ID is required.'}), 400

        # Determine the database file and fingerprint type based on the selected model
        if model_type == 'RDKit':
            db_file = "RDKIT.fps"
            fp_type = "RDKit"
        elif model_type == 'AtomPair':
            db_file = "ATOMPAIR.fps"
            fp_type = "AtomPair"
        elif model_type == 'ECFP4':
            db_file = "ECFP4.fps"
            fp_type = "ECfp4"
        elif model_type == 'Layered':
            db_file = "LAYERED.fps"
            fp_type = "Layered"
        elif model_type == 'MHFP6':
            db_file = "MHFP6.fps"
            fp_type = "MHFP6"
        elif model_type == 'ECFP6':
            db_file = "ECFP6.fps"
            fp_type = "ECfp6"
        elif model_type == 'Fused':
            db_file = "FUSED.fps"
            fp_type = "Fused"
        else:
            return jsonify({'error': f'Unsupported model type: {model_type}'}), 400

        # Calculate fingerprint
        fingerprint_function = {
            'ECFP4': calculate_ecfp4,
            'RDKit': calculate_rdkit,
            'AtomPair': calculate_atompair,
            'Layered': calculate_layered,
            'MHFP6': calculate_mhfp6,
            'ECFP6': calculate_ecfp6,
            'Fused': fuse_fingerprints
        }.get(model_type)

        if not fingerprint_function:
            return jsonify({'error': f'Model type "{model_type}" not recognized.'}), 400

        fingerprint = fingerprint_function(smiles)
        if fingerprint is None:
            return jsonify({'error': 'Failed to generate fingerprint for the given SMILES.'}), 400

        bitstring = ";".join(map(str, list(fingerprint)))  

        # Perform nearest neighbor search
        try:
            output_file_path = find_nearest_neighbors_with_jar(
                smiles=smiles,
                target_id=target_id,
                fingerprint_data=bitstring,
                db_file=db_file,
                num_neighbors=50,
                fp_type=fp_type,
                scoring_method="TANIMOTO")
        except Exception as e:
            return jsonify({'error': 'Failed to compute nearest neighbors.'}), 500

        # Read and process the results from the output file
        nearest_neighbors = []
        try:
            if not os.path.exists(output_file_path):
                return jsonify({'error': 'Nearest neighbor file not found.'}), 500

            with open(output_file_path, "r") as f:
                nearest_neighbors_data = f.readlines()

            print(f"DEBUG: Nearest Neighbors File Content:\n{''.join(nearest_neighbors_data[:20])}")

            if len(nearest_neighbors_data) <= 1:
                return jsonify({'nearest_neighbors': []}), 200

            # Process the nearest neighbors, selecting only the top 5 for the given target
            for line in nearest_neighbors_data[1:]:
                # Ensure correct tab-separated parsing
                parts = line.strip().split("\t")
                if len(parts) == 4:
                    parts.insert(3, "ECfp4")  # Insert "ECfp4" as a placeholder

                if len(parts) < 5:
                    print(f"WARNING: Skipping malformed row (Expected 5 columns, got {len(parts)}): {line.strip()}")
                    continue

                # Extract and clean the target ID
                found_target_id = re.sub(r'[\s\r\n]+', '', parts[2]).upper()  # Remove all spaces, tabs, newlines
                query_target_id = re.sub(r'[\s\r\n]+', '', target_id).upper()

                # Debugging ASCII values for hidden characters
                print(f"DEBUG: ASCII Values of Found Target ID: {[ord(c) for c in found_target_id]}")
                print(f"DEBUG: ASCII Values of Query Target ID: {[ord(c) for c in query_target_id]}")
                print(f"DEBUG: RAW LINE -> {line.strip()}")

                # Check for exact match
                if found_target_id == query_target_id:
                    try:
                        nearest_neighbors.append({
                            "smiles": parts[0],
                            "compound_id": parts[1],
                            "target_id": parts[2],
                            "similarity": float(parts[4]),  # Convert similarity to float
                            "chembl_link": f"https://www.ebi.ac.uk/chembl/compound_report_card/{parts[1]}",
                            "image_path": generate_molecule_image(parts[0], parts[1])})
                    except ValueError:
                        print(f"Skipping invalid row: {line.strip()}")

            print(f"DEBUG: Found {len(nearest_neighbors)} matching neighbors for {target_id}")

            return jsonify({'nearest_neighbors': nearest_neighbors[:10]}), 200  

        except Exception as e:
            return jsonify({'error': 'Failed to process nearest neighbor results.'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True, host="localhost", port=5000)
