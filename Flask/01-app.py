import os
import re
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
from rdkit import Chem
from rdkit.Chem import AllChem, RDKFingerprint, Draw
from rdkit.Chem.MolStandardize import charge
from mhfp.encoder import MHFPEncoder
from tensorflow.keras.models import load_model

# flask app initialization
app = Flask(__name__, static_folder="static")

# load models & target info
try:
    MODELS = {
        'ECFP4': load_model('ecfp4_dnn_final_model.h5', compile=False),
        'AtomPair': load_model('atompair_dnn_model_full_data.h5', compile=False),
        'Layered': load_model('layered_dnn_model_full_data.h5', compile=False),
        'RDKit': load_model('rdkit_dnn_model_full_data.h5', compile=False),
        'MHFP6': load_model('mhfp6_dnn_model_full_data.h5', compile=False),
        'ECFP6': load_model('ecfp6_dnn_model_full_data.h5', compile=False),
        'Fused': load_model('fused11_dnn_model_full_data.h5', compile=False)}
    print("models loaded successfully")
except Exception as e:
    print(f"error loading models: {e}")
    MODELS = {}

try:
    target_details = pd.read_csv('TARGETSDETAILS_2nd.txt', sep="\t")
    target_classification = pd.read_csv('TARGETCLASSIFICATION_2nd.txt', sep="\t")

    TARGET_ID_TO_NAME = target_details.set_index('CHEMBL_ID')['PREF_NAME'].to_dict()
    TARGET_ID_TO_CLASS = target_classification.set_index('CHEMBL_ID')['CLASS'].to_dict()
    TARGET_ID_TO_ORGANISM = target_classification.set_index('CHEMBL_ID')['ORGANISM'].to_dict()
    TARGET_ID_TO_TYPE = target_classification.set_index('CHEMBL_ID')['TYPE'].to_dict()

    with open('DNNTARLABELS_2nd.txt', 'r') as f:
        TARGET_LABELS = [line.strip() for idx, line in enumerate(f) if idx > 0 and line.strip()]
    print("target details and labels loaded successfully")
except Exception as e:
    print(f"error loading target data: {e}")
    TARGET_LABELS = []
    TARGET_ID_TO_NAME, TARGET_ID_TO_CLASS, TARGET_ID_TO_ORGANISM, TARGET_ID_TO_TYPE = {}, {}, {}, {}

# fingerprint functions
mhfp_encoder = MHFPEncoder()

def calculate_ecfp4(smi): 
    mol = Chem.MolFromSmiles(smi); return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 4096) if mol else None
def calculate_ecfp6(smi): 
    mol = Chem.MolFromSmiles(smi); return AllChem.GetMorganFingerprintAsBitVect(mol, 3, 4096) if mol else None
def calculate_atompair(smi): 
    mol = Chem.MolFromSmiles(smi); return AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, 4096) if mol else None
def calculate_layered(smi): 
    mol = Chem.MolFromSmiles(smi); return Chem.RDKFingerprint(mol, fpSize=4096) if mol else None
def calculate_rdkit(smi): 
    mol = Chem.MolFromSmiles(smi); return RDKFingerprint(mol, fpSize=4096) if mol else None
def calculate_mhfp6(smi): 
    return mhfp_encoder.secfp_from_smiles(smi, length=4096, radius=3) if smi else None
def fuse_fingerprints(smi):
    ecfp4, mhfp6 = calculate_ecfp4(smi), calculate_mhfp6(smi)
    return np.concatenate((ecfp4, mhfp6)).astype(np.int8) if ecfp4 is not None and mhfp6 is not None else None

FINGERPRINTS = {
    'ECFP4': calculate_ecfp4,
    'ECFP6': calculate_ecfp6,
    'AtomPair': calculate_atompair,
    'Layered': calculate_layered,
    'RDKit': calculate_rdkit,
    'MHFP6': calculate_mhfp6,
    'Fused': fuse_fingerprints}

# helpers
def preprocess_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if not mol: return None
    non_iso = Chem.MolToSmiles(mol, isomericSmiles=False)
    if not non_iso: return None
    frags = Chem.GetMolFrags(Chem.MolFromSmiles(smi), asMols=True, sanitizeFrags=False)
    largest = max(frags, key=lambda m: m.GetNumAtoms())
    cleaned = Chem.MolToSmiles(largest, isomericSmiles=False)
    mol = Chem.MolFromSmiles(cleaned, sanitize=True)
    if not mol: return None
    valence = Chem.MolToSmiles(mol, isomericSmiles=False)
    uncharger = charge.Uncharger()
    return Chem.MolToSmiles(uncharger.uncharge(Chem.MolFromSmiles(valence)), isomericSmiles=False)

def get_next_prediction_file_path():
    os.makedirs("results", exist_ok=True)
    existing = [f for f in os.listdir("results") if f.startswith("preds_")]
    numbers = [int(re.search(r'preds_(\d+)\.txt', f).group(1)) for f in existing if re.search(r'preds_(\d+)\.txt', f)]
    next_num = max(numbers, default=0) + 1
    return os.path.join("results", f"preds_{next_num}.txt")

def generate_molecule_image(smiles, compound_id):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        path = f"static/molecules/{compound_id}.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Draw.MolToFile(mol, path)
        return path
    return None

# routes
@app.route("/")
@app.route("/home")
def home(): return render_template("index.html")

@app.route("/tutorial")
def tutorial(): return render_template("tutorial.html")

@app.route("/faq")
def faq(): return render_template("faq.html")

@app.route("/contact")
def contact(): return render_template("contact.html")

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route("/result", methods=["POST"])
def result():
    try:
        data = request.json
        smiles_list = data.get("smiles", [])
        if not smiles_list: return jsonify({"error": "no smiles provided"}), 400

        processed = [preprocess_smiles(s) for s in smiles_list]
        valid = [s for s in processed if s]
        invalid = [s for s in smiles_list if s not in valid]
        if not valid: return jsonify({"error": "all smiles invalid"}), 400

        model_type = data.get("model_type", "ECFP4")
        num_predictions = data.get("num_predictions", 20)
        fp_fn, model = FINGERPRINTS.get(model_type), MODELS.get(model_type)
        if not fp_fn or not model: return jsonify({"error": f"model {model_type} not found"}), 400

        results, class_counts, org_counts, type_counts = [], {}, {}, {}
        with open(get_next_prediction_file_path(), "w") as f:
            f.write("smiles\ttarid\n")
            for smi in valid:
                fp = fp_fn(smi)
                if fp is None: continue
                preds = model.predict(np.array([list(fp)]))[0]
                top_idx = preds.argsort()[-num_predictions:][::-1]
                targets = []
                for rank, idx in enumerate(top_idx, start=1):
                    tid = TARGET_LABELS[idx]
                    targets.append({
                        "rank": rank,
                        "target_id": tid,
                        "target_name": TARGET_ID_TO_NAME.get(tid, "unknown"),
                        "confidence": round(float(preds[idx]), 2),
                        "class": TARGET_ID_TO_CLASS.get(tid, "unknown"),
                        "type": TARGET_ID_TO_TYPE.get(tid, "unknown"),
                        "organism": TARGET_ID_TO_ORGANISM.get(tid, "unknown"),
                        "url": f"https://www.ebi.ac.uk/chembl/target_report_card/{tid}"
                    })
                    f.write(f"{smi}\t{tid}\n")
                results.append({"smiles": smi, "predictions": targets})
        img_path = generate_molecule_image(valid[0], "query_molecule") or "static/placeholder.png"
        return render_template("result.html", results=results, invalid_smiles=invalid,
                               class_counts=class_counts, organism_counts=org_counts,
                               type_counts=type_counts, query_image_path=img_path,
                               selected_model=model_type)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# run app
if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=5000)
