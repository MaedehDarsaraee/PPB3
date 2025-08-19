import pandas as pd
from tensorflow.keras.models import load_model
# Load Trained Models
try:
    MODELS = {
        "ECFP4": load_model("ecfp4_dnn_final_model.h5", compile=False),
        "AtomPair": load_model("atompair_dnn_model_full_data.h5", compile=False),
        "Layered": load_model("layered_dnn_model_full_data.h5", compile=False),
        "RDKit": load_model("rdkit_dnn_model_full_data.h5", compile=False),
        "MHFP6": load_model("mhfp6_dnn_model_full_data.h5", compile=False),
        "ECFP6": load_model("ecfp6_dnn_model_full_data.h5", compile=False),
        "Fused": load_model("fused11_dnn_model_full_data.h5", compile=False),
    }
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    MODELS = {}

# Load Target Details
try:
    target_details = pd.read_csv("TARGETSDETAILS_2nd.txt", sep="\t")
    target_classification = pd.read_csv("TARGETCLASSIFICATION_2nd.txt", sep="\t")

    TARGET_ID_TO_NAME = target_details.set_index("CHEMBL_ID")["PREF_NAME"].to_dict()
    TARGET_ID_TO_CLASS = target_classification.set_index("CHEMBL_ID")["CLASS"].to_dict()
    TARGET_ID_TO_ORGANISM = target_classification.set_index("CHEMBL_ID")["ORGANISM"].to_dict()
    TARGET_ID_TO_TYPE = target_classification.set_index("CHEMBL_ID")["TYPE"].to_dict()

    # Load target labels used in DNN outputs
    with open("DNNTARLABELS_2nd.txt", "r") as f:
        TARGET_LABELS = [line.strip() for idx, line in enumerate(f) if idx > 0 and line.strip()]

    print("Target details and labels loaded successfully")

except Exception as e:
    print(f"Error loading target details or labels: {e}")
    TARGET_ID_TO_NAME, TARGET_ID_TO_CLASS, TARGET_ID_TO_ORGANISM, TARGET_ID_TO_TYPE = {}, {}, {}, {}
    TARGET_LABELS = []

# Helper Function
def get_target_info(chembl_id: str) -> dict:
    """Return all metadata for a target ID."""
    return {
        "name": TARGET_ID_TO_NAME.get(chembl_id, "Unknown"),
        "class": TARGET_ID_TO_CLASS.get(chembl_id, "Unknown"),
        "organism": TARGET_ID_TO_ORGANISM.get(chembl_id, "Unknown"),
        "type": TARGET_ID_TO_TYPE.get(chembl_id, "Unknown"),
    }
