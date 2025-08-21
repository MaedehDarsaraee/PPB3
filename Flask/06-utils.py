import os
import re
import uuid

def generate_unique_query_id() -> str:
    return str(uuid.uuid4())
# generating the next prediction file (preds_X.txt)
def get_next_prediction_file_path(results_dir: str = "results") -> str:
    os.makedirs(results_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(results_dir) if f.startswith("preds_") and f.endswith(".txt")]
    numbers = []
    for filename in existing_files:
        match = re.search(r'preds_(\d+)\.txt', filename)
        if match:
            numbers.append(int(match.group(1)))
    next_number = max(numbers, default=0) + 1
    new_filename = f"preds_{next_number}.txt"
    return os.path.join(results_dir, new_filename)

