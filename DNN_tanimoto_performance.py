# overall confidence score tanimoto
import pandas as pd
import numpy as np

def calculate_overall_metrics_by_similarity_cs(df, top_n=5, cs_threshold=0.2):
    """
    Calculates overall recall and precision by tanimoto similarity bins,
    using only predictions with confidence scores >= cs_threshold.
    """
    # Define similarity bins
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    df["similarity_bin"] = pd.cut(df["tanimoto_similarity"], bins=bins, labels=bin_labels, right=False)

    overall_results = []

    # Loop over bins
    for similarity_bin in bin_labels:
        bin_df = df[df["similarity_bin"] == similarity_bin]
        total_TP = 0
        total_FN = 0
        total_FP = 0

        # Loop over folds
        for fold in df["fold_number"].unique():
            fold_df = bin_df[bin_df["fold_number"] == fold]

            for _, row in fold_df.iterrows():
                actual_targets = set(row["target_ids"])
                predicted_targets = row["top_10_predicted_ids"][:top_n]
                confidence_scores = row["confidence_scores"][:top_n]

                # --- Keep only confident predictions ---
                confident_predictions = {
                    tid for tid, score in zip(predicted_targets, confidence_scores)
                    if score >= cs_threshold
                }

                # True Positives (TP): correct predictions
                TP = len(actual_targets.intersection(confident_predictions))
                # False Negatives (FN): true targets not predicted
                FN = len(actual_targets - confident_predictions)
                # False Positives (FP): incorrect predictions above threshold
                FP = len(confident_predictions - actual_targets)

                total_TP += TP
                total_FN += FN
                total_FP += FP

        # Compute overall recall & precision for this similarity bin
        overall_recall = (total_TP / (total_TP + total_FN)) * 100 if (total_TP + total_FN) > 0 else 0
        overall_precision = (total_TP / (total_TP + total_FP)) * 100 if (total_TP + total_FP) > 0 else 0

        overall_results.append({
            "similarity_bin": similarity_bin,
            "Overall Recall (%)": overall_recall,
            "Overall Precision (%)": overall_precision
        })

    return pd.DataFrame(overall_results)


# --- Run it for your dataset ---
overall_df = calculate_overall_metrics_by_similarity_cs(
    ecfp4_tanimoto,
    top_n=5,
    cs_threshold=0.2
)

overall_df
