# average recall and precision
import numpy as np
from collections import defaultdict
import pandas as pd
# nitialize an empty list to store recall and precision results
class_results = []
# Calculate average recall and precision for each top_n value
for top_n in range(1, 11):
    # Dictionary to store TP, FN, and FP per target across folds
    target_metrics_per_fold_recall = defaultdict(lambda: {'TP': [], 'FN': []})
    target_metrics_per_fold_precision = defaultdict(lambda: {'TP': [], 'FP': []})
    
    # Loop over each fold in the kinase_rows
    for fold in predictions_df['fold_number'].unique():
        fold_data = predictions_df[predictions_df['fold_number'] == fold]

        # Dictionary to store TP, FN for recall and TP, FP for precision in this fold
        fold_target_metrics_recall = defaultdict(lambda: {'TP': 0, 'FN': 0})
        fold_target_metrics_precision = defaultdict(lambda: {'TP': 0, 'FP': 0})

        # Loop over each row in the fold
        for _, row in fold_data.iterrows():
            actual_targets = set(row['target_ids'])  # The actual targets for the compound
            predicted_targets = row['top_10_predicted_ids'][:top_n]  # The top-n predicted targets

            # Calculate TP and FN for recall
            for target in actual_targets:
                if target in predicted_targets:
                    fold_target_metrics_recall[target]['TP'] += 1  # True Positive
                else:
                    fold_target_metrics_recall[target]['FN'] += 1  # False Negative

            # Calculate TP and FP for precision
            for target in predicted_targets:
                if target in actual_targets:
                    fold_target_metrics_precision[target]['TP'] += 1  # True Positive
                else:
                    fold_target_metrics_precision[target]['FP'] += 1  # False Positive

        # Aggregate TP, FN for recall and TP, FP for precision across folds
        for target, metrics in fold_target_metrics_recall.items():
            target_metrics_per_fold_recall[target]['TP'].append(metrics['TP'])
            target_metrics_per_fold_recall[target]['FN'].append(metrics['FN'])
        
        for target, metrics in fold_target_metrics_precision.items():
            target_metrics_per_fold_precision[target]['TP'].append(metrics['TP'])
            target_metrics_per_fold_precision[target]['FP'].append(metrics['FP'])

    # Calculate average recall for each target across folds
    recalls = []
    for target, metrics in target_metrics_per_fold_recall.items():
        # Sum TP and FN across the 10 folds
        total_TP = sum(metrics['TP'])
        total_FN = sum(metrics['FN'])

        # Calculate recall for this target across folds
        recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        recalls.append(recall)

    # Calculate average precision for each target across folds
    precisions = []
    for target, metrics in target_metrics_per_fold_precision.items():
        # Sum TP and FP across the 10 folds
        total_TP = sum(metrics['TP'])
        total_FP = sum(metrics['FP'])

        # Calculate precision for this target across folds
        precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        precisions.append(precision)

    # Calculate the average recall and precision across all targets for this top_n
    avg_recall = np.mean(recalls) * 100
    avg_precision = np.mean(precisions) * 100
    class_results.append({
        'Top_n': top_n, 
        'Average Recall (%)': avg_recall,
        'Average Precision (%)': avg_precision
    })
average_recall_precision_df = pd.DataFrame(class_results)
average_recall_precision_df
