#overall precision and recall
overall_tp = 0
overall_fn = 0
overall_fp = 0
overall_results = []
for top_n in range(1, 11):
    # Reset the counts for this top_n
    tp_total = 0
    fn_total = 0
    fp_total = 0

    # Iterate over each fold
    for fold in predictions_df['fold_number'].unique():
        fold_data = predictions_df[predictions_df['fold_number'] == fold]

        # Iterate over each row (compound) in the fold
        for _, row in fold_data.iterrows():
            actual_targets = set(row['target_ids'])  # Actual targets
            predicted_targets = row['top_10_predicted_ids'][:top_n]  # Top-n predicted targets

            # Calculate TP and FN for actual targets
            for target in actual_targets:
                if target in predicted_targets:
                    tp_total += 1  # True Positive
                else:
                    fn_total += 1  # False Negative

            # Calculate FP for predicted targets that are not actual targets
            for target in predicted_targets:
                if target not in actual_targets:
                    fp_total += 1  # False Positive

    # Calculate overall recall and precision
    overall_recall = (tp_total / (tp_total + fn_total)) * 100 if (tp_total + fn_total) > 0 else 0
    overall_precision = (tp_total / (tp_total + fp_total)) * 100 if (tp_total + fp_total) > 0 else 0

    overall_results.append({
        'Top_n': top_n,
        'Overall Recall (%)': overall_recall,
        'Overall Precision (%)': overall_precision
    })
overall_recall_precision_df = pd.DataFrame(overall_results)
overall_recall_precision_df
