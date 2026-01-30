# top 5
# mapping the 6 target types
def map_target_type(target_type):
    type_mapping = {
        "SINGLE PROTEIN": "Single Protein",
        "ORGANISM": "Organism",
        "CELL-LINE": "Cell-line",
        "PROTEIN COMPLEX": "Protein Complex",
        "PROTEIN FAMILY": "Protein Family"
    }
    if isinstance(target_type, str): 
        for key, category in type_mapping.items():
            if key in target_type.upper():  
                return category
    return "Other Target Types"
# handling the multiple classes for target types
exploded_df = id_prediction_df_merged.explode("target_type")
exploded_df["grouped_target_type"] = exploded_df["target_type"].apply(map_target_type)
exploded_df["top_5_predicted_ids"] = exploded_df["top_10_predicted_ids"].apply(lambda x: x[:5])
# calculating the recall
def calculate_recall(row):
    target_names = set(row["target_ids"])
    top_5_predictions = set(row["top_5_predicted_ids"])
    true_positives = len(target_names.intersection(top_5_predictions))
    return true_positives / len(target_names) if len(target_names) > 0 else 0
# calculating precision
def calculate_precision(row):
    target_names = set(row["target_ids"])
    top_5_predictions = set(row["top_5_predicted_ids"])
    true_positives = len(target_names.intersection(top_5_predictions))
    return true_positives / len(top_5_predictions) if len(top_5_predictions) > 0 else 0
exploded_df["recall"] = exploded_df.apply(calculate_recall, axis=1)
exploded_df["precision"] = exploded_df.apply(calculate_precision, axis=1)
average_metrics = exploded_df.groupby("grouped_target_type")[["recall", "precision"]].mean()
target_type_counts = exploded_df["grouped_target_type"].value_counts(normalize=True) * 100
average_metrics["percentage"] = average_metrics.index.map(target_type_counts)
all_types = ["Single Protein", "Organism", "Cell-line", "Protein Complex", "Protein Family", "Other Target Types"]
# averaging the values
average_metrics = average_metrics.reindex(all_types, fill_value=0)
average_metrics = average_metrics.reset_index().rename(columns={
    "grouped_target_type": "target_types",
    "recall": "average_recall",
    "precision": "average_precision"})
# assigning the values
average_metrics["target_types"] = average_metrics.apply(
    lambda row: f"{row['target_types']} ({row['percentage']:.2f}%)", axis=1)
average_metrics = average_metrics[["target_types", "average_recall", "average_precision"]]
average_metrics
