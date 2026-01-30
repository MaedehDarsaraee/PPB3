# top 5
# mapping the 8 protein classes
def map_protein_class(protein_class):
    class_mapping = {
        "kinase": "Kinase",
        "protease": "Protease",
        "enzyme": "Enzyme",
        "membrane receptor": "Membrane Receptor",
        "transporter": "Transporter",
        "transcription factor": "Transcription Factor",
        "ion channel": "Ion Channel"}
    for key, category in class_mapping.items():
        if key in protein_class.lower(): 
            return category
    return "Other Protein Classes"
# handling the multiple classes for protein classes
exploded_df = id_prediction_df_merged.explode("protein_class")
exploded_df["grouped_protein_class"] = exploded_df["protein_class"].apply(map_protein_class)
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
average_metrics = exploded_df.groupby("grouped_protein_class")[["recall", "precision"]].mean()
protein_class_counts = exploded_df["grouped_protein_class"].value_counts(normalize=True) * 100
average_metrics["percentage"] = average_metrics.index.map(protein_class_counts)
all_classes = ["Kinase", "Protease", "Enzyme", "Membrane Receptor", "Transporter", 
               "Transcription Factor", "Ion Channel", "Other Protein Classes"]
# averaging the values
average_metrics = average_metrics.reindex(all_classes, fill_value=0)
average_metrics = average_metrics.reset_index().rename(columns={
    "grouped_protein_class": "protein_classes",
    "recall": "average_recall",
    "precision": "average_precision"})
# assigning the values
average_metrics["protein_classes"] = average_metrics.apply(
    lambda row: f"{row['protein_classes']} ({row['percentage']:.2f}%)", axis=1)
average_metrics = average_metrics[["protein_classes", "average_recall", "average_precision"]]
average_metrics
