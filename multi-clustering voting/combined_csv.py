import pandas as pd

csv_files = {
    'agg': 'confusion_matrix/agg_metrics.csv',
    'birch': 'confusion_matrix/birch_metrics.csv',
    'kmeans': 'confusion_matrix/kmeans_metrics.csv',
    'bagging': 'confusion_matrix/bagging_metrics.csv'
}


combined_metrics = pd.DataFrame()


for method_name, file_path in csv_files.items():
    metrics_df = pd.read_csv(file_path)
    metrics_df['Method'] = method_name
    # Ensuring 'Method' is the first column
    cols = metrics_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    metrics_df = metrics_df[cols]

    combined_metrics = combined_metrics.append(metrics_df, ignore_index=True)

# Saving the combined DataFrame to a CSV file
combined_metrics.to_csv('combined_metrics.csv', index=False)