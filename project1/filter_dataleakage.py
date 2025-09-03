import pandas as pd
import numpy as np
import sys
import os
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def find_clusters(df, tsv_df, tm_threshold):
    train_connections = pd.merge(
        tsv_df,
        df[['ID', 'Train', 'Tm']],
        left_on='Source',
        right_on='ID',
        how='left'
    )
    train_connections = pd.merge(
        train_connections,
        df[['ID', 'Train', 'Tm']],
        left_on='Target',
        right_on='ID',
        how='left',
        suffixes=('_source', '_target')
    )

    train_train_connections = train_connections[
        (train_connections['Train_source'] == 1) & (train_connections['Train_target'] == 1)
    ].copy()

    train_train_connections.loc[:, 'Tm_diff'] = np.abs(
        train_train_connections['Tm_source'] - train_train_connections['Tm_target']
    )

    valid_connections = train_train_connections[
        train_train_connections['Tm_diff'] < tm_threshold
    ][['Source', 'Target']]

    train_ids = df[df['Train'] == 1]['ID'].unique()
    id_to_index = {id_: idx for idx, id_ in enumerate(train_ids)}

    n = len(train_ids)
    adj_matrix = np.zeros((n, n), dtype=int)
    for _, row in valid_connections.iterrows():
        src_idx = id_to_index.get(row['Source'])
        tgt_idx = id_to_index.get(row['Target'])
        if src_idx is not None and tgt_idx is not None:
            adj_matrix[src_idx, tgt_idx] = 1
            adj_matrix[tgt_idx, src_idx] = 1

    graph = csr_matrix(adj_matrix)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    cluster_dict = {train_ids[i]: labels[i] for i in range(len(train_ids))}
    return cluster_dict, n_components

def select_cluster_centroids(df, cluster_dict, n_components):
    centroid_ids = []
    for cluster_label in range(n_components):
        cluster_ids = [id_ for id_, label in cluster_dict.items() if label == cluster_label]
        if not cluster_ids:
            continue
        cluster_df = df[df['ID'].isin(cluster_ids)][['ID', 'Tm']].copy()
        mean_tm = cluster_df['Tm'].mean()
        cluster_df.loc[:, 'Tm_diff_from_mean'] = np.abs(cluster_df['Tm'] - mean_tm)
        centroid_id = cluster_df.loc[cluster_df['Tm_diff_from_mean'].idxmin(), 'ID']
        centroid_ids.append(centroid_id)
    return centroid_ids

def filter_data_leakage(merged_df, tsv_df, tm_threshold):
    merged_df['Tm'] = pd.to_numeric(merged_df['Tm'], errors='coerce')
    merged_df[['Train', 'Test']] = merged_df[['Train', 'Test']].apply(pd.to_numeric, errors='coerce')

    connections = pd.merge(
        tsv_df,
        merged_df[['ID', 'Train', 'Test', 'Tm']],
        left_on='Source',
        right_on='ID',
        how='left'
    )
    connections = pd.merge(
        connections,
        merged_df[['ID', 'Train', 'Test', 'Tm']],
        left_on='Target',
        right_on='ID',
        how='left',
        suffixes=('_source', '_target')
    )

    test_train_connections = connections[
        (connections['Test_source'] == 1) & (connections['Train_target'] == 1)
    ].copy()

    test_train_connections.loc[:, 'Tm_diff'] = np.abs(
        test_train_connections['Tm_source'] - test_train_connections['Tm_target']
    )

    test_ids_to_remove = test_train_connections[
        test_train_connections['Tm_diff'] < tm_threshold
    ]['Source'].unique()

    filtered_df = merged_df[~merged_df['ID'].isin(test_ids_to_remove)]
    return filtered_df, test_ids_to_remove

def merge_and_filter_files(tsv_path, csv_path, tm_threshold, outfolder):
    try:
        tsv_df = pd.read_csv(tsv_path, sep='\t', header=None, names=['Target', 'Source'])
        csv_df = pd.read_csv(csv_path, sep=';')

        if not all(col in csv_df.columns for col in ['ID', 'Train', 'Test', 'Tm']):
            raise ValueError("CSV must contain columns: ID, Train, Test, Tm")

        merged_df = pd.merge(
            csv_df,
            tsv_df[['Source', 'Target']],
            left_on='ID',
            right_on='Source',
            how='left'
        )

        filtered_df, removed_test_ids = filter_data_leakage(merged_df, tsv_df, tm_threshold)
        cluster_dict, n_components = find_clusters(filtered_df, tsv_df, tm_threshold)
        print(f"Identified {n_components} clusters in training data")

        centroid_ids = select_cluster_centroids(filtered_df, cluster_dict, n_components)
        print(f"Selected {len(centroid_ids)} centroids from training clusters")

        train_ids_to_remove = [id_ for id_ in cluster_dict if id_ not in centroid_ids]
        final_filtered_df = filtered_df[~filtered_df['ID'].isin(train_ids_to_remove)]

        # Save filtered CSV
        output_csv_path = f"{outfolder}/filtered_clustered_dataset.csv"
        final_filtered_df.to_csv(output_csv_path, sep=';', index=False)
        print(f"Saved filtered dataset to {output_csv_path}")

        # Save removed test datapoints
        test_file = os.path.join(outfolder, f"filtered_datapoints_test_{tm_threshold}_temp.txt")
        pd.Series(removed_test_ids).to_csv(test_file, index=False, header=False)

        # Save removed training datapoints
        train_file = os.path.join(outfolder, f"filtered_datapoints_train_{tm_threshold}_temp.txt")
        pd.Series(train_ids_to_remove).to_csv(train_file, index=False, header=False)

        print(f"Saved removed test datapoints to {test_file}")
        print(f"Saved removed training datapoints to {train_file}")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python filter_dataleakage.py <tsv_file> <csv_file> <tm_threshold> <outfolder>")
        sys.exit(1)

    tsv_file, csv_file, tm_threshold, outfolder = sys.argv[1:5]

    for path in [tsv_file, csv_file, outfolder]:
        if not os.path.exists(path):
            print(f"Error: Path '{path}' does not exist")
            sys.exit(1)

    try:
        tm_threshold = float(tm_threshold)
        if tm_threshold < 0:
            raise ValueError("Tm threshold must be non-negative")
    except ValueError as e:
        print(f"Error: Tm threshold must be a valid number - {e}")
        sys.exit(1)

    merge_and_filter_files(tsv_file, csv_file, tm_threshold, outfolder)
