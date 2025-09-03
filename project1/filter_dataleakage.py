import pandas as pd
import numpy as np
import sys
import os
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def find_clusters(df, tsv_df, tm_threshold):
    """
    Identify clusters of training points based on connections and Tm differences.
    Return cluster labels for training points.
    """
    # Filter for train-train connections
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
    ]
    
    # Calculate Tm differences
    train_train_connections['Tm_diff'] = np.abs(
        train_train_connections['Tm_source'] - train_train_connections['Tm_target']
    )
    
    # Keep connections where Tm difference < threshold
    valid_connections = train_train_connections[
        train_train_connections['Tm_diff'] < tm_threshold
    ][['Source', 'Target']]
    
    # Get unique training IDs
    train_ids = df[df['Train'] == 1]['ID'].unique()
    id_to_index = {id_: idx for idx, id_ in enumerate(train_ids)}
    
    # Create adjacency matrix for clustering
    n = len(train_ids)
    adj_matrix = np.zeros((n, n), dtype=int)
    for _, row in valid_connections.iterrows():
        src_idx = id_to_index.get(row['Source'])
        tgt_idx = id_to_index.get(row['Target'])
        if src_idx is not None and tgt_idx is not None:
            adj_matrix[src_idx, tgt_idx] = 1
            adj_matrix[tgt_idx, src_idx] = 1  # Undirected graph
    
    # Convert to sparse matrix and find connected components
    graph = csr_matrix(adj_matrix)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    
    # Map labels back to IDs
    cluster_dict = {train_ids[i]: labels[i] for i in range(len(train_ids))}
    return cluster_dict, n_components

def select_cluster_centroids(df, cluster_dict, n_components):
    """
    Select centroid (point closest to mean Tm) for each cluster.
    Return IDs to keep.
    """
    centroid_ids = []
    
    for cluster_label in range(n_components):
        # Get IDs in this cluster
        cluster_ids = [id_ for id_, label in cluster_dict.items() if label == cluster_label]
        if not cluster_ids:
            continue
        
        # Get Tm values for the cluster
        cluster_df = df[df['ID'].isin(cluster_ids)][['ID', 'Tm']]
        if cluster_df.empty:
            continue
        
        # Calculate mean Tm
        mean_tm = cluster_df['Tm'].mean()
        
        # Find ID with Tm closest to mean
        cluster_df['Tm_diff_from_mean'] = np.abs(cluster_df['Tm'] - mean_tm)
        centroid_id = cluster_df.loc[cluster_df['Tm_diff_from_mean'].idxmin(), 'ID']
        centroid_ids.append(centroid_id)
    
    return centroid_ids

def filter_data_leakage(merged_df, tsv_df, tm_threshold):
    """
    Remove test data points connected to train data points with Tm difference < threshold.
    Return filtered DataFrame and count of removed test points.
    """
    # Ensure numeric Tm values
    merged_df['Tm'] = pd.to_numeric(merged_df['Tm'], errors='coerce')
    merged_df[['Train', 'Test']] = merged_df[['Train', 'Test']].apply(pd.to_numeric, errors='coerce')
    
    # Identify test-train connections
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
    
    # Filter for test-source and train-target connections
    test_train_connections = connections[
        (connections['Test_source'] == 1) & (connections['Train_target'] == 1)
    ]
    
    # Calculate Tm differences
    test_train_connections['Tm_diff'] = np.abs(
        test_train_connections['Tm_source'] - test_train_connections['Tm_target']
    )
    
    # Identify test IDs to remove (where Tm difference < threshold)
    test_ids_to_remove = test_train_connections[
        test_train_connections['Tm_diff'] < tm_threshold
    ]['Source'].unique()
    
    # Count removed test points
    num_removed = len(test_ids_to_remove)
    
    # Filter out test points with connections below threshold
    filtered_df = merged_df[~merged_df['ID'].isin(test_ids_to_remove)]
    
    return filtered_df, num_removed

def merge_and_filter_files(tsv_path, csv_path, tm_threshold, outfolder):
    """Merge TSV and CSV files, filter data leakage, remove redundant training points, and save results."""
    try:
        # Read input files
        tsv_df = pd.read_csv(tsv_path, sep='\t', header=None, names=['Target', 'Source'])
        csv_df = pd.read_csv(csv_path, sep=';')
        
        # Validate required columns
        if not all(col in csv_df.columns for col in ['ID', 'Train', 'Test', 'Tm']):
            raise ValueError("CSV must contain columns: ID, Train, Test, Tm")
        
        # Merge data
        merged_df = pd.merge(
            csv_df,
            tsv_df[['Source', 'Target']],
            left_on='ID',
            right_on='Source',
            how='left'
        )
        
        # Filter data leakage
        filtered_df, num_removed_test = filter_data_leakage(merged_df, tsv_df, tm_threshold)
        
        # Find clusters in training data
        cluster_dict, n_components = find_clusters(filtered_df, tsv_df, tm_threshold)
        print(f"Identified {n_components} clusters in training data")
        
        # Select centroids
        centroid_ids = select_cluster_centroids(filtered_df, cluster_dict, n_components)
        print(f"Selected {len(centroid_ids)} centroids from training clusters")
        
        # Filter training points to keep only centroids
        train_ids_to_remove = [id_ for id_, label in cluster_dict.items() if id_ not in centroid_ids]
        final_filtered_df = filtered_df[~filtered_df['ID'].isin(train_ids_to_remove)]
        num_removed_train = len(train_ids_to_remove)
        
        # Save filtered dataset
        output_path = f"{outfolder}/filtered_clustered_dataset.csv"
        final_filtered_df.to_csv(output_path, sep=';', index=False)
        print(f"Saved filtered dataset to {output_path}")
        
        # Print statistics
        print(f"Removed {num_removed_test} test data points due to Tm difference < {tm_threshold}°C")
        print(f"Removed {num_removed_train} redundant training data points, keeping only cluster centroids")
        
        # Print basic statistics
        total_nodes = len(final_filtered_df)
        num_test = final_filtered_df['Test'].sum()
        num_training = final_filtered_df['Train'].sum()
        print(f"\nFILTERED DATASET STATISTICS\n{'='*40}")
        print(f"Total nodes: {total_nodes}")
        print(f"Test datapoints: {num_test} ({num_test/total_nodes*100:.1f}%)")
        print(f"Training datapoints: {num_training} ({num_training/total_nodes*100:.1f}%)")
        print(f"{'='*40}")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python filter_dataleakage.py <tsv_file> <csv_file> <tm_threshold> <outfolder>")
        sys.exit(1)
    
    tsv_file, csv_file, tm_threshold, outfolder = sys.argv[1:5]
    
    # Validate paths
    for path in [tsv_file, csv_file, outfolder]:
        if not os.path.exists(path):
            print(f"Error: Path '{path}' does not exist")
            sys.exit(1)
    
    # Convert tm_threshold to float
    try:
        tm_threshold = float(tm_threshold)
        if tm_threshold < 0:
            raise ValueError("Tm threshold must be non-negative")
    except ValueValueError as e:
        print(f"Error: Tm threshold must be a valid number - {e}")
        sys.exit(1)
    
    merge_and_filter_files(tsv_file, csv_file, tm_threshold, outfolder)