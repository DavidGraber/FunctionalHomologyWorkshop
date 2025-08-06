import pandas as pd
import numpy as np
import sys
import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math

def compute_test_train_tm_diff_stats(cluster_df, tm_column='Tm'):
    """Calculate min/max Tm differences and node pairs between test and train nodes in a cluster."""
    # Separate test and train nodes
    test_nodes = cluster_df[cluster_df['Test'] == 1]
    train_nodes = cluster_df[cluster_df['Train'] == 1]
    
    # Convert Tm values to numeric and drop NaNs
    test_tm = pd.to_numeric(test_nodes[tm_column], errors='coerce').dropna()
    train_tm = pd.to_numeric(train_nodes[tm_column], errors='coerce').dropna()
    
    # If either test or train nodes are missing or empty, return NaNs
    if test_tm.empty or train_tm.empty:
        return np.nan, np.nan, [], [], []
    
    # Compute all pairwise Tm differences using vectorized operation
    test_tm_array = test_tm.to_numpy()
    train_tm_array = train_tm.to_numpy()
    differences = np.abs(test_tm_array[:, np.newaxis] - train_tm_array).flatten()
    
    # Find min and max differences
    min_diff = np.min(differences)
    max_diff = np.max(differences)
    
    # Identify node pairs for min and max differences
    min_pairs = []
    max_pairs = []
    test_indices = test_tm.index
    train_indices = train_tm.index
    for i, test_idx in enumerate(test_indices):
        for j, train_idx in enumerate(train_indices):
            diff = np.abs(test_tm[test_idx] - train_tm[train_idx])
            if diff == min_diff:
                min_pairs.append((test_idx, train_idx, test_tm[test_idx], train_tm[train_idx]))
            if diff == max_diff:
                max_pairs.append((test_idx, train_idx, test_tm[test_idx], train_tm[train_idx]))
    
    return min_diff, max_diff, min_pairs, max_pairs, differences.tolist()

def compute_clustering_metrics(merged_df, tsv_df):
    """Compute clustering metrics and test-train Tm difference statistics."""
    # Basic connectivity metrics
    total_nodes = len(merged_df)
    total_edges = len(tsv_df)  # Each TSV row is an edge
    
    # Test and Training datapoints
    merged_df['Train'] = pd.to_numeric(merged_df['Train'], errors='coerce')
    merged_df['Test'] = pd.to_numeric(merged_df['Test'], errors='coerce')
    test_datapoints = merged_df[merged_df['Test'] == 1]
    training_datapoints = merged_df[merged_df['Train'] == 1]
    num_test = len(test_datapoints)
    num_training = len(training_datapoints)
    
    # Connections between test and training datapoints
    merged_with_train_test = pd.merge(tsv_df, merged_df[['ID', 'Train', 'Test']], 
                                      left_on='Source', right_on='ID', how='left')
    connections = merged_with_train_test[
        (merged_with_train_test['Test'] == 1) & 
        (merged_with_train_test['Target'].isin(training_datapoints['Target']))
    ]
    num_connections = len(connections)
    test_with_connections = len(connections['Source'].unique())
    percent_test_connected = (test_with_connections / num_test * 100) if num_test > 0 else 0
    
    # Test-train Tm difference statistics per cluster
    if 'Tm' in merged_df.columns:
        tm_stats = merged_df.groupby('Target').apply(
            lambda x: compute_test_train_tm_diff_stats(x, 'Tm')
        )
        tm_stats = pd.DataFrame(
            tm_stats.tolist(),
            index=tm_stats.index,
            columns=['Min Tm Diff', 'Max Tm Diff', 'Min Pairs', 'Max Pairs', 'All Differences']
        )
        
        # Compute overall average Tm difference
        all_differences = []
        for differences in tm_stats['All Differences']:
            all_differences.extend(differences)
        overall_avg_tm_diff = np.mean(all_differences) if all_differences else np.nan
        
        # Find overall smallest and largest Tm differences and their node pairs
        valid_stats = tm_stats.dropna(subset=['Min Tm Diff', 'Max Tm Diff'])
        overall_min_diff = valid_stats['Min Tm Diff'].min() if not valid_stats.empty else np.nan
        overall_max_diff = valid_stats['Max Tm Diff'].max() if not valid_stats.empty else np.nan
        
        min_diff_clusters = valid_stats[valid_stats['Min Tm Diff'] == overall_min_diff]
        max_diff_clusters = valid_stats[valid_stats['Max Tm Diff'] == overall_max_diff]
        
        min_diff_info = []
        for cluster, row in min_diff_clusters.iterrows():
            for test_idx, train_idx, test_tm, train_tm in row['Min Pairs']:
                test_id = merged_df.loc[test_idx, 'ID']
                train_id = merged_df.loc[train_idx, 'ID']
                min_diff_info.append(
                    f"Cluster {cluster}: Test node {test_idx} with ID: {test_id} (Tm={test_tm}), "
                    f"Train node {train_idx} with ID: {train_id} (Tm={train_tm})"
                )
        
        max_diff_info = []
        for cluster, row in max_diff_clusters.iterrows():
            for test_idx, train_idx, test_tm, train_tm in row['Max Pairs']:
                test_id = merged_df.loc[test_idx, 'ID']
                train_id = merged_df.loc[train_idx, 'ID']
                max_diff_info.append(
                    f"Cluster {cluster}: Test node {test_idx} with ID: {test_id} (Tm={test_tm}), "
                    f"Train node {train_idx} with ID: {train_id} (Tm={train_tm})"
                )
    else:
        overall_avg_tm_diff = np.nan
        overall_min_diff = overall_max_diff = np.nan
        min_diff_info = max_diff_info = []

    # Print statistics
    print("============================================================\n"
          "CLUSTERING METRICS\n"
          "============================================================\n"
          "📊 BASIC CONNECTIVITY METRICS:\n"
          f"   • Total nodes: {total_nodes}\n"
          f"   • Total edges: {total_edges}\n\n"
          "CLUSTER METRICS:\n"
          f"   • Test datapoints: {num_test} "
          f"({num_test/total_nodes*100:.1f}% of total)\n"
          f"   • Training datapoints: {num_training} "
          f"({num_training/total_nodes*100:.1f}% of total)\n"
          f"   • Connections between test and training datapoints: {num_connections}\n"
          f"   • Percentage of test datapoints with connections to training datapoints: {percent_test_connected:.1f}%\n\n"
          "TEST-TRAIN Tm DIFFERENCE METRICS (within clusters):\n"
          f"   • Overall average Tm difference: {overall_avg_tm_diff:.2f}\n"
          f"   • Smallest Tm difference: {overall_min_diff:.2f}\n"
          f"     Nodes involved:\n")
    for info in min_diff_info:
        print(f"       - {info}")
    print(f"   • Largest Tm difference: {overall_max_diff:.2f}\n"
          f"     Nodes involved:\n")
    for info in max_diff_info:
        print(f"       - {info}")
    print("============================================================\n")

def create_cluster_layout(clusters_by_size, node_attrs):
    """Create a layout that arranges clusters in a grid, sorted by size."""
    all_positions = {}
    
    # Calculate grid dimensions based on number of clusters
    n_clusters = len(clusters_by_size)
    grid_cols = max(1, int(np.ceil(np.sqrt(n_clusters))))
    grid_rows = max(1, int(np.ceil(n_clusters / grid_cols)))
    
    # Calculate spacing between cluster centers
    base_spacing = 40.0  # Massively increased spacing between cluster centers (8.0 × 5)
    
    print(f"Arranging {n_clusters} clusters in a {grid_rows}x{grid_cols} grid")
    
    for idx, (cluster_id, nodes) in enumerate(clusters_by_size):
        # Calculate grid position for this cluster
        row = idx // grid_cols
        col = idx % grid_cols
        
        # Calculate cluster center position
        center_x = col * base_spacing
        center_y = -row * base_spacing  # Negative to have largest clusters at top
        
        # Create subgraph for this cluster
        cluster_graph = nx.Graph()
        cluster_edges = []
        
        # Add edges within this cluster (based on Target grouping)
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i < j:  # Avoid duplicate edges
                    cluster_edges.append((node1, node2))
        
        cluster_graph.add_edges_from(cluster_edges)
        cluster_graph.add_nodes_from(nodes)  # Ensure isolated nodes are included
        
        # Calculate cluster radius based on number of nodes with better spacing
        min_node_distance = 3.0  # Massively increased minimum distance between nodes (0.6 × 5)
        cluster_radius = max(9.0, 3.0 * np.sqrt(len(nodes)))  # Much larger base radius (1.8 × 5)
        
        if len(nodes) == 1:
            # Single node cluster
            all_positions[nodes[0]] = (center_x, center_y)
        elif len(nodes) <= 8:
            # Small clusters: arrange in circle with minimum distance enforcement
            required_radius = (len(nodes) * min_node_distance) / (2 * np.pi)
            cluster_radius = max(cluster_radius, required_radius)
            
            angles = np.linspace(0, 2*np.pi, len(nodes), endpoint=False)
            for i, node in enumerate(nodes):
                x = center_x + cluster_radius * np.cos(angles[i])
                y = center_y + cluster_radius * np.sin(angles[i])
                all_positions[node] = (x, y)
        else:
            # Larger clusters: use spring layout with anti-overlap post-processing
            if cluster_graph.number_of_edges() > 0:
                cluster_pos = nx.spring_layout(
                    cluster_graph, 
                    k=min_node_distance * 15,  # Massive repulsion (3 × 5)
                    iterations=750,  # Many more iterations (150 × 5)
                    scale=cluster_radius
                )
            else:
                # If no edges, arrange in a circular pattern with multiple rings if needed
                nodes_per_ring = 8
                if len(nodes) <= nodes_per_ring:
                    # Single ring
                    angles = np.linspace(0, 2*np.pi, len(nodes), endpoint=False)
                    cluster_pos = {}
                    for i, node in enumerate(nodes):
                        x = cluster_radius * np.cos(angles[i])
                        y = cluster_radius * np.sin(angles[i])
                        cluster_pos[node] = (x, y)
                else:
                    # Multiple concentric rings
                    cluster_pos = {}
                    remaining_nodes = list(nodes)
                    ring_radius = cluster_radius * 0.5
                    
                    while remaining_nodes:
                        ring_nodes = remaining_nodes[:nodes_per_ring]
                        remaining_nodes = remaining_nodes[nodes_per_ring:]
                        
                        angles = np.linspace(0, 2*np.pi, len(ring_nodes), endpoint=False)
                        for i, node in enumerate(ring_nodes):
                            x = ring_radius * np.cos(angles[i])
                            y = ring_radius * np.sin(angles[i])
                            cluster_pos[node] = (x, y)
                        
                        ring_radius += min_node_distance * 15  # Massive space between rings (3 × 5)
            
            # Translate to cluster center and apply anti-overlap post-processing
            temp_positions = {}
            for node, (x, y) in cluster_pos.items():
                temp_positions[node] = (center_x + x, center_y + y)
            
            # Anti-overlap post-processing with much stronger forces
            max_iterations = 500  # Many more iterations (100 × 5)
            for iteration in range(max_iterations):
                moved_any = False
                nodes_list = list(temp_positions.keys())
                
                for i, node1 in enumerate(nodes_list):
                    for j, node2 in enumerate(nodes_list[i+1:], i+1):
                        pos1 = np.array(temp_positions[node1])
                        pos2 = np.array(temp_positions[node2])
                        
                        distance = np.linalg.norm(pos1 - pos2)
                        if distance < min_node_distance and distance > 0:
                            # Move nodes apart with much stronger force
                            direction = (pos1 - pos2) / distance
                            shift = (min_node_distance - distance) / 0.3  # Much stronger push (1.5 ÷ 5)
                            
                            temp_positions[node1] = tuple(pos1 + direction * shift)
                            temp_positions[node2] = tuple(pos2 - direction * shift)
                            moved_any = True
                
                if not moved_any:
                    break
            
            all_positions.update(temp_positions)
    
    return all_positions

def create_subset_network_plot(subset_df):
    """Create a network plot with clusters clearly separated and sorted by size."""
    print(f"Creating improved network plot from subset DataFrame...")
    
    # Initialize Graph
    G = nx.Graph()
    
    # Add edges from Target and Source columns, excluding self-edges
    edges = subset_df[['Target', 'Source']].dropna()
    edges = edges[edges['Target'] != edges['Source']]
    G.add_edges_from(edges.values)
    
    print(f"Total nodes in subset graph: {G.number_of_nodes()}")
    print(f"Total edges in subset graph: {G.number_of_edges()}")
    
    # Get node attributes (Tm values and Test status)
    node_attrs = subset_df.set_index('ID')[['Tm', 'Test', 'Target']].dropna(subset=['Tm']).to_dict('index')
    
    # Filter nodes to those present in the graph and with valid attributes
    valid_nodes = [node for node in G.nodes() if node in node_attrs]
    G = G.subgraph(valid_nodes).copy()
    
    if G.number_of_nodes() == 0:
        print("No valid nodes found for plotting.")
        return
    
    # Group nodes by their Target (cluster) and sort by cluster size
    clusters = {}
    for node in G.nodes():
        target = node_attrs[node]['Target']
        if target not in clusters:
            clusters[target] = []
        clusters[target].append(node)
    
    # Sort clusters by size (largest first)
    clusters_by_size = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"Found {len(clusters_by_size)} clusters:")
    for cluster_id, nodes in clusters_by_size[:10]:  # Show top 10
        print(f"  Cluster {cluster_id}: {len(nodes)} nodes")
    if len(clusters_by_size) > 10:
        print(f"  ... and {len(clusters_by_size) - 10} more clusters")
    
    # Create custom layout with separated clusters
    pos = create_cluster_layout(clusters_by_size, node_attrs)
    
    # Prepare node colors and edge colors
    node_colors = []
    edge_colors = []
    node_sizes = []
    
    for node in G.nodes():
        tm_value = node_attrs[node]['Tm']
        test_status = node_attrs[node]['Test']
        
        node_colors.append(tm_value)
        edge_colors.append('red' if test_status == 1 else 'black')
        node_sizes.append(150)  # Slightly smaller nodes for better visibility
    
    # Setup colormap for Tm values
    cmap = plt.cm.viridis
    vmin = min(node_colors) if node_colors else 0
    vmax = max(node_colors) if node_colors else 1
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    # Create plot with much larger figure size
    plt.figure(figsize=(80, 60))  # Much larger canvas (16×5, 12×5)
    plt.gca().set_facecolor('white')
    plt.axis('off')
    
    # Draw cluster backgrounds (optional - helps visualize clusters)
    cluster_colors = plt.cm.Set3(np.linspace(0, 1, len(clusters_by_size)))
    for idx, (cluster_id, nodes) in enumerate(clusters_by_size):
        if len(nodes) > 1:
            cluster_positions = [pos[node] for node in nodes if node in pos]
            if cluster_positions:
                xs, ys = zip(*cluster_positions)
                center_x, center_y = np.mean(xs), np.mean(ys)
                radius = max(0.6, np.sqrt(len(nodes)) * 0.4)
                circle = Circle((center_x, center_y), radius, 
                              facecolor=cluster_colors[idx], alpha=0.1, 
                              edgecolor=cluster_colors[idx], linewidth=1)
                plt.gca().add_patch(circle)
    
    # Draw edges (make them more visible)
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.6, edge_color='gray')
    
    # Draw nodes
    nodes_collection = nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors=edge_colors,
        linewidths=2,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.8
    )
    
    # Draw labels only for cluster centroids (Target values) - positioned below clusters
    centroid_labels = {}
    centroid_positions = {}
    
    for cluster_id, nodes in clusters_by_size:
        if nodes and all(node in pos for node in nodes):
            # Calculate cluster bounds and centroid position
            cluster_positions = [pos[node] for node in nodes]
            xs, ys = zip(*cluster_positions)
            
            # Position label below the cluster with much more offset
            min_y = min(ys)
            centroid_x = np.mean(xs)
            label_y = min_y - 7.5  # Much more space below cluster (1.5 × 5)
            
            centroid_node = f"centroid_{cluster_id}"
            centroid_labels[centroid_node] = str(cluster_id)
            centroid_positions[centroid_node] = (centroid_x, label_y)
    
    # Draw centroid labels with much smaller font and box
    for node, label in centroid_labels.items():
        plt.text(centroid_positions[node][0], centroid_positions[node][1], 
                label, fontsize=8, fontweight='bold', ha='center', va='center',  # Much smaller font (6 ÷ 5)
                bbox=dict(boxstyle='round,pad=0.04', facecolor='lightyellow', alpha=0.8, edgecolor='gray', linewidth=0.1))  # Much smaller box
    
    # Add colorbar for Tm values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.6, orientation='horizontal')
    cbar.set_label('Tm Values', labelpad=10, fontsize=20)
    cbar.ax.tick_params(labelsize=15)
    
    # Add legend for Test nodes
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markeredgecolor='red', markersize=20, markeredgewidth=2, 
               label='Test nodes'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markeredgecolor='black', markersize=20, markeredgewidth=2, 
               label='Training nodes')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=25)
    
    # Add title with cluster information
    plt.title(f'Network Visualization: {len(clusters_by_size)} clusters, '
              f'{G.number_of_nodes()} nodes, {G.number_of_edges()} edges',
              fontsize=30, fontweight='bold')
    
    # Save the plot
    print(f"Saving network plot to {outfolder}/network_potential_data_leakage.png/svg ...")
    plt.savefig(f"{outfolder}/network_potential_data_leakage.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{outfolder}/network_potential_data_leakage.svg", bbox_inches='tight', facecolor='white')
    plt.close()

def merge_files(tsv_path, csv_path):
    try:
        # Read TSV file (no header, assign column names)
        tsv_df = pd.read_csv(tsv_path, sep='\t', header=None, names=['Target', 'Source'])
        
        # Read CSV file (semicolon-separated, with header)
        csv_df = pd.read_csv(csv_path, sep=';')
        
        # Ensure required columns exist in CSV
        required_columns = ['ID', 'Train', 'Test', 'Tm']
        missing_columns = [col for col in required_columns if col not in csv_df.columns]
        if missing_columns:
            raise ValueError(f"CSV file must contain columns: {', '.join(missing_columns)}")
        
        # Merge dataframes on csv.ID = tsv.Source
        merged_df = pd.merge(csv_df, tsv_df[['Source', 'Target']], 
                           left_on='ID', right_on='Source', how='left')
        
        # Save the full merged result 
        merged_df.to_csv(f"{outfolder}/clustered_dataset.csv", sep=';', index=False)
        print(f"Full output saved to {outfolder}/clustered_dataset.csv")
        
        # Compute and print clustering metrics
        compute_clustering_metrics(merged_df, tsv_df)
        
        # Create subset: keep rows where Target has both Train=1 and Test=1
        merged_df['Train'] = pd.to_numeric(merged_df['Train'], errors='coerce')
        merged_df['Test'] = pd.to_numeric(merged_df['Test'], errors='coerce')
        
        # Group by Target and check for presence of 1 in both Train and Test
        valid_targets = merged_df.groupby('Target').filter(
            lambda x: (x['Train'] == 1).any() and (x['Test'] == 1).any()
        )['Target'].unique()
        
        # Filter rows where Target is in valid_targets
        subset_df = merged_df[merged_df['Target'].isin(valid_targets)]
        
        # Save the subset to a separate file
        subset_df.to_csv(f"{outfolder}/clustered_dataset_potential_data_leakage.csv", sep=';', index=False)
        print(f"Subset output saved to {outfolder}/clustered_dataset_potential_data_leakage.csv")
        
        # Create network plot from subset_df
        if not subset_df.empty:
            create_subset_network_plot(subset_df)
        else:
            print("No network plot generated: subset DataFrame is empty.")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check if correct number of command-line arguments is provided
    if len(sys.argv) !=4:
        print("Usage: python analyze_dataleakage.py <tsv_file generated by mmseqs2 clustering> <csv_file of ProtStab2 dataset>")
        sys.exit(1)
    
    # Get file paths from command-line arguments
    tsv_file = sys.argv[1]
    csv_file = sys.argv[2]
    outfolder = sys.argv[3]
    
    # Verify input files exist
    if not os.path.exists(tsv_file):
        print(f"Error: TSV file '{tsv_file}' does not exist")
        sys.exit(1)
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' does not exist")
        sys.exit(1)
    if not os.path.exists(outfolder):
        print(f"Error: output folder '{csv_file}' does not exist")
        sys.exit(1)    
    # Run the merge and subset
    merge_files(tsv_file, csv_file)