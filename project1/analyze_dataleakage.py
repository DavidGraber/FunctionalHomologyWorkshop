import pandas as pd
import numpy as np
import sys
import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def compute_test_train_tm_diff_stats(cluster_df, tm_column='Tm'):
    """Calculate min/max Tm differences and node pairs between test and train nodes."""
    test_tm = pd.to_numeric(cluster_df[cluster_df['Test'] == 1][tm_column], errors='coerce').dropna()
    train_tm = pd.to_numeric(cluster_df[cluster_df['Train'] == 1][tm_column], errors='coerce').dropna()
    
    if test_tm.empty or train_tm.empty:
        return np.nan, np.nan, [], [], []
    
    differences = np.abs(test_tm.to_numpy()[:, np.newaxis] - train_tm.to_numpy()).flatten()
    min_diff, max_diff = np.min(differences), np.max(differences)
    
    min_pairs = [(ti, ri, test_tm[ti], train_tm[ri]) for ti in test_tm.index for ri in train_tm.index 
                 if np.abs(test_tm[ti] - train_tm[ri]) == min_diff]
    max_pairs = [(ti, ri, test_tm[ti], train_tm[ri]) for ti in test_tm.index for ri in train_tm.index 
                 if np.abs(test_tm[ti] - train_tm[ri]) == max_diff]
    
    return min_diff, max_diff, min_pairs, max_pairs, differences.tolist()

def compute_clustering_metrics(merged_df, tsv_df):
    """Compute clustering metrics and test-train Tm differences."""
    total_nodes, total_edges = len(merged_df), len(tsv_df)
    
    merged_df[['Train', 'Test']] = merged_df[['Train', 'Test']].apply(pd.to_numeric, errors='coerce')
    num_test, num_training = merged_df['Test'].sum(), merged_df['Train'].sum()
    
    connections = pd.merge(tsv_df, merged_df[['ID', 'Train', 'Test']], left_on='Source', right_on='ID', how='left')
    connections = connections[(connections['Test'] == 1) & connections['Target'].isin(merged_df[merged_df['Train'] == 1]['Target'])]
    num_connections = len(connections)
    test_with_connections = len(connections['Source'].unique())
    percent_test_connected = (test_with_connections / num_test * 100) if num_test > 0 else 0
    
    tm_stats = pd.DataFrame()  # Initialize empty DataFrame
    overall_avg_tm_diff = overall_min_diff = overall_max_diff = np.nan
    min_diff_info = max_diff_info = []

    if 'Tm' in merged_df.columns:
        tm_stats = merged_df.groupby('Target').apply(
            lambda x: compute_test_train_tm_diff_stats(x)
        ).apply(pd.Series, index=['Min Tm Diff', 'Max Tm Diff', 'Min Pairs', 'Max Pairs', 'All Differences'])
        
        all_differences = [d for diffs in tm_stats['All Differences'] for d in diffs]
        overall_avg_tm_diff = np.mean(all_differences) if all_differences else np.nan
        
        valid_stats = tm_stats.dropna(subset=['Min Tm Diff', 'Max Tm Diff'])
        if not valid_stats.empty:
            overall_min_diff, overall_max_diff = valid_stats['Min Tm Diff'].min(), valid_stats['Max Tm Diff'].max()
            min_diff_info = [
                f"Cluster {c}: Test node {ti} (ID: {merged_df.loc[ti, 'ID']}, Tm={ttm}), Train node {ri} (ID: {merged_df.loc[ri, 'ID']}, Tm={rtm})"
                for c, row in valid_stats[valid_stats['Min Tm Diff'] == overall_min_diff].iterrows()
                for ti, ri, ttm, rtm in row['Min Pairs']
            ]
            max_diff_info = [
                f"Cluster {c}: Test node {ti} (ID: {merged_df.loc[ti, 'ID']}, Tm={ttm}), Train node {ri} (ID: {merged_df.loc[ri, 'ID']}, Tm={rtm})"
                for c, row in valid_stats[valid_stats['Max Tm Diff'] == overall_max_diff].iterrows()
                for ti, ri, ttm, rtm in row['Max Pairs']
            ]

    print(f"CLUSTERING METRICS\n{'='*60}\n"
          f"Nodes: {total_nodes}, Edges: {total_edges}\n"
          f"Test datapoints: {num_test} ({num_test/total_nodes*100:.1f}%)\n"
          f"Training datapoints: {num_training} ({num_training/total_nodes*100:.1f}%)\n"
          f"Test-training connections: {num_connections}\n"
          f"Test nodes connected to training: {percent_test_connected:.1f}%\n"
          f"Average Tm difference: {overall_avg_tm_diff:.2f}\n"
          f"Smallest Tm difference: {overall_min_diff:.2f}\n" +
          "\n".join([f"  - {info}" for info in min_diff_info]) + "\n"
          f"Largest Tm difference: {overall_max_diff:.2f}\n" +
          "\n".join([f"  - {info}" for info in max_diff_info]) + "\n"
          f"{'='*60}\n")

def create_cluster_layout(clusters_by_size, node_attrs):
    """Arrange clusters in a grid layout with reduced spacing."""
    all_positions = {}
    n_clusters = len(clusters_by_size)
    grid_cols = max(1, int(np.ceil(np.sqrt(n_clusters))))
    grid_rows = max(1, int(np.ceil(n_clusters / grid_cols)))
    
    base_spacing = 10.0 * (5 / max(1, np.sqrt(n_clusters))) 
    
    for idx, (cluster_id, nodes) in enumerate(clusters_by_size):
        row, col = idx // grid_cols, idx % grid_cols
        center_x, center_y = col * base_spacing, -row * base_spacing
        
        cluster_graph = nx.Graph([(n1, n2) for i, n1 in enumerate(nodes) for n2 in nodes[i+1:]] + [(n, n) for n in nodes])
        min_node_distance = (5 / max(1, np.sqrt(len(nodes)))) 
        cluster_radius = max(4.5, 1.5 * np.sqrt(len(nodes)))
        
        if len(nodes) == 1:
            all_positions[nodes[0]] = (center_x, center_y)
        elif len(nodes) <= 8:
            angles = np.linspace(0, 2*np.pi, len(nodes), endpoint=False)
            for i, node in enumerate(nodes):
                all_positions[node] = (center_x + cluster_radius * np.cos(angles[i]), 
                                      center_y + cluster_radius * np.sin(angles[i]))
        else:
            pos = nx.spring_layout(cluster_graph, k=min_node_distance * 7.5, iterations=500, scale=cluster_radius) if cluster_graph.number_of_edges() > 0 else {
                node: (cluster_radius * 0.5 * np.cos(2*np.pi*i/len(nodes)), cluster_radius * 0.5 * np.sin(2*np.pi*i/len(nodes))) 
                for i, node in enumerate(nodes)
            }
            temp_positions = {node: (center_x + x, center_y + y) for node, (x, y) in pos.items()}
            
            for _ in range(200):  # Fewer iterations
                moved = False
                for i, n1 in enumerate(temp_positions):
                    for n2 in list(temp_positions)[i+1:]:
                        p1, p2 = np.array(temp_positions[n1]), np.array(temp_positions[n2])
                        dist = np.linalg.norm(p1 - p2)
                        if dist < min_node_distance and dist > 0:
                            direction = (p1 - p2) / dist
                            shift = (min_node_distance - dist) / 0.3
                            temp_positions[n1], temp_positions[n2] = tuple(p1 + direction * shift), tuple(p2 - direction * shift)
                            moved = True
                if not moved:
                    break
            
            all_positions.update(temp_positions)
    
    return all_positions

def create_subset_network_plot(subset_df, seq_ids, identity_matrix):
    """Create a network plot with UniProt labels below nodes, Tm values above nodes, and pairwise sequence identities on edges."""
    G = nx.Graph()
    edges = subset_df[['Target', 'Source']].dropna()
    edges = edges[edges['Target'] != edges['Source']].values
    print(f"Initial edges from subset_df: {len(edges)}")
    G.add_edges_from(edges)
    print(f"Edges in graph after adding: {G.number_of_edges()}")
    
    node_attrs = subset_df.set_index('ID')[['Tm', 'Test', 'Target']].dropna(subset=['Tm']).to_dict('index')
    valid_nodes = [n for n in G.nodes() if n in node_attrs]
    excluded_nodes = [n for n in G.nodes() if n not in node_attrs]
    if excluded_nodes:
        print(f"Warning: {len(excluded_nodes)} nodes excluded due to missing Tm values: {excluded_nodes}")
    G = G.subgraph(valid_nodes).copy()
    print(f"Edges in graph after subgraph filtering: {G.number_of_edges()}")
    
    if not G.number_of_nodes():
        print("No valid nodes for plotting.")
        return
    
    clusters = {}
    for node in G.nodes():
        clusters.setdefault(node_attrs[node]['Target'], []).append(node)
    clusters_by_size = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"Clusters: {len(clusters_by_size)}" + 
          (f" ({len(clusters_by_size[:10])} shown): " + ", ".join(f"{c}: {len(n)} nodes" for c, n in clusters_by_size[:10]) +
           (f", +{len(clusters_by_size)-10} more" if len(clusters_by_size) > 10 else "")))
    
    pos = create_cluster_layout(clusters_by_size, node_attrs)
    
    n_nodes = G.number_of_nodes()
    node_scale = min(1.0, 100 / max(1, n_nodes))  # Scale nodes inversely with node count
    node_size = 5000 * node_scale
    font_scale = min(1.0, 50 / max(1, np.sqrt(n_nodes)))  # Scale fonts with square root of node count
    font_size = 45 * font_scale
    
    node_colors = [node_attrs[n]['Tm'] for n in G.nodes()]
    edge_colors = ['red' if node_attrs[n]['Test'] == 1 else 'black' for n in G.nodes()]
    node_labels = {n: str(n) for n in G.nodes()}
    tm_labels = {n: f"{node_attrs[n]['Tm']:.1f}" for n in G.nodes()}
    label_positions_below = {n: (x, y - 2 * node_scale) for n, (x, y) in pos.items()}
    label_positions_above = {n: (x, y + 2 * node_scale) for n, (x, y) in pos.items()}
    
    # Create a dictionary to map UniProt IDs to indices in the identity matrix
    seq_id_to_index = {seq_id: idx for idx, seq_id in enumerate(seq_ids)}
    missing_ids = [n for n in G.nodes() if n not in seq_id_to_index]
    if missing_ids:
        print(f"Warning: {len(missing_ids)} node IDs not found in seq_ids: {missing_ids}")
    
    # Prepare edge labels for pairwise sequence identities
    edge_labels = {}
    for u, v in G.edges():
        if u in seq_id_to_index and v in seq_id_to_index:
            idx_u = seq_id_to_index[u]
            idx_v = seq_id_to_index[v]
            identity = identity_matrix[idx_u, idx_v]
            edge_labels[(u, v)] = "<30" if identity == 0 else f"{identity:.1f}"
    
    plt.figure(figsize=(100, 80))
    plt.gca().set_facecolor('white')
    plt.axis('off')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=3.5 * node_scale, alpha=0.6, edge_color='gray')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, edgecolors=edge_colors,
                           linewidths=5 * node_scale, cmap=plt.cm.viridis, vmin=min(node_colors, default=0),
                           vmax=max(node_colors, default=1), alpha=1.0)
    
    # Draw UniProt labels below nodes (light yellow box)
    for node, label in node_labels.items():
        plt.text(label_positions_below[node][0], label_positions_below[node][1], label, fontsize=font_size, fontweight='bold',
                 ha='center', va='top', bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow',
                                                  alpha=0.8, edgecolor='gray', linewidth=0.5 * node_scale))
    
    # Draw Tm values above nodes (pale violet box)
    for node, tm_label in tm_labels.items():
        plt.text(label_positions_above[node][0], label_positions_above[node][1], tm_label, fontsize=font_size, fontweight='bold',
                 ha='center', va='bottom', color='black',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='lavenderblush', alpha=0.8, edgecolor='none'))
    
    # Draw sequence identity labels on edges (grey transparent box)
    for (u, v), label in edge_labels.items():
        x_mid = (pos[u][0] + pos[v][0]) / 2
        y_mid = (pos[u][1] + pos[v][1]) / 2
        plt.text(x_mid, y_mid, label, fontsize=font_size * 0.8, fontweight='bold',
                 ha='center', va='center', color='black',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='grey', alpha=0.5, edgecolor='none'))
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(node_colors, default=0), vmax=max(node_colors, default=1)))
    cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.6, orientation='vertical', pad=0.05)
    cbar.set_label('Tm [°C]', labelpad=font_size, fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)
    
    # Legend including label boxes
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markeredgecolor='red',
               markersize=font_size, markeredgewidth=5 * node_scale, label='Test nodes'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markeredgecolor='black',
               markersize=font_size, markeredgewidth=5 * node_scale, label='Train nodes'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='lightyellow', markeredgecolor='gray',
               markersize=font_size * 0.8, label='UniProt ID (below nodes)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='lavenderblush', markeredgecolor='none',
               markersize=font_size * 0.8, label='Tm value (above nodes)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='grey', alpha=0.5, markeredgecolor='none',
               markersize=font_size * 0.8, label='Sequence identity (on edges)')
    ]
    
    plt.legend(handles=legend_elements, fontsize=font_size, 
              loc='upper center', bbox_to_anchor=(1.05, 0.1), 
              frameon=True, framealpha=0.8, edgecolor='grey')
    
    print(f"Saving plot to {outfolder}/network_potential_data_leakage.png/svg")
    plt.savefig(f"{outfolder}/network_potential_data_leakage.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{outfolder}/network_potential_data_leakage.svg", bbox_inches='tight', facecolor='white')
    plt.close()
    
def merge_files(tsv_path, csv_path, npz_path):
    """Merge TSV and CSV files, load identity matrix, compute metrics, and generate plots."""
    try:
        tsv_df = pd.read_csv(tsv_path, sep='\t', header=None, names=['Target', 'Source'])
        csv_df = pd.read_csv(csv_path, sep=';')
        
        if not all(col in csv_df.columns for col in ['ID', 'Train', 'Test', 'Tm']):
            raise ValueError("CSV must contain columns: ID, Train, Test, Tm")
        
        # Load identity matrix
        npz_data = np.load(npz_path)
        seq_ids = npz_data['seq_ids']
        identity_matrix = npz_data['identity_matrix']
        
        merged_df = pd.merge(csv_df, tsv_df[['Source', 'Target']], left_on='ID', right_on='Source', how='left')
        merged_df.to_csv(f"{outfolder}/clustered_dataset.csv", sep=';', index=False)
        print(f"Saved full dataset to {outfolder}/clustered_dataset.csv")
        
        compute_clustering_metrics(merged_df, tsv_df)
        
        valid_targets = merged_df.groupby('Target').filter(
            lambda x: (x['Train'].eq(1).any()) & (x['Test'].eq(1).any())
        )['Target'].unique()
        
        subset_df = merged_df[merged_df['Target'].isin(valid_targets)]
        subset_df.to_csv(f"{outfolder}/clustered_dataset_potential_data_leakage.csv", sep=';', index=False)
        print(f"Saved subset to {outfolder}/clustered_dataset_potential_data_leakage.csv")
        
        if not subset_df.empty:
            create_subset_network_plot(subset_df, seq_ids, identity_matrix)
        else:
            print("No plot generated: subset is empty.")
            
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python analyze_dataleakage.py <tsv_file> <csv_file> <npz_file> <outfolder>")
        sys.exit(1)
    
    tsv_file, csv_file, npz_file, outfolder = sys.argv[1:5]
    for path in [tsv_file, csv_file, npz_file, outfolder]:
        if not os.path.exists(path):
            print(f"Error: Path '{path}' does not exist")
            sys.exit(1)
    
    merge_files(tsv_file, csv_path=csv_file, npz_path=npz_file)
