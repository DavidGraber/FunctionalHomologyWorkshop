import os
import json
import pandas as pd
import numpy as np
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import pickle

'''
This script creates a network graph from a similarity matrix.

Parameters:
    --adjacency_matrix: Path to the similarity matrix file.
    --output_path: Output path for the graph.
    --mask: Boolean mask indicating which nodes (and their neighbors) should be plotted (npy file)
    --ids: Path to a file assigning ids to the columns/rows of the adjacency matrix (json file).
    --labels: Path to a file with numerical labels for color coding (npy file).

Example usage:
    python create_graph.py --clustering adjacency_matrix.npy --output_path similarity_graph.png --mask mask.npy --ids ids.json --labels labels.npy
    python create_graph.py --clustering clustering.tsv --output_path similarity_graph.png --mask mask.npy --ids ids.json --labels labels.npy
'''


def parse_arguments():
    parser = argparse.ArgumentParser(description="Create a network graph from a clustering file.")
    parser.add_argument('--clustering', '-c', type=str, required=True, help='Path to clustering file, either adjacency matrix (.npy) or mmseqs2-style clustering (.tsv).')
    parser.add_argument('--mask', '-m', type=str, help='Boolean mask indicating which nodes (and their neighbors) should be plotted (npy file)')
    parser.add_argument('--ids', '-i', type=str, help='Path to a file assigning ids to the columns/rows of the adjacency matrix (json file).')
    parser.add_argument('--labels', '-l', type=str, help='Path to a file with numerical labels for color coding (npy file).')
    parser.add_argument('--output_path', '-o', type=str, default='similarity_graph.png', help='Output path for graph')
    parser.add_argument('--graph_output_path', '-g', type=str, default=None, help='Optional output path to save the networkx file to.')
    args = parser.parse_args()
    return args


def clustering_to_adjacency_matrix(clustering, ids=None):

    if ids is None:
        raise ValueError("IDs list is required for clustering to adjacency matrix conversion")
    
    # Initialize adjacency matrix with zeros
    n = len(ids)
    adjacency_matrix = np.zeros((n, n), dtype=int)
    id_to_index = {id_val: idx for idx, id_val in enumerate(ids)} # Map from ID to index for fast lookup
    
    # Read the clustering file and populate the adjacency matrix
    with open(clustering, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                id1, id2 = line.split('\t')
                if id1 in id_to_index and id2 in id_to_index:
                    idx1 = id_to_index[id1]
                    idx2 = id_to_index[id2]
                    adjacency_matrix[idx1, idx2] = 1
                    adjacency_matrix[idx2, idx1] = 1
                else:
                    print(f"Warning: Skipping pair ({id1}, {id2}) - one or both IDs not found in provided IDs list")
    
    print(f"Created adjacency matrix with shape {adjacency_matrix.shape}")
    return adjacency_matrix



def create_nx_graph(adjacency_matrix, ids=None, mask=None, labels=None, output_path="similarity_graph.png", graph_output_path=None):

    print(f"Creating network graph from adjacency matrix with shape {adjacency_matrix.shape}...")
    num_total_nodes = adjacency_matrix.shape[0]
        
    # Setup color scale, node sizes and edgecolors arrays based on the mask and the labels
    # ------------------------------------------------------------------------------------------------
    if labels is not None:
        cmap = plt.cm.viridis
        vmin = min(labels)
        vmax = max(labels)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    else: 
        cmap = None
        vmin = None
        vmax = None
        norm = None

    # Get indices of nodes that should be included (mask=1)
    if mask is not None:
        mask = np.atleast_1d(mask)
        edgecolors = ['red' if i else 'white' for i in mask]
        node_sizes = [500   if i else 300     for i in mask]
        included_indices = np.where(mask)[0]
        print(f"Processing {len(included_indices)} nodes with mask=1 {[ids[x] for x in included_indices[0:5]]}...", flush=True)

    else:
        edgecolors = ['white'] * num_total_nodes
        node_sizes = [200] * num_total_nodes
        included_indices = np.arange(num_total_nodes)
        print(f"Processing all {len(included_indices)} nodes (no mask provided) {[ids[x] for x in included_indices[0:5]]}...", flush=True)  
    # ------------------------------------------------------------------------------------------------


    # Initialize Graph, add nodes and edges
    # ------------------------------------------------------------------------------------------------
    G = nx.Graph()

    # First pass: add edges for included nodes and their neighbors
    for i in included_indices:
        for j in range(adjacency_matrix.shape[1]):
            if adjacency_matrix[i, j] > 0:
                G.add_edge(ids[i], ids[j])

    # # Second pass: add edges between neighbors of included nodes
    # # This ensures we capture the full neighborhood structure
    # for i in included_indices:
    #     for j in range(adjacency_matrix.shape[1]):
    #         if adjacency_matrix[i, j] > 0:
    #             # Add edges between neighbors of included nodes
    #             for k in range(adjacency_matrix.shape[1]):
    #                 if adjacency_matrix[j, k] > 0 and j != k:
    #                     G.add_edge(ids[j], ids[k])

    print(f"Total nodes in graph: {G.number_of_nodes()}")
    print(f"Total edges in graph: {G.number_of_edges()}")
    # ------------------------------------------------------------------------------------------------


    # Create final node_colors, node_sizes and edgecolors using indexing
    # ------------------------------------------------------------------------------------------------

    node_colors_final = []
    node_sizes_final = []
    edgecolors_final = []
    
    for node in G.nodes():
        index = ids.index(node)
        node_colors_final.append(labels[index] if labels is not None else 'cornflowerblue')
        node_sizes_final.append(node_sizes[index])
        edgecolors_final.append(edgecolors[index])
    # ------------------------------------------------------------------------------------------------
        
    
    # Draw the graph and save it
    # ------------------------------------------------------------------------------------------------
    print("Computing graph layout...")
    pos = nx.spring_layout(G, k=0.05)
    # pos = nx.kamada_kawai_layout(G)
    # pos = nx.circular_layout(G)
    # pos = nx.shell_layout(G)
    
    plt.figure(figsize=(50, 50))
    plt.gca().set_facecolor('white')
    plt.axis('off')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                        node_size=node_sizes_final, 
                        node_color=node_colors_final, 
                        edgecolors=edgecolors_final, 
                        cmap=cmap, 
                        vmin=vmin, 
                        vmax=vmax, 
                        alpha=0.8)

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='gray')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')
    

    # Add colorbar if labels are provided
    if labels is not None and len(node_colors_final) > 0:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8, orientation='horizontal')
        cbar.set_label('Label Values', labelpad=15, fontsize=20)
        cbar.ax.tick_params(labelsize=20)  # Set tick label font size to 20
    
    # Save the graph
    print(f"Saving graph to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("Graph creation completed!")

    # Print metrics about the clustering
    print_metrics(adjacency_matrix, mask)
    # ------------------------------------------------------------------------------------------------

    # Optionally write the NetworkX graph to a file
    if graph_output_path:
        print(f"Writing NetworkX graph to {graph_output_path}...")
        with open(graph_output_path, "wb") as f:
            pickle.dump(G, f)
        print("Graph file saved!")
    # ------------------------------------------------------------------------------------------------


def print_metrics(adjacency_matrix, mask=None):

    print("\n" + "="*60)
    print("CLUSTERING METRICS")
    print("="*60)
    
    # Basic connectivity metrics (always available)
    n_nodes = adjacency_matrix.shape[0]
    degrees = np.sum(adjacency_matrix, axis=1)
    n_edges = np.sum(adjacency_matrix) // 2  # Divide by 2 for undirected graph
    
    print(f"📊 BASIC CONNECTIVITY METRICS:")
    print(f"   • Total nodes: {n_nodes}")
    print(f"   • Total edges: {n_edges}")
    
    if mask is not None:
        print(f"\n MASKED CLUSTERS METRICS:")
        n_masked = np.sum(mask)
        print(f"   • Masked nodes (test set complexes): {n_masked} ({n_masked/n_nodes*100:.1f}% of total)")
        print(f"   • External nodes (train set complexes): {np.sum(~mask)} ({np.sum(~mask)/n_nodes*100:.1f}% of total)")
        
        # # Connections to external nodes
        external_edges = adjacency_matrix[mask][:, ~mask]
        print(f"   • Connections between masked and external nodes: {np.sum(external_edges)}")
        print(f"   • Percentage of masked nodes with connections to external nodes: {((external_edges == 1).any(axis=1).sum() / n_masked) * 100:.1f}%")
        
        # Masked node degree statistics
        masked_degrees = degrees[mask]
        print(f"\n Degree (number of connections) of masked nodes (test set nodes):")
        print(f"   • Avg degree: {np.mean(masked_degrees):.2f}")
        print(f"   • Min degree: {np.min(masked_degrees)}")
        print(f"   • Max degree: {np.max(masked_degrees)}")

        # External node degree statistics
        external_degrees = degrees[~mask]
        print(f"\n Degree (number of connections) of external nodes (train nodes):")
        print(f"   • Avg degree: {np.mean(external_degrees):.2f}")
        print(f"   • Min degree: {np.min(external_degrees)}")
        print(f"   • Max degree: {np.max(external_degrees)}")
    print("="*60 + "\n")


def main():
    args = parse_arguments()

    ids = None
    if args.ids:
        with open(args.ids, 'r') as f:
            ids = json.load(f)
        print(f"Loaded ids with length {len(ids)} ({ids[0:5]}...)")

    mask = None
    if args.mask:
        mask = np.load(args.mask)
        print(f"Loaded mask with {np.sum(mask)} active nodes out of {len(mask)} total nodes")

    labels = None
    if args.labels:
        labels = np.load(args.labels)
        print(f"Loaded labels with shape {labels.shape}")

    # Load clustering/adjacency matrix
    if args.clustering.endswith('.tsv'):
        adjacency_matrix = clustering_to_adjacency_matrix(args.clustering, ids)
    elif args.clustering.endswith('.npy'):
        adjacency_matrix = np.load(args.clustering)
    else:
        raise ValueError(f"Unsupported file type: {args.clustering}")

    create_nx_graph(adjacency_matrix, ids, mask, labels, args.output_path, args.graph_output_path)

if __name__ == "__main__":
    main()