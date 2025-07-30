import os
import json
import pandas as pd
import numpy as np
import argparse
import networkx as nx
import matplotlib.pyplot as plt

'''
This script creates a network graph from a similarity matrix.

Parameters:
    --adjacency_matrix: Path to the similarity matrix file.
    --output_path: Output path for the graph.
    --mask: Boolean mask indicating which nodes (and their neighbors) should be plotted (npy file)
    --ids: Path to a file assigning ids to the columns/rows of the adjacency matrix (json file).
    --labels: Path to a file with numerical labels for color coding (npy file).

Example usage:
    python create_graph.py --adjacency_matrix adjacency_matrix.npy --output_path similarity_graph.png --mask mask.npy --ids ids.json --labels labels.npy
'''


def parse_arguments():
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Create a network graph from an adjacency matrix, including selected nodes and their neighbors.")
    parser.add_argument('--adjacency_matrix', type=str, required=True, help='Path to the adjacency matrix (npy file).')
    parser.add_argument('--mask', type=str, help='Boolean mask indicating which nodes (and their neighbors) should be plotted (npy file)')
    parser.add_argument('--ids', type=str, help='Path to a file assigning ids to the columns/rows of the adjacency matrix (json file).')
    parser.add_argument('--labels', type=str, help='Path to a file with numerical labels for color coding (npy file).')
    parser.add_argument('--output_path', '-o', type=str, default='similarity_graph.png', help='Output path for the graph.')
    args = parser.parse_args()
    return args



def create_nx_graph(adjacency_matrix, ids=None, mask=None, labels=None, output_path="similarity_graph.png"):

    print(f"Creating network graph from {adjacency_matrix}...")
    
    # Load adjacency matrix
    adjacency_matrix = np.load(adjacency_matrix)
    num_total_nodes = adjacency_matrix.shape[0]
    print(f"Loaded adjacency matrix with shape {adjacency_matrix.shape}")
        
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
        node_sizes = [300   if i else 200     for i in mask]
        included_indices = np.where(mask)[0]
        print(f"Processing {len(included_indices)} nodes with mask=1 {[ids[x] for x in included_indices[0:5]]}...", flush=True)

    else:
        edgecolors = ['black'] * num_total_nodes
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
        node_colors_final.append(labels[index] if labels is not None else 0)
        node_sizes_final.append(node_sizes[index])
        edgecolors_final.append(edgecolors[index])
    # ------------------------------------------------------------------------------------------------
        
    
    # Draw the graph and save it
    # ------------------------------------------------------------------------------------------------
    print("Computing graph layout...")
    pos = nx.spring_layout(G)
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
    # ------------------------------------------------------------------------------------------------




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

    create_nx_graph(args.adjacency_matrix, ids, mask, labels, args.output_path)

if __name__ == "__main__":
    main()