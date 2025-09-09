import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

def create_nx_graph(adjacency_matrix, ids=None, mask=None, labels=None, output_path="similarity_graph.png", graph_output_path=None):
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

    else:
        edgecolors = ['white'] * num_total_nodes
        node_sizes = [200] * num_total_nodes
        included_indices = np.arange(num_total_nodes)
    # ------------------------------------------------------------------------------------------------


    # Initialize Graph, add nodes and edges
    # ------------------------------------------------------------------------------------------------
    G = nx.Graph()

    for i in included_indices:
        for j in range(adjacency_matrix.shape[1]):
            if adjacency_matrix[i, j] > 0:
                G.add_edge(ids[i], ids[j])

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
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Optionally write the NetworkX graph to a file
    if graph_output_path:
        with open(graph_output_path, "wb") as f:
            pickle.dump(G, f)


def create_graph(adj, ids=None, mask=None, labels=None, output_folder = Path('./'), prefix = 'similarity_graph'):
    ids = json.load(open(ids)) if ids else None
    mask = np.load(mask) if mask else None
    labels = np.load(labels) if labels else None

    # Define output path for graph figure and file.
    plot_out = str(output_folder / (prefix + '.png'))
    network_out = str(output_folder / (prefix + '.pkl'))

    create_nx_graph(adj, ids, mask, labels, plot_out, network_out)
