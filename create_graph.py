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
    --similarity_matrix: Path to the similarity matrix file.
    --output_path: Output path for the graph.

Example usage:
    python create_graph.py --similarity_matrix similarity_matrix.tsv --output_path similarity_graph.png
'''


def parse_arguments():
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Extract sequences from PDB files and cluster them using MMseqs.")
    parser.add_argument('--similarity_matrix', type=str, default='similarity_matrix.tsv', help='Path to the similarity matrix file.')
    parser.add_argument('--output_path', '-o', type=str, default='similarity_graph.png', help='Output path for the graph.')
    args = parser.parse_args()
    return args



def create_nx_graph(similarity_matrix, output_path="similarity_graph.png"):

    print(f"Creating network graph from {similarity_matrix}...")

    similarity_matrix = pd.read_csv(similarity_matrix, sep='\t', header=None)
    G = nx.Graph()

    # Add edges from the DataFrame
    for index, row in similarity_matrix.iterrows():
        G.add_edge(row[0], row[1])

    # Set node positions using the spring layout
    pos = nx.spring_layout(G)

    # Set a clear background
    plt.figure(figsize=(50, 50))
    plt.gca().set_facecolor('white')

    # Draw nodes with adjusted size and color
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color='skyblue', alpha=0.8)

    # Draw edges with adjusted thickness and color
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='gray')

    # Draw labels with adjusted font size
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')

    plt.axis('off')  # Turn off axis
    plt.savefig(output_path, dpi=300, bbox_inches='tight')



def main():

    args = parse_arguments()
    create_nx_graph(args.similarity_matrix, args.output_path)

if __name__ == "__main__":
    main()