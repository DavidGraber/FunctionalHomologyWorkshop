import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from collections import defaultdict
import numpy as np
import math
import warnings
from matplotlib import MatplotlibDeprecationWarning
import argparse
import seaborn as sns

# Hide Matplotlib deprecation warnings
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

def load_similarity_matrix(npz_file, substrate_ids):
    """
    Load pairwise similarity matrix from .npz file and map substrate IDs to S-prefixed format.
    Handles cases where substrate IDs may be strings or integers (e.g., numpy.int64) and aligns
    CSV IDs (e.g., S008) with .npz IDs (e.g., 8 or 008).
    """
    try:
        data = np.load(npz_file, allow_pickle=True)
        if 'similarity_matrix' not in data.files or 'substrate_ids' not in data.files:
            missing_keys = [key for key in ['similarity_matrix', 'substrate_ids'] if key not in data.files]
            raise KeyError(f"Missing required keys in .npz file: {missing_keys}")
        matrix = data['similarity_matrix']
        npz_substrate_ids = data['substrate_ids']
        
        # Normalize .npz substrate IDs: convert to strings and strip leading zeros
        npz_substrate_ids_normalized = [
            str(int(str(id).lstrip('0'))) if str(id).isdigit() else str(id) 
            for id in npz_substrate_ids
        ]
        # Map to S-prefixed IDs for visualization (e.g., '8' -> 'S8')
        mapped_substrate_ids = [f"S{id}" for id in npz_substrate_ids_normalized]
        
        # Normalize input substrate IDs: strip 'S' prefix and leading zeros
        input_substrate_ids_normalized = [
            str(int(id.lstrip('S').lstrip('0'))) if id.lstrip('S').isdigit() else id.lstrip('S')
            for id in substrate_ids
        ]
        
        # Ensure all normalized input substrate IDs are in the normalized .npz IDs
        missing_ids = [substrate_ids[i] for i, id in enumerate(input_substrate_ids_normalized)
                       if id not in npz_substrate_ids_normalized]
        if missing_ids:
            print(f"Warning: Some substrate IDs {missing_ids} not found in similarity matrix.")
            return None, None
        
        # Filter matrix to include only substrates in the current cluster
        indices = [npz_substrate_ids_normalized.index(id) for id in input_substrate_ids_normalized]
        filtered_matrix = matrix[np.ix_(indices, indices)]
        
        # Use mapped (S-prefixed) IDs for visualization
        ordered_substrate_ids = [mapped_substrate_ids[i] for i in indices]
        
        return filtered_matrix, ordered_substrate_ids
    except Exception as e:
        print(f"Error loading similarity matrix: {str(e)}")
        return None, None

def create_heatmap(enzyme_id, substrate_ids, similarity_matrix, substrate_status, activity_filter=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    heatmap = sns.heatmap(
        similarity_matrix,
        annot=True,
        cmap='Blues',
        vmin=0, vmax=1,
        xticklabels=substrate_ids,
        yticklabels=substrate_ids,
        cbar_kws={'label': 'Tanimoto Similarity'},
        ax=ax,
        annot_kws={'size': 10},
    )
    
    n = len(substrate_ids)
    x_labels = []
    y_labels = []
    
    for i in range(n):
        sub_id = substrate_ids[i].lstrip('S')  # Remove leading 'S'
        # Pad the numeric part to 3 digits, preserving leading zeros
        normalized_sub_id = sub_id.zfill(3) if sub_id.isdigit() else sub_id
        
        # Debug: Print ID and status
        status = substrate_status.get(normalized_sub_id, {'Test': False, 'Train': False})

        is_test = status['Test']
        label_color = 'red' if is_test else 'black'
        label_weight = 'bold' if is_test else 'normal'
        
        x_label = ax.get_xticklabels()[i]
        x_label.set_color(label_color)
        x_label.set_fontweight(label_weight)
        x_label.set_fontsize(12)
        x_labels.append(x_label)
        
        y_label = ax.get_yticklabels()[i]
        y_label.set_color(label_color)
        y_label.set_fontweight(label_weight)
        y_label.set_fontsize(12)
        y_labels.append(y_label)
    
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    
    plt.title(f'Pairwise Tanimoto Similarity Heatmap for Enzyme E{enzyme_id}\n({len(substrate_ids)} substrates)', fontsize=14)
    plt.xlabel('Substrate ID', fontsize=12)
    plt.ylabel('Substrate ID', fontsize=12)
    
    cbar = heatmap.collections[0].colorbar
    cbar.ax.set_ylabel('Tanimoto Similarity', fontsize=12)
    
    plt.tight_layout()
    
    output_file = f'enzyme_E{enzyme_id}_similarity_heatmap_{activity_filter or "all"}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap: {output_file}")
    plt.close()

def create_substrate_cluster_network(csv_file_path, activity_filter=None, similarity_file=None):
    """
    Create a network visualization with separate subplots for each enzyme cluster
    where the cluster contains substrates with both Test and Train entries.
    Substrates from Test have red borders, Train substrates have black borders.
    Nodes are colored by reaction type with a legend.
    If activity_filter is provided, only enzyme-substrate pairs with the specified
    reaction type are included.
    If similarity_file is provided, generate heatmaps for large clusters.
    """
    try:
        df = pd.read_csv(csv_file_path, sep=';', dtype={'Enzyme': str, 'Substrate': str})
        df.columns = df.columns.str.strip()

        if 'Reaction Type' not in df.columns:
            print("Warning: 'Reaction Type' column not found, using 'Unknown' for all reactions.")
            df['Reaction Type'] = 'Unknown'

        # Apply activity filter if specified
        if activity_filter:
            df = df[df['Reaction Type'].str.strip() == activity_filter]
            if df.empty:
                print(f"No entries found with Reaction Type '{activity_filter}'.")
                return None, None, None, None, None

        enzyme_substrates = defaultdict(set)
        substrate_status = defaultdict(lambda: {'Test': False, 'Train': False})
        substrate_reaction = {}

        for _, row in df.iterrows():
            enzyme_id = str(row['Enzyme']).strip()
            substrate_id = str(row['Substrate']).strip()
            test_status = str(row.get('Test', '')).strip().upper() == 'T'
            train_status = str(row.get('Train', '')).strip().upper() == 'T'
            reaction_type = str(row.get('Reaction Type', 'Unknown')).strip()

            if enzyme_id and substrate_id and enzyme_id != 'nan' and substrate_id != 'nan':
                enzyme_substrates[enzyme_id].add(substrate_id)
                if test_status:
                    substrate_status[substrate_id]['Test'] = True
                if train_status:
                    substrate_status[substrate_id]['Train'] = True
                if substrate_id not in substrate_reaction:
                    substrate_reaction[substrate_id] = reaction_type

        valid_enzymes = []
        for enzyme_id, substrates in enzyme_substrates.items():
            has_test = any(substrate_status[s]['Test'] for s in substrates)
            has_train = any(substrate_status[s]['Train'] for s in substrates)
            if has_test and has_train:
                valid_enzymes.append(enzyme_id)

        enzyme_substrates = {
            enzyme_id: substrates for enzyme_id, substrates in enzyme_substrates.items()
            if enzyme_id in valid_enzymes
        }

        clusters_by_size = sorted(enzyme_substrates.items(), key=lambda x: len(x[1]), reverse=True)
        if not clusters_by_size:
            print("No enzyme clusters found with substrates having both Test and Train entries.")
            return None, None, None, None, None

        reaction_types = sorted(set(substrate_reaction.values()))
        colormap = plt.cm.get_cmap('tab10', max(len(reaction_types), 1))
        reaction_colors = {rt: colormap(i) for i, rt in enumerate(reaction_types)}

        n_clusters = len(clusters_by_size)
        cols = math.ceil(math.sqrt(n_clusters))
        rows = math.ceil(n_clusters / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))

        if n_clusters == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if n_clusters == 1 else axes
        else:
            axes = axes.flatten()

        for idx, (enzyme_id, substrate_ids) in enumerate(clusters_by_size):
            ax = axes[idx]

            G = nx.Graph()
            substrate_labels = [f"S{sub_id}" for sub_id in substrate_ids]
            G.add_nodes_from(substrate_labels)

            for i in range(len(substrate_labels)):
                for j in range(i + 1, len(substrate_labels)):
                    G.add_edge(substrate_labels[i], substrate_labels[j])

            if len(substrate_labels) == 1:
                pos = {substrate_labels[0]: (0, 0)}
            elif len(substrate_labels) <= 8:
                angles = np.linspace(0, 2*np.pi, len(substrate_labels), endpoint=False)
                radius = max(1, len(substrate_labels) * 0.3)
                pos = {label: (radius*np.cos(angles[i]), radius*np.sin(angles[i]))
                       for i, label in enumerate(substrate_labels)}
            else:
                pos = nx.spring_layout(G, k=2, iterations=50)

            if G.number_of_edges() > 0:
                nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.6, width=2)

            for sub_label in substrate_labels:
                sub_id = sub_label[1:]
                node_color = reaction_colors.get(substrate_reaction.get(sub_id, 'Unknown'), 'lightblue')
                edge_color = 'red' if substrate_status[sub_id]['Test'] else 'black'
                nx.draw_networkx_nodes(
                    G, pos, nodelist=[sub_label], ax=ax,
                    node_color=[node_color],  # wrap in list
                    node_size=800, alpha=0.8,
                    edgecolors=edge_color, linewidths=2)

            nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')

            ax.set_title(f'Enzyme E{enzyme_id}\n({len(substrate_ids)} substrates)',
                         fontsize=12, fontweight='bold', pad=10)
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth=1
                spine.set_edgecolor='lightgray'

        for idx in range(n_clusters, len(axes)):
            axes[idx].axis('off')
            axes[idx].set_visible=False

        # One legend for the whole figure
        legend_patches = [mpatches.Patch(color=color, label=rt)
                          for rt, color in reaction_colors.items()]
        legend_patches.append(mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                                           markersize=10, markeredgewidth=2, label='Test'))
        legend_patches.append(mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                                           markersize=10, markeredgewidth=2, label='Train'))

        fig.legend(handles=legend_patches, loc='lower center',
                   bbox_to_anchor=(0.5, -0.02),
                   ncol=len(legend_patches), fontsize=12)

        plt.tight_layout()
        output_file = f'enzyme_substrate_pair_network_{activity_filter or "all"}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()

        print("Data Leakage Statistics:")
        print("=" * 40)

        print(f"\nEnzyme Clusters with Test and Train Substrates ({len(clusters_by_size)}):")
        print("=" * 40)
        for enzyme_id, substrate_ids in clusters_by_size:
            print(f"Enzyme E{enzyme_id} ({len(substrate_ids)} substrates)")

        return enzyme_substrates, clusters_by_size, substrate_status, substrate_reaction, similarity_file

    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        return None, None, None, None, None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None, None, None, None, None

def create_single_enzyme_plot(enzyme_id, substrate_ids, substrate_status, substrate_reaction, save_individual=False, activity_filter=None, similarity_file=None):
    """
    Create a single plot for one enzyme cluster with Test substrates having red borders,
    Train substrates having black borders, and nodes colored by reaction type with a legend.
    If similarity_file is provided, generate a heatmap for the pairwise similarity matrix.
    """
    G = nx.Graph()
    substrate_labels = [f"S{sub_id}" for sub_id in substrate_ids]
    G.add_nodes_from(substrate_labels)

    for i in range(len(substrate_labels)):
        for j in range(i + 1, len(substrate_labels)):
            G.add_edge(substrate_labels[i], substrate_labels[j])

    fig, ax = plt.subplots(figsize=(8, 6))

    if len(substrate_labels) == 1:
        pos = {substrate_labels[0]: (0, 0)}
    elif len(substrate_labels) <= 8:
        angles = np.linspace(0, 2*np.pi, len(substrate_labels), endpoint=False)
        radius = max(1, len(substrate_labels) * 0.4)
        pos = {label: (radius*np.cos(angles[i]), radius*np.sin(angles[i]))
               for i, label in enumerate(substrate_labels)}
    else:
        pos = nx.spring_layout(G, k=3, iterations=100)

    reaction_types = sorted(set(substrate_reaction.values()))
    colormap = plt.cm.get_cmap('tab10', max(len(reaction_types), 1))
    reaction_colors = {rt: colormap(i) for i, rt in enumerate(reaction_types)}

    if G.number_of_edges() > 0:
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.6, width=2)

    for sub_label in substrate_labels:
        sub_id = sub_label[1:]
        node_color = reaction_colors.get(substrate_reaction.get(sub_id, 'Unknown'), 'lightblue')
        edge_color = 'red' if substrate_status[sub_id]['Test'] else 'black'
        nx.draw_networkx_nodes(
            G, pos, nodelist=[sub_label], ax=ax,
            node_color=[node_color],  # wrap in list
            node_size=1200, alpha=0.8,
            edgecolors=edge_color, linewidths=2)

    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_weight='bold')

    ax.set_title(f'Enzyme E{enzyme_id} Substrate Network\n({len(substrate_ids)} substrates)',
                 fontsize=14, fontweight='bold')
    ax.axis('off')

    legend_patches = [mpatches.Patch(color=color, label=rt)
                      for rt, color in reaction_colors.items()]
    legend_patches.append(mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                                       markersize=10, markeredgewidth=2, label='Test'))
    legend_patches.append(mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                                       markersize=10, markeredgewidth=2, label='Train'))

    fig.legend(handles=legend_patches, loc='lower center',
               bbox_to_anchor=(0.5, -0.02),
               ncol=len(legend_patches), fontsize=8)

    plt.tight_layout()

    if save_individual:
        output_file = f'enzyme_E{enzyme_id}_network_{activity_filter or "all"}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved individual plot: {output_file}")

    plt.show()

    # Generate heatmap if similarity file is provided and cluster is large
    if similarity_file and len(substrate_ids) >= 5:
        similarity_matrix, ordered_substrate_ids = load_similarity_matrix(similarity_file, substrate_labels)
        if similarity_matrix is not None:
            create_heatmap(enzyme_id, ordered_substrate_ids, similarity_matrix, substrate_status, activity_filter)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create substrate cluster network visualization.")
    parser.add_argument('--activity', type=str, help='Filter by reaction type (e.g., Chlorination)')
    parser.add_argument('--pairwise_similarity', type=str, help='Path to pairwise similarity matrix (.npz file)')
    args = parser.parse_args()

    csv_file_path = 'SI_ReactionTable_10012024.csv'

    print("Creating substrate cluster network visualization...")
    result = create_substrate_cluster_network(csv_file_path, activity_filter=args.activity, similarity_file=args.pairwise_similarity)

    if result[0] is None:
        print("No valid clusters found, exiting.")
    else:
        enzyme_data, clusters_by_size, substrate_status, substrate_reaction, similarity_file = result
        output_file = f"substrate_cluster_subplots_test_train_{args.activity or 'all'}.png"
        print(f"\nNetwork visualization saved as '{output_file}'")

        large_clusters = [item for item in clusters_by_size if len(item[1]) >= 5]
        if large_clusters:
            print(f"\nFound {len(large_clusters)} enzyme(s) with 5+ substrates.")
            create_individual = input("Create individual detailed plots for large clusters? (y/n): ").lower().strip()
            if create_individual == 'y':
                for enzyme_id, substrate_ids in large_clusters:
                    create_single_enzyme_plot(enzyme_id, substrate_ids, substrate_status,
                                             substrate_reaction, save_individual=True,
                                             activity_filter=args.activity,
                                             similarity_file=similarity_file)