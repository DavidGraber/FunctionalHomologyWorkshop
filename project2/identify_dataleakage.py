import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse
import warnings
from matplotlib import MatplotlibDeprecationWarning
from rdkit import Chem
from rdkit.Chem import Draw
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
import tempfile

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

def load_similarity_matrix(npz_file, substrate_ids):
    """
    Load pairwise similarity matrix and map substrate IDs, dropping 'S' prefix.
    """
    try:
        data = np.load(npz_file, allow_pickle=True)
        if 'similarity_matrix' not in data.files or 'substrate_ids' not in data.files:
            missing_keys = [key for key in ['similarity_matrix', 'substrate_ids'] if key not in data.files]
            raise KeyError(f"Missing required keys in .npz file: {missing_keys}")
        matrix = data['similarity_matrix']
        npz_substrate_ids = data['substrate_ids']
        
        npz_substrate_ids = [str(id) for id in npz_substrate_ids]
        npz_substrate_ids_normalized = []
        for id in npz_substrate_ids:
            clean_id = id.lstrip('S')
            if clean_id.isdigit():
                normalized_id = str(int(clean_id.lstrip('0')))
            else:
                normalized_id = clean_id
            npz_substrate_ids_normalized.append(normalized_id)
        
        input_substrate_ids_normalized = []
        for id in substrate_ids:
            clean_id = id.lstrip('S')
            if clean_id.isdigit():
                normalized_id = str(int(clean_id.lstrip('0')))
            else:
                normalized_id = clean_id
            input_substrate_ids_normalized.append(normalized_id)
        
        missing_ids = [substrate_ids[i] for i, id in enumerate(input_substrate_ids_normalized)
                       if id not in npz_substrate_ids_normalized]
        if missing_ids:
            print(f"Error: Substrate IDs {missing_ids} not found in similarity matrix.")
            raise ValueError(f"Substrate ID mismatch: {missing_ids}")
        
        indices = [npz_substrate_ids_normalized.index(id) for id in input_substrate_ids_normalized]
        filtered_matrix = matrix[np.ix_(indices, indices)]
        ordered_substrate_ids = [id for id in input_substrate_ids_normalized if id in npz_substrate_ids_normalized]
        
        return filtered_matrix, ordered_substrate_ids
    except Exception as e:
        print(f"Error loading similarity matrix: {str(e)}")
        return None, None

def extract_cluster_data(csv_file_path, similarity_file, tanimoto_cutoffs=[0.7, 0.8, 0.9], activity_filter=None):
    """
    Extract enzyme clusters, substrate status, reaction types, similarity data, and top 5 test-train pairs.
    Return counts of train/test substrates per reaction type, top 5 pairs, and other data.
    Skip clusters with missing substrates in the similarity matrix.
    """
    try:
        df = pd.read_csv(csv_file_path, sep=';', dtype={'Enzyme': str, 'Substrate': str})
        df.columns = df.columns.str.strip()

        # Load SMILES from substrate_smiles.csv
        smiles_map = {}
        try:
            smiles_df = pd.read_csv('substrate_smiles.csv', sep="\t", dtype={'Substrate ID': str})
            smiles_df.columns = smiles_df.columns.str.strip()
            for _, row in smiles_df.iterrows():
                sub_id = str(row['Substrate ID']).lstrip('S')
                if sub_id.isdigit():
                    sub_id = str(int(sub_id.lstrip('0')))
                smiles_map[sub_id] = row['SMILES']
        except FileNotFoundError:
            print("Warning: 'substrate_smiles.csv' not found, cannot retrieve SMILES codes.")
        except Exception as e:
            print(f"Warning: Error loading SMILES data: {str(e)}")

        if 'Reaction Type' not in df.columns:
            print("Warning: 'Reaction Type' column not found, using 'Unknown'.")
            df['Reaction Type'] = 'Unknown'

        if activity_filter:
            df = df[df['Reaction Type'].str.strip() == activity_filter]
            if df.empty:
                print(f"No entries found with Reaction Type '{activity_filter}'.")
                return None, None, None, None, None

        enzyme_substrates = defaultdict(set)
        substrate_status = defaultdict(lambda: {'Test': False, 'Train': False})
        substrate_reaction = {}
        top_pairs = []

        for _, row in df.iterrows():
            enzyme_id = str(row['Enzyme']).strip()
            substrate_id = str(row['Substrate']).strip()
            test_status = str(row.get('Test', '')).strip().upper() == 'T'
            train_status = str(row.get('Train', '')).strip().upper() == 'T'
            reaction_type = str(row.get('Reaction Type', 'Unknown')).strip()

            if enzyme_id and substrate_id and enzyme_id != 'nan' and substrate_id != 'nan':
                clean_substrate_id = substrate_id.lstrip('S')
                if clean_substrate_id.isdigit():
                    clean_substrate_id = str(int(clean_substrate_id.lstrip('0')))
                enzyme_substrates[enzyme_id].add(clean_substrate_id)
                if test_status:
                    substrate_status[clean_substrate_id]['Test'] = True
                if train_status:
                    substrate_status[clean_substrate_id]['Train'] = True
                substrate_reaction[clean_substrate_id] = reaction_type

        # Filter to only keep enzymes with both test and train substrates
        valid_enzymes = []
        for enzyme_id, substrates in enzyme_substrates.items():
            has_test = any(substrate_status[s]['Test'] for s in substrates)
            has_train = any(substrate_status[s]['Train'] for s in substrates)
            if has_test and has_train:
                valid_enzymes.append(enzyme_id)

        enzyme_substrates = {eid: subs for eid, subs in enzyme_substrates.items() if eid in valid_enzymes}
        if not enzyme_substrates:
            print("No enzyme clusters with both Test and Train substrates.")
            return None, None, None, None, None

        # Initialize counts dictionary
        counts_by_cutoff = {cutoff: defaultdict(lambda: {'Train': 0, 'Test': 0}) for cutoff in tanimoto_cutoffs}
        
        # Process each enzyme cluster
        skipped_enzymes = []
        all_pairs = []  # Store all test-train pairs for later sorting
        
        for enzyme_id, substrate_ids in enzyme_substrates.items():
            substrate_labels = list(substrate_ids)
            similarity_matrix, ordered_substrate_ids = load_similarity_matrix(similarity_file, substrate_labels)
            if similarity_matrix is None:
                print(f"Skipping enzyme E{enzyme_id} due to missing substrates in similarity matrix.")
                skipped_enzymes.append(enzyme_id)
                continue

            # Count pairs that meet similarity cutoffs
            for cutoff in tanimoto_cutoffs:
                for i in range(len(ordered_substrate_ids)):
                    for j in range(i + 1, len(ordered_substrate_ids)):
                        if similarity_matrix[i, j] >= cutoff:
                            sub_id1 = ordered_substrate_ids[i]
                            sub_id2 = ordered_substrate_ids[j]
                            reaction_type1 = substrate_reaction.get(sub_id1, 'Unknown')
                            reaction_type2 = substrate_reaction.get(sub_id2, 'Unknown')
                            
                            # Only count if same reaction type
                            if reaction_type1 == reaction_type2:
                                # Count each substrate once per pair
                                if substrate_status[sub_id1]['Train']:
                                    counts_by_cutoff[cutoff][reaction_type1]['Train'] += 1
                                elif substrate_status[sub_id1]['Test']:
                                    counts_by_cutoff[cutoff][reaction_type1]['Test'] += 1
                                    
                                if substrate_status[sub_id2]['Train']:
                                    counts_by_cutoff[cutoff][reaction_type1]['Train'] += 1
                                elif substrate_status[sub_id2]['Test']:
                                    counts_by_cutoff[cutoff][reaction_type1]['Test'] += 1

            # Collect test-train pairs for similarity table
            for i in range(len(ordered_substrate_ids)):
                for j in range(i + 1, len(ordered_substrate_ids)):
                    sub_id1 = ordered_substrate_ids[i]
                    sub_id2 = ordered_substrate_ids[j]
                    reaction_type1 = substrate_reaction.get(sub_id1, 'Unknown')
                    reaction_type2 = substrate_reaction.get(sub_id2, 'Unknown')
                    similarity = similarity_matrix[i, j]
                    
                    # Only consider pairs with same reaction type
                    if reaction_type1 == reaction_type2:
                        # Check if one is test and one is train
                        if (substrate_status[sub_id1]['Test'] and substrate_status[sub_id2]['Train']):
                            test_id = sub_id1
                            train_id = sub_id2
                            all_pairs.append((test_id, train_id, similarity, reaction_type1))
                        elif (substrate_status[sub_id1]['Train'] and substrate_status[sub_id2]['Test']):
                            test_id = sub_id2
                            train_id = sub_id1
                            all_pairs.append((test_id, train_id, similarity, reaction_type1))

        if skipped_enzymes:
            print(f"Warning: Skipped {len(skipped_enzymes)} enzyme clusters due to missing substrates: {skipped_enzymes}")
        
        if not any(any(counts_by_cutoff[cutoff][rt].values()) for cutoff in tanimoto_cutoffs for rt in counts_by_cutoff[cutoff]):
            print("No valid substrate pairs found for any cutoff after processing.")
            return None, None, None, None, None

        # Get top 5 pairs and add SMILES
        top_pairs = sorted(all_pairs, key=lambda x: x[2], reverse=True)[:5]
        top_pairs_with_smiles = []
        for test_id, train_id, similarity, reaction_type in top_pairs:
            test_smiles = smiles_map.get(test_id, 'Unknown')
            train_smiles = smiles_map.get(train_id, 'Unknown')
            top_pairs_with_smiles.append((test_id, train_id, test_smiles, train_smiles, similarity, reaction_type))

        # Get all unique reaction types
        reaction_types = sorted(set(substrate_reaction.values()))
        return counts_by_cutoff, reaction_types, enzyme_substrates, substrate_status, top_pairs_with_smiles

    except FileNotFoundError as e:
        print(f"Error: File not found: {str(e)}")
        return None, None, None, None, None
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return None, None, None, None, None

def create_barplot(counts_by_cutoff, reaction_types, tanimoto_cutoffs=[0.7, 0.8, 0.9], activity_filter=None):
    """
    Create a bar plot showing the number of train and test substrates per reaction type
    for each Tanimoto similarity cutoff.
    """
    if not counts_by_cutoff or not reaction_types:
        print("No data available for bar plot.")
        return

    fig, axes = plt.subplots(1, len(tanimoto_cutoffs), figsize=(6*len(tanimoto_cutoffs), 8), sharey=True)
    if len(tanimoto_cutoffs) == 1:
        axes = [axes]
    
    max_count = 0
    for cutoff in tanimoto_cutoffs:
        for rt in reaction_types:
            if rt in counts_by_cutoff[cutoff]:
                train_count = counts_by_cutoff[cutoff][rt]['Train']
                test_count = counts_by_cutoff[cutoff][rt]['Test']
                max_count = max(max_count, train_count, test_count)
    
    if max_count == 0:
        max_count = 1
    
    max_tick = int(np.ceil(max_count / 2.0)) * 2
    y_ticks = np.arange(0, max_tick + 2, max(1, max_tick//10))
    
    for idx, cutoff in enumerate(tanimoto_cutoffs):
        ax = axes[idx]
        train_counts = []
        test_counts = []
        
        for rt in reaction_types:
            if rt in counts_by_cutoff[cutoff]:
                train_counts.append(counts_by_cutoff[cutoff][rt]['Train'])
                test_counts.append(counts_by_cutoff[cutoff][rt]['Test'])
            else:
                train_counts.append(0)
                test_counts.append(0)
        
        x = np.arange(len(reaction_types))
        width = 0.35
        
        ax.bar(x - width/2, train_counts, width, label='Train', color='blue', alpha=0.7)
        ax.bar(x + width/2, test_counts, width, label='Test', color='red', alpha=0.7)
        
        ax.set_title(f'Tanimoto Cutoff: {cutoff}', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(reaction_types, rotation=45, ha='right', fontsize=14)
        if idx == 0:
            ax.set_ylabel('Number of Substrate Pairs', fontsize=14)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis='y', labelsize=12)
        ax.legend(fontsize=12)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'Substrate Pair Counts by Reaction Type and Train/Test Status\n(Activity Filter: {activity_filter or "All"})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_file = f'similar_substrates_{activity_filter or "all"}.png'
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved bar plot: {output_file}")
    except Exception as e:
        print(f"Error saving bar plot: {str(e)}")
    
    plt.show()

def create_similarity_table(top_pairs, activity_filter=None):
    """
    Create a table figure showing the top 5 most similar test-train substrate pairs with images of substrate structures.
    """
    if not top_pairs:
        print("No pairs available for similarity table.")
        return

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    
    columns = ['Test Substrate ID', 'Train Substrate ID', 'Test Structure', 'Train Structure', 'Tanimoto Similarity', 'Reaction Type']
    cell_text = []
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        for idx, (test_id, train_id, test_smiles, train_smiles, similarity, reaction_type) in enumerate(top_pairs):
            row = [test_id, train_id, '', '', f"{similarity:.3f}", reaction_type]
            
            # Handle test structure
            if test_smiles and test_smiles != 'Unknown':
                try:
                    mol = Chem.MolFromSmiles(test_smiles)
                    if mol:
                        row[2] = 'Structure'
                    else:
                        row[2] = 'Invalid SMILES'
                except Exception as e:
                    row[2] = f'Error: {str(e)[:20]}...'
                    print(f"Error processing test SMILES for {test_id}: {str(e)}")
            else:
                row[2] = 'No SMILES'
            
            # Handle train structure
            if train_smiles and train_smiles != 'Unknown':
                try:
                    mol = Chem.MolFromSmiles(train_smiles)
                    if mol:
                        row[3] = 'Structure'
                    else:
                        row[3] = 'Invalid SMILES'
                except Exception as e:
                    row[3] = f'Error: {str(e)[:20]}...'
                    print(f"Error processing train SMILES for {train_id}: {str(e)}")
            else:
                row[3] = 'No SMILES'
            
            cell_text.append(row)
        
        # Create table with specific positioning
        table = ax.table(cellText=cell_text, colLabels=columns, cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 3.0)  # Increased height for better image display
        
        # Set column widths
        col_widths = [0.15, 0.15, 0.2, 0.2, 0.15, 0.15]
        for i, width in enumerate(col_widths):
            for j in range(len(cell_text) + 1):  # +1 for header
                table.get_celld()[j, i].set_width(width)
                table.get_celld()[j, i].set_height(0.15)
        
        # Force table to render to get accurate positions
        fig.canvas.draw()
        
        # Add molecular structure images after table is rendered
        for idx, (test_id, train_id, test_smiles, train_smiles, similarity, reaction_type) in enumerate(top_pairs):
            row_idx = idx + 1  # +1 for header row
            
            # Add test structure image
            if test_smiles and test_smiles != 'Unknown':
                try:
                    mol = Chem.MolFromSmiles(test_smiles)
                    if mol:
                        test_img_path = os.path.join(temp_dir, f'test_{test_id}_{idx}.png')
                        Draw.MolToFile(mol, test_img_path, size=(120, 120), kekulize=True)
                        
                        # Get cell position
                        cell = table.get_celld()[row_idx, 2]
                        bbox = cell.get_bbox()
                        
                        # Transform bbox coordinates to figure coordinates
                        bbox_fig = ax.transData.transform([[bbox.x0, bbox.y0], [bbox.x1, bbox.y1]])
                        bbox_fig = fig.transFigure.inverted().transform(bbox_fig)
                        
                        center_x = (bbox_fig[0][0] + bbox_fig[1][0]) / 2
                        center_y = (bbox_fig[0][1] + bbox_fig[1][1]) / 2
                        
                        # Create inset axes for the image
                        img_ax = fig.add_axes([center_x - 0.08, center_y - 0.06, 0.16, 0.12])
                        img_ax.axis('off')
                        
                        img = plt.imread(test_img_path)
                        img_ax.imshow(img)
                        
                except Exception as e:
                    print(f"Error adding test structure image: {str(e)}")
            
            # Add train structure image
            if train_smiles and train_smiles != 'Unknown':
                try:
                    mol = Chem.MolFromSmiles(train_smiles)
                    if mol:
                        train_img_path = os.path.join(temp_dir, f'train_{train_id}_{idx}.png')
                        Draw.MolToFile(mol, train_img_path, size=(120, 120), kekulize=True)
                        
                        # Get cell position
                        cell = table.get_celld()[row_idx, 3]
                        bbox = cell.get_bbox()
                        
                        # Transform bbox coordinates to figure coordinates
                        bbox_fig = ax.transData.transform([[bbox.x0, bbox.y0], [bbox.x1, bbox.y1]])
                        bbox_fig = fig.transFigure.inverted().transform(bbox_fig)
                        
                        center_x = (bbox_fig[0][0] + bbox_fig[1][0]) / 2
                        center_y = (bbox_fig[0][1] + bbox_fig[1][1]) / 2
                        
                        # Create inset axes for the image
                        img_ax = fig.add_axes([center_x - 0.08, center_y - 0.06, 0.16, 0.12])
                        img_ax.axis('off')
                        
                        img = plt.imread(train_img_path)
                        img_ax.imshow(img)
                        
                except Exception as e:
                    print(f"Error adding train structure image: {str(e)}")
        
        plt.suptitle(f'Top 5 Most Similar Test-Train Substrate Pairs\n(Activity Filter: {activity_filter or "All"})', 
                    fontsize=14, y=0.95)
        
        output_file = f'top5_similar_pairs_{activity_filter or "all"}.png'
        try:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved similarity table: {output_file}")
        except Exception as e:
            print(f"Error saving similarity table: {str(e)}")
        
        plt.show()
        
    finally:
        # Clean up temporary files
        try:
            for file in os.listdir(temp_dir):
                if file.endswith('.png'):
                    os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
        except:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create bar plot and table for substrate counts and top similar pairs by Tanimoto similarity.")
    parser.add_argument('--activity', type=str, help='Filter by reaction type (e.g., Chlorination)')
    parser.add_argument('--pairwise_similarity', type=str, required=True, help='Path to pairwise similarity matrix (.npz file)')
    args = parser.parse_args()

    csv_file_path = 'SI_ReactionTable_10012024.csv'
    tanimoto_cutoffs = [0.5, 0.7, 0.9]

    print("Extracting cluster data and computing substrate counts...")
    try:
        counts_by_cutoff, reaction_types, enzyme_substrates, substrate_status, top_pairs = extract_cluster_data(
            csv_file_path, args.pairwise_similarity, tanimoto_cutoffs, args.activity
        )

        if counts_by_cutoff is None:
            print("No valid data found, exiting.")
        else:
            print("Creating bar plot...")
            create_barplot(counts_by_cutoff, reaction_types, tanimoto_cutoffs, args.activity)
            
            if top_pairs:
                print("Creating similarity table...")
                create_similarity_table(top_pairs, args.activity)
            else:
                print("No test-train pairs found for similarity table.")
                
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()