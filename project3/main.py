import os
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt

"""
This script creates the adjacency and distance matrices for the PDBbind database, then computes predictions for CASF2016 test set with training data lookup. 
The thresholds for the similarity metrics are set in the parse_args function.

The necessary input data include three distance matrices, for the TM-scores, Tanimoto ligand similarity and RMSD ligand positioning similarity metrics.

- TM-scores: A measure of the similarity of the protein fold (alignment-based).
- Tanimoto: A measure of the ligand similarity (fingerprint-based). 
- RMSD: A measure of the similarity of the ligand positioning in the pocket (pocket-aligned ligand RMSD).
- Sequence similarity: A measure of the similarity of the protein sequence (sequence-based).

The script is run with the following command:
python main.py --TM_threshold 0.8 --Tanimoto_threshold 0.8 --rmsd_threshold 2.0 --sequence_similarity_threshold 0.8

The thresholds can be changed to create different adjacency and distance matrices.
The script saves the adjacency and distance matrices to npy files.
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Create the adjacency and distance matrices for PDBbind, then compute predictions for CASF2016 test set with training data lookup')

    # Change thresholds here
    parser.add_argument('--TM_threshold', type=float, default=0.8, help='TM-score threshold')
    parser.add_argument('--Tanimoto_threshold', type=float, default=0.8, help='Tanimoto threshold')
    parser.add_argument('--rmsd_threshold', type=float, default=2.0, help='RMSD threshold')
    parser.add_argument('--sequence_similarity_threshold', type=float, default=0.8, help='Sequence similarity threshold')
    
    # File paths (No change needed)
    parser.add_argument('--psm_tanimoto', type=str, default='pairwise_similarity_matrix_tanimoto.npy', help='Path to the Tanimoto similarity matrix')
    parser.add_argument('--psm_tm', type=str, default='pairwise_similarity_matrix_tm.npy', help='Path to the TM-score similarity matrix')
    parser.add_argument('--psm_rmsd', type=str, default='pairwise_similarity_matrix_rmsd.npy', help='Path to the RMSD similarity matrix')
    parser.add_argument('--psm_sequence', type=str, default='pairwise_similarity_matrix_sequence.npy', help='Path to the sequence similarity matrix')
    parser.add_argument('--complexes', type=str, default='pairwise_similarity_complexes.json', help='Path to the list of complexes')
    parser.add_argument('--affinity_data', type=str, default='PDBbind_data_dict.json', help='Path to the affinity data')
    parser.add_argument('--data_split', type=str, default='PDBbind_data_split.json', help='Path to the data split')

    return parser.parse_args()


def create_adjacency_matrix(similarity_matrix_tm, 
                            similarity_matrix_tanimoto, 
                            similarity_matrix_rmsd,
                            # similarity_matrix_sequence,
                            TM_threshold, 
                            Tanimoto_threshold, 
                            rmsd_threshold, 
                            sequence_similarity_threshold):


    # CREATE BINARY ADJACENCY MATRIX BASED ON THRESHOLDS FOR TANIMOTO, TM-SCORE AND LABEL DIFFERENCES
    print(f"Creating adjacency matrix with TM-score threshold {TM_threshold}, Tanimoto threshold {Tanimoto_threshold}, RMSD threshold {rmsd_threshold}, and sequence similarity threshold {sequence_similarity_threshold}...")
    adjacency_matrix = (
        (similarity_matrix_tm > TM_threshold)
        & (similarity_matrix_tanimoto > Tanimoto_threshold)
        & (similarity_matrix_rmsd < rmsd_threshold)
        # & (similarity_matrix_sequence > sequence_similarity_threshold)
    )
    
    # Finalize the adjacency matrix
    print(f"Finalizing adjacency matrix...")
    adjacency_matrix = adjacency_matrix.astype(int)
    adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T) # Make sure the adjacency matrix is symmetric
    np.fill_diagonal(adjacency_matrix, 0)
    return adjacency_matrix



def create_distance_matrix(similarity_matrix_tm, 
                            similarity_matrix_tanimoto,
                            similarity_matrix_rmsd,
                            #similarity_matrix_sequence
                            ):

    # CREATE DISTANCE MATRIX BASED ON THRESHOLDS FOR TANIMOTO, TM-SCORE AND LABEL DIFFERENCES
    print(f"Creating distance matrix based on TM-score and Tanimoto similarity matrices...")
    distance_matrix = (similarity_matrix_tm) + (similarity_matrix_tanimoto)

    # Finalize the distance matrix
    print(f"Finalizing distance matrix...")
    distance_matrix = distance_matrix.astype(float)
    distance_matrix = np.maximum(distance_matrix, distance_matrix.T) # Make sure the distance matrix is symmetric
    np.fill_diagonal(distance_matrix, 0)

    return distance_matrix


def plot_predictions(y_true, y_pred, title, label):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, c='blue', label=label)
    axislim = 16
    plt.plot([0, axislim], [0, axislim], color='red', linestyle='--')
    plt.xlabel('True pK Values', fontsize=12)
    plt.ylabel('Predicted pK Values', fontsize=12)
    plt.ylim(0, axislim)
    plt.xlim(0, axislim)
    plt.axhline(0, color='grey', linestyle='--')
    plt.axvline(0, color='grey', linestyle='--')
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.show()



def main():
    args = parse_args()

    # List of complexes in the pairwise similarity matrix
    with open(args.complexes, 'r') as f:
        complexes = json.load(f)

    # TM-SCORE SIMILARITY MATRIX
    if os.path.exists(args.psm_tm):
        similarity_matrix_tm = np.load(args.psm_tm)
    else:
        raise FileNotFoundError(f"TM-score similarity matrix file not found: {args.psm_tm}")

    # TANIMOTO SIMILARITY MATRIX
    if os.path.exists(args.psm_tanimoto):
        similarity_matrix_tanimoto = np.load(args.psm_tanimoto)
    else:
        raise FileNotFoundError(f"Tanimoto similarity matrix file not found: {args.psm_tanimoto}")

    # RMSD SIMILARITY MATRIX
    if os.path.exists(args.psm_rmsd):
        similarity_matrix_rmsd = np.load(args.psm_rmsd)
    else:
        raise FileNotFoundError(f"RMSD similarity matrix file not found: {args.psm_rmsd}")

    # SEQUENCE SIMILARITY MATRIX
    # if os.path.exists(args.psm_sequence):
    #     similarity_matrix_sequence = np.load(args.psm_sequence)
    # else:
    #     raise FileNotFoundError(f"Sequence similarity matrix file not found: {args.psm_sequence}")


    # Create the adjacency and distance matrices
    adjacency_matrix = create_adjacency_matrix(similarity_matrix_tm, 
                                               similarity_matrix_tanimoto, 
                                               similarity_matrix_rmsd, 
                                            #  similarity_matrix_sequence, 
                                               args.TM_threshold, 
                                               args.Tanimoto_threshold, 
                                               args.rmsd_threshold, 
                                               args.sequence_similarity_threshold)

    distance_matrix = create_distance_matrix(similarity_matrix_tm, 
                                               similarity_matrix_tanimoto, 
                                               similarity_matrix_rmsd, 
                                            #  similarity_matrix_sequence
                                               )

    # Save the matrices to npy files
    np.save('adjacency_matrix.npy', adjacency_matrix)
    print(f"Saved adjacency matrix to adjacency_matrix.npy")
    np.save('distance_matrix.npy', distance_matrix)
    print(f"Saved distance matrix to distance_matrix.npy")


if __name__ == "__main__":
    main()