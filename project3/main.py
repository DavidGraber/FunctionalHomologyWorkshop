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
The script saves the adjacency and distance matrices to npy files, along with the test_mask array.
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


def compute_lookup_predictions(distance_matrix, complexes, affinity_data, test_or_not, top_n):

    """
    Compute the predictions for the test dataset using the lookup method.
    The predictions are based on the average label of the top n most similar training complexes.
    The similarity is computed using the pairwise similarity matrices (PSM) for Tanimoto and TM-score.
    """

    # Loop over the test complexes and look for the most similar training complexes
    # ---------------------------------------------------------------------------------
    print(f"\n\nComputing predictions for CASF2016 test set with training data lookup")

    
    test_complex_indices = np.where(test_or_not)[0]
    test_complexes = [complexes[i] for i in test_complex_indices]
    true_labels = [affinity_data[complex]['log_kd_ki'] for complex in test_complexes]

    predicted_labels = {}
    for complex_idx, complex in zip(test_complex_indices, test_complexes):

        print(f"\nFinding similar training complexes for {complex}")

        distances = distance_matrix[complex_idx, :]
        distances[complex_idx] = -np.inf # Set the metrics of the complex itself to small number
        distances[test_or_not == 1] = -np.inf # Set the metrics of all complexes in the test dataset to small number

        sorted_indices = np.argsort(distances)
        sorted_indices = list(reversed(sorted_indices))

        # Get the top n similar and average their labels
        top_indices = sorted_indices[:top_n]
        names = [complexes[idx] for idx in top_indices]
        affinities = np.array([affinity_data[complex]['log_kd_ki'] for complex in names])
        weights = distances[top_indices]
        weighted_average = np.average(affinities, weights=weights)
        predicted_labels[complex] = weighted_average.item()

        print(f"Most similar complexes: {names}")
        print(f"Distances: {distances[top_indices]}")
        print(f"Predicted affinity: {weighted_average}")


    # Compute the evaluation metrics
    predicted_labels = np.array([predicted_labels[complex] for complex in test_complexes])
    corr_matrix = np.corrcoef(true_labels, predicted_labels)
    r = corr_matrix[0, 1]
    rmse = np.sqrt(np.mean((np.array(true_labels) - np.array(predicted_labels))**2))

    plot_predictions(true_labels, 
                     predicted_labels, 
                     f"CASF2016 predictions\nWeighted average of labels of top {top_n} similar training complexes\nR = {r:.3f}, RMSE = {rmse:.3f}",
                     f"CASF2016 Predictions")
    
    plt.savefig(f'CASF2016_predictions_top{top_n}_compl', dpi=300)



def main():
    args = parse_args()
    TM_threshold = args.TM_threshold
    Tanimoto_threshold = args.Tanimoto_threshold
    rmsd_threshold = args.rmsd_threshold
    sequence_similarity_threshold = args.sequence_similarity_threshold


    # List of complexes in the pairwise similarity matrix
    with open(args.complexes, 'r') as f:
        complexes = json.load(f)

    # Import affinity dict and get true affinity for each complex
    with open(args.affinity_data, 'r') as f:
        affinity_data = json.load(f)

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
                                            #    similarity_matrix_sequence, 
                                               TM_threshold, 
                                               Tanimoto_threshold, 
                                               rmsd_threshold, 
                                               sequence_similarity_threshold)

    distance_matrix = create_distance_matrix(similarity_matrix_tm, 
                                               similarity_matrix_tanimoto, 
                                               similarity_matrix_rmsd, 
                                            #    similarity_matrix_sequence
                                               )

    # Save the matrices to npy files
    np.save('adjacency_matrix.npy', adjacency_matrix)
    np.save('distance_matrix.npy', distance_matrix)


    # Compute the predictions for the test dataset using the lookup method
    with open(args.data_split, 'r') as f:
        data_split = json.load(f)
    test_or_not = np.array([True if complex in data_split['casf2016'] else False for complex in complexes])
    np.save('test_mask.npy', test_or_not)

    compute_lookup_predictions(distance_matrix, complexes, affinity_data, test_or_not, 5)


if __name__ == "__main__":
    main()