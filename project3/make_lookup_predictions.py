import os
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Compute predictions for CASF2016 test set with training data lookup')

    parser.add_argument('--matrix', type=str, default='distance_matrix.npy', help='Path to the distance matrix')
    parser.add_argument('--complexes', type=str, default='pairwise_similarity_complexes.json', help='Path to the list of complexes')
    parser.add_argument('--affinity_data', type=str, default='affinities.npy', help='Path to the affinity data')
    parser.add_argument('--data_split', type=str, default='test_train_mask.npy', help='Path to the data split')
    parser.add_argument('--top_n', type=int, default=5, help='Number of top similar complexes to use for prediction')

    return parser.parse_args()


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
    print(f"Computing predictions for CASF2016 test set with training data lookup")

    test_complex_indices = np.where(test_or_not)[0]
    test_complexes = [complexes[i] for i in test_complex_indices]
    

    predicted_labels = {}
    for complex_idx, complex in zip(test_complex_indices, test_complexes):

        # print(f"\nFinding similar training complexes for {complex}")

        distances = distance_matrix[complex_idx, :]
        distances[complex_idx] = -np.inf # Set the metrics of the complex itself to small number
        distances[test_or_not == 1] = -np.inf # Set the metrics of all complexes in the test dataset to small number

        sorted_indices = np.argsort(distances)
        sorted_indices = list(reversed(sorted_indices))

        # Get the top n similar and average their labels
        top_indices = sorted_indices[:top_n]
        names = [complexes[idx] for idx in top_indices]
        affinities = [affinity_data[idx] for idx in top_indices]
        # affinities = np.array([affinity_data[complex]['log_kd_ki'] for complex in names])
        weights = distances[top_indices]
        weighted_average = np.average(affinities, weights=weights)
        predicted_labels[complex] = weighted_average.item()

        # print(f"Most similar complexes: {names}")
        # print(f"Distances: {distances[top_indices]}")
        # print(f"Predicted affinity: {weighted_average}")


    # Compute the evaluation metrics
    true_labels = [affinity_data[idx] for idx in test_complex_indices]
    predicted_labels = np.array([predicted_labels[complex] for complex in test_complexes])
    corr_matrix = np.corrcoef(true_labels, predicted_labels)
    r = corr_matrix[0, 1]
    rmse = np.sqrt(np.mean((np.array(true_labels) - np.array(predicted_labels))**2))

    plot_predictions(true_labels, 
                     predicted_labels, 
                     f"CASF2016 predictions\nWeighted average of labels of top {top_n} similar training complexes\nR = {r:.3f}, RMSE = {rmse:.3f}",
                     f"CASF2016 Predictions")
    
    plt.savefig(f'CASF2016_predictions_top{top_n}', dpi=300)
    print(f"Saved scatterplot to CASF2016_predictions_top{top_n}.png")


def main():
    args = parse_args()

    # Load the list of complexes
    with open(args.complexes, 'r') as f:
        complexes = json.load(f)

    # Load the data split
    test_or_not = np.load(args.data_split)

    # Import true affinity for each complex
    affinity_data = np.load(args.affinity_data)

    # Load the distance matrix
    distance_matrix = np.load(args.matrix)

    compute_lookup_predictions(distance_matrix, complexes, affinity_data, test_or_not, args.top_n)


if __name__ == "__main__":
    main()