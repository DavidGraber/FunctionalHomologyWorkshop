import os
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt


def compute_lookup_predictions(distance_matrix, complexes, affinity_data, test_or_not, top_n):

    """
    Compute the predictions for the test dataset using the lookup method.
    The predictions are based on the average label of the top n most similar training complexes.
    The similarity is computed using the pairwise similarity matrices (PSM) for Tanimoto and TM-score.
    """

    # Loop over the test complexes and look for the most similar training complexes
    # ---------------------------------------------------------------------------------
    #(f"Computing predictions for CASF2016 test set with training data lookup")

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

    return true_labels, predicted_labels


def get_predictions(dist, data_split, affinity_data, complexes, top_n = 5, TM_threshold = 1.0, Tanimoto_threshold = 1.0):
    # Load the list of complexes
    with open(complexes, 'r') as f:
        complexes = json.load(f)
    # Load the data split
    test_or_not = np.load(data_split)
    # Import true affinity for each complex
    affinity_data = np.load(affinity_data)

    cutoff_distance = Tanimoto_threshold + TM_threshold
    dist[dist > cutoff_distance] = 0
    # Compute the predictions
    return compute_lookup_predictions(
        dist, complexes, affinity_data, test_or_not, top_n
    )
