# Project 3: Protein-Ligand Binding Affinity Prediction

## Overview

This project focuses on similarity detection within the PDBbind database. You will perform the following steps:
- **Clustering:** You will cluster the protein-ligand complexes in PDBbind using different combinations of similarity metrics and thresholds.
- **Visualization:** For each clustering, we can visualize the connectivity of test complexes with training complexes, giving insights into train-test data leakage present in this database. 
- **Binding Affinity Prediction:** You will predict protein-ligand binding affinities using a similarity search. The goal is to predict the binding affinity (pK values = log Kd/Ki) of protein-ligand complexes in the CASF2016 test set by finding similar complexes in the training data and using their known affinities. If you choose good similarity metrics and thresholds, you will be able to outperform some published deep-learning-based binding affinity prediction models.

## Key Concepts

### PDBbind Database
This database of affinity-labelled protein-ligand complexes is most frequently used for the training and evaluation of structure-based binding affinity prediction models. It is subdivided into a general set for training and a testing set called Comparative Assessment of Scoring Functions 2016 (CASF2016) dataset.

### Binding Affinity
- **Binding affinity** measures how strongly a ligand binds to a protein
- Expressed as -log Kd/Ki values (higher values indicate stronger binding)
- Critical for drug discovery and understanding protein-ligand interactions

### Similarity Metrics
The project uses multiple similarity measures to find similar protein-ligand complexes:

1. **TM-Score (Template Modeling Score)**: Measures structural similarity between proteins (0-1, higher is more similar)
2. **Tanimoto Coefficient**: Measures molecular fingerprint similarity between ligands (0-1, higher is more similar)
3. **Ligand RMSD (Root Mean Square Deviation)**: Measures positional deviation of ligands (lower values indicate more similar conformations)

### Prediction Method
The approach uses a **k-nearest neighbors** strategy:
- For each test complex, find the most similar training complexes
- Predict binding affinity as a weighted average of the top-k most similar complexes
- Weights are based on similarity scores

## Data Requirements

The project requires several data files containing pre-computed similarity matrices and metadata:

- `pairwise_similarity_matrix_tm.npy`: TM-score similarity matrix
- `pairwise_similarity_matrix_tanimoto.npy`: Tanimoto similarity matrix  
- `pairwise_similarity_matrix_rmsd.npy`: RMSD similarity matrix
- `pairwise_similarity_complexes.json`: List of complex identifiers
- `PDBbind_data_dict.json`: Binding affinity data for each complex
- `PDBbind_data_split.json`: Train/test split information

## Run the analysis

Navigate to the project directory, and install the dependencies.
```bash
cd project3
pip install .
```
Optionally, you can change values in the `config.toml` file.  

Download the data
```
fhw_p3 download
```
Run the parameter sweep.
```
fhw_p3 process
```

Note that it is assumed that the config.toml file is present in the directory of execution. If this is not the case, specify it with `-c path/to/config.toml`.
Based on the parameters specified in the toml, similarity graphs (pkl & plotted) will be written to the output directory. Additionally, the Pearson correlation and RMSE associated to the parameter terms are written under results_summary.tsv as well.
