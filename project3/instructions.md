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

## Getting Started

### 1. Navigate to Project Directory
```bash
cd project3
```

### 2. Download Data
```bash
wget https://g-eac64e.765b9d.09d9.data.globus.org/project3_data.tar.gz
```

### 3. Extract Data
```bash
tar -xzvf project3_data.tar.gz
```

## Environment Requirements

### Required Libraries
The project requires the following Python libraries:

- **scipy**: For statistical analysis
- **numpy**: For numerical computations and array operations
- **matplotlib**: For creating plots and visualizations
- **networkx**: For creating and analyzing network graphs
- **pandas**: For data manipulation (used in create_graph.py)


### Installation
You can install the required packages using pip:

```bash
pip install scipy numpy matplotlib networkx pandas
```

### Conda Environment
If you're using conda, you can create a dedicated environment:

```bash
conda create -n project3 python=3.8
conda activate project3
conda install scipy numpy matplotlib networkx pandas
```

### System Requirements
- **Python**: Version 3.7 or higher
- **Memory**: At least 8GB RAM recommended for large similarity matrices
- **Storage**: ~2GB free space for data files and outputs

## Step 1: Clustering

Executing the command below will use the downloaded pairwise similarity matrices and the thresholds given for each metric to compute a binary adjacency matrix and a continuous similarity matrix.
```bash
python main.py
```

This runs the code with default parameters:
- **TM-score threshold:** 0.8
- **Tanimoto threshold:** 0.8  
- **RMSD threshold:** 2.0


### Parameter Explanation:
- **TM_threshold:** Minimum TM-score for considering proteins similar (default: 0.8)
- **Tanimoto_threshold:** Minimum Tanimoto coefficient for considering ligands similar (default: 0.8)
- **rmsd_threshold:** Maximum RMSD for considering ligand positioning similar (default: 2.0)

### Customizing parameters
You can adjust the similarity thresholds to control how similar complexes need to be to be clustered together:

```bash
python main.py --TM_threshold 0.7 --Tanimoto_threshold 0.9 --rmsd_threshold 1.5
```

### Output Files

- **adjacency_matrix.npy:** Binary matrix indicating which complexes are considered similar
- **distance_matrix.npy:** Continuous similarity scores between all complex pairs (used in Step 3)





## Step 2: Visualization

Now we use the **adjacency matrix** from Step 1 to create a network graph visualization to explore the similarities **between the training and testing protein-ligand complexes** in PDBbind based on the similarity thresholds defined during the clustering. This helps us identify potential data leakage between training and test set complexes. Each test complex will appear in a small sub-network with its similar training complexes.

Executing the command below will generate a network graph visualization:
```bash
python create_graph.py --clustering adjacency_matrix.npy --mask test_train_mask.npy --labels affinities.npy --ids pairwise_similarity_complexes.json --output_path similarity_graph.png
```

### Parameter Explanation
- **adjacency_matrix** (npy format, required): Path to the `adjacency_matrix` file, which contains the similarity relationships previously identified in Step 1 (Clustering)
- **mask** (npy format, optional): Path to boolean mask file indicating which nodes should be plotted (along with their neighbors). We will use the `train_test_mask` downloaded in Step 1, which contains 1 for all test complexes and 0 for all training complexes. Like this, only the test complexes with their neighbors will be plotted.
- **labels** (npy file, optional): Path to a file with numerical labels for color coding nodes (affinities in our case). We will use the `affinities` array downloaded in Step 1, which contains affinity values (pKs) for all complexes. In this way, our complexes in the plot will be color-coded according to their affinity. 
- **ids** (json file, optional): Path to a file assigning IDs to the columns/rows of the adjacency matrix. We will use the `pairwise_similarity_complexes` json file downloaded in Step 1, because we want the markers in the plot to be labelled with PDB IDs.

- **output_path** (optional, default: similarity_graph.png): Output path for the graph image 


### Output Files
- **similarity_graph.png:** Network graph visualization, showing test complexes using larger markers with red edges and similar training complexes using smaller markers with white edges. 

### Graph Features
  - Nodes: Protein-ligand complexes
  - Edges: Similarity connections (based on adjacency matrix)
  - Node colors: Based on labels/affinities (if provided)
  - Edge colors: **Red for test complexes nodes, white for others** (if test_train_mask_provided)
  - Node sizes: Larger for nodes representing test complexes (if test_train_mask provided)


### Points to pay attention to: 

1. **Similarity Distribution:** How complexes are connected based on your chosen thresholds
2. **Data Leakage:** Do the test complexes and the training complexes in a cluster usually have similar affinities?
3. **Test Complex Connectivity:** How well test complexes connect to training data
4. **Train-test splits:** How would you avoid train-test data leakage when training on PDBbind and benchmarking on CASF2016?





## Step 3: CASF2016 predictions using Lookup
Using the pairwise distance matrix for PDBbind that we computed in Step 1, we can now make predictions for each complex of the CASF2016 test set by searching the N most similar training complexes and averaging their label. For this, run the command below - If you achieve a **Pearson Correlation higher than 0.507** or an **RMSE lower than 1.592**, then you have already outperformed a published binding affinity prediction model!

```bash
python make_lookup_predictions.py
```
This runs the code with default parameters:
- **top_n:** 5 (Number of top most similar training complexes to take into account)
- **distance_matrix:** distance_matrix.npy 
- **complexes:** pairwise_similarity_complexes.json  
- **affinity_data:** affinities.npy
- **data_split:** train_test_mask.npy

If you get any FileNotFoundErrors, please check all paths and run the script with corrected paths:

```bash
python make_lookup_predictions.py --matrix <path> --complexes <path> --affinity_data <path> --data_split <path>
```

### Customizing Parameters
You can adjust the `top_n` parameter to change the number of similar training complexes that are used to make a prediction for the test complexes:

```bash
python make_lookup_predictions.py --top_n 5
```


### Evaluation Metrics

The model performance is evaluated using:

- **Pearson Correlation (R)**: Measures linear correlation between predicted and true values
- **RMSE (Root Mean Square Error)**: Measures prediction accuracy

The generated plot shows:
- Blue dots: Predicted vs true binding affinities for test complexes
- Red dashed line: Perfect prediction line (y=x)
- Title includes correlation and RMSE values


## Step 4: Adding Dataset Filtering to the Lookup
As our distance matrix is based on the sum of the Tanimoto and the TM scores, we can simulate training dataset filtering based on these scores. The optional parameters `TM_threshold` and `Tanimoto_threshold` can be used to filter out training complexes with high similarity to a test complex. 

**Example:** Tanimoto_threshold = 0.8 and TM_threshold 0.8 removes all training complex with scores > 0.8 from the lookup search (sets the corresponding distance matrix elements to zero). 

Run the command below and observe how the performance of the lookup algorithm develops. Try also other thresholds. How much do you have to lower the thresholds to remove train-test data leakage?

```bash
python make_lookup_predictions.py --top_n 5 --TM_threshold 0.8 --Tanimoto_threshold 0.8
```

## Step 5: Prepare presentation
Please collect your results from the terminal outputs and figures. You will present them to the other groups in the next session. Please push your results to your branch of the repository (see instructions in readme file). We aim to publish the results including GitHub repository on a preprint server.




