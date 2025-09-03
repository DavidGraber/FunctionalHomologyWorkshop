# Project 2: Substrate-informed Predictive Model for Biocatalysis

## Overview

This project focuses on similarity detection within the CATNIP train and test dataset (https://chemrxiv.org/engage/chemrxiv/article-details/670c192f51558a15eff5c275). You will perform the following steps:
- **Clustering:** You will cluster the enzyme-substrate pairs to find out which enzyme accept which substrates.
- **Visualization:** These networks can be visualized and the different catalyzed reactions (). Which test datapoints would you remove? 
- **Tanimoto similarity calculation:** To identify potential train-test dataleakage, you will calculate a pairwise tanimoto similarity matrix that shows chemical similarity of all vs. all substrates.
- **Heatmap:** To visualize potential train-test dataleakage, you will plot heatmaps of ligand similarity for different reactions.  
- **Dataset filtering:** OPTIONAL
  
## Key Concepts

### CATNIP
CATNIP is an open-access web interface that uses machine learning to predict which enzymes are most compatible with a given small-molecule substrate, streamlining the design of biocatalytic synthesis. It is built on the BioCatSet1 dataset, which was generated from high-throughput experiments testing the reactivity of α-ketoglutarate-dependent non-heme iron enzymes with over 100 substrates, capturing enzyme–substrate compatibility. By connecting chemical space to protein sequence space, CATNIP aims  to efficiently rank and select enzymes for new synthetic transformations.

### Similarity Metrics
The project utilizes Morgan fingerprints and tanimoto similarity to quantify substrate similarity:

1. **Morgan fingerprints**: Morgan fingerprints are circular molecular fingerprints representing atom-centered substructures. 
2. **Tanimoto similarity**: Tanimoto similarity measures the overlap between two fingerprint sets as a ratio of shared to total features. (0-1, higher is more similar)

## Getting Started

### 1. Navigate to Project Directory
```bash
cd project2
```

### 2. Download Data

```bash
wget https://g-eac64e.765b9d.09d9.data.globus.org/project2_data.tar.gz
```

### 3. Extract Data
```bash
tar --strip-components=1 -xvzf project2_data.tar.gz -C .
```

This should provide you with the following files:
- `substrate_smiles.csv`: Training and test data used in ProtStab paper
- `SI_ReactionTable_10012024.csv`: Protein sequence fasta file for clustering
- `tanimoto_similarity.npz`: Pre-computed sequence identity matrix (all vs. all) as fallback
- 
## Environment Requirements

### Required Libraries
The project requires the following Python libraries:

- **numpy**: For numerical computations and array operations
- **matplotlib**: For creating plots and visualizations
- **networkx**: For creating and analyzing network graphs
- **pandas**: For data manipulation
- **scipy**: For finding connected components in graph.
- **mmseqs2**: For fast sequence identity calculation & clustering

### Installation via Conda:
You can install the required packages using conda:

```bash
conda create -n project1 python=3.8
conda activate project1
conda install numpy matplotlib networkx pandas scipy
conda install -c conda-forge -c bioconda mmseqs2
```

### System Requirements
- **Python**: Version 3.7 or higher
- **Memory**: At least 8GB RAM recommended for large similarity matrices
- **Storage**: ~2GB free space for data files and outputs

## Step 1: Clustering

Executing the command below will use mmseqs2's functionality easy-linlust to cluster the sequences based on there sequence identity.

This example utilizes the parameters:

- **Sequence identity threshold:** --min-seq-id 0.6 (60% sequence identity threshold per cluster)
- **Sequence coverage threshold:** -c 0.6 (60% sequence coverage threshold per cluster - at least 60% of the sequence length should be shared)

Please do not change --cov-mode 0 


```bash
mkdir -p result_60_60 && mmseqs easy-linclust protstab_sequences.fasta result_60_60/output result_60_60 --min-seq-id 0.6 -c 0.6 --cov-mode 0
```
## Step 2: Visualization & Analysis
The resulting tsv file clusteres the input sequences into sequence identity clusters. This cluster information can be added to the ProtStab dataset.
With this information, we can investigate train and test set for data leakage, for example by visualizing as network. Please run the following command:

```bash
 python analyze_dataleakage.py result_60_60/output_cluster.tsv protstab_dataset.csv identity_matrix.npz result_60_60
```
### How to define data leakage?
Please think about the following: Which datapoints could be dentrimental and which beneficial for model performance? (hint: compare the Tm values within the clusters). Which would you remove from train/test set?

## Step 3: Dataset curation
In the last step, the ProtStab dataset will be curated to remove similar datapoints from the test dataset. Additionally, similar datapoints are also from the training dataset as those could lead to an overestimation of the performance determined via cross-validation.

For this purpose, we investigate all datapoints (test & train) from the same similarity cluster (e.g. those sharing 60% sequence identity). Only datapoints with less than X °C (e.g. 5 °C) temperature difference are removed as the others might contain interesting information about differences in thermostability.

The dataset can be curated with the following command. Feel free to utilize different clustering and temperature thresholts.

```bash
 python filter_dataleakage.py result_60_60/output_cluster.tsv protstab_dataset.csv 5 result_60_60
```

## Step 4: Prepare presentation
Please collect your results from the terminal outputs and figures. It will be especially interesting how much of the dataset were filtered depending of different thresholds. You will present them to the other groups in the next session. Please create a pull request to upload your results to the GitHub repository. We aim to publish the results including GitHub repository on a preprint server.










