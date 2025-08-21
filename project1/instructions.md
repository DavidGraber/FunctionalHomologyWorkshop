# Project 1: Sequence-based Protein Thermostability Prediction

## Overview

This project focuses on similarity detection within the ProtStab train and test dataset ([https://doi.org/10.3390/ijms231810798](https://link.springer.com/article/10.1186/s12864-019-6138-7)). You will perform the following steps:
- **Clustering:** You will cluster the protein sequences using different sequence identity and sequence coverage thresholds.
- **Visualization:** For each clustering, you can visualize the connectivity of test sequences with training complexes, giving insights into train-test data leakage present in this database. Which test datapoints would you remove? 
- **Dataset curation:** After identifying train-test dataleakage, you will also curate the train dataset to ensure more realistic cross-validation results.
- **Training of XGBoost regressors:** You will train a simple XGBoost regressor with 5-fold cross-validation and evaluate its performance based on the unfiltered test dataset and the filtered train dataset. How do the performance metrics change?

## Key Concepts

### ProtStab
The ProtStab algorithm utilized a dataset of protein thermal stability measurements (https://www.science.org/doi/10.1126/science.aai7825), specifically focused on protein melting temperatures (Tₘ). It contains experimentally determined Tₘ values for a wide range of wild-type proteins under various experimental conditions. This dataset is widely used for training and evaluating computational models that predict protein thermal stability and its determinants. 

### Similarity Metrics
The project uses two different similarity metrics to cluster similar protein sequences:

1. **Sequence identity**: Measures sequence similarity between protein sequences (0-100, higher is more similar)
2. **Sequence coverage**: Measures how much of the two sequences can be aligned (0-1, higher is more similar)

### Prediction Method
The approach uses a **XGBoost** regressor:
- The Regressor is first trained on the full ProtStab training dataset (without hyperparameter tuning) 
- The Regressor is then trained on a filtered ProtStab training dataset (without hyperparameter tuning) 
- The resulting cross-validation performances of both datasets are compared

## Data Requirements

The project requires several data files containing pre-computed similarity matrices and metadata:

- `pairwise_similarity_matrix.npy`: Sequence identity matrix
# ADAPT THIS PART HERE
- `pairwise_similarity_matrix_tanimoto.npy`: Tanimoto similarity matrix  
- `pairwise_similarity_matrix_rmsd.npy`: RMSD similarity matrix
- `pairwise_similarity_complexes.json`: List of complex identifiers
- `PDBbind_data_dict.json`: Binding affinity data for each complex
- `PDBbind_data_split.json`: Train/test split information

## Getting Started

### 1. Navigate to Project Directory
```bash
cd project1
```

### 2. Download Data
# PROVIDE DAVID WITH NEEDED FILES AND ADAPT GLOBUS LINK
```bash
wget https://g-eac64e.765b9d.09d9.data.globus.org/project1_data.tar.gz
```

### 3. Extract Data
```bash
tar -xzvf project1_data.tar.gz
```

This should provide you with the following files:
protstab2_dataset.csv (training and test data used in ProtStab2 paper)
protstab2.fasta (protein sequence file for clustering)

## Environment Requirements

### Required Libraries
The project requires the following Python libraries:

- **numpy**: For numerical computations and array operations
- **matplotlib**: For creating plots and visualizations
- **networkx**: For creating and analyzing network graphs
- **pandas**: For data manipulation (used in create_graph.py)


### Installation via Conda:
You can install the required packages using conda:
```

```bash
conda create -n project3 python=3.8
conda activate project3
conda install numpy matplotlib networkx pandas
conda install -c conda-forge -c bioconda mmseqs2
```

### System Requirements
- **Python**: Version 3.7 or higher
- **Memory**: At least 8GB RAM recommended for large similarity matrices
- **Storage**: ~2GB free space for data files and outputs

## Step 1: Clustering

Executing the command below will use mmseqs2's functionality easy-linlust to cluster the.

This example utilizes the parameters:

- **Sequence identity threshold:** --min-seq-id 0.6 (60% sequence identity threshold per cluster)

Please do not change --cov-mode 0 (60% sequence coverage threshold per cluster - at least 60% of the sequence length should be shared)

```bash
mkdir -p result_60_60 && mmseqs easy-linclust protstab2.fasta result_60_60/output result_60_60 --min-seq-id 0.6 -c 0.6 --cov-mode 0
```

The resulting tsv file clusteres the input sequences into sequence identity clusters. This cluster information can be added to the ProtStab2 dataset.
With this information, we can investigate train and test set for data leakage, for example by visualizing as network. Please run the following command:

```bash
 python analyze_dataleakage.py result_60_60/output_cluster.tsv protstab_dataset.csv result_60_60
```
### Customizing parameters
Please also try a higher threshold and compare the networks and clustering metrics, e.g.:

```bash
gmkdir -p result_80_80 && mmseqs easy-linclust protstab_sequences.fasta result_80_80/output test --min-seq-id 0.8 -c 0.8 --cov-mode 0
```
```bash
 python analyze_dataleakage.py result_80_80/output_cluster.tsv protstab_dataset.csv result_80_80
```


### Customizing parameters
You can adjust the similarity thresholds to control how similar complexes need to be to be clustered together:

```bash
python main.py --sequence_identity_threshold 0.9 --sequence_coverage_threshold 0.9
```

### Output Files



