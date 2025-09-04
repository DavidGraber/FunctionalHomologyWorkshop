# Project 2: Substrate-informed Predictive Model for Biocatalysis

## Overview

This project focuses on similarity detection within the CATNIP train and test dataset (https://chemrxiv.org/engage/chemrxiv/article-details/670c192f51558a15eff5c275). You will perform the following steps:
- **Clustering:** You will cluster the enzyme-substrate pairs to find out which enzyme accepts which substrates.
- **Visualization:** These networks will be visualized higlighting train or test datapoints and the diversity of catalyzed reactions (e.g. Chlorination). 
- **Tanimoto similarity calculation:** To identify potential train-test dataleakage, you will calculate a pairwise tanimoto similarity matrix that shows chemical similarity of all vs. all substrates.
- **Dataleakage detection:** To visualize potential train-test dataleakage, you will plot heatmaps of ligand similarity and count the occurence of similar train-test pairs for different reactions.  
  
## Key Concepts

### CATNIP
CATNIP is an open-access web interface that uses machine learning to predict which enzymes are most compatible with a given small-molecule substrate, streamlining the design of biocatalytic synthesis. It is built on the BioCatSet1 dataset, which was generated from high-throughput experiments testing the reactivity of α-ketoglutarate-dependent non-heme iron enzymes with over 100 substrates, capturing enzyme–substrate compatibility. By connecting chemical space to protein sequence space, CATNIP aims to efficiently rank and select enzymes for new synthetic transformations.

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
- **seaborn**: For creating plots and visualizations
- **networkx**: For creating and analyzing network graphs
- **pandas**: For data manipulation
- **rdkit**: For handling smiles and calculating tanimoto similarity

### Installation via Conda:
You can install the required packages using conda:

```bash
conda create -n project2 python=3.8
conda activate project2
conda install numpy matplotlib networkx pandas seaborn rdkit
```

### System Requirements
- **Python**: Version 3.7 or higher
- **Memory**: At least 8GB RAM recommended for large similarity matrices
- **Storage**: ~2GB free space for data files and outputs

## Step 1: Clustering

Executing the command below will create clusters of substrates that are accepted by the same enzyme.
It will also provide you with the numberof "Enzyme Clusters" that contain substrates from both Test and Train datasets. 

```bash
python identify_clusters.py
```

You can expect quite a number of clusters. To get more detailed insights, clusters with 5+ substrates can optionally be saved by confirming: 

```bash
Create individual detailed plots for large clusters? (y/n): y
```
**Please take a look at the individual cluster image files and find a few examples where substrates from Test and Train datasets in the same cluster also show the same catalyzed reaction (e.g. Chlorination)**.

## Step 2: Visualization
For these examples that might result in dataleakage, generate the same plots with **one reaction only (e.g. Chlorination)** by running the following command:

```bash
python identify_clusters.py --activity Chlorination
```
Please take a look at all the output images generated for different reaction types.

## Step 3: Tanimoto similarity calculation
Clearly, the same enzyme occuring in Test and Train dataset is not a sufficient criterium for calling it "dataleakage". However, what if the same enzyme performs the same reaction on a very similar substrate? Please execute the following command to calculate the so-called tanimoto similarity (based on Morgan Fingerprints) between all substrates (all vs. all). 

```bash
python calculate_similarity.py
```
The output will be a similarity matrix in .npz format. You can provide this matrix to the cluster identification script using the following command and it will additionally generate a heatmap for the individual clusters to give you some insights into the substrate similarity of this cluster

```bash
python identify_clusters.py --activity Chlorination --pairwise_similarity tanimoto_similarity.npz
```
How does it look like for the most common reaction?
```bash
python identify_clusters.py --activity OH --pairwise_similarity tanimoto_similarity.npz
```


## Step 5: Dataleakage detection
Until now, we have only explored the potential for dataleakage in some selected clusters and for some selected enzymatic activities. Let's do this a bit more systematically by counting the number of similar substrates between training and test datasets but also within the training dataset as training set redundancy might have a critical impact on cross-validation performance estimation. 
**The following command will provide three separater barplots for different tanimoto similarity cutoffs. Additionally it will generate a table with the top 5 most similar substrate pairs from train and test dataset.**

```bash
python identify_dataleakage.py --pairwise_similarity tanimoto_similarity.npz
```
Analyze similar_substrates_all.png and top5_similar_pairs_all.png. Based on this analysis, decide which enzyme function specific analysis could be interesting, e.g.:

```bash
python identify_dataleakage.py --pairwise_similarity tanimoto_similarity.npz --activity OH
python identify_dataleakage.py --pairwise_similarity tanimoto_similarity.npz --activity Desat
python identify_dataleakage.py --pairwise_similarity tanimoto_similarity.npz --activity Chlorination
```


## Step 6: Prepare presentation
Please collect your results from the terminal outputs and figures. It will be especially interesting how much of the dataset were filtered depending of different thresholds. You will present them to the other groups in the next session. Please create a pull request to upload your results to the GitHub repository. We aim to publish the results including GitHub repository on a preprint server.
