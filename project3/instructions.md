# Project 3: Protein-Ligand Binding Affinity Prediction

## Overview

This project focuses on predicting protein-ligand binding affinities using similarity-based approaches. The goal is to predict the binding affinity (log Kd/Ki) of protein-ligand complexes in the CASF2016 test set by finding similar complexes in the training data and using their known affinities.

## Key Concepts

### Binding Affinity
- **Binding affinity** measures how strongly a ligand binds to a protein
- Expressed as log Kd/Ki values (lower values indicate stronger binding)
- Critical for drug discovery and understanding protein-ligand interactions

### Similarity Metrics
The project uses multiple similarity measures to find similar protein-ligand complexes:

1. **TM-Score (Template Modeling Score)**: Measures structural similarity between proteins (0-1, higher is more similar)
2. **Tanimoto Coefficient**: Measures molecular fingerprint similarity between ligands (0-1, higher is more similar)
3. **RMSD (Root Mean Square Deviation)**: Measures structural deviation (lower values indicate more similar structures)
4. **Sequence Similarity**: Measures amino acid sequence similarity (currently disabled in the code)

### Prediction Method
The approach uses a **k-nearest neighbors** strategy:
- For each test complex, find the most similar training complexes
- Predict binding affinity as a weighted average of the top-k similar complexes
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

## Running the Code

### Basic Usage
```bash
python main.py
```

This runs the code with default parameters:
- TM-score threshold: 0.8
- Tanimoto threshold: 0.8  
- RMSD threshold: 2.0
- Sequence similarity threshold: 0.8
- Top-k neighbors: 5

### Customizing Parameters
You can adjust the similarity thresholds to control how similar complexes need to be:

```bash
python main.py --TM_threshold 0.7 --Tanimoto_threshold 0.9 --rmsd_threshold 1.5
```

### Parameter Explanation
- **TM_threshold**: Minimum TM-score for considering proteins similar (default: 0.8)
- **Tanimoto_threshold**: Minimum Tanimoto coefficient for considering ligands similar (default: 0.8)
- **rmsd_threshold**: Maximum RMSD for considering structures similar (default: 2.0)
- **sequence_similarity_threshold**: Minimum sequence similarity (default: 0.8)

## Output Files

The script generates several output files:

1. **adjacency_matrix.npy**: Binary matrix indicating which complexes are considered similar
2. **distance_matrix.npy**: Continuous similarity scores between all complex pairs
3. **test_mask.npy**: Boolean array identifying test set complexes
4. **CASF2016_predictions_top5_compl.png**: Scatter plot of predicted vs true binding affinities

## Evaluation Metrics

The model performance is evaluated using:

- **Pearson Correlation (R)**: Measures linear correlation between predicted and true values
- **RMSE (Root Mean Square Error)**: Measures prediction accuracy


The generated plot shows:
- Blue dots: Predicted vs true binding affinities for test complexes
- Red dashed line: Perfect prediction line (y=x)
- Title includes correlation and RMSE values



## Code Structure

### Key Functions

1. **`create_adjacency_matrix()`**: Creates binary similarity matrix based on thresholds
2. **`create_distance_matrix()`**: Creates continuous similarity scores
3. **`compute_lookup_predictions()`**: Implements k-nearest neighbors prediction
4. **`plot_predictions()`**: Visualizes prediction results

### Data Flow
1. Load similarity matrices and metadata
2. Create adjacency and distance matrices
3. Identify test set complexes
4. For each test complex, find top-k similar training complexes
5. Predict affinity using weighted average
6. Evaluate and visualize results

## Troubleshooting

### Common Issues
- **File not found errors**: Ensure all required data files are in the project directory
- **Memory errors**: Large similarity matrices may require significant RAM
- **Poor predictions**: Try adjusting similarity thresholds

### Performance Tips
- Lower thresholds increase the number of similar complexes found
- Higher thresholds ensure more similar complexes but may miss predictions
- The current implementation uses top-5 neighbors; you can modify this in the code

## Next Steps

To extend this project, consider:
- Implementing different similarity metrics
- Testing various k values for k-nearest neighbors
- Adding sequence similarity back into the similarity calculation
- Comparing with machine learning approaches
- Analyzing which similarity metrics contribute most to prediction accuracy
