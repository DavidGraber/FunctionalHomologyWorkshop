import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Read CSV file
def read_smiles_csv(file_path):
    df = pd.read_csv(file_path, sep="\t")
    return df['Substrate ID'].values, df['SMILES'].values

# Calculate Morgan fingerprints
def get_fingerprint(smiles, substrate_id):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Error: Invalid SMILES for Substrate ID {substrate_id}: {smiles}")
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    except Exception as e:
        print(f"Error processing Substrate ID {substrate_id}: {smiles}. Exception: {str(e)}")
        return None

# Calculate Tanimoto similarity matrix
def calculate_tanimoto_matrix(substrate_ids, smiles_list):
    n = len(smiles_list)
    fingerprints = []
    valid_indices = []
    valid_ids = []
    
    # Generate fingerprints and track valid entries
    for i, (smiles, sub_id) in enumerate(zip(smiles_list, substrate_ids)):
        fp = get_fingerprint(smiles, sub_id)
        if fp is not None:
            fingerprints.append(fp)
            valid_indices.append(i)
            valid_ids.append(sub_id)
    
    # Calculate similarity for valid fingerprints
    m = len(fingerprints)
    similarity_matrix = np.zeros((m, m))
    
    for i in range(m):
        for j in range(i, m):
            similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Symmetric matrix
    
    return similarity_matrix, valid_ids, valid_indices

# Print a subset of the similarity matrix to console
def print_similarity_matrix(similarity_matrix, substrate_ids):
    print("\nTanimoto Similarity Matrix (First and Last Few Rows/Columns):")
    n = len(substrate_ids)
    # Set number of rows/columns to display at start and end
    display_n = 5
    
    # If matrix is small, print it entirely
    if n <= 2 * display_n:
        header = "Substrate ID\t" + "\t".join(str(id) for id in substrate_ids)
        print(header)
        print("-" * len(header))
        for i, substrate_id in enumerate(substrate_ids):
            row = f"{substrate_id}\t" + "\t".join(f"{similarity_matrix[i, j]:.4f}" for j in range(n))
            print(row)
    else:
        # Select indices for first and last display_n entries
        indices = list(range(min(display_n, n))) + list(range(max(n - display_n, 0), n))
        display_ids = [substrate_ids[i] for i in indices]
        
        # Print header with selected substrate IDs
        header = "Substrate ID\t" + "\t".join(str(id) for id in display_ids)
        print(header)
        print("-" * len(header))
        
        # Print selected rows
        for i in indices:
            substrate_id = substrate_ids[i]
            row = f"{substrate_id}\t" + "\t".join(f"{similarity_matrix[i, j]:.4f}" for j in indices)
            print(row)

def main():
    # Input file path (modify as needed)
    file_path = 'substrate_smiles.csv'
    
    # Read substrate IDs and SMILES
    substrate_ids, smiles = read_smiles_csv(file_path)
    
    # Calculate Tanimoto similarity matrix
    similarity_matrix, valid_ids, valid_indices = calculate_tanimoto_matrix(substrate_ids, smiles)
    
    # Save to .npz file
    np.savez('tanimoto_similarity.npz', 
             similarity_matrix=similarity_matrix, 
             substrate_ids=valid_ids,
             original_indices=valid_indices)
    
    print(f"Processed {len(valid_ids)}/{len(substrate_ids)} valid SMILES strings.")
    if len(valid_ids) < len(substrate_ids):
        print("Some SMILES strings were invalid and skipped. Check the console output for details.")
    
    # Print the similarity matrix subset to console
    print_similarity_matrix(similarity_matrix, valid_ids)

if __name__ == '__main__':
    main()