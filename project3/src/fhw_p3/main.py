import os
import json
import numpy as np
import pandas as pd
from importlib import metadata
import click
import requests
import toml
from pathlib import Path
import hashlib
import requests
import subprocess
import sys
from joblib import Parallel, delayed
import itertools
from fhw_p3.create_graph import create_graph
from fhw_p3.make_lookup_predictions import get_predictions

"""
This script creates the adjacency and distance matrices for the PDBbind database, then computes predictions for CASF2016 test set with training data lookup. 
The thresholds for the similarity metrics are set in the parse_args function.

The necessary input data include three distance matrices, for the TM-scores, Tanimoto ligand similarity and RMSD ligand positioning similarity metrics.

- TM-scores: A measure of the similarity of the protein fold (alignment-based).
- Tanimoto: A measure of the ligand similarity (fingerprint-based). 
- RMSD: A measure of the similarity of the ligand positioning in the pocket (pocket-aligned ligand RMSD).

The script is run with the following command:
python main.py --TM_threshold 0.8 --Tanimoto_threshold 0.8 --rmsd_threshold 2.0

The thresholds can be changed to create different adjacency and distance matrices.
The script saves the adjacency and distance matrices to npy files.
"""

@click.group()
@click.help_option("-h", "--help")
@click.version_option(version=metadata.version('fhw_p3'), prog_name='fhw_p3')
def main():
    pass

@main.command(context_settings={"show_default": True})
@click.help_option("-h", "--help")
@click.option('--config_toml', '-c', default='./config.toml',  type=click.Path(exists=True), help='Path to the config.toml file.')
@click.pass_context
def download(ctx, **kwargs) -> None:
    if not kwargs.get('config_toml'):
        print("Please provide a path to the config.toml file using the -c / --config_toml option.")
        sys.exit(1)
    config = toml.load(kwargs['config_toml'])
    url = config['download_url']
    outdir = Path(config['outdir'])
    outdir.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    if (outdir / 'dataset.tar.gz').exists():
        print(f"File {outdir / 'dataset.tar.gz'} already exists. Veryfying checksum..")
        checksum = hashlib.md5()
        with open(outdir / 'dataset.tar.gz', 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                checksum.update(chunk)
        if checksum.hexdigest() == '43f69e6d26bff0bdc4e20896a2542e11':
            print("Checksum matches. File is valid. Skipping download.")
        else:
            print("Checksum does not match. Redownloading file.")
            os.remove(outdir / 'dataset.tar.gz')
    if not (outdir / 'dataset.tar.gz').exists():
        with open(outdir / 'dataset.tar.gz', 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    subprocess.run(f"tar -xzf dataset.tar.gz -C {str(outdir.resolve())}", shell=True, cwd=outdir)

@main.command(context_settings={"show_default": True})
@click.help_option("-h", "--help")
@click.option('--config_toml', '-c', default='./config.toml', type=click.Path(exists=True), help='Path to the config.toml file.')
@click.pass_context
def process(ctx, **kwargs) -> None:
    if not kwargs.get('config_toml'):
        print("Please provide a path to the config.toml file using the -c / --config_toml option.")
        sys.exit(1)
    print("Setting config, checking paths.")
    config = toml.load(kwargs['config_toml'])
    outdir = Path(config['outdir'])
    if not outdir.exists():
        print(f"Output directory {outdir} does not exist. Please run `fhw_p3 download` first.")
        sys.exit(1)
    
    # File paths - create adj & dist matrices
    psm_tanimoto = outdir / "pairwise_similarity_matrix_tanimoto.npy"
    psm_tm = outdir / "pairwise_similarity_matrix_tm.npy"
    psm_rmsd = outdir / "pairwise_similarity_matrix_rmsd.npy"
    complexes = outdir / "pairwise_similarity_complexes.json"
    for _ in [psm_tanimoto, psm_tm, psm_rmsd, complexes]:
        if not _.exists():
            print(f"file {_} not found. Please run `fhw_p3 download` first.")
            sys.exit(1)
    
    # Load data
    print("Reading data.")
    with open(complexes, 'r') as f:
        complexes = json.load(f)

    sim_tm = np.load(psm_tm)
    sim_tanimoto = np.load(psm_tanimoto)
    sim_rmsd = np.load(psm_rmsd)
    # Get thresholds
    parameter_sets = [
        (tm_thr, tan_thr, rmsd_thr, sim_tm, sim_tanimoto, sim_rmsd, outdir)
        for tm_thr, tan_thr, rmsd_thr in itertools.product(
            config['tm_threshold'], config['tan_threshold'], config['rmsd_threshold']
        )
    ]
    results = Parallel(n_jobs=config['threads'])(delayed(worker)(pset) for pset in parameter_sets)
    # Collate results to dataframe
    df = collate_results(results, parameter_sets)
    df.to_csv(outdir / "results_summary.tsv", sep='\t', index=False)

def worker(pset):
    _thresh = pset[0:3]
    _simmats = pset[3:6]
    outdir = pset[6]

    print(f'Worker with tm = {_thresh[0]}, tan = {_thresh[1]}, rmsd = {_thresh[2]}')
    adj = create_adjacency_matrix(_simmats, _thresh)

    # ids, mask, labels are downloaded, and should be present in output folder.
    ids = outdir / "pairwise_similarity_complexes.json"
    mask = outdir / "test_train_mask.npy"
    labels = outdir / "affinities.npy"
    create_graph(
        adj,
        ids=ids,
        mask=mask,
        labels=labels,
        output_folder=outdir,
        prefix=f'similarity_graph_tm_{_thresh[0]}_tan_{_thresh[1]}_rmsd_{_thresh[2]}'
    )
    # Data_split, affinity data & complexes.
    dist = create_distance_matrix(_simmats)
    truelabels, predlabels = get_predictions(dist, mask, labels, ids)
    return (truelabels, predlabels)

def create_adjacency_matrix(sims, thresholds):
    tm_sim, tan_sim, rmsd_sim = sims
    tm_thr, tan_thr, rmsd_thr = thresholds
    adjm = np.empty(tm_sim.shape, dtype=bool)
    np.logical_and(tm_sim > tm_thr, tan_sim > tan_thr, out=adjm)
    np.logical_and(adjm, rmsd_sim < rmsd_thr, out=adjm)
    adjm = adjm.astype(np.int8, copy=False)
    adjm |= adjm.T
    np.fill_diagonal(adjm, 0)
    return adjm

def create_distance_matrix(sims):
    tm_sim, tan_sim, _ = sims
    dist = tm_sim.astype(float, copy=True)
    dist += tan_sim
    np.maximum(dist, dist.T, out=dist)
    np.fill_diagonal(dist, 0.0)
    return dist

def collate_results(res, parameter_sets):
    '''
    res is list containing[(truelabels, predlabels), (), ...]
    '''
    resl = []
    for labset, pset in zip(res, parameter_sets):
        true_labels, pred_labels = labset
        tm_thr, tan_thr, rmsd_thr = pset[0:3]
        corr_matrix = np.corrcoef(true_labels, pred_labels)
        r = corr_matrix[0, 1]
        rmse = np.sqrt(np.mean((np.array(true_labels) - np.array(pred_labels))**2))
        resl.append({
            'tm_thr': tm_thr,
            'tan_thr': tan_thr,
            'rmsd_thr': rmsd_thr,
            'r': r,
            'rmse': rmse
        })
    return (pd.DataFrame(resl))