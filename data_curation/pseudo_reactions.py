import argparse
import os
import subprocess
import numpy as np
from ase.io import read, write
from tqdm import tqdm
from ase.build.rotate import minimize_rotation_and_translation
from ase.calculators.singlepoint import SinglePointCalculator
import toml
import random


def get_pseudo_reaction(conformers):
    
    # get energies saved in info from CREST
    
    energies = [float(list(c.info.keys())[0]) for c in conformers]
    
    # Sort conformers by energy
    sorted_idx = np.argsort(energies)

    if len(sorted_idx) < 3:
        print("Not enough conformers found by CREST for reaction.")
        return None

    # Pick conformers
    reactant_idx = sorted_idx[-2]  # reactant (2nd highest)
    ts_idx = sorted_idx[-1]        # TS (highest energy)
    product_idx = sorted_idx[0]    # product (lowest energy)

    # Add info and calculators
    reactant = conformers[reactant_idx]
    reactant.info = {"type": "reactant"}
    reactant.calc = SinglePointCalculator(reactant, energy=energies[reactant_idx]*27.2114)
    transition_state = conformers[ts_idx]
    transition_state.info = {"type": "transition_state"}
    transition_state.calc = SinglePointCalculator(transition_state, energy=energies[ts_idx]*27.2114)
    product = conformers[product_idx]
    product.info = {"type": "product"}
    product.calc = SinglePointCalculator(product, energy=energies[product_idx]*27.2114)

    # align TS and product to reactant
    minimize_rotation_and_translation(reactant, transition_state)
    minimize_rotation_and_translation(reactant, product)

    pseudo_reaction = [reactant, transition_state, product]
    
    return pseudo_reaction

def get_crest_input(n_threads, charge, uhf, use_gfnff_md=False):
    crest_settings = {
        'input': 'input.xyz', 
        'runtype': 'imtd-gc', 
        'threads': n_threads, 
        'calculation': {
            'elog': 'energies.log', 
        }, 
        'cregen': {'ewin': 20.0}
    }
    if use_gfnff_md:
        # crest_settings['calculation']['level'] = [
        #     {'method': 'gfnff', 'uhf': uhf, 'chrg': charge},
        #     {'method': 'gfn2', 'uhf': uhf, 'chrg': charge}
        # ]
        # crest_settings['dynamics'] = {'active': [1]}
        crest_settings['calculation']['level'] = [{'method': 'gfnff', 'uhf': uhf, 'chrg': charge}]
    else:
        crest_settings['calculation']['level'] = [{'method': 'gfn2', 'uhf': uhf, 'chrg': charge}]
    return crest_settings

def perform_crest(atoms, use_gfnff_md=False):
    
    origin_rxn = atoms.info["rxn"]
    origin_type = atoms.info["type"]
    origin_tm = atoms.info["tm"]
    origin_source = atoms.info["source"]
    chrg = int(atoms.info.get("charge", 0))
    uhf = int(atoms.info.get("uhf", 0))

    folder = f"{datasource}/crest/rxn_{origin_rxn}_type_{origin_type}_tm_{origin_tm}_source_{origin_source}"
    os.makedirs(folder, exist_ok=True)
    atoms.write(f"{folder}/input.xyz")

    crest_settings = get_crest_input(n_threads, chrg, uhf, use_gfnff_md=use_gfnff_md)

    with open(f"{folder}/input.toml", "w") as f:
        toml.dump(crest_settings, f)

    with open(f"{folder}/out.log", "w") as f:
        result = subprocess.run(
            ["crest", "input.toml"],
            cwd=folder,
            stdout=f,
            stderr=f,
        )
    
    pseudo_reaction = None
    if result.returncode == 0:
        conformers = read(f"{folder}/crest_conformers.xyz", index=":")
        pseudo_reaction = get_pseudo_reaction(conformers)
    
    if pseudo_reaction is None:
        print(f"CREST failed for reaction {origin_rxn}.")
    else:
        for a in pseudo_reaction:
            a.info["origin_rxn"] = origin_rxn
            a.info["origin_type"] = origin_type
            a.info["origin_tm"] = origin_tm
            a.info["origin_source"] = origin_source
            a.info["charge"] = chrg
            a.info["uhf"] = uhf

    return pseudo_reaction

def run_crest(atoms, use_gfnff_md=False):
    crest_db = []
    origin_rxn = atoms.info["rxn"]
    origin_type = atoms.info["type"]
    origin_tm = atoms.info["tm"]
    origin_source = atoms.info["source"]
    print(f"Running CREST for reaction {origin_rxn} and type {origin_type} from tm {origin_tm} and source {origin_source}...")
    pseudo_reaction = perform_crest(atoms, use_gfnff_md=use_gfnff_md)
    if pseudo_reaction is not None:
        crest_db.extend(pseudo_reaction)

    write(f"{datasource}/crest_db.xyz", crest_db, append=True)
    
    return 1 if pseudo_reaction is not None else 0
    

parser = argparse.ArgumentParser(
    description="Optimize samples from a given datasource."
)
parser.add_argument("--datasource", type=str, default="rh_dataset", help="Datasource file")
parser.add_argument("--start_idx", type=int, default=0, help="Start index")
parser.add_argument("--end_idx", type=int, default=None, help="End index")
parser.add_argument("--n_threads", type=int, default=1, help="Threads to use")
parser.add_argument("--use_gfnff_md", action="store_true", help="Use GFNFF MD")

args = parser.parse_args()
start_idx = args.start_idx
end_idx = args.end_idx
datasource = args.datasource
n_threads = args.n_threads
use_gfnff_md = args.use_gfnff_md


main_folder = "raw_db"
datasource_db = read(f"{main_folder}/{datasource}.xyz", index=":")

# Shuffle dataset
random.seed(42)
random.shuffle(datasource_db)

# Take smaller subset for start
datasource_db = datasource_db[start_idx:end_idx]

print(
    f"Starting optimization for {len(datasource_db)} samples (idx {start_idx} to {end_idx})"
)
if use_gfnff_md:
    print("Using GFNFF MD in CREST calculations.")
    
os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
pbar = tqdm(datasource_db, total=len(datasource_db), desc="Calculations")

conv_count = 0
for i, sample in enumerate(pbar):
    conv_count += run_crest(sample, use_gfnff_md=use_gfnff_md)

    # Update tqdm postfix with convergence rate
    conv_rate = conv_count / (i + 1) * 100
    pbar.set_postfix({
        "conv": f"{conv_count}",
        "rate": f"{conv_rate:.2f}%"
    })