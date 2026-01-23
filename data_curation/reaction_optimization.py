import argparse
import os

from math import sqrt

import ase.units as units
import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import read, write
from ase.io.formats import UnknownFileTypeError
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from sella import IRC, Sella
from tqdm import tqdm
from tblite.ase import TBLite
from ase.build.rotate import minimize_rotation_and_translation

def compute_rmsd(atoms1, atoms2):
    """Compute RMSD between two ASE Atoms objects after optimal alignment."""
    # Ensure both have the same number of atoms
  
    assert np.all(atoms1.numbers == atoms2.numbers), "Atoms objects must have the same number of atoms to compute RMSD."
    
    # Create copies to avoid modifying original objects
    atoms1_copy = atoms1.copy()
    atoms2_copy = atoms2.copy()
    
    # Minimize rotation and translation
    minimize_rotation_and_translation(atoms1_copy, atoms2_copy)
    
    # Compute RMSD
    diff = atoms1_copy.get_positions() - atoms2_copy.get_positions()
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    return rmsd

def get_energies_and_modes(atoms, hessian):
    n_atoms = len(atoms)

    assert hessian.shape == (n_atoms * 3, n_atoms * 3), (
        "Hessian must be a square matrix of size 3N x 3N where N is the number of atoms."
    )
    mass = torch.from_numpy(atoms.get_masses())
    coords = torch.from_numpy(atoms.get_positions())
    P = _calculate_translational_rotational_projection(mass, coords)
    hessian = torch.from_numpy(hessian)
    sqrtmmm = mass.repeat_interleave(3).sqrt()
    sqrtmmminv = 1.0 / sqrtmmm
    H = P.T @ torch.einsum("i,ij,j->ij", sqrtmmminv, hessian, sqrtmmminv) @ P

    omega2, vectors = torch.linalg.eigh(H)

    # Sort out modes and eigenvalues that belong to translations and rotations
    idx_nm = torch.ones(len(omega2), dtype=torch.bool, device=omega2.device)
    idx_nm[omega2.abs().argsort()[:6]] = False

    # Filter out unwanted modes (row equals normal mode, unit: eV/A^2/amu)
    omega2 = omega2[idx_nm].numpy()
    vectors = vectors[:, idx_nm].numpy()

    unit_conversion = units._hbar * units.m / sqrt(units._e * units._amu)
    energies = unit_conversion * omega2.astype(complex) ** 0.5

    modes = vectors.T.reshape(n_atoms * 3 - 6, n_atoms, 3)
    modes = modes * atoms.get_masses()[np.newaxis, :, np.newaxis] ** -0.5

    return energies, modes


def _calculate_translational_rotational_projection(mass, coords):
    """
    Calculate the Projection matrix to project out translational and rotational modes
    for a given set of masses and coordinates.

    Compare:
    https://github.com/pyscf/pyscf/blob/master/pyscf/hessian/thermo.py

    Args:
        mass: The atomic masses in units of amu. Shape is (N,).
        coords: The atomic positions in units of Å. Shape is (N, 3).

    Returns:
        Projection matrix P.

    Procedure:
        1. Calculate the center of mass of the system.
        2. Translate the coordinates to the center of mass frame.
        3. Compute the translational modes (Tx, Ty, Tz).
        4. Calculate the moment of inertia tensor and its eigenvalues and eigenvectors.
        5. Determine the principal axes of rotation.
        6. Compute the rotational modes (Rx, Ry, Rz) in the principal axes frame.
    """
    mass_center = torch.einsum("z,zx->x", mass, coords) / mass.sum()
    coords = coords - mass_center
    massp = mass**0.5

    # translational mode
    Tx = torch.einsum(
        "m,x->mx", massp, torch.tensor([1, 0, 0], dtype=mass.dtype, device=mass.device)
    )
    Ty = torch.einsum(
        "m,x->mx", massp, torch.tensor([0, 1, 0], dtype=mass.dtype, device=mass.device)
    )
    Tz = torch.einsum(
        "m,x->mx", massp, torch.tensor([0, 0, 1], dtype=mass.dtype, device=mass.device)
    )

    im = torch.einsum("m,mx,my->xy", mass, coords, coords)
    im = torch.eye(3, dtype=mass.dtype, device=mass.device) * im.trace() - im
    w, paxes = torch.linalg.eigh(im)

    # make the z-axis be the rotation vector with the smallest moment of inertia
    w = w.flip(0)
    paxes = paxes.flip(1)
    ex, ey, ez = paxes.T

    coords_in_rot_frame = coords @ paxes
    cx, cy, cz = coords_in_rot_frame.T
    Rx = massp[:, None] * (cy[:, None] * ez - cz[:, None] * ey)
    Ry = massp[:, None] * (cz[:, None] * ex - cx[:, None] * ez)
    Rz = massp[:, None] * (cx[:, None] * ey - cy[:, None] * ex)

    # Get projection matrix
    TRspace = torch.vstack(
        (
            Tx.flatten(),
            Ty.flatten(),
            Tz.flatten(),
            Rx.flatten(),
            Ry.flatten(),
            Rz.flatten(),
        )
    )

    q, r = torch.linalg.qr(TRspace.T)
    P = torch.eye(mass.shape[0] * 3, dtype=mass.dtype, device=mass.device) - q @ q.T

    return P


def get_minimum_from_trajectory(trajectory, folder):
    # Extract energies of the IRC trajectory
    energies = [atom.get_potential_energy() for atom in trajectory]

    # Search for discontinuity in the trajectory to find where the first IRC stopped
    diffs = np.diff(energies)
    split_index = np.argmax(diffs) + 1

    traj_part1 = trajectory[:split_index]
    traj_part2 = trajectory[split_index:]

    traj_energy_1 = [atom.get_potential_energy() for atom in traj_part1]
    traj_energy_2 = [atom.get_potential_energy() for atom in traj_part2]

    min_index_1 = np.argmin(traj_energy_1)
    min_index_2 = np.argmin(traj_energy_2)
    min_atom_1 = traj_part1[min_index_1]
    min_atom_2 = traj_part2[min_index_2]

    plt.figure(figsize=(8, 5))
    plt.plot(energies, marker="o")
    plt.axvline(x=split_index, color="r", linestyle="--", label="IRC Split")
    plt.scatter(min_index_1, traj_energy_1[min_index_1], color="g", label="Min 1")
    plt.scatter(
        min_index_2 + split_index, traj_energy_2[min_index_2], color="b", label="Min 2"
    )
    plt.xlabel("IRC Step")
    plt.ylabel("Energy (Hartree)")
    plt.title("IRC Energy Profile")
    plt.legend()
    plt.savefig(f"{folder}/irc_energy_profile.png")
    plt.close()

    return min_atom_1, min_atom_2


def ts_opt(sample, calc, ts_opt_kwargs, max_steps=500, verbose=False):
    rxn = sample.info["rxn"]
    folder = f"{datasource}/rxn_{rxn}/ts_opt"
    os.makedirs(folder, exist_ok=True)

    if verbose:
        print(f"Optimizing sample {rxn}")
    
    # Set calculator
    sample.calc = calc

    # Set up a Sella Dynamics object
    failed = False        
    try:
        dyn = Sella(
            sample,
            trajectory=f"{folder}/ts_opt.traj",
            logfile=f"{folder}/ts_opt.log",
            **ts_opt_kwargs
        )
        converged_ = dyn.run(fmax=1e-3, steps=max_steps)
    except Exception as e:
        print(f"\nFailed for {rxn} during TS optimization!\n")
        print("Error message:", e, "\n")
        failed = True

    try:
        # Read in the trajectory and log file
        traj = read(f"{folder}/ts_opt.traj", ":")
        with open(f"{folder}/ts_opt.log", "r") as f:
            lines = f.readlines()
            n_opt_steps = int(lines[-1].split()[1])

        # Read in optimized structure
        opt_structure = traj[-1]
        opt_structure.info["type"] = "transition_state"
        opt_structure.info["len_traj"] = len(traj)
        opt_structure.info["n_steps_opt"] = n_opt_steps
    except UnknownFileTypeError:
        print(f"\nEmpty trajectory for {rxn}! Using initial structure.\n")
        opt_structure = sample
        opt_structure.info["len_traj"] = 0
        opt_structure.info["n_steps_opt"] = 0

    if failed:
        opt_structure.info["converged"] = False
    else:
        try:
            # Compute Hessian
            vib = Vibrations(sample, name=f"{folder}/vib")
            vib.run()

            # standard settings are: method='standard', direction='central':
            # https://ase-lib.org/_modules/ase/vibrations/vibrations.html#Vibrations
            if verbose:
                vib.summary()
            else:
                vib.read()

            np.save(f"{folder}/hessian.npy", vib.H)

            # Extract energies in eV and modes in Angstrom
            energies, modes = get_energies_and_modes(opt_structure, vib.H)
            np.save(f"{folder}/energies.npy", energies)
            np.save(f"{folder}/frequencies.npy", energies / units.invcm)
            np.save(f"{folder}/modes.npy", modes)

            # Check if only one imaginary frequency (all frequencies below 20 cm^-1 are ignored)
            freq_threshold = 20  # cm^-1
            frequencies = energies / units.invcm
            converged = int((frequencies.imag > freq_threshold).sum()) == 1
            opt_structure.info["converged"] = converged

        except Exception as e:
            print(f"\nFailed for {rxn} during Hessian calculation!\n")
            print("Error message:", e, "\n")
            failed = True
            opt_structure.info["converged"] = False

    opt_structure.write(f"{folder}/opt_ts.xyz")

    # Write the optimized structure to the datasource file
    os.makedirs(datasource, exist_ok=True)
    write(f"{datasource}/ts_opt_db.xyz", opt_structure, append=True)

    return opt_structure


def irc_(sample, irc_kwargs, max_steps, folder, calc):
    rxn = sample.info["rxn"]
    try:
        opt = IRC(
            sample,
            trajectory=f"{folder}/irc.traj",
            **irc_kwargs,
        )
        opt.run(fmax=0.1, steps=max_steps, direction="forward")
        opt.run(fmax=0.1, steps=max_steps, direction="reverse")
    except Exception as e:
        print(f"\nFailed for {rxn} during IRC!\n")
        print("Error message:", e, "\n")
        min1, min2 = sample.copy(), sample.copy()
        min1.calc = calc
        min2.calc = calc
        return None, min1, min2

    # Analyze the trajectory to find the minimum points
    try:
        trajectory = read(f"{folder}/irc.traj", index=":")
        minimum_1, minimum_2 = get_minimum_from_trajectory(trajectory, folder)
    except Exception as e:
        print(f"\nCouldn't find two minima for {rxn} in IRC trajectory!\n")
        print("Error message:", e, "\n")
        min1, min2 = sample.copy(), sample.copy()
        min1.calc = calc
        min2.calc = calc
        return None, min1, min2

    # Minimize the IRC endpoints to ensure they are at a minimum
    try:
        minimum_1.calc = calc
        dyn = BFGS(minimum_1, trajectory=f"{folder}/minimum_1.traj")
        converged = dyn.run(fmax=0.05, steps=max_steps)

        min_1 = read(f"{folder}/minimum_1.traj", index="-1")
        min_1.info["type"] = "minimum"
        min_1.info["converged"] = converged
        min_1.write(f"{folder}/minimum_1.xyz")

        minimum_2.calc = calc
        dyn = BFGS(minimum_2, trajectory=f"{folder}/minimum_2.traj")
        converged = dyn.run(fmax=0.05, steps=max_steps)

        min_2 = read(f"{folder}/minimum_2.traj", index="-1")
        min_2.info["type"] = "minimum"
        min_2.info["converged"] = converged
        min_2.write(f"{folder}/minimum_2.xyz")

        rmsd = compute_rmsd(min_1, min_2)
        print(f"RMSD between minima: {rmsd:.3f} Å")
    except Exception as e:
        print(f"\nProblem in optimizing the found minima for {rxn}!\n")
        print("Error message:", e, "\n")
        min1, min2 = sample.copy(), sample.copy()
        min1.calc = calc
        min2.calc = calc
        return None, min1, min2
    
    return rmsd, min_1, min_2


def irc(
    sample,
    calc,
    irc_kwargs={},
    max_steps=500,
    max_iter_irc=3,
    rmsd_threshold=0.1,
    verbose=False,
):
    rxn = sample.info["rxn"]

    folder = f"{datasource}/rxn_{rxn}/irc"
    os.makedirs(folder, exist_ok=True)
    if verbose:
        print(f"IRC for sample {rxn}")

    # Set calculator
    sample.calc = calc

    # If an IRC inner iteration fails, the IRC calculation stops by default. However,
    # if the keyword argument keep_going is set as keep_going=True, the IRC optimization
    # will print a warning that the trajectory is no longer accurate, but it will
    # continue with the IRC. This helps when IRC is used as a way of finding connected
    # minima, and the exactness of the entire IRC trajectory is of lesser relevance.

    # Repeat IRC with smaller dx until two distinct minima are found or max iterations reached
    converged = False
    dx_start = irc_kwargs.get("dx", 0.05)
    for i in range(max_iter_irc):
        irc_kwargs["dx"] = dx_start / (2**i)
        rmsd, min1, min2 = irc_(sample, irc_kwargs, max_steps, folder, calc)
        if rmsd is None:
            break
        if rmsd > rmsd_threshold:
            if min1.info["converged"] and min2.info["converged"]:
                converged = True
            break
        else:
            print(
                f"RMSD below threshold {rmsd_threshold}, reducing dx {irc_kwargs['dx']} further"
            )

    # If structures collapsed to same minima, then consider as not converged
    min1.info["converged"] = converged
    min2.info["converged"] = converged

    return min1, min2


def run_ts_opt_and_irc(sample, max_steps_ts_opt=500, max_steps_geo_opt=500, max_iter_irc=3, rmsd_threshold=0.1):
    ts_opt_kwargs = {}
    irc_kwargs = {
        "dx": 0.05,
        "eta": 1e-4,
        "gamma": 0.4,
        "keep_going": True,
    }

    # Define calculator
    xtb_kwargs = {
        "max_iterations": 500,
        "charge": sample.info.get("charge", 0),
        "uhf": sample.info.get("multiplicity", 1)-1,
    }
    
    print(
        f"\nUsing charge {xtb_kwargs['charge']} and multiplicity "
        f"{xtb_kwargs['uhf']+1} for rxn={sample.info['rxn']}\n"
    )
    
    calc = TBLite(atoms=sample, method="GFN2-xTB", **xtb_kwargs)
    
    # Opt TS
    opt_structure = ts_opt(sample, calc, ts_opt_kwargs, max_steps=max_steps_ts_opt)

    if opt_structure.info["converged"] is False:
        print(f"Skipping IRC for {sample.info['rxn']} as TS optimization didn't converge")
        min1, min2 = opt_structure.copy(), opt_structure.copy()
        min1.calc = calc
        min2.calc = calc
        min1.info["converged"] = False
        min2.info["converged"] = False
        rxn_data = [min1, opt_structure, min2]
    else:
        # Perform IRC on found TS (in case minima has RMSD below threshold, redo IRC with lower dx)
        min1, min2 = irc(
            sample,
            calc,
            irc_kwargs,
            max_steps=max_steps_geo_opt,
            max_iter_irc=max_iter_irc,
            rmsd_threshold=rmsd_threshold,
        )

        # In case any minima did not converge, label the whole reaction as not converged
        if min1.info["converged"] is False or min2.info["converged"] is False:
            opt_structure.info["converged"] = False
            
        # Define reactant as lower energy minimum
        try:
            min1_energy = min1.get_potential_energy()
            min2_energy = min2.get_potential_energy()
        except Exception as e:
            print(f"\nFailed for {sample.info['rxn']} when calculating minimum energies!\n")
            print("Error message:", e, "\n")
            min1_energy = float("inf")
            min2_energy = float("inf")
        
        if min1_energy < min2_energy:
            min1.info["type"] = "product"
            min2.info["type"] = "reactant"
            rxn_data = [min1, opt_structure, min2]
        else:
            min1.info["type"] = "reactant"
            min2.info["type"] = "product"
            rxn_data = [min2, opt_structure, min1]

    assert rxn_data[0].info["converged"] == rxn_data[1].info["converged"] == rxn_data[2].info["converged"], "Inconsistent convergence status in final reaction data"
    write(f"{datasource}/irc_db.xyz", rxn_data, append=True)

    return rxn_data



parser = argparse.ArgumentParser(
    description="Optimize samples from a given datasource."
)
parser.add_argument("--datasource", type=str, default="rh_dataset", help="Datasource file")
parser.add_argument("--start_idx", type=int, default=0, help="Start index")
parser.add_argument("--end_idx", type=int, default=None, help="End index")


args = parser.parse_args()
start_idx = args.start_idx
end_idx = args.end_idx
datasource = args.datasource

main_folder = "raw_db"
datasource_db = read(f"{main_folder}/{datasource}.xyz", index=":",)[start_idx:end_idx]

print(
    f"Starting optimization for {len(datasource_db)} samples (idx {start_idx} to {end_idx})"
)

os.environ["OMP_NUM_THREADS"] = "1"

for sample in tqdm(datasource_db, total=len(datasource_db), desc="Calculations"):
    run_ts_opt_and_irc(sample)
