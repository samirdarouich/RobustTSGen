import logging
import math
from typing import List, Tuple, Union

import numpy as np
from ase import Atoms
from ase.io import read
from pymatgen.analysis.molecule_matcher import (
    BruteForceOrderMatcher,
    GeneticOrderMatcher,
    HungarianOrderMatcher,
    KabschMatcher,
)
from pymatgen.core import Molecule
from pymatgen.io.xyz import XYZ


def xh2pmg(xh):
    mol = Molecule(
        species=xh[:, -1].long().cpu().numpy(),
        coords=xh[:, :3].cpu().numpy(),
    )
    return mol


def xyz2pmg(xyzfile):
    xyz_converter = XYZ(mol=None)
    mol = xyz_converter.from_file(xyzfile).molecule
    return mol


log = logging.getLogger(__name__)
logging.getLogger("pymatgen").setLevel(logging.CRITICAL)


def get_total_permutations(
    atomic_numbers: Union[List[int], Tuple[int], np.ndarray],
) -> int:
    """
    Calculate the total number of permutations based on the occurrence of each unique atom.

    Args::
        atomic_numbers (Union[List[int], np.ndarray]):
          List of atomic numbers.

    Returns:
        int:
          Total number of permutations.
    """
    # Get occurence of each unique atom
    _, count = np.unique(atomic_numbers, return_counts=True)
    total_permutations = 1
    for c in count:
        total_permutations *= math.factorial(c)
    return total_permutations


def pymatgen_align(source: Atoms, target: Atoms, same_order: bool = False) -> Atoms:
    """
    Aligns the source molecule to the target molecule using various matching algorithms.

    In case the same atom order is given, simply use the Kabsch Matcher algorithm.
    Otherwise 3 alternative algorithms are used:
      1) Brute force order matching if the total number of possible permuations of atoms
         is lower than 1e5. This test every permutation of atoms with the same species.
         For each trial, the Kabsch Matcher algorithm is used, and the permutation with
         the lowest rmsd is taken

    Args:
        source (Atoms):
          The source molecule to be aligned.
        target (Atoms):
          The target molecule to align the source molecule to.
        same_order (bool, optional):
          If True, aligns the source molecule to the target molecule assuming the atoms
          are in the same order. Defaults to False.

    Returns:
        Atoms:
          The aligned source molecule.
    """
    source_pymatgen = Molecule(
        species=source.get_atomic_numbers(), coords=source.get_positions()
    )
    target_pymatgen = Molecule(
        species=target.get_atomic_numbers(), coords=target.get_positions()
    )

    if same_order:
        assert np.all(source.get_atomic_numbers() == target.get_atomic_numbers()), (
            "Expected source and target to have the same atom ordering."
        )
        log.debug("Use Kabsch Matcher matching.")
        bfm = KabschMatcher(target_pymatgen)
        aligned_source, _ = bfm.fit(source_pymatgen)
    else:
        total_permutations = get_total_permutations(source_pymatgen.atomic_numbers)  # type: ignore

        if total_permutations < 1e4:
            log.debug("Use brute force matching.")
            bfm = BruteForceOrderMatcher(target_pymatgen)
            aligned_source, _ = bfm.fit(source_pymatgen)
        else:
            bfm = GeneticOrderMatcher(target_pymatgen, threshold=0.5)
            pairs = bfm.fit(source_pymatgen)
            if len(pairs) == 0:
                log.debug("Use hungarian order matching.")
                bfm = HungarianOrderMatcher(target_pymatgen)
                aligned_source, _ = bfm.fit(source_pymatgen)
            else:
                log.debug("Use genetic order matching.")
                min_idx = np.argmin([p[1] for p in pairs])
                aligned_source = [p[0] for p in pairs][min_idx]

    return Atoms(
        numbers=aligned_source.atomic_numbers, positions=aligned_source.cart_coords
    )


def _compute_rmsd(source: Atoms, target: Atoms) -> float:
    """Calculates the Root Mean Square Deviation (RMSD) between two molecular structures.

    Compute RMSD (https://en.wikipedia.org/wiki/Root_mean_square_deviation_of_atomic_positions)

    RMSD(v,w) = sqrt( 1/n sum_i^n ||v_i-w_i||^2 )
              = sqrt( 1/n sum_i^n ( (v_i,x-w_i,x)^2 + (v_i,y-w_i,y)^2 + (v_i,z-w_i,z)^2 )

    Args:
        source (Atoms):
          The source which will be translated/rotated to match the target with lowest
          RMSD.
        target (Atoms):
          Target to match the source with lowest RMSD.

    Returns:
        float:
          The calculated RMSD.
    """
    # Compute RMSD(v,w):
    # get ||v_i-w_i||^2: np.sum((v-w)**2,axis=1)
    # 1/n sum_i_n ||v_i-w_i||^2: np.mean(||v_i-w_i||^2)
    # RMSD = sqrt(1/n sum_i_n ||v_i-w_i||^2)
    return np.sqrt(
        np.mean(
            np.sum(
                (source.get_positions() - target.get_positions()) ** 2,
                axis=1,
            )
        )
    )


def compute_rmsd(
    source: Atoms, target: Atoms, same_order: bool = True
) -> Tuple[Atoms, float]:
    """Calculates the Root Mean Square Deviation (RMSD) between two molecular structures.

    Pymatgen library is used to align the two structures (https://pymatgen.org).

    Compute RMSD (https://en.wikipedia.org/wiki/Root_mean_square_deviation_of_atomic_positions)

    RMSD(v,w) = sqrt( 1/n sum_i^n ||v_i-w_i||^2 )
              = sqrt( 1/n sum_i^n ( (v_i,x-w_i,x)^2 + (v_i,y-w_i,y)^2 + (v_i,z-w_i,z)^2 )

    Args:
        source (Atoms):
          The source which will be translated/rotated to match the target with lowest
          RMSD.
        target (Atoms):
          Target to match the source with lowest RMSD.
        same_order (bool, optional):
          Whether the atoms in both molecules are in the same order. Default is True.

    Returns:
        Tuple[Atoms,float]:
          The aligned molecule and calculated RMSD.
    """
    assert len(source) == len(target), (
        "Expected source and target to thave the same number of atoms."
    )

    # Align molecule using different algorithms
    aligned_molecule = pymatgen_align(
        source=source, target=target, same_order=same_order
    )

    # Compute RMSD
    rmsd = _compute_rmsd(source=aligned_molecule, target=target)

    return aligned_molecule, rmsd


def get_rmsd(
    source: Atoms,
    target: Atoms,
    same_order: bool = True,
    ignore_chirality: bool = False,
) -> Tuple[Atoms, float]:
    """Calculates the RMSD between two molecular structures with optional chirality reflection.


    Args:
        source (Atoms):
          The source which will be translated/rotated to match the target with lowest
          RMSD.
        target (Atoms):
          Target to match the source with lowest RMSD.
        same_order (bool, optional):
          Whether the atoms in both molecules are in the same order. Default is True.
        ignore_chirality (bool, optional):
          If True, the function also calculates RMSD for the molecule reflected along
          the z-axis, and returns the minimum RMSD. Default is False.

    Returns:
        Tuple[Atoms, float]:
          The aligned molecule and minimum RMSD, considering chirality if specified.
    """
    aligned_molecule, rmsd = compute_rmsd(source, target, same_order)
    log.debug(f"RMSD: {rmsd:.4f}")

    if ignore_chirality:
        # Reflect coordinates
        source_reflected = source.copy()
        reflected_pos = source_reflected.get_positions()
        reflected_pos[:, -1] = -reflected_pos[:, -1]
        source_reflected.set_positions(reflected_pos)

        # Compure RMSD for refrecleted molecule
        aligned_molecule_reflect, rmsd_reflect = compute_rmsd(
            source_reflected, target, same_order
        )
        log.debug(f"RMSD reflected: {rmsd_reflect:.4f}")

        # Return lower RMSD
        if rmsd_reflect < rmsd:
            rmsd = rmsd_reflect
            aligned_molecule = aligned_molecule_reflect

    return aligned_molecule, rmsd


rmsd_list = []
for repeat in range(0, 10):
    rmsds = []
    for batch in range(17):
        if batch == 16:
            no_samples = 49
        else:
            no_samples = 64

        for i in range(no_samples):
            gen = f"repeat_{repeat}/batch_{batch}/gen_{i}_ts.xyz"
            ref = f"repeat_{repeat}/batch_{batch}/sample_{i}_ts.xyz"
            aligned, rmsd = get_rmsd(
                source=read(gen),
                target=read(ref),
                same_order=False,
                ignore_chirality=True,
            )
            aligned.write(f"repeat_{repeat}/batch_{batch}/gen_{i}_aligned_ts.xyz")

            # rmsd_false, aligned_ = pymatgen_rmsd(xyz2pmg(gen), xyz2pmg(ref), ignore_chirality=True, threshold=0.5, return_aligned=True)

            print(f"batch {batch}, sample {i}, RMSD: {rmsd:.4f}")

            if rmsd > 1.0 * 3**0.5:
                print("RMSD too high! Use 1.0*3**0.5 instead.")
            rmsds.append(min(rmsd, 1.0 * 3**0.5))

    print(
        f"Repeat {repeat}, Mean RMSD: {np.mean(rmsds):.4f}, Std RMSD: {np.std(rmsds):.4f}"
    )
    np.save(f"repeat_{repeat}/rmsds_correct.npy", np.array(rmsds))
    rmsd_list.append(rmsds)
# np.save("rmsd_correct.npy", np.array(rmsd_list))
