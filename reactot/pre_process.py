from pymatgen.io.xyz import XYZ
import numpy as np
from pymatgen.analysis.molecule_matcher import KabschMatcher
from ase.io import read
import os


def xyz2pmg(xyzfile):
    # Converts an XYZ file to a pymatgen Molecule object
    xyz_converter = XYZ(mol=None)
    mol = xyz_converter.from_file(xyzfile).molecule
    return mol

def translate_molecule(mol):
    # Translates the molecule so that its center of mass is at the origin
    coordinates = np.array([[site.x, site.y, site.z] for site in mol.sites])
    avg_coordinates = np.mean(coordinates, axis=0)
    translated_coordinates = coordinates - avg_coordinates
    for i, site in enumerate(mol.sites):
        site.x, site.y, site.z = translated_coordinates[i]
    return mol    

def write_xyz(mol, filename):
    # Writes a pymatgen Molecule object to an XYZ file
    num_atoms = len(mol.sites)
    comment = "have a nice day"
    with open(filename, 'w',encoding='utf-8') as f:
        f.write(f"{num_atoms}\n")
        f.write(f"{comment}\n")
        for site in mol.sites:
            f.write(f"{site.specie} {site.x:.6f} {site.y:.6f} {site.z:.6f}\n")
    return filename

def pre_treatment(rxyz,pxyz):
    """
    Pre-treatment function to optimize the reactant and product molecules
    """
    mol1 = xyz2pmg(rxyz)
    mol2 = xyz2pmg(pxyz)
    mol1_opt=translate_molecule(mol1)
    bfm = KabschMatcher(mol1_opt)
    mol2_opt, rmsd = bfm.fit(mol2)
    mol1_opt_path = rxyz.replace('.xyz', '-opt.xyz')
    mol2_opt_path = pxyz.replace('.xyz', '-opt.xyz')
    
    write_xyz(mol1_opt, mol1_opt_path)
    write_xyz(mol2_opt, mol2_opt_path)
    
    mol1_data=read(mol1_opt_path)
    mol2_data=read(mol2_opt_path)
    os.remove(mol1_opt_path)
    os.remove(mol2_opt_path)
    return mol1_data, mol2_data, rmsd
