import ast
from molSimplify.Classes.atom3D import atom3D
from molSimplify.Scripts.geometry import rotate_around_axis, vecangle
from molSimplify.Classes.mol3D import mol3D
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

def cross_product(u1, u2):
    cross = [u1[1] * u2[2] - u1[2] * u2[1],
             u1[2] * u2[0] - u1[0] * u2[2],
             u1[0] * u2[1] - u1[1] * u2[0]]
    return cross

def align_axis(mol,Rp,u1,u2):
    # mol: complex you want to rotate
    # Rp is reference point
    # u1 and u2 are vectors, rotates u1 to align with u2
    axis = cross_product(u1,u2)
    angle = vecangle(u1,u2)
    rotate_around_axis(mol,Rp,axis,angle)
    
# im now filtering for "small", "common", "chemically relevant", and "negatively charged so we can make it neutral by ligand dissociation"
tmc_data = pd.read_csv('tmcs_from_csd.csv')
query_dict = {'Ti': ['807353bea90914391f361aaa72e6883d'], # hexafluoride. maybe add hexachloride, f7e3dd2f0dde15f2998ef1276ffbaae3
              'Zn': ['cfaa3b0fe7f5faf5fa9809e0d982d50c'], # tetrachloride
              'Ru': ['5e7889fa0782fbb06e91e3d62aa8034a'], # hexachloride
              'Rh': ['e340b42167afa8aae5401ea2d0a3620b'],#, 'babadf6cc67f81960ddf67356082399a'], # dichloro dicarbonyl and dichloro cyclooctadiene
              'Pd': ['9b827bdd5ac3c0c3cecab941e528aed7'], # tetrachloride
              'Ag': ['a8ac5a216d459fa839a4cfbd545a549a'], # dicyanide
              'Os': ['c311422fd8eaf3441da88797cdb74187'], # hexachloride
              'Ir': ['772497f2aaa0353ec55238071e60d811'], # hexachloride
              'Pt': ['84004d8e69b9a12b8b5f48b22f6b7262'], # hexachloride
              'Au': ['d33762471fe78eb307c013c1daaa7d76']} # tetrachloride

# sort data by r-factor
tmc_data = tmc_data.sort_values(by='csd_r_factor')

for query_metal in query_dict.keys():
    hashes = query_dict[query_metal]
    for query_hash in hashes:
        # find corresponding metal and graph
        query_subset = tmc_data[(tmc_data['metal']==query_metal) & (tmc_data['graph_hash']==query_hash)].reset_index(drop=True)
        # parse mol2 string with molSimplify, return xyz coordinates
        mol2_string = query_subset['csd_mol2string'][0]
        mol = mol3D()
        mol.readfrommol2(filename=mol2_string, readstring=True)
        xyz_string = mol.writexyz(filename='', writestring=True)
        refcode = query_subset['refcode_plus'][0]

        print(f"transition metal: {query_metal}")
        print(f"oxidation state: {query_subset['name_oxidation_states'][0]}")
        print(f"tmc charge: {query_subset['complex_charge'][0]}")
        print(f"number of occurrences: {len(query_subset)}")
        print(f"best quality structure: {refcode}")
        print(f"r-factor: {query_subset['csd_r_factor'][0]}\n")
        # print(f"xyz coordinates:\n{xyz_string}")

        mol.writexyz(filename='TMCs/'+query_metal+'_'+refcode+'.xyz')
        
        
# add TS to metals as ligands
ts_structures = pd.read_csv('molecular_graphs_cleaned.csv')
ts_preds = pd.read_csv('pydentate_predictions/coordination_preds.csv')
ts_data = pd.concat([ts_structures, ts_preds.drop(columns=['smiles'])], axis=1)
ts_data['predicted_coordinating_atoms'] = ts_data['predicted_coordinating_atoms'].apply(ast.literal_eval)

metal_property_dict = {'Ti': {'ox': 4, 'charge': -2, 'ligand_to_remove': 'F', 'indices_to_remove': [1, 4]},
                       'Zn': {'ox': 2, 'charge': -2, 'ligand_to_remove': 'Cl', 'indices_to_remove': [1, 2]},
                       'Ru': {'ox': 3, 'charge': -3, 'ligand_to_remove': 'Cl', 'indices_to_remove': [1, 2, 6]},
                       'Rh': {'ox': 1, 'charge': -1, 'ligand_to_remove': 'Cl', 'indices_to_remove': [1]},
                       'Pd': {'ox': 2, 'charge': -1, 'ligand_to_remove': 'Cl', 'indices_to_remove': [1]},
                       'Ag': {'ox': 1, 'charge': -1, 'ligand_to_remove': 'CN', 'indices_to_remove': [3, 1]},
                       'Os': {'ox': 4, 'charge': -2, 'ligand_to_remove': 'Cl', 'indices_to_remove': [1, 4]},
                       'Ir': {'ox': 4, 'charge': -2, 'ligand_to_remove': 'Cl', 'indices_to_remove': [1, 3]},
                       'Pt': {'ox': 4, 'charge': -2, 'ligand_to_remove': 'Cl', 'indices_to_remove': [1, 2]},
                       'Au': {'ox': 3, 'charge': -1, 'ligand_to_remove': 'Cl', 'indices_to_remove': [1]}}
TMCs = [f for f in os.listdir('TMCs') if not f.startswith('.')]

for idx in tqdm(range(len(ts_data))):
    rxn_idx = ts_data['reaction_index'][idx]
    coord_atoms = ts_data['predicted_coordinating_atoms'][idx]
    ts_xyz = f"single_fragment_ts/rxn_{rxn_idx}.xyz"
    # only consider ligands predicted to coordinate to the metal with one atom
    if len(coord_atoms)==1:
        for TMC in TMCs:
            tmc_mol = mol3D()
            tmc_mol.readfromxyz(filename='TMCs/'+TMC)
            metal_idx = tmc_mol.findMetal()[0]
            metal = TMC.split('_')[0]

            # find ligands to delete
            ligand_to_delete = metal_property_dict[metal]['ligand_to_remove']
            num_ligands_to_delete = abs(metal_property_dict[metal]['charge'])
            atoms_to_delete = metal_property_dict[metal]['indices_to_remove']
            anchor_atom = atoms_to_delete[0]

            # instantiate mol3D of TS
            ts_mol = mol3D()
            ts_mol.readfromxyz(filename=ts_xyz)
            ts_natoms = ts_mol.natoms

            # add dummy atom to TS at coordination centroid
            coord_centroid = np.mean([ts_mol.atoms[atom_idx].coords() for atom_idx in coord_atoms], axis=0)
            x_atom = atom3D()
            x_atom.mutate('X')
            x_atom.setcoords(xyz=coord_centroid)
            ts_mol.addAtom(atom=x_atom)
            x_idx = ts_mol.natoms-1

            # align predicted coordinating atoms with TMC anchor atoms
            ts_mol.alignmol(atom1=x_atom, atom2=tmc_mol.atoms[anchor_atom])

            # rotate TS so representative axis is aligned relative to TMC
            # TS representative axis (v1): center of mass to coordination centroid
            v1 = ts_mol.atoms[x_idx].coords() - np.array(ts_mol.centermass())
            u1 = v1 / np.linalg.norm(v1)
            # TMC representative axis (v2): metal to anchor atom
            v2 = np.array(tmc_mol.atoms[metal_idx].coords()) - np.array(tmc_mol.atoms[anchor_atom].coords())
            u2 = v2 / np.linalg.norm(v2)
            align_axis(mol=ts_mol, Rp=ts_mol.atoms[ts_mol.natoms-1].coords(), u1=u1, u2=u2)

            # combine molecule, remove excess ligands
            ts_mol.deleteatoms([x_idx])
            tmc_mol.combine(mol=ts_mol)
            tmc_mol.deleteatoms(atoms_to_delete)

            # save tmc
            tmc_mol.writexyz(filename=f'ts_with_tmc/{metal}/{metal}_rxn_{rxn_idx}.xyz')