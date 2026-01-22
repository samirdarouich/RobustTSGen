from typing import List
import torch
import numpy as np
from reactot.utils import bond_analyze

# atomic number
# source: http://www.webelements.com
atomic_number = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100}
atomic_number_to_symbol = {v: k for k, v in atomic_number.items()}

def write_xyz(mol, dataset_info, xyzfile="tmp.xyz"):
    atom_decoder = dataset_info['atom_decoder']
    n_atom = mol["atom"].size(0)
    with open(xyzfile, "w") as fo:
        fo.write(str(n_atom) + "\n\n")
        for ii in range(n_atom):
            pos = mol["pos"][ii].cpu().numpy()
            ele = atom_decoder[mol["atom"][ii]]
            _x = " ".join([str(__x) for __x in pos])
            fo.write(f"{ele} {_x}\n")


def check_stability(mol, dataset_info, debug=False):
    positions, atom_type = mol["pos"], mol["atom"]
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    atom_decoder = dataset_info['atom_decoder']
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype='int')

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[atom_type[j]]
            pair = sorted([atom_type[i], atom_type[j]])
            if dataset_info['name'] == 'qm9':
                order = bond_analyze.get_bond_order(atom1, atom2, dist)
            else:
                raise KeyError("only qm9 is allowed!")
            # if i == 3 or j == 3:
            #     print(i, j, dist, order)
            nr_bonds[i] += order
            nr_bonds[j] += order
    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        possible_bonds = bond_analyze.allowed_bonds[atom_decoder[atom_type_i]]
        # print(atom_decoder[atom_type_i], nr_bonds_i)
        if type(possible_bonds) == int:
            is_stable = possible_bonds >= nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        if not is_stable and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type_i], nr_bonds_i))
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)
    return int(molecule_stable), nr_stable_bonds, len(x)


def assemble_sample_inputs(
    atoms: List,
    device: torch.device = torch.device("cuda"),
    n_samples: int = 1,
    frag_type: bool = False,
):
    empty_site = torch.tensor([[1, 0, 0, 0, 0, 1]], device=device)
    if not frag_type:
        decoders = [
                {
                    "H": [1, 0, 0, 0, 0, 1],
                    "C": [0, 1, 0, 0, 0, 6],
                    "N": [0, 0, 1, 0, 0, 7],
                    "O": [0, 0, 0, 1, 0, 8],
                    "F": [0, 0, 0, 0, 1, 9]
                }
            ] * 2
    else:
        decoders = [
                {
                    "H": [1, 0, 0, 0, 0, 1, 0],
                    "C": [0, 1, 0, 0, 0, 6, 0],
                    "N": [0, 0, 1, 0, 0, 7, 0],
                    "O": [0, 0, 0, 1, 0, 8, 0],
                    "F": [0, 0, 0, 0, 1, 9, 0]
                },
                {
                    "H": [1, 0, 0, 0, 0, 1, 1],
                    "C": [0, 1, 0, 0, 0, 6, 1],
                    "N": [0, 0, 1, 0, 0, 7, 1],
                    "O": [0, 0, 0, 1, 0, 8, 1],
                    "F": [0, 0, 0, 0, 1, 9, 1]
                }
            ]

    h0 = [
        torch.cat(
            [
                torch.tensor(
                    [decoders[ii % 2][atom] for atom in atoms],
                    device=device
                )
                for _ in range(n_samples)
            ]
        ) for ii in range(3)
    ]
    return h0

def write_single_xyz(xyzfile, natoms, out):
    with open(xyzfile, "w") as fo:
            fo.write(str(natoms) + "\n\n")
            for ele in out:
                x = ele[:3].cpu().numpy()
                _a = atomic_number_to_symbol[ele[-1].long().item()]
                _x = " ".join([str(__x) for __x in x])
                fo.write(f"{_a} {_x}\n")


def write_tmp_xyz(fragments_nodes, out_samples, idx=[0], prefix="gen", localpath="tmp", ex_ind=0):
    TYPEMAP = {
        0: "react",
        1: "ts",
        2: "prod",

    }
    for ii in idx:
        st = TYPEMAP[ii]
        start_ind, end_ind = 0, 0
        for jj, natoms in enumerate(fragments_nodes[0]):
            _jj = jj + ex_ind

            # xyzfile = f"{localpath}/{_jj}.xyz"
            xyzfile = f"{localpath}/{prefix}_{_jj}_{st}.xyz"
    
            end_ind += natoms.item()
            write_single_xyz(
                xyzfile,
                natoms.item(),
                out=out_samples[ii][start_ind: end_ind],
            )
            start_ind = end_ind
