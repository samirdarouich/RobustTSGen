import torch
import numpy as np
import os
from ase.io import read
from reactot.trainer.pl_trainer import SBModule
from reactot.pre_process import pre_treatment
import logging

logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def onehot_convert(atomic_numbers):
    """
    Convert a list of atomic numbers into an one-hot matrix
    """
    encoder= {1: [1, 0, 0, 0, 0], 6: [0, 1, 0, 0, 0], 7: [0, 0, 1, 0, 0], 8: [0, 0, 0, 1, 0]}
    onehot = [encoder[i] for i in atomic_numbers]
    return np.array(onehot)

def format_error(error_code, message, task_id):
    return [
            {
                "task_id": task_id,
                "error_code": error_code,
                "error_display_text": message,
                "error_raw_message": repr(message)
            }
        ]
    

def check_xyz_files(rxyz, pxyz,task_id):
    """
    Check if the two xyz files have the same number of atoms.
    Raises an exception if they do not match.
    """
    def parse_xyz(xyz_file):
        with open(xyz_file, 'r') as file:
            lines = file.readlines()
            atom_count = int(lines[0].strip())
            atom_types = [line.split()[0] for line in lines[2:2 + atom_count]]
            allowed_atom_types = {'C', 'H', 'O', 'N'}
            for atom in atom_types:
                if atom not in allowed_atom_types:
                    raise ValueError(f"This model currently supports compounds containing C, H, O, and N only.")

            if atom_count > 50:
                print("WARNING: number of atoms > 50, system size much larger than the current React-OT training set, predictions might be inaccurate.")
            
            return atom_count, atom_types
        


    try:
        rxyz_atom_count, rxyz_atom_types = parse_xyz(rxyz)
    except Exception as e:
        print(format_error(10001, f"Error parsing xyz files: {e}", task_id))  
    
    try:
        pxyz_atom_count, pxyz_atom_types = parse_xyz(pxyz)
    except Exception as e:
        print(format_error(10002, f"Error parsing xyz files: {e}", task_id))

    if rxyz_atom_count != pxyz_atom_count:
        print(format_error(10003, "The number of atoms in the reactant and product do not match.", task_id))

    if rxyz_atom_types != pxyz_atom_types:
        print(format_error(10004, "The atom types in the reactant and product do not match.", task_id))


def input_split(r_path,p_path,batch_size=50):
    """
    Split zip of reactants and products into suitable batches
    """
    import zipfile

    # extract xyz files from a zip file
    extract_dir = os.path.join(os.getcwd(),'inp_xyzs')
    with zipfile.ZipFile(r_path, 'r') as zip_ref: zip_ref.extractall(extract_dir)
    with zipfile.ZipFile(p_path, 'r') as zip_ref: zip_ref.extractall(extract_dir)

    # create list pf xyz files    
    rxyz_files = sorted(os.listdir(os.path.join(extract_dir,r_path.split('/')[-1].split('.zip')[0])))
    pxyz_files = sorted(os.listdir(os.path.join(extract_dir,p_path.split('/')[-1].split('.zip')[0])))

    # make sure r and p lists have matched name
    rxyzs, pxyzs = [], []
    for rxyz in rxyz_files:
        name = rxyz.split('-r.xyz')[0]
        if name+'-p.xyz' in pxyz_files:
            rxyzs.append(rxyz)
            pxyzs.append(name+'-p.xyz')
        else:
            print(f'Reaction {name} does not have matched reactant and product input xyz files, skip this reaction...')

    # bundle jobs to batches
    num_folds = (len(rxyzs) + batch_size - 1) // batch_size

    # Create empty folds            
    all_rxyzs = [[] for _ in range(num_folds)]
    all_pxyzs = [[] for _ in range(num_folds)]

    for ind in range(len(rxyzs)):
        fold_index = ind % num_folds
        all_rxyzs[fold_index].append(os.path.join(extract_dir,r_path.split('/')[-1].split('.zip')[0],rxyzs[ind]))
        all_pxyzs[fold_index].append(os.path.join(extract_dir,p_path.split('/')[-1].split('.zip')[0],pxyzs[ind]))

    return all_rxyzs, all_pxyzs

def parse_rxn_xyzs(rxyzs, pxyzs, device):
    """
    Function used to parse a xyz file and convert it into input entry format
    """
    assert len(rxyzs) == len(pxyzs)
    nrxns = len(rxyzs)
    size, mask, charge = [], [], []
    for count in range(nrxns):
        rmol, pmol , rmsd = pre_treatment(rxyzs[count],pxyzs[count])
        assert np.array_equal(rmol.get_atomic_numbers(), pmol.get_atomic_numbers())
        atomic_numbers = rmol.get_atomic_numbers()
        rcoords = rmol.get_positions()
        pcoords = pmol.get_positions()
        tscooors = (rcoords + pcoords) / 2
        size.append(len(atomic_numbers))
        mask += [count for _ in atomic_numbers]
        charge += [[i] for i in atomic_numbers]
        if count == 0:
            rpos = rcoords
            ppos = pcoords
            tspos = tscooors
            one_hot = onehot_convert(atomic_numbers)
        else:
            rpos = np.vstack([rpos,rcoords]) 
            ppos = np.vstack([ppos,pcoords])
            tspos = np.vstack([tspos,tscooors])
            one_hot = np.vstack([one_hot,onehot_convert(atomic_numbers)])

    # generate representations
    representation = [{'size':   torch.tensor(np.array(size), dtype=torch.int64, device=device),\
                       'pos':    torch.tensor(rpos, dtype=torch.float32, device=device),\
                       'one_hot':torch.tensor(one_hot, dtype=torch.int64, device=device),\
                       'charge': torch.tensor(np.array(charge),dtype=torch.int32, device=device),\
                       'mask':   torch.tensor(np.array(mask), dtype=torch.int64, device=device)},
                      {'size':   torch.tensor(np.array(size), dtype=torch.int64, device=device),\
                       'pos':    torch.tensor(tspos, dtype=torch.float32, device=device),\
                       'one_hot':torch.tensor(one_hot, dtype=torch.int64, device=device),\
                       'charge': torch.tensor(np.array(charge),dtype=torch.int32, device=device),\
                       'mask':   torch.tensor(np.array(mask), dtype=torch.int64, device=device)},
                      {'size':   torch.tensor(np.array(size), dtype=torch.int64, device=device),\
                       'pos':    torch.tensor(ppos, dtype=torch.float32, device=device),\
                       'one_hot':torch.tensor(one_hot, dtype=torch.int64, device=device),\
                       'charge': torch.tensor(np.array(charge),dtype=torch.int32, device=device),\
                       'mask':   torch.tensor(np.array(mask), dtype=torch.int64, device=device)}
    ]
    return representation, rmsd

def write_ts_xyz(fragments_nodes, pred_outputs, name_index, localpath='tmp'):
    '''
    Write predicted TS structure
    '''
    start_ind, end_ind = 0, 0
    for jj, natoms in enumerate(fragments_nodes[0]):
        xyzfile = f"{localpath}/{name_index[jj].split('_')[0]}_ts.xyz"
        end_ind += natoms.item()
        write_single_xyz(
            xyzfile,
            atomic_number=pred_outputs[0][start_ind: end_ind],
            coords=pred_outputs[2][start_ind: end_ind],
        )
        start_ind = end_ind
    return

def write_rxn_xyz(fragments_nodes, pred_outputs, name_index, localpath='tmp'):
    '''
    Write input reactant, predicted TS structure, input product in sequence
    '''
    start_ind, end_ind = 0, 0
    for jj, natoms in enumerate(fragments_nodes[0]):
        xyzfile = f"{localpath}/{name_index[jj].split('_')[0]}_rxn.xyz"
        end_ind += natoms.item()
        write_single_xyz(
            xyzfile,
            atomic_number=pred_outputs[0][start_ind: end_ind],
            coords=pred_outputs[1][start_ind: end_ind],
        )
        write_single_xyz(
            xyzfile,
            atomic_number=pred_outputs[0][start_ind: end_ind],
            coords=pred_outputs[2][start_ind: end_ind],
            write_type="a"
        )
        write_single_xyz(
            xyzfile,
            atomic_number=pred_outputs[0][start_ind: end_ind],
            coords=pred_outputs[3][start_ind: end_ind],
            write_type="a"
        )
        start_ind = end_ind
    return

def write_single_xyz(xyzfile, atomic_number, coords, write_type = 'w'):
    C2A = {
        1: "H",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
    }
    with open(xyzfile, write_type) as fo:
        fo.write(str(len(atomic_number)) + "\n\n")
        for i in range(len(atomic_number)):
            ele = C2A[atomic_number[i].long().item()]
            x = coords[i].cpu().numpy()
            _x = " ".join([str(__x) for __x in x])
            fo.write(f"{ele} {_x}\n")
    return

def make_pred(representations, device, opt):
    """
    Call React-OT to generate TSs
    """
    model = SBModule.load_from_checkpoint(
        checkpoint_path=opt.checkpoint_path,
        map_location=device,
    )
    model = model.eval()
    model = model.to(device)

    # set config
    model.training_config["use_sampler"] = False
    model.training_config["swapping_react_prod"] = False
    model.nfe = opt.nfe
    model.ddpm.opt = opt
    
    # precoess input representations
    n_samples = representations[0]["size"].size(0)
    fragments_nodes = [repre["size"] for repre in representations]
    conditions = torch.tensor([[0] for _ in range(n_samples)], device=device)
    
    # run react-ot
    r_pos, ts_pos, p_pos, x0_size, x0_other, rmsds = model.eval_sample_batch(
            (representations, conditions),
            return_all=True,
        )

    return fragments_nodes, (x0_other[:, -1].unsqueeze(1), r_pos, ts_pos, p_pos)

def pred_ts(rxyz, pxyz, opt, output_path):
    """
    Apply React-OT to provide a TS structure based on input R and P
    input:
    rxyzs: a list of reactant xyz files
    pxyzs: a list of product xyz files
    opt: parameters
    output_path: path to save the output
    """

    # find device
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    # check input type, if input type is xyz, run a single prediction,
    # if input type is zip, unzip the folder and bundle xyz files into batches
    if '.xyz' in rxyz:
        all_rxyzs = [[rxyz]]
        all_pxyzs = [[pxyz]]
    else:
        all_rxyzs, all_pxyzs = input_split(rxyz, pxyz, batch_size=opt.batch_size)

    for batch_id in range(len(all_rxyzs)):
        # load xyz files in batch
        rxyzs = all_rxyzs[batch_id]
        pxyzs = all_pxyzs[batch_id]
        name_index = [i.split('/')[-1].split('-r.xyz')[0] for i in rxyzs]

        # parse input rxn
        representations, rmsd = parse_rxn_xyzs(rxyzs, pxyzs, device=device)
        if rmsd > 1:
            print("WARNING: RMSD is too large, please double check the input xyz files")

        # make predictions
        fragments_nodes, outputs = make_pred(representations, device, opt)
        # write down ts xyz
        write_ts_xyz(
            fragments_nodes, 
            outputs,
            name_index,
            localpath=output_path,
        )

        # write down rxn xyz
        write_rxn_xyz(
            fragments_nodes, 
            outputs,
            name_index,
            localpath=output_path,
        )
    
    return True  # Return True if the function runs successfully

