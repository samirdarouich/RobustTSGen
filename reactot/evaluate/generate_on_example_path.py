from typing import List
import time
import os
import numpy as np
import torch
import pickle
import argparse
from uuid import uuid4

from torch.utils.data import DataLoader

from reactot.trainer.pl_trainer import DDPMModule
from reactot.dataset.transition1x import ProcessedTS1x
from reactot.analyze.rmsd import batch_rmsd
from reactot.evaluate.utils import (
    set_new_schedule,
    inplaint_batch,
    samples_to_pos_charge,
)
from reactot.utils.sampling_tools import (
    assemble_sample_inputs,
    write_single_xyz,
    write_tmp_xyz,
)
from reactot.diffusion._normalizer import FEATURE_MAPPING

EV2KCALMOL = 23.06
AU2KCALMOL = 627.5


def assemble_filename(config):
    _id = str(uuid4()).split("-")[0]
    filename = f"conf-uuid-{_id}-"
    for k, v in config.items():
        filename += f"{k}-{v}_"
    filename += ".pkl"
    print(filename)
    return filename


parser = argparse.ArgumentParser(description="get training params")
parser.add_argument(
    "--bz", dest="bz", default=64, type=int, help="batch size"
)
parser.add_argument(
    "--timesteps", dest="timesteps", default=150, type=int, help="timesteps"
)
parser.add_argument(
    "--resamplings", dest="resamplings", default=15, type=int, help="resamplings"
)
parser.add_argument(
    "--jump_length", dest="jump_length", default=15, type=int, help="jump_length"
)
parser.add_argument(
    "--repeats", dest="repeats", default=2, type=int, help="repeats"
)
parser.add_argument(
    "--partition", dest="partition", default="valid", type=str, help="partition"
)
parser.add_argument(
    "--single_frag_only", dest="single_frag_only", default=0, type=int, help="single_frag_only"
)
parser.add_argument(
    "--model", dest="model", default="leftnet_2074", type=str, help="model"
)
parser.add_argument(
    "--power", dest="power", default="2", type=str, help="power"
)

args = parser.parse_args()
print("args: ", args)

name = "sample_all_12"
chosen_idx = int(name.split("_")[-1])

if not os.path.isdir(name):
    os.makedirs(name)

config = dict(
    model=args.model,
    partition=args.partition,
    timesteps=args.timesteps,
    bz=args.bz,
    resamplings=args.resamplings,
    jump_length=args.jump_length,
    repeats=args.repeats,
    max_batch=-1,
    shuffle=False,
    single_frag_only=args.single_frag_only,
    noise_schedule="polynomial_" + args.power,
)

print("loading ddpm trainer...")
device = torch.device("cuda")
tspath = "/anfhome/crduan/diff/TSDiffusion/reactot/trainer/checkpoint/TSDiffusion-TS1x"
checkpoints = {
    "leftnet_2074": f"{tspath}-All/leftnet-8-70b75beeaac1/ddpm-epoch=2074-val-totloss=531.18.ckpt",
}
ddpm_trainer = DDPMModule.load_from_checkpoint(
    checkpoint_path=checkpoints[config["model"]],
    map_location=device,
)
ddpm_trainer = set_new_schedule(
    ddpm_trainer,
    timesteps=config["timesteps"],
    noise_schedule=config["noise_schedule"]
)

dataset = ProcessedTS1x(
    npz_path="../data/transition1x/valid.pkl",
    center=True,
    pad_fragments=0,
    device="cuda",
    zero_charge=False,
    remove_h=False,
    single_frag_only=False,
    swapping_react_prod=False,
    use_by_ind=True,
    # confidence_model=True,
)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=dataset.collate_fn
)

speices = ["reactant", "transition_state", "product"]
keys = ['num_atoms', 'charges', 'positions']
if not os.path.isfile(f"../data/transition1x/examples/{name}.pkl"):
    data = {}
    for s in speices:
        data[s] = {}
        for k in keys:
            data[s][k] = []
    for s in ["target", "rmsd", "single_fragment"]:
        data[s] = []
else:
    data = pickle.load(open(f"../data/transition1x/examples/{name}.pkl", "rb"))

print("sampling...")
n_samples = 128
ex_ind = 0

for batch_idx, batch in enumerate(loader):
    if batch_idx != chosen_idx:
        continue

    representations, res = batch
    duplicates = n_samples
    representations_duplicated = []
    for ii, repr in enumerate(representations):
        tmp = {}
        for k, v in repr.items():
            # print(ii, v)
            if not k == "mask":
                tmp[k] = torch.cat([v] * duplicates)
            else:
                tmp[k] = torch.arange(duplicates).repeat_interleave(repr["size"].item())
        representations_duplicated.append(tmp)

    xh_fixed = [
        torch.cat(
            [repre[feature_type] for feature_type in FEATURE_MAPPING],
            dim=1,
        )
        for repre in representations_duplicated
    ]
    n_samples = representations_duplicated[0]["size"].size(0)
    fragments_nodes = [
        repre["size"] for repre in representations_duplicated
    ]
    conditions = torch.tensor([[0] for _ in range(duplicates)], device=device)
    write_tmp_xyz(fragments_nodes, xh_fixed, idx=[0, 1, 2], prefix="sample", localpath=name, ex_ind=ex_ind)

    out_samples, out_masks = ddpm_trainer.ddpm.inpaint(
        n_samples=n_samples,
        fragments_nodes=fragments_nodes,
        conditions=conditions,
        return_frames=1,
        resamplings=config["resamplings"],
        jump_length=config["jump_length"],
        timesteps=None,
        xh_fixed=xh_fixed,
        frag_fixed=[0, 2],
    )

out_samples = out_samples[0]
write_tmp_xyz(fragments_nodes, out_samples, idx=[0, 1, 2], localpath=name, ex_ind=ex_ind)
pos, z, natoms = samples_to_pos_charge(out_samples, fragments_nodes)
for s in speices:
    data[s]["positions"] += pos[s]
    data[s]["charges"] += z
    data[s]["num_atoms"] += natoms
data["rmsd"] += [-1] * n_samples
data["target"] += [-1] * n_samples
data["single_fragment"] += [0] * n_samples

with open(f"../data/transition1x/examples/{name}.pkl", "wb") as fo:
    pickle.dump(data, fo)
