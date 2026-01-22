import sys
sys.path.append("/home/samirdar/projects/react-ot")

from reactot.dataset import ProcessedTS1x
from reactot.utils import (
    get_n_frag_switch
)
import os
import torch

FEATURE_MAPPING = ["pos", "one_hot", "charge"]
datadir = "data/transition1x_solvent/no_solvation_gbsa_water"
dataset = ProcessedTS1x(
    npz_path=os.path.join(datadir, "train_rpsb_all.pkl"),
    remove_h=False,
    device="cpu",
    bz=14,
    num_workers=0,
    clip_grad=True,
    gradient_clip_val=None,
    ema=True,
    ema_decay=0.999,
    swapping_react_prod=False,
    append_frag=False,
    use_by_ind=True,
    reflection=False,
    single_frag_only=False,
    position_key="positions_xtb_gbsa",  # use xtb+gbsa positions
    only_ts=False,
    conditioning_key="epsilon_xtb_gbsa",  # use solvent epsilon as condition
)


batch, conditions = dataset.collate_fn([dataset[i] for i in range(3)])

masks = [repre["mask"] for repre in batch]
combined_mask = torch.cat(masks)

condition_mask = torch.cat([ mask+i*(max(mask)+1) for i,mask in enumerate(masks)])

cond = torch.concat(
    [
        conditions[ii].clone()
        for ii in range(3)
    ],
    dim=0,
)


pos = torch.concat(
            [b["pos"].clone() for b in batch],
            dim=0,
        )


h_condition = cond[condition_mask]
breakpoint()
h = torch.cat([pos, h_condition], dim=1)

print(cond.shape)
