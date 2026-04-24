import sys

from typing import List, Optional, Tuple
from uuid import uuid4
import os
import shutil
import torch

from pl_trainer import DDPMModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from oa_reactdiff.trainer.ema import EMACallback
from oa_reactdiff.model import EGNN, LEFTNet


model_type = "leftnet"

# version = "pretrained_with_crest_swap_small" # 500 CREST samples
# version = "pretrained_with_crest_swap_medium" # 2500 CREST samples

version = "pretrained_with_crest_swap_ood_small" # 100 CREST samples

# version = "pretrained_with_crest_tmc_small" # 500 CREST samples
# version = "pretrained_with_crest_tmc_medium" # 2500 CREST samples

# version = "pretrained_with_crest_tmc_pt_ood_small" # 100 CREST samples
# version = "pretrained_with_crest_tmc_rh_ood_small" # 100 CREST samples

# version = "pretrained_with_crest_tmc_dft_medium" # 1500 CREST samples

project = "MIT"
# ---EGNNDynamics---
egnn_config = dict(
    in_node_nf=8,  # embedded dim before injecting to egnn
    in_edge_nf=0,
    hidden_nf=256,
    edge_hidden_nf=64,
    act_fn="swish",
    n_layers=9,
    attention=True,
    out_node_nf=None,
    tanh=True,
    coords_range=15.0,
    norm_constant=1.0,
    inv_sublayers=1,
    sin_embedding=True,
    normalization_factor=1.0,
    aggregation_method="mean",
)
leftnet_config = dict(
    pos_require_grad=False,
    cutoff=10.0,
    num_layers=6,
    hidden_channels=196,
    num_radial=96,
    in_hidden_channels=8,
    reflect_equiv=True,
    legacy=True,
    update=True,
    pos_grad=False,
    single_layer_output=True,
    object_aware=True,
)

if "swap" in version:
    leftnet_config["sum_aggr"] = True
elif "tmc" in version:
    #! change to mean aggregation to avoid overfloat for big systems!
    leftnet_config["sum_aggr"] = False

if not leftnet_config["sum_aggr"]:
    version += "_mean_aggr"
    
if model_type == "leftnet":
    model_config = leftnet_config
    model = LEFTNet
elif model_type == "egnn":
    model_config = egnn_config
    model = EGNN
else:
    raise KeyError("model type not implemented.")

optimizer_config = dict(
    lr=2.5e-4,
    # lr=1.0e-4,
    betas=[0.9, 0.999],
    weight_decay=0,
    amsgrad=True,
)

bz = 16
if "dft" in version:
    datadir = "../data/transition1x_tmc/crest_dft/"
elif "tmc_pt_ood" in version:
    datadir = "../data/transition1x_tmc/crest_ood/pt/"
    bz = 4
elif "tmc_rh_ood" in version:
    datadir = "../data/transition1x_tmc/crest_ood/rh/"
    bz = 4
elif "swap_ood" in version:
    datadir = "../data/transition1x_swap/crest_ood/"
elif "swap" in version:
    datadir = "../data/transition1x_swap/crest/"
elif "tmc" in version:
    datadir = "../data/transition1x_tmc/crest/"
    
print("\nUsing data from:", datadir, "\n")

if "small" in version:
    datadir += "small"
elif "medium" in version:
    datadir += "medium"
else:
    raise ValueError("Please specify data size in version name.")

T_0 = 200
T_mult = 2
training_config = dict(
    datadir=datadir,
    remove_h=False,
    bz=bz, # small dataset only use small batch size
    num_workers=0,
    clip_grad=True,
    gradient_clip_val=None,
    ema=False,
    ema_decay=0.999,
    swapping_react_prod=True,
    append_frag=False,
    use_by_ind=True,
    reflection=False,
    single_frag_only=False,
    only_ts=False,
    lr_schedule_type=None,
    lr_schedule_config=dict(
        gamma=0.8,
        step_size=100,
    ),  # step
)
training_data_frac = 1.0


z_embedding_dim = 196
feature_mapping = ["pos", "charge"]
node_nfs: List[int] = [3+z_embedding_dim] * 3  # 3 (pos) + x (z_embedding_dim)
version += f"_z_{z_embedding_dim}_embedding"

edge_nf: int = 0  # edge type
condition_nf: int = 1
fragment_names: List[str] = ["R", "TS", "P"]
pos_dim: int = 3
update_pocket_coords: bool = True
condition_time: bool = True
edge_cutoff: Optional[float] = None
loss_type = "l2"
pos_only = True
process_type = "TS1x"
enforce_same_encoding = None
scales = [1.0, 2.0, 1.0]
fixed_idx: Optional[List] = None
eval_epochs = 1000 # don't eval too often

# ----Normalizer---
norm_values: Tuple = (1.0, 1.0, 1.0)
norm_biases: Tuple = (0.0, 0.0, 0.0)

# ---Schedule---
noise_schedule: str = "cosine"
timesteps: int = 5000
precision: float = 1e-5

norms = "_".join([str(x) for x in norm_values])
run_name = f"{model_type}-{version}-" + str(uuid4()).split("-")[-1]

print(f"\nRun name:\n  {run_name}")

## Use pretrained model

if not leftnet_config["sum_aggr"]:
    checkpoint_path = "checkpoint/OAReactDiff/pretrained_mean_aggr_z_196_embedding-f2bdf8200857/last.ckpt"
else:
    checkpoint_path = "checkpoint/OAReactDiff/pretrained_z_196_embedding-832f68584409/last.ckpt"

#checkpoint_path = None

# ! Use pretrained model and let everything free
use_pretrain: bool = True if checkpoint_path is not None else False
use_en_decoder = True
freeze_model = False
freeze_embedding = False

source = None
if use_pretrain:
    ddpm_trainer = DDPMModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location="cpu",
    )

    print(f"\nUsing checkpoint from {checkpoint_path}")

    if use_en_decoder:
        source = {
            "model": ddpm_trainer.ddpm.dynamics.model.state_dict(),
            "encoders": ddpm_trainer.ddpm.dynamics.encoders.state_dict(),
            "decoders": ddpm_trainer.ddpm.dynamics.decoders.state_dict(),
        }
        print("Using all weights from pretrained model.\n")
    else:
        source = {"model": ddpm_trainer.ddpm.dynamics.model.state_dict()}
        print("Not using en/decoder weights from pretrained model.\n")
    
        
    if z_embedding_dim > 0 and hasattr(ddpm_trainer.ddpm, "z_embedding"):
        print("Using z embedding weights from pretrained model.\n")
        source.update(
            {
                "z_embedding": ddpm_trainer.ddpm.z_embedding.state_dict(),
                "nuc_embedding": ddpm_trainer.ddpm.nuc_embedding.state_dict(),
            }
        )
    
    training_config.update(
        {
            "checkpoint_path": checkpoint_path,
            "use_pretrain": use_pretrain,
            "use_en_decoder": use_en_decoder,
            "freeze_model": freeze_model,
        }
    )



seed_everything(42, workers=True)
ddpm = DDPMModule(
    model_config,
    optimizer_config,
    training_config,
    node_nfs,
    edge_nf,
    condition_nf,
    fragment_names,
    pos_dim,
    update_pocket_coords,
    condition_time,
    edge_cutoff,
    norm_values,
    norm_biases,
    noise_schedule,
    timesteps,
    precision,
    loss_type,
    pos_only,
    process_type,
    model,
    enforce_same_encoding,
    scales,
    source=source,
    fixed_idx=fixed_idx,
    eval_epochs=eval_epochs,
    feature_mapping=feature_mapping,
    z_embedding_dim=z_embedding_dim,
)

if freeze_model:
    for param in ddpm.ddpm.dynamics.model.parameters():
        param.requires_grad = False
    print("\nBackbone model frozen for training. Only training en/decoder.\n")

    # Count trainable and frozen parameters
    trainable_params = sum(p.numel() for p in ddpm.ddpm.dynamics.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in ddpm.ddpm.dynamics.parameters() if not p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    print(f"Frozen parameters: {frozen_params}")

if freeze_embedding:
    if ddpm.ddpm.z_embedding is not None:
        for param in ddpm.ddpm.z_embedding.parameters():
            param.requires_grad = False
        print("\nZ embedding frozen for training.\n")
        
config = model_config.copy()
config.update(optimizer_config)
config.update(training_config)
trainer = None
if trainer is None or (isinstance(trainer, Trainer) and trainer.is_global_zero):
    wandb_logger = WandbLogger(
        project=project,
        log_model=False,
        name=run_name,
        #offline=True,
    )
    try:  # Avoid errors for creating wandb instances multiple times
        wandb_logger.experiment.config.update(config)
    except:
        pass

ckpt_path = f"checkpoint/{project}/{wandb_logger.experiment.name}"
earlystopping = EarlyStopping(
    monitor="val-totloss",
    patience=2000,
    verbose=True,
    log_rank_zero_only=True,
)
checkpoint_callback = ModelCheckpoint(
    monitor="val-totloss",
    dirpath=ckpt_path,
    filename="ddpm-{epoch:03d}-{val-totloss:.2f}",
    every_n_epochs=100,
    save_top_k=1, #! only save the best model not every one
    save_last=True, #! save the last model
)
lr_monitor = LearningRateMonitor(logging_interval="step")
callbacks = [earlystopping, checkpoint_callback, TQDMProgressBar(), lr_monitor]
if training_config["ema"]:
    callbacks.append(EMACallback(decay=training_config["ema_decay"]))
print("config: ", config)

strategy = None
devices = [0]
strategy = DDPStrategy(find_unused_parameters=True)
if strategy is not None:
    devices = list(range(torch.cuda.device_count()))
if len(devices) == 1:
    strategy = None
trainer = Trainer(
    max_epochs=200,
    accelerator="gpu",
    deterministic=False,
    devices=devices,
    strategy=strategy,
    log_every_n_steps=1,
    callbacks=callbacks,
    profiler=None,
    logger=wandb_logger,
    accumulate_grad_batches=1,
    gradient_clip_val=training_config["gradient_clip_val"],
    # limit_train_batches=200,# !!!!! remove it
    limit_val_batches=10,
    # max_time="00:10:00:00",
)

trainer.fit(ddpm)
#trainer.save_checkpoint("pretrained-ts1x-diff.ckpt")
