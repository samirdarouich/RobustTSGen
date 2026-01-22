import sys

from typing import List, Optional, Tuple
from uuid import uuid4
import torch

from oa_reactdiff.trainer.pl_trainer import DDPMModule
from reactot.trainer.pl_trainer import SBModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from reactot.trainer.ema import EMACallback
from reactot.model import LEFTNet

class OPT:
    def __init__(
        self,
        solver,
        method,
    ):
        self.solver = solver
        self.method = method
        self.atol = 1e-2
        self.rtol = 1e-2
    
opt = OPT(solver="ddpm", method="midpoint")

model_type = "leftnet"

# version = "finetune_no_tmc"
# version = "finetune_no_tmc_10_pct"
# version = "finetune_no_tmc_25_pct"
# version = "finetune_no_tmc_50_pct"
# version = "finetune_no_tmc_100_pct"

# version = "finetune_crest_tmc_small"
# version = "finetune_crest_tmc_small_10_pct"
# version = "finetune_crest_tmc_small_25_pct"
# version = "finetune_crest_tmc_small_50_pct"
# version = "finetune_crest_tmc_small_100_pct"

# version = "finetune_crest_tmc_medium"
# version = "finetune_crest_tmc_medium_10_pct_repeat"
# version = "finetune_crest_tmc_medium_10_pct_2nd_repeat"
# version = "finetune_crest_tmc_medium_25_pct"
# version = "finetune_crest_tmc_medium_50_pct_repeat"
# version = "finetune_crest_tmc_medium_100_pct"

# version = "finetune_crest_tmc_dft_medium_100_pct" #!Using 1500 DFT optimized CREST reactions for pretraining

# version = "vanilla_crest_tmc_ood_small_new_tmc_rh" #! Using 100 OOD CREST SAMPLES
version = "vanilla_crest_tmc_ood_small_new_pt_act" #! Using 100 OOD CREST SAMPLES


subfolder = "crest_tmc"

project = "MIT"
device = "cuda"
# ---EGNNDynamics---
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
    sum_aggr=False, #! change to mean aggregation to avoid overfloat for big systems!
)

if model_type == "leftnet":
    model_config = leftnet_config
    model = LEFTNet
else:
    raise KeyError("model type not implemented.")

optimizer_config = dict(
    lr=2.5e-4,
    betas=[0.9, 0.999],
    weight_decay=0,
    amsgrad=True,
)

T_0 = 10
T_mult = 1

position_key = "positions"
swap_charges_key = None
if "10_pct" in version:
    datadir="../data/transition1x_tmc/t1x_tmc/10_pct" #! only 10% of tmc included
    print("\nUsing 10% of tmc included data.\n")
elif "25_pct" in version and "tmc_ood" not in version:
    datadir="../data/transition1x_tmc/t1x_tmc/25_pct" #! only 25% of tmc included
    print("\nUsing 25% of tmc included data.\n")
elif "50_pct" in version:
    datadir="../data/transition1x_tmc/t1x_tmc/50_pct" #! only 50% of tmc included
    print("\nUsing 50% of tmc included data.\n")
elif "100_pct" in version:
    datadir="../data/transition1x_tmc/t1x_tmc/100_pct" #! all tmc data included
    print("\nUsing 100% of tmc included data.\n")
else:
    datadir="../data/transition1x_swap/transition1x_swap/no_swap" #! just plain GFN2-xTB T1x dataset
    position_key = "positions_swap"
    swap_charges_key = "atomic_numbers_swap"
    print("\nUsing no tmc included data.\n")

training_config = dict(
    datadir=datadir,
    remove_h=False,
    bz=48,
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
    position_key=position_key,
    swap_charges_key=swap_charges_key,
    only_ts=False,
    lr_schedule_type=None,
    lr_schedule_config=dict(
        gamma=0.8,
        step_size=10,
    ),  # step
    # lr_schedule_config=dict(
    #     T_0=T_0,
    #     T_mult=T_mult,
    #     eta_min=0,
    # ),  #cos
    use_sampler=False,
    sampler_config=dict(
        max_num=2800,  # This is for 16GB GPU; Scale linearly with memory
        mode="node^2",
        shuffle=True,
        ddp=False,
    ),
    device=device
)
training_data_frac = 1.0 if not training_config["reflection"] else 0.5


main_path = "checkpoint/OAReactDiff"

z_embedding_dim = 196
if not leftnet_config["sum_aggr"]:
    version += "_mean_aggr"
    # mean aggregation
    if "crest_tmc_ood" in version:
        if "tmc_ood_small_new_pt" in version:
            checkpoint_path = f"{main_path}/leftnet-pretrained_with_crest_tmc_pt_ood_small_mean_aggr_z_196_embedding-b5eea0f691aa/last.ckpt" #! 100 crest samples
        elif "tmc_ood_small_new_tmc_rh" in version:
            checkpoint_path = f"{main_path}/leftnet-pretrained_with_crest_tmc_rh_ood_small_mean_aggr_z_196_embedding-9f972fb8c191/last.ckpt" #! 100 crest samples
    elif "crest_tmc" in version:
        if "small" in version:
            checkpoint_path = f"{main_path}/leftnet-pretrained_with_crest_tmc_small_mean_aggr_z_196_embedding-4ffba61bbd26/last.ckpt" #! 500 crest samples
        elif "medium" in version:
            checkpoint_path = f"{main_path}/leftnet-pretrained_with_crest_tmc_medium_mean_aggr_z_196_embedding-aff685c1d6c0/last.ckpt" #! 2500 crest samples
        elif "dft_medium" in version:
            checkpoint_path = f"{main_path}/leftnet-pretrained_with_crest_tmc_dft_medium_mean_aggr_z_196_embedding-10ede24b5bfb/last.ckpt" #! 1500 crest samples
    else:
        checkpoint_path = f"{main_path}/leftnet-pretrained_mean_aggr_z_196_embedding-f2bdf8200857/last.ckpt"
else:
    raise ValueError("No model trained with sum aggregation.")

unique_numbers = [1, 6, 7, 8, 9, 17, 22, 30, 44, 45, 46, 47, 76, 77, 78, 79]
atom_mapping = {num: i for i, num in enumerate(unique_numbers)}

feature_mapping = ["pos", "charge"]
node_nfs: List[int] = [3+z_embedding_dim] * 3  # 3 (pos) + x (z_embedding_dim)
version += f"_z_{z_embedding_dim}_embedding"
training_config["atom_mapping"] = atom_mapping


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
scales = [1., 2., 1.] #! overscales TS loss (is the only loss taking)
fixed_idx = [0, 2]
eval_epochs = 1
save_epochs = 1

# ----Normalizer---
norm_values: Tuple = (1., 1., 1.)
norm_biases: Tuple = (0., 0., 0.)

# ---Schedule---
timesteps: int = 3000
beta_max: float = 0.3
power: float = 0.5
inv_power: float = 1
precision: float = 1e-5  # not used
noise_schedule: str = "cosine"  # not used

# ---SB---
mapping: str = "R+P->TS"
mapping_initial: str = "RP"  # RP for (r+p)/2, GUESS for guessing 
# mapping_initial: str = "GUESS"  #! USE IDPP
# ts_guess: bool = "idpp_prior"  #! USE IDPP
nfe: int = 10
ot_ode: bool = True
sigma: float = 0.
ts_guess: bool = None

norms = "_".join([str(x) for x in norm_values])
run_name = f"{model_type}-{version}-" + str(uuid4()).split("-")[-1]

print(f"\nStarting run: {run_name}\n")

## === Fine tuning from a ckpt ===

#checkpoint_path = "/mnt/beegfs/home/ac137577/projects/react-ot_new/reactdiff-pretrained.ckpt"
# pretrained en/decoder with fixed LEFTNet weights
use_pretrain: bool = True if checkpoint_path is not None else False
use_endecoder = True

source = None
if use_pretrain:
    
    print(f"\nLoading pretrained model from: '{checkpoint_path}'\n")
    
    ddpm_trainer = DDPMModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location="cpu",
    )
    
    if use_endecoder:
        source = {
            "model": ddpm_trainer.ddpm.dynamics.model.state_dict(),
            "encoders": ddpm_trainer.ddpm.dynamics.encoders.state_dict(),
            "decoders": ddpm_trainer.ddpm.dynamics.decoders.state_dict(),
        }
    else:
        source = {
            "model": ddpm_trainer.ddpm.dynamics.model.state_dict(),
            
        }
        
    if z_embedding_dim > 0:
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
            "use_endecoder": use_endecoder,
            "z_embedding_dim": z_embedding_dim,
        }
    )

seed_everything(42, workers=True)
ddpm = SBModule(
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
    mapping=mapping,
    mapping_initial=mapping_initial,
    nfe=nfe,
    beta_max=beta_max,
    ot_ode=ot_ode,
    power=power,
    inv_power=inv_power,
    sigma=sigma,
    ts_guess=ts_guess,
    validate_full_rmsd_every_epochs=50,
    feature_mapping=feature_mapping,
    z_embedding_dim=z_embedding_dim,
)
ddpm.ddpm.opt = opt 

config = model_config.copy()
config.update(optimizer_config)
config.update(training_config)
trainer = None
if trainer is None or (isinstance(trainer, Trainer) and trainer.is_global_zero):
    wandb_logger = WandbLogger(
        project=project,
        log_model=False,
        name=run_name,
    )
    try:  # Avoid errors for creating wandb instances multiple times
        wandb_logger.experiment.config.update(config)
        # wandb_logger.watch(
        #     ddpm.ddpm.dynamics, log="all", log_freq=100, log_graph=False
        # )
    except:
        pass

ckpt_path = f"checkpoint/{project}/{subfolder}/{wandb_logger.experiment.name}" #!!
earlystopping = EarlyStopping(
    monitor="val_ep_scaled_err",
    patience=2000,
    verbose=True,
    log_rank_zero_only=True,
)
checkpoint_callback = ModelCheckpoint(
    monitor="val_ep_scaled_err",
    dirpath=ckpt_path,
    save_last=True,
    filename="sb-{epoch:03d}-{val_ep_scaled_err:.4f}",
    every_n_epochs=save_epochs,
    save_top_k=1, #onnly save the best model
)
lr_monitor = LearningRateMonitor(logging_interval='step')
callbacks = [checkpoint_callback, TQDMProgressBar(), lr_monitor]

strategy = None
devices = [0]
strategy = DDPStrategy(find_unused_parameters=True)
if strategy is not None:
    devices = list(range(torch.cuda.device_count()))
if len(devices) == 1:
    strategy = None

if training_config["ema"]:
    callbacks.append(
        EMACallback(
            pl_module=ddpm,
            decay=training_config["ema_decay"])

    )

print("config: ", config)
trainer = Trainer(
    max_epochs=200,
    accelerator=device,
    deterministic=False,
    #devices=devices,
    #strategy=strategy,
    log_every_n_steps=20,
    callbacks=callbacks,
    profiler=None,
    logger=wandb_logger,
    accumulate_grad_batches=1,
    gradient_clip_val=training_config["gradient_clip_val"],
    num_sanity_val_steps=0,
    #limit_train_batches=200,
    limit_val_batches=10,
    replace_sampler_ddp=False,
    # resume_from_checkpoint=training_checkpoint,
    # max_time="00:10:00:00",
)

trainer.fit(ddpm)
