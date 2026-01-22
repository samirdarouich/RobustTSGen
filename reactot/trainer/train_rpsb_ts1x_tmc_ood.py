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

# version = "vanilla_tmc_rh_3_pct"
# version = "finetune_tmc_rh_3_pct"
version = "finetune_crest_tmc_ood_small_new_tmc_rh_3_pct"

# version = "vanilla_pt_act_25_pct"
# version = "finetune_pt_act_25_pct"
# version = "finetune_crest_tmc_ood_small_new_pt_act_25_pct"


project = "MIT"
subfolder = "crest_tmc_finetune"
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

if not leftnet_config["sum_aggr"]:
    version += "_mean_aggr"
    
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

device = "cuda"
T_0 = 10
T_mult = 1

if "tmc_rh" in version:
    if "3_pct" in version:
        datadir = "../data/transition1x_tmc/rh_cata/split_3pct" # 50 samples
elif "pt_act" in version:
    if "25_pct" in version:
        datadir = "../data/transition1x_tmc/pt_cata/split_25pct" # 50 samples

training_config = dict(
    datadir=datadir,
    remove_h=False,
    bz=1, #! only finetuning with 50 structures, so batch size 1
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
    position_key="positions",
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

#! Change number of node features for different embedding style

# Provide atom mapping for original one-hot, even though its not used
atom_mapping = {i: i for i in range(100)}

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
    elif "finetune_tmc":
        checkpoint_path = f"{main_path}/leftnet-finetune_no_tmc_100_pct_mean_aggr_z_196_embedding-fdb525d67cf7/last.ckpt" #! T1x-TMC finetuned
    else:
        checkpoint_path = f"{main_path}/leftnet-pretrained_mean_aggr_z_196_embedding-f2bdf8200857/last.ckpt"
else:
    raise ValueError("No model trained with sum aggregation.")

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
ts_guess: bool = None
# mapping_initial: str = "GUESS"  #! USE IDPP
# ts_guess: bool = "idpp_prior"  #! USE IDPP
nfe: int = 10
ot_ode: bool = True
sigma: float = 0.

if ts_guess is not None:
    version += f"_{ts_guess}"

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
    validate_full_rmsd_every_epochs=100,
    feature_mapping=feature_mapping,
    z_embedding_dim=z_embedding_dim,
)
ddpm.ddpm.opt = opt  # heck for the new optimizer

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
    dirpath=ckpt_path,
    filename="sb-{epoch:03d}",
    save_top_k=1,          # keep only the best checkpoint
    every_n_epochs=50,      # save every 50 epochs
    save_last=True,         # also save 'last.ckpt'
    save_on_train_epoch_end=True
)
lr_monitor = LearningRateMonitor(logging_interval='step')
callbacks = [checkpoint_callback, TQDMProgressBar(), lr_monitor]

# strategy = None
# devices = [0]
# strategy = DDPStrategy(find_unused_parameters=True)
# if strategy is not None:
#     devices = list(range(torch.cuda.device_count()))
# if len(devices) == 1:
#     strategy = None

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
    #limit_train_batches=200, #! why?
    limit_val_batches=10, #! why?
    replace_sampler_ddp=False,
    # resume_from_checkpoint=training_checkpoint,
    # max_time="00:10:00:00",
)

trainer.fit(ddpm)
