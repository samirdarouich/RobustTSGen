import argparse
import logging
import pathlib
import sys

import numpy as np
import torch
import yaml
from reactot.trainer.pl_trainer import SBModule
from rich.console import Console
from rich.logging import RichHandler

device = "cuda" if torch.cuda.is_available() else "cpu"


def setup_logger(log_dir: pathlib.Path) -> None:
    log_dir.mkdir(exist_ok=True, parents=True)

    log_file = open(log_dir / "log.txt", "w")
    file_console = Console(file=log_file, width=150)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        force=True,
        handlers=[RichHandler(), RichHandler(console=file_console)],
    )


def load_model(
    checkpoint_path="",
    datadir="./reactot/data/transition1x",
    position_key="positions",
    max_num_neighbors=None,
    use_cutoff_for_edges=False,
    **training_config_kwargs,
):
    print(f"\nLoading model from checkpoint:\n  {checkpoint_path}\n")
    model = SBModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location=device,
    )
    model = model.eval()
    model = model.to(device)
    
    # Crop edge index with cutoff for large systems to avoid overflowing of messages due to sum aggregation.
    model.ddpm.max_num_neighbors = max_num_neighbors
    if max_num_neighbors is not None:
        print(f"\nSetting max_num_neighbors to {max_num_neighbors}\n")
    
    model.ddpm.use_cutoff_for_edges = use_cutoff_for_edges
    if use_cutoff_for_edges:
        print("\nUsing cutoff for edge construction\n")
    
    
    print(
        "\nModel uses these features for encoding:\n   ",
        ", ".join(model.ddpm.feature_mapping),
        "\n",
    )

    mapping = model.ddpm.mapping
    mapping_initial = model.ddpm.mapping_initial
    ts_guess = model.ddpm.ts_guess

    print(
        f"\nModel uses mapping: '{mapping}' with initial mapping: '{mapping_initial}' "
        + (f"and TS guess: '{ts_guess}'\n" if ts_guess is not None else "\n")
    )
    model.training_config["device"] = device
    model.training_config["use_sampler"] = False
    model.training_config["swapping_react_prod"] = False
    model.training_config["datadir"] = datadir
    model.training_config["position_key"] = position_key

    print("\n")
    for k, v in training_config_kwargs.items():
        if k in model.training_config:
            print(
                f"Overriding training config:\n  {k}: {model.training_config[k]} -> {v}"
            )
        else:
            print(f"Adding new training config:\n  {k}: {v}")
        model.training_config[k] = v
    print("\n")

    model.setup(stage="fit", device=device, swapping_react_prod=False)
    return model


def main(opt):
    setup_logger(pathlib.Path(".log"))
    log = logging.getLogger(__name__)

    log.info("===== Start =====")
    log.info("Using device: %s", device)
    log.info("Command used:\n{}".format(" ".join(sys.argv)))

    if opt.training_config_kwargs.get("ext_atom_mapping", False):
        opt.training_config_kwargs.pop("ext_atom_mapping")
        # dummy atom mapping for one-hot encoding of external atoms. Will not be used
        opt.training_config_kwargs["atom_mapping"] = {i: i for i in range(100)}

    if opt.training_config_kwargs:
        for k, v in opt.training_config_kwargs.items():
            log.info(f"Additional training config: {k}={v}")
    model = load_model(
        opt.checkpoint, 
        opt.datadir, 
        opt.position_key, 
        opt.max_num_neighbors,
        opt.use_cutoff_for_edges,
        **opt.training_config_kwargs
    )

    val_loader = model.val_dataloader(bz=opt.batch_size, shuffle=False)
    model.nfe = opt.nfe
    model.ddpm.opt = opt  # hack :)

    if opt.dryrun:
        # dryrun: just first batch
        batch = next(iter(val_loader))
        r_pos, ts_pos, p_pos, x0_size, x0_other, rmsds = model.eval_sample_batch(
            batch,
            return_all=True,
        )  # 30s for nfe=100
    else:
        # full val set
        if opt.write_xyz:
            print(
                "\nSaving xyz files at:",
                f"{opt.save_path}/{opt.solver}-{opt.method}/nfe{opt.nfe}\n",
            )
            save_args_path = (
                pathlib.Path(opt.save_path)
                / f"{opt.solver}-{opt.method}/nfe{opt.nfe}/args.yaml"
            )
            save_args_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_args_path, "w") as f:
                yaml.safe_dump(vars(opt), f)
        res, rmsds = model.eval_rmsd(
            val_loader,
            write_xyz=opt.write_xyz,
            bz=opt.batch_size,
            refpath=f"{opt.save_path}/ref_ts",
            localpath=f"{opt.save_path}/{opt.solver}-{opt.method}/nfe{opt.nfe}/",
            # max_num_batch=10,
        )
        # np.savez(f"data/{opt.save}.npz", res=res, rmsds=rmsds)

    log.info(f"mean={np.mean(rmsds):.5f}, median={np.median(rmsds):.5f}, {len(rmsds)=}")
    log.info("===== End =====")


def parse_kwargs(s):
    if not s:  # catches "" or None
        return {}
    d = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        v_strip = v.strip().lower()
        if v_strip == "true":
            v_cast = True
        elif v_strip == "false":
            v_cast = False
        elif v_strip == "none":
            v_cast = None
        else:
            try:
                v_cast = int(v)
            except ValueError:
                try:
                    v_cast = float(v)
                except ValueError:
                    v_cast = v  # leave as string
        d[k] = v_cast
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument("--nfe", type=int, default=100)
    parser.add_argument("--save", type=str, default="debug")
    parser.add_argument("--dryrun", action="store_true")

    parser.add_argument("--write-xyz", action="store_true")
    parser.add_argument("--save_path", type=str, default=".")
    parser.add_argument("--solver", type=str, choices=["ddpm", "ei", "ode"])
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--datadir", type=str, default="./reactot/data/transition1x")
    parser.add_argument("--position_key", type=str, default="positions")
    parser.add_argument("--training_config_kwargs", type=parse_kwargs, default={})
    parser.add_argument("--max_num_neighbors", type=int, default=None)
    parser.add_argument("--use_cutoff_for_edges", action="store_true")
    
    # ei
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--diz", type=str, default="linear", choices=["linear", "quad"])
    parser.add_argument("--normalize", action="store_true")

    # ode
    parser.add_argument("--method", type=str, default="midpoint")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)

    opt = parser.parse_args()

    main(opt)
