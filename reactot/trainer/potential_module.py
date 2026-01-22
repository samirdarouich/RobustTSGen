from typing import Dict, List, Optional, Tuple

from pathlib import Path
import torch
from torch import nn

from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR
from pytorch_lightning import LightningModule
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, CosineSimilarity
from sklearn.metrics.pairwise import cosine_similarity

from reactot.dataset.ff_lmdb import LmdbDataset
from reactot.dynamics import Potential
from reactot.trainer._metrics import average_over_batch_metrics, pretty_print
import reactot.utils.training_tools as utils

LR_SCHEDULER = {
    "cos": CosineAnnealingWarmRestarts,
    "step": StepLR,
}


class PotentialModule(LightningModule):
    def __init__(
        self,
        model_config: Dict,
        optimizer_config: Dict,
        training_config: Dict,
        node_nfs: List[int] = [9] * 1,
        edge_nf: int = 4,
        condition_nf: int = 1,
        fragment_names: List[str] = ["struct"],
        pos_dim: int = 3,
        edge_cutoff: Optional[float] = None,
        model: nn.Module = None,
        enforce_same_encoding: Optional[List] = None,
        source: Optional[Dict] = None,
        use_autograd: bool = False,
        timesteps: int = 5000,
        condition_time: bool = True,
    ) -> None:
        super().__init__()
        self.potential = Potential(
            model_config=model_config,
            node_nfs=node_nfs,
            edge_nf=edge_nf,
            condition_nf=condition_nf,
            fragment_names=fragment_names,
            pos_dim=pos_dim,
            edge_cutoff=edge_cutoff,
            model=model,
            enforce_same_encoding=enforce_same_encoding,
            source=source,
            timesteps=timesteps,
            condition_time=condition_time,
        )

        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.training_config = training_config
        self.n_fragments = len(fragment_names)
        self.use_autograd = use_autograd

        self.clip_grad = training_config["clip_grad"]
        if self.clip_grad:
            self.gradnorm_queue = utils.Queue()
            self.gradnorm_queue.add(3000)
        self.save_hyperparameters()

        self.loss_fn = nn.MSELoss()
        self.MAEEval = MeanAbsoluteError()
        self.MAPEEval = MeanAbsolutePercentageError()
        self.cosineEval = CosineSimilarity(reduction="mean")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.potential.parameters(),
            **self.optimizer_config
        )
        if not self.training_config["lr_schedule_type"] is None:
            scheduler_func = LR_SCHEDULER[self.training_config["lr_schedule_type"]]
            scheduler = scheduler_func(
                optimizer=optimizer,
                **self.training_config["lr_schedule_config"]
            )
            return [optimizer], [scheduler]
        return optimizer

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.train_dataset = LmdbDataset(
                Path(self.training_config["datadir"], f"ff_valid.lmdb"),
                **self.training_config,
            )
            self.val_dataset = LmdbDataset(
                Path(self.training_config["datadir"], f"ff_valid.lmdb"),
                **self.training_config,
            )
            print("# of training data: ", len(self.train_dataset))
            print("# of validation data: ", len(self.val_dataset))
        elif stage == "test":
            self.test_dataset = LmdbDataset(
                Path(self.training_config["datadir"], f"ff_test.lmdb"),
                **self.training_config,
            )
        else:
            raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.training_config["bz"],
            shuffle=True,
            num_workers=self.training_config["num_workers"],
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.training_config["bz"] * 3,
            shuffle=False,
            num_workers=self.training_config["num_workers"],
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.training_config["bz"],
            shuffle=False,
            num_workers=self.training_config["num_workers"],
        )

    @torch.enable_grad()
    def compute_loss(self, batch):
        if not self.use_autograd:
            hat_ae, hat_forces = self.potential.forward(
                batch.to(self.device),
            )
        else:
            hat_ae, hat_forces = self.potential.forward_autograd(
                batch.to(self.device),
            )
        hat_ae = hat_ae.to(self.device)
        hat_forces = hat_forces.view(-1, ).to(self.device)
        ae = batch.ae.to(self.device)
        forces = batch.forces.view(-1, ).to(self.device)

        eloss = self.loss_fn(ae, hat_ae)
        floss = self.loss_fn(forces, hat_forces)
        info = {
            "MAE_E": self.MAEEval(hat_ae, ae).item(),
            "MAE_F": self.MAEEval(hat_forces, forces).item(),
            "MAPE_E": self.MAPEEval(hat_ae, ae).item(),
            "MAPE_F": self.MAPEEval(hat_forces, forces).item(),
            "MAE_Fcos": 1 - self.cosineEval(hat_forces.detach().cpu(), forces.detach().cpu()),
            "Loss_E": eloss.item(),
            "Loss_F": floss.item(),
        }

        # loss = floss * 100 + eloss
        loss = floss * 100
        return loss, info

    def training_step(self, batch, batch_idx):
        loss, info = self.compute_loss(batch)
        self.log("train-totloss", loss, rank_zero_only=True)

        for k, v in info.items():
            self.log(f"train-{k}", v, rank_zero_only=True)
        return loss

    def _shared_eval(self, batch, batch_idx, prefix, *args):
        loss, info = self.compute_loss(batch)
        info["totloss"] = loss.item()

        info_prefix = {}
        for k, v in info.items():
            info_prefix[f"{prefix}-{k}"] = v
        return info_prefix

    def validation_step(self, batch, batch_idx, *args):
        return self._shared_eval(batch, batch_idx, "val", *args)

    def test_step(self, batch, batch_idx, *args):
        return self._shared_eval(batch, batch_idx, "test", *args)

    def validation_epoch_end(self, val_step_outputs):
        val_epoch_metrics = average_over_batch_metrics(val_step_outputs)
        if self.trainer.is_global_zero:
            pretty_print(self.current_epoch, val_epoch_metrics, prefix="val")
        val_epoch_metrics.update({"epoch": self.current_epoch})
        for k, v in val_epoch_metrics.items():
            self.log(k, v, sync_dist=True)

    def configure_gradient_clipping(
        self,
        optimizer,
        optimizer_idx,
        gradient_clip_val,
        gradient_clip_algorithm
    ):

        if not self.clip_grad:
            return

        # Allow gradient norm to be 150% + 1.5 * stdev of the recent history.
        max_grad_norm = 1.5 * self.gradnorm_queue.mean() + \
            3 * self.gradnorm_queue.std()

        # Get current grad_norm
        params = [p for g in optimizer.param_groups for p in g['params']]
        grad_norm = utils.get_grad_norm(params)

        # Lightning will handle the gradient clipping
        self.clip_gradients(optimizer, gradient_clip_val=max_grad_norm,
                            gradient_clip_algorithm='norm')

        if float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.add(float(max_grad_norm))
        else:
            self.gradnorm_queue.add(float(grad_norm))

        if float(grad_norm) > max_grad_norm:
            print(f'Clipped gradient with value {grad_norm:.1f} '
                  f'while allowed {max_grad_norm:.1f}')