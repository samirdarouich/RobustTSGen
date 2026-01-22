
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.autograd import grad
from torch_scatter import scatter_sum, scatter_mean
from torch_geometric.data import Data

from reactot.model import EGNN
from reactot.model.core import GatedMLP
from reactot.utils import (
    get_subgraph_mask,
    get_n_frag_switch,
    get_mask_for_frag,
    get_edges_index,
)
from ._base import BaseDynamics


FEATURE_MAPPING = ["pos", "one_hot", "charges"]


def remove_mean_batch(x, indices):
    mean = scatter_mean(x, indices, dim=0)
    x = x - mean[indices]
    return x


class Potential(BaseDynamics):
    def __init__(
        self,
        model_config: Dict,
        fragment_names: List[str],
        node_nfs: List[int],
        edge_nf: int,
        condition_nf: int = 0,
        pos_dim: int = 3,
        edge_cutoff: Optional[float] = None,
        model: nn.Module = EGNN,
        device: torch.device = torch.device("cuda"),
        enforce_same_encoding: Optional[List] = None,
        source: Optional[Dict] = None,
        timesteps: int = 5000,
        condition_time: bool = True,
        **kwargs,
    ) -> None:
        r"""Confindence score for generated samples.

        Args:
            model_config (Dict): config for the equivariant model.
            fragment_names (List[str]): list of names for fragments
            node_nfs (List[int]): list of number of input node attributues.
            edge_nf (int): number of input edge attributes.
            condition_nf (int): number of attributes for conditional generation.
            Defaults to 0.
            pos_dim (int): dimension for position vector. Defaults to 3.
            update_pocket_coords (bool): whether to update positions of everything.
                Defaults to True.
            condition_time (bool): whether to condition on time. Defaults to True.
            edge_cutoff (Optional[float]): cutoff for building intra-fragment edges.
                Defaults to None.
            model (Optional[nn.Module]): Module for equivariant model. Defaults to None.
        """
        model_config.update({"for_conf": False, "ff": True})
        update_pocket_coords = True
        super().__init__(
            model_config,
            fragment_names,
            node_nfs,
            edge_nf,
            condition_nf,
            pos_dim,
            update_pocket_coords,
            condition_time,
            edge_cutoff,
            model,
            device,
            enforce_same_encoding,
            source=source,
        )

        hidden_channels = model_config["hidden_channels"]
        self.readout = GatedMLP(
            in_dim=hidden_channels,
            out_dims=[hidden_channels, hidden_channels, 1],
            activation="swish",
            bias=True,
            last_layer_no_activation=True,
        )
        self.timesteps = timesteps

    def _forward(
        self,
        xh: List[Tensor],
        edge_index: Tensor,
        t: Tensor,
        conditions: Tensor,
        n_frag_switch: Tensor,
        combined_mask: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        r"""predict confidence.

        Args:
            xh (List[Tensor]): list of concatenated tensors for pos and h
            edge_index (Tensor): [n_edge, 2]
            t (Tensor): time tensor. If dim is 1, same for all samples;
                otherwise different t for different samples
            conditions (Tensor): condition tensors
            n_frag_switch (Tensor): [n_nodes], fragment index for each nodes
            combined_mask (Tensor): [n_nodes], sample index for each node
            edge_attr (Optional[Tensor]): [n_edge, dim_edge_attribute]. Defaults to None.

        Raises:
            NotImplementedError: The fragement-position-fixed mode is not implement.

        Returns:
            Tensor: binary probability of confidence fo each graph.
        """
        pos = torch.concat(
            [_xh[:, : self.pos_dim] for _xh in xh],
            dim=0,
        )
        h = torch.concat(
            [
                self.encoders[ii](xh[ii][:, self.pos_dim :])
                for ii, name in enumerate(self.fragment_names)
            ],
            dim=0,
        )
        if self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)

        condition_dim = 0
        if self.condition_time:
            if len(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t[combined_mask]
            h = torch.cat([h, h_time], dim=1)
            condition_dim += 1

        if self.condition_nf > 0:
            h_condition = conditions[combined_mask]
            h = torch.cat([h, h_condition], dim=1)
            condition_dim += self.condition_nf

        subgraph_mask = get_subgraph_mask(edge_index, n_frag_switch)
        if self.update_pocket_coords:
            update_coords_mask = None
        else:
            raise NotImplementedError  # no need to mask pos for inpainting mode.

        node_features, forces = self.model(
            h,
            pos,
            edge_index,
            edge_attr,
            node_mask=None,
            edge_mask=None,
            update_coords_mask=update_coords_mask,
            subgraph_mask=subgraph_mask[:, None],
        )  # (n_node, n_hidden)

        node_features = self.readout(node_features)
        ae = scatter_sum(
            node_features,
            index=combined_mask,
            dim=0,
        )  # (n_system, n_hidden)
        return ae.squeeze(), forces

    def forward(
        self,
        pyg_batch: Data,
        conditions: Optional[Tensor] = None,
    ):
        masks = [pyg_batch.batch]
        combined_mask = torch.cat(masks)
        edge_index = get_edges_index(combined_mask, remove_self_edge=True)
        fragments_nodes = [pyg_batch.natoms]
        n_frag_switch = get_n_frag_switch(fragments_nodes)
        conditions = conditions or torch.zeros(pyg_batch.ae.size(0), 1, dtype=torch.long)
        conditions = conditions.to(pyg_batch.batch.device)

        pyg_batch.pos = remove_mean_batch(pyg_batch.pos, pyg_batch.batch)

        xh = [
            torch.cat(
                [pyg_batch.pos, pyg_batch.one_hot, pyg_batch.charges.view(-1, 1)],
                dim=1,
            )
        ]
        
        t = torch.randint(0, self.timesteps, size=(1,)) / self.timesteps

        ae, forces = self._forward(
            xh=xh,
            edge_index=edge_index,
            t=torch.tensor([0.]),
            conditions=conditions,
            n_frag_switch=n_frag_switch,
            combined_mask=combined_mask,
            edge_attr=None,
        )
        return ae, forces

    def _forward_autograd(
        self,
        h: List[Tensor],
        pos: Tensor,
        edge_index: Tensor,
        t: Tensor,
        conditions: Tensor,
        n_frag_switch: Tensor,
        combined_mask: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        r"""predict confidence.

        Args:
            xh (List[Tensor]): list of concatenated tensors for pos and h
            edge_index (Tensor): [n_edge, 2]
            t (Tensor): time tensor. If dim is 1, same for all samples;
                otherwise different t for different samples
            conditions (Tensor): condition tensors
            n_frag_switch (Tensor): [n_nodes], fragment index for each nodes
            combined_mask (Tensor): [n_nodes], sample index for each node
            edge_attr (Optional[Tensor]): [n_edge, dim_edge_attribute]. Defaults to None.

        Raises:
            NotImplementedError: The fragement-position-fixed mode is not implement.

        Returns:
            Tensor: binary probability of confidence fo each graph.
        """
        h = torch.concat(
            [
                self.encoders[ii](h[ii])
                for ii, name in enumerate(self.fragment_names)
            ],
            dim=0,
        )
        if self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)

        condition_dim = 0
        if self.condition_time:
            if len(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t[combined_mask]
            h = torch.cat([h, h_time], dim=1)
            condition_dim += 1

        if self.condition_nf > 0:
            h_condition = conditions[combined_mask]
            h = torch.cat([h, h_condition], dim=1)
            condition_dim += self.condition_nf

        subgraph_mask = get_subgraph_mask(edge_index, n_frag_switch)
        if self.update_pocket_coords:
            update_coords_mask = None
        else:
            raise NotImplementedError  # no need to mask pos for inpainting mode.

        node_features, forces = self.model(
            h,
            pos,
            edge_index,
            edge_attr,
            node_mask=None,
            edge_mask=None,
            update_coords_mask=update_coords_mask,
            subgraph_mask=subgraph_mask[:, None],
        )  # (n_node, n_hidden)

        node_features = self.readout(node_features)
        ae = scatter_sum(
            node_features,
            index=combined_mask,
            dim=0,
        )  # (n_system, n_hidden)
        return ae.squeeze(), forces

    @torch.enable_grad()
    def forward_autograd(
        self,
        pyg_batch: Data,
        conditions: Optional[Tensor] = None,
    ):
        masks = [pyg_batch.batch]
        combined_mask = torch.cat(masks)
        edge_index = get_edges_index(combined_mask, remove_self_edge=True)
        fragments_nodes = [pyg_batch.natoms]
        n_frag_switch = get_n_frag_switch(fragments_nodes)
        conditions = conditions or torch.zeros(pyg_batch.ae.size(0), 1, dtype=torch.long)
        conditions = conditions.to(pyg_batch.batch.device)

        pyg_batch.pos = remove_mean_batch(pyg_batch.pos, pyg_batch.batch)
        pyg_batch.pos.requires_grad_(True)

        h = [
            torch.cat(
                [pyg_batch.one_hot, pyg_batch.charges.view(-1, 1)],
                dim=1,
            ).float()
        ]

        t = torch.randint(0, self.timesteps, size=(1,)) / self.timesteps
        
        ae, forces = self._forward_autograd(
            h=h,
            pos=pyg_batch.pos,
            edge_index=edge_index,
            t=torch.tensor([0.]),
            conditions=conditions,
            n_frag_switch=n_frag_switch,
            combined_mask=combined_mask,
            edge_attr=None,
        )
        forces = -grad(
            torch.sum(ae),
            pyg_batch.pos,
            create_graph=self.training,
        )[0]
        return ae, forces
