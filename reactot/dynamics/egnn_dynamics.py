from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torch_scatter import scatter_mean
from reactot.model import EGNN
from reactot.utils._graph_tools import get_subgraph_mask
from reactot.model.util_funcs import radius_graph_pbc, get_pbc_distances, get_max_neighbors_mask
from ._base import BaseDynamics


class EGNNDynamics(BaseDynamics):
    def __init__(
        self,
        model_config: Dict,
        fragment_names: List[str],
        node_nfs: List[int],
        edge_nf: int,
        condition_nf: int = 0,
        pos_dim: int = 3,
        update_pocket_coords: bool = True,
        condition_time: bool = True,
        edge_cutoff: Optional[float] = None,
        model: nn.Module = EGNN,
        device: torch.device = torch.device("cuda"),
        enforce_same_encoding: Optional[List] = None,
        source: Optional[Dict] = None,
        fixed_idx: Optional[List] = None,
        pbc: bool = False,
        allow_nan_vel: bool = False,
    ) -> None:
        r"""Base dynamics class set up for denoising process.

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
        self.fixed_idx = fixed_idx or []
        self.pbc = pbc
        self.allow_nan_vel = allow_nan_vel

    def forward(
        self,
        xh: List[Tensor],
        edge_index: Tensor,
        t: Tensor,
        conditions: List[Tensor], #! IMPORTANT CHANGE COMPARED TO ORIGINAL CODE. PROVIDE PER FRAGMENT A CONDITION
        n_frag_switch: Tensor,
        combined_mask: Tensor,
        condition_mask: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        natoms: Optional[Tensor] = None,
        pbc: bool = False,
    ) -> Tuple[List[Tensor], Tensor]:
        r"""predict noise /mu.

        Args:
            xh (List[Tensor]): list of concatenated tensors for pos and h
            edge_index (Tensor): [n_edge, 2]
            t (Tensor): time tensor. If dim is 1, same for all samples;
                otherwise different t for different samples
            conditions (Tensor): condition tensors
            n_frag_switch (Tensor): [n_nodes], fragment index for each nodes
            combined_mask (Tensor): [n_nodes], sample index for each node
            edge_attr (Optional[Tensor]): [n_edge, dim_edge_attribute]. Defaults to None.
            natoms (Optional[Tensor]): [n_sample * n_fragment]. Defaults to None.

        Raises:
            NotImplementedError: The fragement-position-fixed mode is not implement.

        Returns:
            Tuple[List[Tensor], Tensor]: updated pos-h and edge attributes
        """
        frag_index = self.compute_frag_index(n_frag_switch)
        
        pos = torch.concat(
            [_xh[:, : self.pos_dim].clone() for _xh in xh],
            dim=0,
        )
        
        h = torch.concat(
            [
                self.encoders[ii](xh[ii][:, self.pos_dim :].clone())
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
            # Conditions are provided for each fragment. So concat them.
            conditions = torch.concat(
                [
                    conditions[ii].clone()
                    for ii, name in enumerate(self.fragment_names)
                ],
                dim=0,
            )
            assert (condition_mask.max()+1) == len(conditions), f"Conditions mismatch mask: {(condition_mask.max()+1)} vs {len(conditions)}"
            
            # append condition to node feature and remove it later on again
            h_condition = conditions[condition_mask]
            h = torch.cat([h, h_condition], dim=1)
            condition_dim += self.condition_nf
            
        subgraph_mask = get_subgraph_mask(edge_index, n_frag_switch)
        #breakpoint()
        if self.update_pocket_coords:
            update_coords_mask = None
        else:
            raise NotImplementedError  # no need to mask pos for inpainting mode.
        
        distance_vectors = None
        if pbc or self.pbc:
            length_of_cell = 20
            single_cell = torch.tensor(np.eye(3) * length_of_cell).reshape(1, 3, 3)
            
            single_cell = torch.tensor([
                [6.926, 0, 0],
                [0, 6.926, 0],
                [0, 0, 6.410],
            ]).reshape(1, 3, 3).double().to(pos.device)
            
            cell = torch.vstack([
                single_cell for _ in range(natoms.size(0))
            ])
            
            pbc_edge_index, cell_offsets, neighbors = radius_graph_pbc(
                pos=pos,
                cell=cell,
                natoms=natoms,
                radius=self.model.cutoff,
                max_num_neighbors_threshold=50,
            )
            pbc_edge_index, pbc_distances, pbc_distance_vectors, offsets = get_pbc_distances(
                pos,
                pbc_edge_index,
                cell,
                cell_offsets,
                neighbors,
                out_cell_only=False,
            )
            # print("pos: ", pos)
            # print("natoms: ", natoms)
            # print("offsets: ", torch.min(cell_offsets), torch.max(cell_offsets))
            
            pbc_subgraph_mask = torch.ones_like(pbc_distances)
            
            idx_cross_fragment = torch.where(subgraph_mask == 0)[0]
            edge_index = edge_index.T[idx_cross_fragment].T
            subgraph_mask = subgraph_mask[idx_cross_fragment]

            i, j = edge_index
            distance_vectors = torch.cat([pos[i] - pos[j], pbc_distance_vectors], axis=0)
            edge_index = torch.cat([edge_index, pbc_edge_index], axis=1)
            subgraph_mask = torch.cat([subgraph_mask.float(), pbc_subgraph_mask.float()])
            
            pos = self.pbc_pos(pos, [length_of_cell] * 3)

        h_final, pos_final, edge_attr_final = self.model(
            h,
            pos,
            edge_index,
            edge_attr,
            node_mask=None,
            edge_mask=None,
            update_coords_mask=update_coords_mask,
            subgraph_mask=subgraph_mask[:, None],
            pbc=pbc,
            distance_vectors=distance_vectors,
            condition=h_condition,
        )
        vel = pos_final - pos
        if torch.any(torch.isnan(vel)):
            if self.allow_nan_vel:
                print("Warning: detected nan in pos, but letting it pass.")
            else:
                # vel = torch.randn_like(vel)
                no_atoms_per_batch = condition_mask[frag_index[0] : frag_index[1]].bincount().max()
                no_edges_per_batch = edge_index[0].bincount().max()
                raise ValueError(
                "nan detected in EGNN pos output. These are the max size of the molecules "
                "and the max number of edges in the batch: {} {}".format(no_atoms_per_batch, no_edges_per_batch)
                )
            
        if torch.any(torch.isnan(h_final)):
            raise ValueError("nan detected in EGNN scalar output.")

        h_final = h_final[:, :-condition_dim]
 
        xh_final = [
            torch.cat(
                [
                    self.remove_mean_batch(
                        vel[frag_index[ii] : frag_index[ii + 1]],
                        combined_mask[frag_index[ii] : frag_index[ii + 1]],
                    ),
                    self.decoders[ii](h_final[frag_index[ii] : frag_index[ii + 1]]),
                ],
                dim=-1,
            )
            for ii, name in enumerate(self.fragment_names)
        ]

        for ii in self.fixed_idx:
            xh_final[ii] = torch.zeros_like(
                xh_final[ii][:, :],
                device=xh_final[ii].device,
            )
        
        # xh_final = self.enpose_pbc(xh_final)

        if edge_attr_final is None or edge_attr_final.size(1) <= max(1, self.dist_dim):
            edge_attr_final = None
        else:
            edge_attr_final = self.edge_decoder(edge_attr_final)
        return xh_final, edge_attr_final

    @staticmethod
    def enpose_pbc(xh: List[Tensor], magnitude=10.0) -> List[Tensor]:
        xrange = magnitude * 2
        xh = [
            torch.remainder(_xh + magnitude, xrange) - magnitude
            for _xh in xh
        ]
        return xh
    
    @staticmethod
    def pbc_pos(pos: List[Tensor], cell=[10, 10, 10]) -> List[Tensor]:  # TODO: generalize to non-cubic
        for ii in range(3):
            pos[:, ii] = torch.fmod(
                pos[:, ii], cell[ii]
            )
            loc = torch.where(torch.abs(pos[:, ii]) > cell[ii]/2)[0]
            pos[loc, ii] += -1 * cell[ii] * pos[loc, ii].sign()
        return pos

    @staticmethod
    def compute_frag_index(n_frag_switch: Tensor) -> np.ndarray:
        counts = [
            torch.where(n_frag_switch == ii)[0].numel()
            for ii in torch.unique(n_frag_switch)
        ]
        return np.concatenate([np.array([0]), np.cumsum(counts)])

    @torch.no_grad()
    def adjust_edge_attr_on_new_eij(
        self,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_index_new: Tensor,
    ) -> Tensor:
        r"""Get ready new edge attributes (e_ij) given old {ij, e_ij} and new {ij}

        Args:
            edge_index (Tensor): ij
            edge_attr (Tensor): e_ij
            edge_index_new (Tensor): new ij

        Raises:
            ValueError: finding multiple entries for the same ij pair

        Returns:
            Tensor: new e_ij
        """
        edge_index_T = torch.transpose(edge_index, 1, 0)
        edge_index_new_T = torch.transpose(edge_index_new, 1, 0)

        edge_attr_new = []
        for _ind, ij in enumerate(edge_index_new_T):
            ind = torch.where((ij == edge_index_T).all(dim=1))[0]
            if ind.size(0) > 1:
                raise ValueError(f"ind should only be 0 or 1, getting {ind}")

            if ind.size(0) == 0:
                self.create_new_edge_attr(
                    ind_new=_ind,
                    ij_new=ij,
                    edge_index_new_T=edge_index_new_T,
                    edge_attr_new=edge_attr_new,
                    edge_attr=edge_attr,
                )
            else:
                edge_attr_new.append(edge_attr[ind.item()].detach())
        return torch.stack(edge_attr_new, dim=0)

    @staticmethod
    def init_edge_attr(sample_edge_attr):
        r"""initialize edge attributes."""
        return torch.rand_like(sample_edge_attr)

    def create_new_edge_attr(
        self,
        ind_new: Tensor,
        ij_new: Tensor,
        edge_index_new_T: Tensor,
        edge_attr_new: List[Tensor],
        edge_attr: Tensor,
    ) -> List[Tensor]:
        r"""Create new edge attrbution for ij that is not present in old connections

        Args:
            ind_new (Tensor): natural index of new ij
            ij_new (Tensor): new ij
            edge_index_new_T (Tensor): new edge indexes, [n_edge, 2]
            edge_attr_new (List[Tensor]): list of new edge attributes
            edge_attr (Tensor): old edge attributes

        Raises:
            ValueError: not ji found for ij in new indexes

        Returns:
            List[Tensor]: list of new edge attributes
        """
        ij_new_reverse = ij_new[torch.tensor([1, 0])]
        ind_new_reverse = torch.where(
            (ij_new_reverse == edge_index_new_T).all(dim=1))[0]
        print(ind_new_reverse)
        if ind_new_reverse.size(0) == 0:
            raise ValueError(f"should always find a reverse ind.")
        # print(ij_new, ind_new, ind_new_reverse)
        if ind_new_reverse.item() >= ind_new:
            edge_attr_new.append(self.init_edge_attr(edge_attr[0]))
        else:
            edge_attr_new.append(edge_attr_new[ind_new_reverse.item()])
        return edge_attr_new

    @staticmethod
    def remove_mean_batch(x, indices):
        mean = scatter_mean(x, indices, dim=0)
        x = x - mean[indices]
        return x
