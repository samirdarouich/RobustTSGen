"""Utility functions for graphs."""
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor
from torch_geometric.nn import radius_graph

def edge_set(edge_index):
    pairs = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    return set(pairs)

def get_edges_index(
    combined_mask: Tensor,
    pos: Optional[Tensor] = None,
    edge_cutoff: Optional[float] = None,
    remove_self_edge: bool = False,
    max_num_neighbors: Optional[int] = None,
) -> Tensor:
    r"""

    The fully connected graph for the fragments R, TS, P scales with O(N^2).
    Thus if R has 20 atoms (TS, and P has the same), we get a fully connected graph of
    60 atoms, which results in 60 * 59 = 3540 edges.
    
    
    Args:
        combined_mask (Tensor): Combined mask for all fragments.
            Edges are built for nodes with the same indexes in the mask.
        pos (Optional[Tensor]): 3D coordinations of nodes. Defaults to None.
        edge_cutoff (Optional[float]): cutoff for building edges within a fragment.
            Defaults to None.
        remove_self_edge (bool): whether to remove self-connecting edge (i.e., ii).
            Defaults to False.

    Returns:
        Tensor: [2, n_edges], i for node index.
    """
    if max_num_neighbors is None and edge_cutoff is None:
        adj = combined_mask[:, None] == combined_mask[None, :]
        if edge_cutoff is not None:
            adj = adj & (torch.cdist(pos, pos) <= edge_cutoff)
        if remove_self_edge:
            adj = adj.fill_diagonal_(False)
        edges = torch.stack(torch.where(adj), dim=0)
    else:
        assert pos is not None, "positions must be given."
        cutoff = 1e+50 if edge_cutoff is None else edge_cutoff
        max_num_neighbors = 10000 if max_num_neighbors is None else max_num_neighbors
        
        # Seems like the CUDA version does some weird things
        input_device = pos.device
        
        # Sort combined_mask to use radius_graph (otherwise it wont work with batch)
        sort_idx = torch.argsort(combined_mask, stable=True)
        
        j_sorted, i_sorted = radius_graph(
            pos[sort_idx].cpu(), 
            r=cutoff, 
            batch=combined_mask[sort_idx].cpu(), 
            loop=not remove_self_edge,
            max_num_neighbors=max_num_neighbors
        )
        
        # Get back to original edge index
        i = sort_idx[i_sorted.to(input_device)]
        j = sort_idx[j_sorted.to(input_device)]
        edges = torch.stack((i, j), dim=0)
        assert torch.all(combined_mask[edges[0]] == combined_mask[edges[1]])

    return edges


def get_subgraph_mask(edge_index: Tensor, n_frag_switch: Tensor) -> Tensor:
    r"""Filter out edges that have inter-fragment connections.
    Example:
    edge_index: [
        [0, 0, 1, 1, 2, 2],
        [1, 2, 0, 2, 0, 1],
        ]
    n_frag_switch: [0, 0, 1]
    -> [1, 0, 1, 0, 0, 0]

    Args:
        edge_index (Tensor): e_ij
        n_frag_switch (Tensor): fragment that a node belongs to

    Returns:
        Tensor: [n_edge], 1 for inner- and 0 for inter-fragment edge
    """
    subgraph_mask = torch.zeros(edge_index.size(1)).long()
    in_same_frag = n_frag_switch[edge_index[0]] == n_frag_switch[edge_index[1]]
    subgraph_mask[torch.where(in_same_frag)] = 1
    return subgraph_mask.to(edge_index.device)


def get_n_frag_switch(natm_list: List[Tensor]) -> Tensor:
    r"""Get the type of fragments to which each node belongs
    Example: [Tensor(1, 1), Tensor(2, 1)] -> [0, 0, 1, 1 ,1]

    Args:
        natm_list (List[Tensor]): [Tensor([number of atoms per small fragment])]

    Returns:
        Tensor: [n_nodes], type of fragment each node belongs to
    """
    shapes = [natm.shape[0] for natm in natm_list]
    assert np.std(shapes) == 0, "Tensor must be the same length for <natom_list>"
    n_frag_switch = torch.repeat_interleave(
        torch.arange(len(natm_list), device=natm_list[0].device),
        torch.tensor(
            [torch.sum(natm).item() for natm in natm_list],
            device=natm_list[0].device,
        ),
    )
    return n_frag_switch.to(natm_list[0].device)


def get_mask_for_frag(natm: Tensor) -> Tensor:
    r"""Get fragment index for each node
    Example: Tensor([2, 0, 3]) -> [0, 0, 2, 2, 2]

    Args:
        natm (Tensor): number of nodes per small fragment

    Returns:
        Tensor: [n_node], the natural index of fragment a node belongs to
    """
    return torch.repeat_interleave(
        torch.arange(natm.size(0), device=natm.device), natm
    ).to(natm.device)


def get_inner_edge_index(subgraph_mask: Tensor):
    return torch.stack(torch.where(subgraph_mask), dim=0)
