from typing import List

import math
import torch
from torch import Tensor
from torch_scatter import scatter_add, scatter_mean

import ase
from ase.calculators.emt import EMT
from ase.neb import NEB
from ase import Atoms


def remove_mean_batch(x, indices):
    mean = scatter_mean(x, indices, dim=0)
    x = x - mean[indices]
    return x


def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    largest_value = x.abs().max().item()
    error = scatter_add(x, node_mask, dim=0).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f"Mean is not zero, relative_error {rel_error}"


def sample_center_gravity_zero_gaussian_batch(
    size: List[int], indices: List[Tensor]
) -> Tensor:
    assert len(size) == 2
    x = torch.randn(size, device=indices[0].device)

    # This projection only works because Gaussian is rotation invariant
    # around zero and samples are independent!
    x_projected = remove_mean_batch(x, torch.cat(indices))
    return x_projected


def sum_except_batch(x, indices, dim_size):
    return scatter_add(x.sum(-1), indices, dim=0, dim_size=dim_size)


def cdf_standard_gaussian(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))


def sample_gaussian(size, device):
    x = torch.randn(size, device=device)
    return x


def num_nodes_to_batch_mask(n_samples, num_nodes, device):
    assert isinstance(num_nodes, int) or len(num_nodes) == n_samples

    if isinstance(num_nodes, torch.Tensor):
        num_nodes = num_nodes.to(device)

    sample_inds = torch.arange(n_samples, device=device)

    return torch.repeat_interleave(sample_inds, num_nodes)


def unsqueeze_xdim(z, xdim):
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]


def space_indices(num_steps, count):
    assert count <= num_steps

    if count <= 1:
        frac_stride = 1
    else:
        frac_stride = (num_steps - 1) / (count - 1)

    cur_idx = 0.0
    taken_steps = []
    for _ in range(count):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride

    return taken_steps


def idpp_guess(r_pos, p_pos, x0_size, x0_other, n_images=3, interpolate="idpp"):
    _r_pos = torch.tensor_split(
        r_pos,
        torch.cumsum(x0_size, dim=0).to("cpu")[:-1]
    )
    _p_pos = torch.tensor_split(
        p_pos,
        torch.cumsum(x0_size, dim=0).to("cpu")[:-1]
    )
    z = torch.tensor_split(
        x0_other[:, -1],
        torch.cumsum(x0_size, dim=0).to("cpu")[:-1]
    )
    z = [_z.long().cpu().numpy() for _z in z]

    ts_pos = []
    for x_r, x_p, atom_number in zip(_r_pos, _p_pos, z):
        mol_r = Atoms(
            numbers=atom_number,
            positions=x_r.cpu().numpy(),
        )
        mol_p = Atoms(
            numbers=atom_number,
            positions=x_p.cpu().numpy(),
        )

        images = [mol_r.copy()]
        for _ in range(n_images - 2):
            images.append(mol_r.copy())
        images.append(mol_p.copy())

        for image in images:
            image.calc = EMT()

        neb = NEB(images)
        if interpolate == "idpp":
            neb.idpp_interpolate(
                traj=None, log=None, fmax=1000, optimizer=ase.optimize.MDMin, mic=False, steps=0)
        elif interpolate == "linear":
            neb.interpolate('linear')
        else:
            raise ValueError("interpolate can only be idpp or linear")
        x_ts = torch.tensor(
            neb.images[n_images // 2].arrays["positions"],
            dtype=torch.float32,
        )
        ts_pos.append(x_ts)

    ts_pos = torch.concat(ts_pos).to(x0_size.device)
    return ts_pos