from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_scatter import scatter_mean

from reactot.dynamics import EGNNDynamics
from reactot.utils import (
    get_n_frag_switch,
    get_mask_for_frag,
    get_edges_index,
    get_subgraph_mask,
)

import reactot.diffusion._utils as utils
from reactot.diffusion._schedule import SBSchedule, compute_gaussian_product_coef
from reactot.diffusion._normalizer import Normalizer, FEATURE_MAPPING
from reactot.model.embedding import NuclearEmbedding
from torchdiffeq import odeint

def compute_scaled_err(x, y):
    max_y = torch.max(torch.abs(y))
    return torch.mean(torch.abs((x - y) / max_y))



class EnSB(nn.Module):
    """
    The E(n) Schrodinger Bridge Module.
    """

    def __init__(
        self,
        dynamics: EGNNDynamics,
        schdule: SBSchedule,
        normalizer: Normalizer,
        size_histogram: Optional[Dict] = None,
        loss_type: str = "l2",
        pos_only: bool = False,
        fixed_idx: Optional[List] = None,
        mapping: str = "R+P->TS",
        mapping_initial: str = "RP",
        sigma: float = 0.0,
        ts_guess: bool = False,
        idx: int = 1,
        feature_mapping: Optional[List[str]] = None,
        z_embedding_dim: int = 0,
        max_num_neighbors: Optional[int] = None,
        use_cutoff_for_edges: bool = False,
    ):
        super().__init__()
        assert loss_type in {"vlb", "l2"}

        self.dynamics = dynamics
        self.schedule = schdule
        self.normalizer = normalizer
        self.size_histogram = size_histogram
        self.loss_type = loss_type
        self.pos_only = pos_only
        self.fixed_idx = fixed_idx or []

        self.pos_dim = dynamics.pos_dim
        self.node_nfs = dynamics.node_nfs
        self.fragment_names = dynamics.fragment_names
        self.T = schdule.timesteps
        self.norm_values = normalizer.norm_values
        self.norm_biases = normalizer.norm_biases

        self.mapping = mapping
        self.mapping_initial = mapping_initial
        self.sigma = sigma
        self.ts_guess = ts_guess
        self.idx = idx
        self.max_num_neighbors = max_num_neighbors
        self.use_cutoff_for_edges = use_cutoff_for_edges

        if idx == 1:
            assert mapping.split(">")[-1] == "TS"
        elif idx == 2:
            assert mapping.split(">")[-1] == "P"
        elif idx == 0:
            assert mapping.split(">")[-1] == "R"
        else:
            pass
        
        if feature_mapping is None:
            self.feature_mapping = FEATURE_MAPPING
        else:
            self.feature_mapping = feature_mapping
        
        # embed Z with nuclear embedding
        if z_embedding_dim > 0:
            self.z_embedding = nn.Embedding(100, z_embedding_dim)
            self.nuc_embedding = NuclearEmbedding(z_embedding_dim)
        else:
            self.z_embedding = None
            self.nuc_embedding = None

    # ------ FORWARD PASS ------
    def sample_batch(
        self,
        representations,
        conditions,
        return_timesteps: bool = False,
        training: bool = False,
    ):
        # representations: List[Dict]

        def parse_features(features):
            # features.keys() = ["pos", "one_hot", "charge", "size", "mask"]
            # n_atoms = sum([sample.natom for sample in batch])
            # pos: (n_atoms, 3)
            # other: (n_atoms, 6)
            pos = features["pos"]
            size = features["size"]
            other = torch.cat([features["one_hot"], features["charge"]], dim=1).float()
            return pos, size, other

        r_pos, r_size, r_other = parse_features(representations[0]) # index 0
        t_pos, t_size, t_other = parse_features(representations[1]) # index 1
        p_pos, p_size, p_other = parse_features(representations[2]) # index 2

        if self.mapping == "R->P":
            x1 = r_pos
            x0 = p_pos
            cond = {
                "hs": torch.stack([r_other, p_other]), # (2, n_atoms, 6) TODO r_other == p_other == t_other can be pruned out?
                "r_pos": r_pos.detach(),
            }
            fragments_nodes = [r_size, p_size]
            x0_size, x0_other = p_size, p_other # for eval

        elif self.mapping == "R+P->TS":
            if self.mapping_initial == 'RP':
                # if training:
                #     factor = torch.randn(1)[0] * self.sigma + 0.5
                #     factor = 1 if factor > 1 else factor
                #     factor = 0 if factor < 0 else factor
                #     x1 = r_pos * factor + p_pos * (1 - factor)
                # else:
                #     x1 = (r_pos+p_pos) / 2
                x1 = (r_pos+p_pos) / 2
            elif self.mapping_initial == 'GUESS' and self.ts_guess:
                x1 = conditions["ts_guess"].float().to(r_pos.device)
            elif self.mapping_initial == 'R':
                x1 = r_pos
            elif self.mapping_initial == 'P':
                x1 = p_pos
            elif self.mapping_initial == 'Gaussian':
                x1 = torch.randn(r_pos.size(), device=r_pos.device)
            elif self.mapping_initial == 'Zeros':
                x1 = torch.zeros(r_pos.size(), device=r_pos.device)
            else:
                raise ValueError(f"mapping_initial {self.mapping_initial} not recgonized!")
            x0 = t_pos
            hs = torch.stack([r_other, t_other, p_other])
            cond = {
                "hs": hs, # (3, n_atoms, 6)
                "r_pos": r_pos.detach(),
                "p_pos": p_pos.detach(),
            }
            fragments_nodes = [r_size, t_size, p_size]
            x0_size, x0_other = t_size, t_other # for eval
        elif self.mapping == "R+TS->P":
            if self.mapping_initial == 'RTS':
                x1 = 2 * t_pos - r_pos
            elif self.mapping_initial == 'TS':
                x1 = t_pos
            elif self.mapping_initial == 'GUESS' and self.ts_guess:
                x1 = conditions["ts_guess"].float().to(r_pos.device)
            elif self.mapping_initial == 'R':
                x1 = r_pos
            else:
                raise ValueError(f"mapping_initial {self.mapping_initial} not recgonized!")
            x0 = p_pos
            hs = torch.stack([r_other, t_other, p_other])
            cond = {
                "hs": hs, # (3, n_atoms, 6)
                "r_pos": r_pos.detach(),
                "ts_pos": t_pos.detach(),
            }
            fragments_nodes = [r_size, t_size, p_size]
            x0_size, x0_other = p_size, p_other # for eval
        elif self.mapping == "TS+P->R":
            if self.mapping_initial == 'TSP':
                x1 = 2 * t_pos - p_pos
            elif self.mapping_initial == 'TS':
                x1 = t_pos
            elif self.mapping_initial == 'GUESS' and self.ts_guess:
                x1 = conditions["ts_guess"].float().to(r_pos.device)
            elif self.mapping_initial == 'P':
                x1 = p_pos
            else:
                raise ValueError(f"mapping_initial {self.mapping_initial} not recgonized!")
            x0 = r_pos
            hs = torch.stack([r_other, t_other, p_other])
            cond = {
                "hs": hs, # (3, n_atoms, 6)
                "p_pos": p_pos.detach(),
                "ts_pos": t_pos.detach(),
            }
            fragments_nodes = [r_size, t_size, p_size]
            x0_size, x0_other = p_size, p_other # for eval
        else:
            raise ValueError(f"mapping {self.mapping} not recgonized!")

        # if training:
            # x0 = x0 + self.sigma * torch.randn(x0.size(), device=x0.device)
            # x1 = x1 + self.sigma * torch.randn(x1.size(), device=x1.device)

        # compute edge_index, subgraph_mask
        # fragments_nodes: [R:[sample0,1,2,3], P:[sample0,1,2,3]]
        fragments_masks = [get_mask_for_frag(natm_nodes) for natm_nodes in fragments_nodes] # indices of sample in batch
        # combined_mask = torch.cat(fragments_masks)


        # edge_index = get_edges_index(combined_mask, pos,remove_self_edge=True, max_num_neighbors=self.max_num_neighbors) # each atom links to other atoms in the moleculars of fragments
        # # ts_edge_index = get_edges_index(fragments_masks[1], remove_self_edge=True)
        # n_frag_switch = get_n_frag_switch(fragments_nodes) # 0:R, 1:P
        # subgraph_mask = get_subgraph_mask(edge_index, n_frag_switch) # 1: edge from same fragment, otherwise 0
        
        
        
        # cond["edge_index"] = edge_index
        # cond["subgraph_mask"] = subgraph_mask
        cond["ts_mask"] = fragments_masks[self.idx]

        x1 = utils.remove_mean_batch(
            x1,
            cond["ts_mask"],
        ) .to(r_pos.device) # remove mean from batch in case of Gaussian

        if return_timesteps:
            # sample timestep for each sample among batch
            # timestep = torch.randint(0, self.interval, (self.batch_size,))
            timestep = torch.randint(0, self.interval, (self.batch_size // self.n_gpu_per_node,))

            # repeat each sampled timestep to match num of atoms in each sample
            timestep = torch.repeat_interleave(timestep, r_size)
            # timestep = timestep.to(self.device)
            return x0, x1, cond, timestep, x0_size, x0_other

        #! x0_other will be used in sampling process for RMSD computation
        #! the rmsd function will take the new predicted positions and keep the x0_other
        #! features for target and pred the same.
        return x0, x1, cond, x0_size, x0_other

    def q_sample(self, step, x0, x1, ot_ode=True, mask=None):
        """ Sample q(x_t | x_0, x_1), i.e. eq 11 """

        assert x0.shape == x1.shape
        device=self.schedule.mu_x0.device
        step = step.to(device)

        mu_x0 = self.schedule.mu_x0[step].to(x0.device)
        mu_x1 = self.schedule.mu_x1[step].to(x0.device)
        std_sb = self.schedule.std_sb[step].to(x0.device)

        xt = mu_x0 * x0 + mu_x1 * x1
        if not ot_ode:
            xt = xt + std_sb * torch.randn_like(xt)

        if mask is not None:
            xt = utils.remove_mean_batch(xt, mask)
        return xt

    def compute_label(self, step, x0, xt):
        """ Eq 12 """
        std_fwd = self.schedule.get_std_fwd(step, xdim=x0.shape[1:]).to(x0.device)
        label = (xt - x0) / std_fwd
        # label = (xt - x0)
        return label

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False, val=10.):
        """ Given network output, recover    x0. This should be the inverse of Eq 12 """
        std_fwd = self.schedule.get_std_fwd(step, xdim=xt.shape[1:]).to(xt.device)
        pred_x0 = xt - std_fwd * net_out
        # pred_x0 = xt - net_out
        if clip_denoise: pred_x0.clamp_(-val, val)
        return pred_x0

    def get_xh(self, representations, masks):
        # This assumes that only pos and Z are given
        if self.z_embedding is not None:
            assert len(self.feature_mapping) == 2 and "pos" in self.feature_mapping and "charge" in self.feature_mapping, "When using z_embedding, only pos and charge should be used as features"
            
            # Concatenate x, and h.
            h0 = [
                torch.cat(
                    [repre[feature_type] for feature_type in self.feature_mapping if feature_type != "pos"],
                    dim=1,
                )
                for repre in representations
            ]
        
            # assume only z as node feature
            h0_z_emb = [
                self.z_embedding(h0[ii].squeeze(-1).long()) for ii in range(len(masks))
            ]
            h0_nuc_emb = [
                self.nuc_embedding(h0[ii].long()) for ii in range(len(masks))
            ]
            h0 = [
                h0_z_emb[ii] + h0_nuc_emb[ii] for ii in range(len(masks))
            ]
            
            pos = [repre["pos"] for repre in representations]
            xh_t = [
                torch.cat(
                    [pos[ii], h0[ii]],
                    dim=1,
                )
                for ii in range(len(masks))
            ]
        else:
            # Concatenate x, and h[categorical].
            xh_t = [
                torch.cat(
                    [repre[feature_type] for feature_type in self.feature_mapping],
                    dim=1,
                )
                for repre in representations
            ]
        return xh_t
    
    def get_edge_indices(self, representations, combined_mask ):
        pos = torch.cat([ repre["pos"] for repre in representations], dim=0)
        edge_index = get_edges_index(
            combined_mask=combined_mask, 
            pos=pos, 
            edge_cutoff=self.dynamics.model.cutoff if self.use_cutoff_for_edges else None,
            remove_self_edge=True,
            max_num_neighbors=self.max_num_neighbors,
        )
        return edge_index

    def forward(
        self,
        representations: List[Dict],
        conditions: Union[Tensor, Dict],
        ot_ode: bool = True,
    ):
        r"""
        Computes the loss and NLL terms.

        #TODO: edge_attr not considered at all
        """
        num_sample = representations[0]["size"].size(0)
        device = representations[0]["pos"].device
        masks = [repre["mask"] for repre in representations]
        combined_mask = torch.cat(masks)
        edge_index = self.get_edge_indices(representations, combined_mask)
        fragments_nodes = [repr["size"] for repr in representations]
        n_frag_switch = get_n_frag_switch(fragments_nodes)

        # Normalize data, take into account volume change in x.
        representations = self.normalizer.normalize(representations)

        # Sample a timestep t for each example in batch
        # At evaluation time, loss_0 will be computed separately to decrease
        # variance in the estimator (costs two forward passes)
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(
            lowest_t, self.T, size=(num_sample, 1), device=device
        )
        t = t_int / self.T

        x0, x1, cond, x0_size, x0_other = self.sample_batch(
            representations, conditions, return_timesteps=False, training=True)

        timestep = torch.repeat_interleave(t_int, x0_size)
        timestep = self.schedule.inflate_batch_array(
            timestep, representations[0]["pos"]
        )

        xt = self.q_sample(timestep, x0, x1, ot_ode=ot_ode, mask=cond["ts_mask"])
        
        # Get xh for all fragments
        xh_t = self.get_xh(representations, masks)
        
        # Replace position of the diffused fragment by xt
        xh_t[self.idx][:, : self.pos_dim] = xt

        # --- use x1_t for everything ---
        # xh_t[0][:, : self.pos_dim] = xt
        # xh_t[2][:, : self.pos_dim] = xt

        # ---- ts_guess to R/P ---
        # xt_r = self.q_sample(timestep, cond["r_pos"], x1, ot_ode=ot_ode, mask=cond["ts_mask"])
        # xt_p = self.q_sample(timestep, cond["p_pos"], x1, ot_ode=ot_ode, mask=cond["ts_mask"])
        # xh_t[0][:, : self.pos_dim] = xt_r.to(xt.device)
        # xh_t[2][:, : self.pos_dim] = xt_p.to(xt.device)

        # Neural net prediction.
        # ! This conditions mask is needed in order to make sure that each fragment has its
        # ! own condition and its correctly broadcastest
        # in case a ts_guess is given, conditions is a dict and contains the condition
        # as well as the ts guess
        if self.ts_guess:
            cond = conditions["condition"] 
            condition_mask = torch.cat([ mask+i*(max(mask)+1) for i,mask in enumerate(masks)])
        else:
            cond = conditions
            condition_mask = torch.cat([ mask+i*(max(mask)+1) for i,mask in enumerate(masks)])

        net_eps_xh, _ = self.dynamics(
            xh=xh_t,
            edge_index=edge_index,
            t=t,
            conditions=cond,
            n_frag_switch=n_frag_switch,
            combined_mask=combined_mask,
            condition_mask=condition_mask,
            edge_attr=None,  # TODO: no edge_attr is considered now
        )

        pred = net_eps_xh[self.idx][:, : self.pos_dim]
        label = self.compute_label(timestep.squeeze(), x0, xt)

        loss = F.mse_loss(pred, label)
        scaled_err = compute_scaled_err(pred, label)
        loss_terms = {
            "loss": loss,
            "scaled_err": scaled_err,
            "pred": pred,
            "label": label,
        }
        return loss_terms

    # ------ INVERSE PASS ------
    def p_posterior(self, nprev, n, x_n, x0, ot_ode=True):
        """ Sample p(x_{nprev} | x_n, x_0), i.e. eq 4"""

        assert nprev < n
        std_n     = self.schedule.std_fwd[n]
        std_nprev = self.schedule.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

        # alpha_n = self.alphas[n]
        # mu_x0, mu_xn, var = compute_gaussian_product_coef(alpha_n)

        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not ot_ode and nprev > 0: # TODO
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)
        # print(nprev, n, not ot_ode and nprev > 0)
        return xt_prev


    def ddpm_sampling(self, steps, pred_x0_fn, x1,
                      ot_ode=False, log_steps=None, verbose=False, cog_mask=None):
        xt = x1.detach()

        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        steps = steps[::-1]

        assert steps[-1] == 0

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1) if verbose else pair_steps
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"

            pred_x0 = pred_x0_fn(xt, step)
            xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)
            if cog_mask is not None:
                xt = utils.remove_mean_batch(xt, cog_mask)

            if step in log_steps or prev_step == 0:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)

    @torch.no_grad()
    def ode_sampling(self, steps, net_out_fn, pred_x0_fn, x1, t_size,
                     log_steps=None, verbose=False, cog_mask=None):
        xt = x1.detach()

        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        steps = steps[::-1]

        assert steps[-1] == 0

        assert torch.allclose(self.schedule.betas[:-1], self.schedule.betas[1:])
        beta = self.schedule.betas[0] * self.T

        # nfe = 0
        def f(t, xt):
            # return (xt - x0) / t
            tt = t.repeat(t_size).reshape(-1, 1).to(xt)
            net_out = net_out_fn(xt, tt) # = (X_t - X_0) / sigma_t
            # nonlocal nfe
            # nfe += 1

            # sigma = torch.sqrt(beta * t)
            sigma_div_t = torch.sqrt(beta / t)
            return net_out * sigma_div_t

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='ODE sampling', total=len(steps)-1) if verbose else pair_steps
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"

            prev_t = max(1e-5, prev_step / self.T)
            t = step / self.T
            assert prev_t < t, f"{prev_t=}, {t=}"

            ode_out = odeint(f, xt, torch.tensor([t, prev_t]).to(xt),
                method=self.opt.method, atol=self.opt.atol, rtol=self.opt.rtol
            )
            xt = ode_out[-1]

            # print("step", step)

            if cog_mask is not None:
                xt = utils.remove_mean_batch(xt, cog_mask)

            if step in log_steps or prev_step == 0:
                xs.append(xt.detach().cpu())
                # pred_x0s.append(pred_x0.detach().cpu())

        # print(nfe)

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(xs)

    @torch.no_grad()
    def sample(self, x1, representations, conditions,
               clip_denoise=True, nfe=None, log_count=10, verbose=False, ot_ode=True):

        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        nfe = nfe or self.T - 1
        assert 0 < nfe < self.T == len(self.schedule.betas)
        steps = utils.space_indices(self.T, nfe + 1)

        # create log steps
        log_count = min(len(steps)-1, log_count)
        log_steps = [steps[i] for i in utils.space_indices(len(steps)-1, log_count)]
        assert log_steps[0] == 0

        # Prepare data for inference
        masks = [repre["mask"] for repre in representations]
        combined_mask = torch.cat(masks)
        edge_index = self.get_edge_indices(representations, combined_mask)
        fragments_nodes = [repr["size"] for repr in representations]
        n_frag_switch = get_n_frag_switch(fragments_nodes)

        # Get prior sample for batch x1 = (R+P)/2
        x0, x1, cond, x0_size, x0_other = self.sample_batch(
            representations, conditions, return_timesteps=False, training=True)
        
        # Get xh for all fragments
        xh_t = self.get_xh(representations, masks)

        # ! This conditions mask is needed in order to make sure that each fragment has its
        # own condition and its correctly broadcastest
        if self.ts_guess:
            _cond = conditions["condition"] 
            condition_mask = torch.cat([ mask+i*(max(mask)+1) for i,mask in enumerate(masks)])
        else:
            _cond = conditions
            condition_mask = torch.cat([ mask+i*(max(mask)+1) for i,mask in enumerate(masks)])

        def net_out_fn(xt, t):
            xh_t[self.idx][:, : self.pos_dim] = xt

            # --- use x1_t for everything ---
            # xh_t[0][:, : self.pos_dim] = xt
            # xh_t[2][:, : self.pos_dim] = xt

            # ---- ts_guess to R/P ---
            # print("cond[r_pos]: ", cond["r_pos"].shape)
            # print("x1: ", x1.shape)
            # xt_r = self.q_sample(timestep, cond["r_pos"], x1, ot_ode=ot_ode, mask=cond["ts_mask"])
            # xt_p = self.q_sample(timestep, cond["p_pos"], x1, ot_ode=ot_ode, mask=cond["ts_mask"])
            # xh_t[0][:, : self.pos_dim] = xt_r.to(xt.device)
            # xh_t[2][:, : self.pos_dim] = xt_p.to(xt.device)
            net_eps_xh, _ = self.dynamics(
                xh=xh_t,
                edge_index=edge_index,
                t=t,
                conditions=_cond,
                n_frag_switch=n_frag_switch,
                combined_mask=combined_mask,
                condition_mask=condition_mask,
                edge_attr=None,  # TODO: no edge_attr is considered now
            )
            return net_eps_xh[self.idx][:, :self.pos_dim]

        def pred_x0_fn(xt, step):
            step = torch.full(
                (representations[self.idx]["size"].size(0),),
                step,
                dtype=torch.long,
                device=xt.device,
            ).unsqueeze(1)
            t = step / self.T

            timestep = torch.repeat_interleave(step, representations[self.idx]["size"])
            timestep = self.schedule.inflate_batch_array(
                timestep, representations[0]["pos"]
            )

            out = net_out_fn(xt, t)
            return self.compute_pred_x0(timestep.squeeze(), xt, out, clip_denoise=clip_denoise)

        if self.opt.solver == "ddpm":
            xs, pred_x0 = self.ddpm_sampling(
                steps, pred_x0_fn, x1, ot_ode=ot_ode,
                log_steps=log_steps, verbose=verbose, cog_mask=representations[self.idx]["mask"],
            )
        elif self.opt.solver == "ode":
            xs, pred_x0 = self.ode_sampling(
                steps, net_out_fn, pred_x0_fn, x1, t_size=representations[self.idx]["size"].size(0),
                log_steps=log_steps, verbose=verbose, cog_mask=representations[self.idx]["mask"],
            )
        elif self.opt.solver == "ei":
            xs, pred_x0 = self.EI_sampling(
                steps, pred_x0_fn, x1, r=self.opt.order, ot_ode=ot_ode,
                log_steps=log_steps, verbose=verbose, cog_mask=representations[self.idx]["mask"],
            )

        torch.cuda.empty_cache()

        return xs, pred_x0

    ###################Exponential Integrator
    def EI_ODE(self, t, x_n, x0, idx, r, ot_ode=False):
        """ Sample p(x_{nprev} | x_n, x_0), i.e. eq 4"""
        assert ot_ode
        # assert nprev < n
        # std_n     = self.schedule.std_fwd[n]
        # std_nprev = self.schedule.std_fwd[nprev]
        # std_delta = (std_n**2 - std_nprev**2).sqrt()

        scale = 1. if self.opt.normalize else t
        fv = (x_n - x0) / scale #Time is negative

        accumulated_fx = 0
        max_order = min(idx, r)

        for jj in range(max_order+1):
            coef = self.intgral_norm[idx][jj]
            if jj==0:
                coef_fv = coef*fv
            else:
                assert jj-1>=0 and len(self.prev_fv)==max_order
                coef_fv = self.prev_fv[jj-1]*coef

            accumulated_fx += coef_fv

        # xt_prev=x_n+fv*dt
        xt_prev=x_n+accumulated_fx

        if len(self.prev_fv) < r:
            self.prev_fv.insert(0,fv)
        elif len(self.prev_fv)==0:
            pass
        else:
            self.prev_fv.pop(-1)
            self.prev_fv.insert(0,fv)

        return xt_prev


    def EI_sampling(self, steps, pred_x0_fn, x1, r,
                    ot_ode=False, log_steps=None, verbose=True, cog_mask=None):

        xt = x1.detach()

        xs = []
        pred_x0s = []

        # log_steps = log_steps or steps
        # assert steps[0] == log_steps[0] == 0

        # steps = steps[::-1]

        # pair_steps = zip(steps[1:], steps[:-1])
        # pair_steps = tqdm(pair_steps, desc='EI sampling', total=len(steps)-1) if verbose else pair_steps

        # EI specific
        # ts  = torch.Tensor(steps[:-1]) / self.T
        # dts = torch.Tensor(steps[:-1]) / self.T - torch.Tensor(steps[1:]) / self.T
        if self.opt.diz == "linear":
            ts = torch.linspace(1, 0, len(steps))
        elif self.opt.diz == "quad":
            ts = torch.linspace(1, np.sqrt(0.1), len(steps))
            ts = ts**2
        dts = ts[0:-1]-ts[1:]
        ts  = ts[0:-1]

        # Recompute steps
        # TODO(ghliu) steps should be inited in sample not here...
        steps = (ts * (self.T - 1)).round().long()
        log_idx = np.linspace(0, len(ts)-1, len(log_steps)).astype(int)

        self.intgral_norm = self.AB_fn(ts, dts, r=self.opt.order)
        self.prev_fv      = []

        for idx, (t, step) in enumerate(zip(ts, steps)):

            pred_x0 = pred_x0_fn(xt, step)
            xt = self.EI_ODE(t, xt, pred_x0, idx, r, ot_ode=ot_ode)
            if cog_mask is not None:
                xt = utils.remove_mean_batch(xt, cog_mask)

            if idx in log_idx:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)

    def monte_carlo_integral(self, fn,t0,t1,num_sample):
        ts = torch.linspace(t0,t1,num_sample, device='cuda')[:,None]
        return (fn(ts)).sum(0)*(t1-t0)/num_sample

    def extrapolate_fn(self, ts,i,j,r):
        def _fn_r(t):
            #=====time coef=========
            prod= torch.ones_like(t)
            for k in range(r+1):
                assert i-k>=0 and i-j>=0
                if k!=j:
                    prod= prod* ((t-ts[i-k])/(ts[i-j]-ts[i-k]))
            #=====time coef=========
            return prod / (t if self.opt.normalize else 1.)
        return _fn_r

    def AB_fn(self, ts, dts, r=0):
        intgral_norm = {}
        num_monte_carlo_sample = 10000
        for idx,(t,dt) in enumerate(zip(ts,dts)):
            max_order       = min(idx,r)
            intgral_norm[idx] = {}
            for jj in range(max_order+1):
                coef_fn         = self.extrapolate_fn(ts,idx,j=jj,r=max_order)
                coef            = self.monte_carlo_integral(coef_fn,t,t-dt,num_monte_carlo_sample)
                intgral_norm[idx][jj]=coef
        return intgral_norm