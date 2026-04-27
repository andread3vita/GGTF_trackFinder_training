"""CGATr training script — Conformal Geometric Algebra (Cl(4,1)) variant of GATr.

Differences from upstream `train_lightning.py` / GATr:
- Algebra: Conformal GA Cl(4,1), 32-component multivectors (vs PGA Cl(3,0,1), 16
  components in `gatr_v111`). All equivariant primitives live in `src/cgatr/`.
- Single-channel CGA hit encoding:
    * VTX hits  -> CGA null point        (grade-1, `embed_point`)
    * DC  hits  -> IPNS drift circle     (grade-2, `embed_circle_ipns(wire, R)`)
  CGA puts both points and circles into the same multivector space, so the
  drift-chamber wire+radius is preserved natively (vs the original "left-right
  point pair" workaround used by upstream's CDC encoding).
- Object-condensation loss carries an extra term:
    L_var = mean_k mean_{i in track k} ||x_i - centroid_k||^2
  (within-cluster variance regularizer). It addresses the "elongated cluster"
  failure mode we observed in our embedding analyses, where the standard
  attractive `log(d^2+1)` term has weak gradients at the tail and lets clusters
  drift along the track. L_var is unweighted by beta, quadratic in distance,
  anchored at the centroid (not the alpha point), and warmed up linearly.
- Default clustering embedding dimension = 5 (driven by a PCA analysis of
  trained embeddings; see model_training/README_CGATr.md).

Single-process CLI:
    PYTHONPATH=. python src/train_cgatr_parquet.py --data_dir <dir> --gpus 0

Distributed (DDP via torchrun):
    PYTHONPATH=. torchrun --nproc_per_node=4 src/train_cgatr_parquet.py \\
        --data_dir <dir> --num_epochs 40
"""

import os
import sys
import time
import math
import copy
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
from torch_scatter import scatter_max, scatter_add, scatter_mean

from src.cgatr.nets.cgatr import CGATr
from src.cgatr.layers.attention.config import SelfAttentionConfig
from src.cgatr.layers.mlp.config import MLPConfig
from src.cgatr.interface.point import embed_point
from src.cgatr.interface.scalar import embed_scalar
from src.cgatr.interface.circle import embed_circle_ipns
from src.cgatr.interface.plane import embed_plane
from src.cgatr.interface.sphere import embed_sphere
from src.cgatr.primitives.linear import _compute_pin_equi_linear_basis
from src.cgatr.primitives.attention import _build_dist_basis
from src.cgatr.primitives.invariants import compute_inner_product_mask
from src.cgatr.primitives.dual import _DualCache
from src.dataset.parquet_dataset import IDEAParquetDataset, collate_idea_events


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average) wrapper
# ---------------------------------------------------------------------------
class EMAModel:
    """Maintains an exponential moving average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model):
        for name, param in model.state_dict().items():
            if param.is_floating_point():
                self.shadow[name].mul_(self.decay).add_(param, alpha=1.0 - self.decay)
            else:
                self.shadow[name].copy_(param)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = copy.deepcopy(state_dict)


# ---------------------------------------------------------------------------
# Model: Single-channel CGA with original-style 3D+beta output
# ---------------------------------------------------------------------------
class CGATrParquetModel(nn.Module):
    """Single-channel CGA model.

    VTX hits  -> CGA null point  (grade-1 vector,   in 32-dim MV)
    DC  hits  -> IPNS circle     (grade-2 bivector,  in 32-dim MV)
    Both live in ONE multivector channel — the core CGA premise.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bn_pos = nn.BatchNorm1d(3, momentum=0.1)
        self.bn_wire = nn.BatchNorm1d(3, momentum=0.1)
        self.bn_drift = nn.BatchNorm1d(1, momentum=0.1)

        gp_sparse = torch.load("cga_utils/cga_geometric_product.pt", weights_only=False)
        self.register_buffer("basis_gp", gp_sparse.to_dense().to(dtype=torch.float32))

        op_sparse = torch.load("cga_utils/cga_outer_product.pt", weights_only=False)
        self.register_buffer("basis_outer", op_sparse.to_dense().to(dtype=torch.float32))

        metadata = torch.load("cga_utils/cga_metadata.pt", weights_only=False)
        _DualCache.init_from_metadata(metadata, device=torch.device("cpu"))

        pin_basis = _compute_pin_equi_linear_basis(device=torch.device("cpu"), dtype=torch.float32)
        basis_q, basis_k = _build_dist_basis(device=torch.device("cpu"), dtype=torch.float32)
        basis_ip_weights = compute_inner_product_mask(self.basis_gp, device=torch.device("cpu"))

        self.cgatr = CGATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=args.hidden_mv_channels,
            in_s_channels=None,
            out_s_channels=None,
            hidden_s_channels=args.hidden_s_channels,
            num_blocks=args.num_blocks,
            attention=SelfAttentionConfig(),
            mlp=MLPConfig(),
            basis_gp=self.basis_gp,
            basis_ip_weights=basis_ip_weights,
            basis_outer=self.basis_outer,
            basis_pin=pin_basis,
            basis_q=basis_q,
            basis_k=basis_k,
        )

        self.embed_dim = args.embed_dim
        self.clustering = nn.Linear(32, args.embed_dim, bias=False)
        if getattr(args, "beta_mlp", False):
            self.beta = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
        else:
            self.beta = nn.Linear(32, 1)

    def forward(self, features, seq_lens):
        from xformers.ops.fmha import BlockDiagonalMask

        pos = features[:, :3]
        hit_type = features[:, 3:4]
        pos_normed = self.bn_pos(pos)

        is_vtx = (hit_type.squeeze(-1) == 0)
        is_dc = (hit_type.squeeze(-1) == 1)

        mv = torch.zeros(features.shape[0], 32, device=features.device, dtype=features.dtype)

        if is_vtx.any():
            mv[is_vtx] = embed_point(pos_normed[is_vtx])

        if is_dc.any():
            dc = features[is_dc]
            wire_normed = self.bn_wire(dc[:, 4:7])
            drift_normed = self.bn_drift(dc[:, 7:8]).squeeze(-1)
            wire_pos = wire_normed
            cos_s = torch.cos(dc[:, 9])
            sin_s = torch.sin(dc[:, 9])
            cos_a = torch.cos(dc[:, 8])
            sin_a = torch.sin(dc[:, 8])
            wire_dir = torch.stack([sin_s * cos_a, sin_s * sin_a, cos_s], dim=-1)
            wire_dir = wire_dir / (torch.norm(wire_dir, dim=-1, keepdim=True) + 1e-8)
            mv[is_dc] = embed_circle_ipns(wire_pos, wire_dir, drift_normed, self.basis_outer)

        mv = mv + embed_scalar(hit_type)

        if self.args.normalize_mv_inputs:
            mv_norm = torch.norm(mv, dim=-1, keepdim=True).clamp(min=1e-6)
            mv = mv / mv_norm

        mv = mv.unsqueeze(1)  # (N, 1, 32) — single channel

        mask = BlockDiagonalMask.from_seqlens(seq_lens)
        out_mv, _ = self.cgatr(mv, scalars=None, attention_mask=mask)
        out = out_mv[:, 0, :]
        return torch.cat([self.clustering(out), self.beta(out)], dim=1)


# ---------------------------------------------------------------------------
# Object condensation loss — torch_scatter port of hgcalimplementation
# (no DGL dependency)
# ---------------------------------------------------------------------------
def object_condensation_loss(
    coords, beta, mc_index, batch,
    noise_index=0, qmin=0.1,
    attr_weight=1.0, repul_weight=1.0, fill_loss_weight=0.0,
    use_average_cc_pos=0.0, s_B=1.0,
    beta_suppress_weight=0.0,
    var_weight=0.0,
    return_components=False,
):
    """Object condensation loss (hgcalimplementation-style) using torch_scatter.

    Matches the original calc_LV_Lbeta semantics:
    - Attraction: signal hits pulled toward their own condensation point.
    - Repulsion: ALL hits (incl. noise) pushed away from non-own-cluster objects.
    - Repulsion computed per-event to control memory.
    - Beta suppression: penalizes high beta for non-alpha signal hits.

    Additions on top of the OC loss as used in upstream:
    - Within-cluster variance regularizer
          L_var = mean_k mean_i ||x_i - mu_k||^2
      where mu_k is the mean of signal-hit coords assigned to track k. This
      directly shrinks the tail of each track cluster (an elongation pattern
      we observed in embedding analyses, caused by the `log(d^2+1)` attractive
      term + beta-weighted charge having weak gradients at the cluster edges).
      Controlled by `var_weight`; if 0, the term is ignored and the loss
      reduces to the standard OC formulation.
    - If `return_components=True`, returns (total, dict of components) for
      logging / diagnostics.
    """
    device = coords.device
    beta = torch.nan_to_num(beta, nan=0.0)

    is_noise = mc_index == noise_index
    is_sig = ~is_noise

    n_hits = coords.shape[0]
    n_hits_sig = is_sig.sum().item()
    if n_hits_sig < 4:
        return torch.tensor(0.0, device=device, requires_grad=True)

    sig_coords = coords[is_sig]
    sig_beta = beta[is_sig]
    sig_mc = mc_index[is_sig]
    sig_batch = batch[is_sig]

    # Per-event reincrementalization of signal labels -> contiguous 0..K_e-1
    object_index = torch.empty_like(sig_mc)
    n_objects_per_event_list = []
    unique_events = sig_batch.unique()
    for evt in unique_events:
        evt_mask = sig_batch == evt
        _, inv = sig_mc[evt_mask].unique(return_inverse=True)
        object_index[evt_mask] = inv
        n_objects_per_event_list.append(inv.max().item() + 1)

    n_objects_per_event = torch.tensor(n_objects_per_event_list, device=device, dtype=torch.long)

    # Make object_index globally unique across events
    offsets = torch.zeros_like(n_objects_per_event)
    offsets[1:] = n_objects_per_event[:-1].cumsum(dim=0)
    _, event_remap = sig_batch.unique(return_inverse=True)
    object_index = object_index + offsets[event_remap]

    n_objects = n_objects_per_event.sum().item()
    if n_objects < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # q for ALL hits (repulsion uses noise hits too, matching original)
    q_all = (beta.clip(0.0, 1 - 1e-4).arctanh() / 1.01) ** 2 + qmin
    q_sig = q_all[is_sig]

    # Alpha points (condensation points)
    q_alpha, index_alpha = scatter_max(q_sig, object_index)
    x_alpha = sig_coords[index_alpha]
    beta_alpha = sig_beta[index_alpha]

    # --- Attractive potential (signal hits only, per-hit) ---
    e1 = torch.exp(torch.tensor(1.0, device=device))
    d_sq_own = ((sig_coords - x_alpha[object_index]) ** 2).sum(dim=1)
    norms_att = torch.log(e1 * d_sq_own / 2 + 1)
    V_att_per_hit = q_sig * q_alpha[object_index] * norms_att

    V_att_per_obj = scatter_add(V_att_per_hit, object_index)
    n_hits_per_obj = scatter_add(torch.ones(n_hits_sig, device=device), object_index)
    V_att_per_obj = V_att_per_obj / (n_hits_per_obj + 1e-3)
    L_V_att = V_att_per_obj.mean()

    # --- Within-cluster variance regularizer ---
    # For each track k, pull every assigned hit toward the track centroid with a
    # quadratic penalty. Unlike the attractive term it is (a) unweighted by beta
    # so low-beta tail hits still feel the pull, (b) quadratic in distance so
    # gradients grow at the tail, (c) anchored at the centroid not the alpha
    # point so it is shift-invariant. Controlled by `var_weight`.
    x_centroid = scatter_mean(sig_coords, object_index, dim=0)
    d_sq_centroid = ((sig_coords - x_centroid[object_index]) ** 2).sum(dim=1)
    L_var_per_obj = scatter_mean(d_sq_centroid, object_index)
    L_var = L_var_per_obj.mean()

    # --- Repulsive potential (per-event, ALL hits incl. noise, matching original) ---
    # Build global object assignment: signal hits get their object_index, noise gets -1
    all_object_index = torch.full((n_hits,), -1, device=device, dtype=torch.long)
    all_object_index[is_sig] = object_index

    rep_sum = torch.tensor(0.0, device=device)
    rep_n_objects = 0
    obj_offset = 0

    for i, evt_val in enumerate(unique_events):
        n_evt_obj = n_objects_per_event[i].item()
        if n_evt_obj < 2:
            obj_offset += n_evt_obj
            continue

        evt_mask = batch == evt_val
        evt_coords = coords[evt_mask]
        evt_q = q_all[evt_mask]
        evt_obj = all_object_index[evt_mask]

        evt_x_alpha = x_alpha[obj_offset:obj_offset + n_evt_obj]
        evt_q_alpha = q_alpha[obj_offset:obj_offset + n_evt_obj]

        d_sq = ((evt_coords.unsqueeze(1) - evt_x_alpha.unsqueeze(0)) ** 2).sum(-1)
        exp_rep = torch.exp(-d_sq / 2)

        # M_inv: 1 for non-own-cluster pairs. Noise hits (obj=-1) repel from ALL objects.
        local_obj = evt_obj.clone()
        has_obj = local_obj >= 0
        local_obj[has_obj] -= obj_offset
        own_mask = torch.zeros(evt_coords.shape[0], n_evt_obj, device=device)
        if has_obj.any():
            own_mask[has_obj] = torch.nn.functional.one_hot(
                local_obj[has_obj], num_classes=n_evt_obj
            ).float()
        M_inv = 1.0 - own_mask

        V_rep = evt_q.unsqueeze(1) * evt_q_alpha.unsqueeze(0) * exp_rep * M_inv
        V_rep_per_obj = V_rep.sum(dim=0)
        n_rep_terms = M_inv.sum(dim=0).clamp(min=1.0)
        V_rep_per_obj = V_rep_per_obj / n_rep_terms

        rep_sum = rep_sum + V_rep_per_obj.sum()
        rep_n_objects += n_evt_obj
        obj_offset += n_evt_obj

    L_V_rep = rep_sum / max(rep_n_objects, 1)
    L_V = attr_weight * L_V_att + repul_weight * L_V_rep

    # --- L_beta signal ---
    beta_sum_per_obj = scatter_add(sig_beta, object_index)
    L_beta_sig = torch.mean(1 - beta_alpha + 1 - torch.clip(beta_sum_per_obj, 0, 1))

    # --- L_beta noise (matches original: .sum() / batch_size) ---
    batch_size = batch.unique().numel()
    L_beta_noise = torch.tensor(0.0, device=device)
    if is_noise.any():
        noise_beta = beta[is_noise]
        noise_batch = batch[is_noise]
        _, noise_evt_remap = noise_batch.unique(return_inverse=True)
        n_noise_per_evt = scatter_add(
            torch.ones_like(noise_evt_remap, dtype=torch.float), noise_evt_remap
        ).clamp(min=1.0)
        beta_noise_per_evt = scatter_add(noise_beta, noise_evt_remap)
        L_beta_noise = s_B * (beta_noise_per_evt / n_noise_per_evt).sum() / batch_size

    # --- Beta suppression: push non-alpha signal betas toward 0 ---
    L_beta_suppress = torch.tensor(0.0, device=device)
    if beta_suppress_weight > 0 and n_hits_sig > n_objects:
        is_alpha = torch.zeros(n_hits_sig, dtype=torch.bool, device=device)
        is_alpha[index_alpha] = True
        L_beta_suppress = beta_suppress_weight * sig_beta[~is_alpha].mean()

    total = L_V + L_beta_sig + L_beta_noise + L_beta_suppress + var_weight * L_var
    if return_components:
        components = {
            "L_V_att": L_V_att.detach(),
            "L_V_rep": L_V_rep.detach(),
            "L_beta_sig": L_beta_sig.detach() if torch.is_tensor(L_beta_sig) else torch.tensor(float(L_beta_sig), device=device),
            "L_beta_noise": L_beta_noise.detach() if torch.is_tensor(L_beta_noise) else torch.tensor(float(L_beta_noise), device=device),
            "L_beta_suppress": L_beta_suppress.detach() if torch.is_tensor(L_beta_suppress) else torch.tensor(float(L_beta_suppress), device=device),
            "L_var": L_var.detach(),
            "var_weight": torch.tensor(float(var_weight), device=device),
        }
        return total, components
    return total


# ---------------------------------------------------------------------------
# Beta-greedy evaluation (pure numpy, from original get_clustering_np)
# ---------------------------------------------------------------------------
def get_clustering_np(betas, X, tbeta=0.5, td=0.5):
    n_points = betas.shape[0]
    select = betas > tbeta
    indices = np.nonzero(select)[0]
    indices = indices[np.argsort(-betas[select])]
    unassigned = np.arange(n_points)
    clustering = -1 * np.ones(n_points, dtype=np.int32)
    for idx in indices:
        d = np.linalg.norm(X[unassigned] - X[idx], axis=-1)
        assigned = unassigned[d < td]
        clustering[assigned] = idx
        unassigned = unassigned[~(d < td)]
    return clustering


def _seq_lens_to_batch(seq_lens, device):
    return torch.repeat_interleave(
        torch.arange(len(seq_lens), device=device, dtype=torch.long),
        torch.tensor(seq_lens, device=device, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def print_rank0(msg):
    if get_rank() == 0:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------
def _compute_var_weight(epoch, args):
    """Linear warmup of var_weight from 0 to args.var_weight over the first
    `args.var_warmup_epochs` epochs (inclusive of the final epoch)."""
    if args.var_warmup_epochs <= 0:
        return float(args.var_weight)
    # epoch is 1-indexed in this codebase
    frac = min(1.0, max(0.0, (epoch - 1) / max(args.var_warmup_epochs, 1)))
    return float(args.var_weight) * frac


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, args, ema=None):
    model.train()
    total_loss = 0.0
    n_batches = 0
    t0 = time.time()
    raw_model = model.module if isinstance(model, DDP) else model

    vw = _compute_var_weight(epoch, args)
    comp_sums = {"L_V_att": 0.0, "L_V_rep": 0.0, "L_beta_sig": 0.0,
                 "L_beta_noise": 0.0, "L_beta_suppress": 0.0, "L_var": 0.0}

    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()

        if batch is None:
            dummy = torch.zeros(2, 10, device=device)
            dummy[:, 3] = 1.0
            out = model(dummy, [2])
            (out.sum() * 0.0).backward()
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(raw_model)
            continue

        features = batch["features"].to(device, non_blocking=True)
        mc_index = batch["mc_index"].to(device, non_blocking=True)
        is_secondary = batch["is_secondary"].to(device, non_blocking=True)
        seq_lens = batch["seq_lens"]
        batch_ids = _seq_lens_to_batch(seq_lens, device)

        output = model(features, seq_lens)

        ed = args.embed_dim
        mc_index_loss = mc_index.clone()
        mc_index_loss[is_secondary] = 0  # secondaries become noise (mc_index=0)

        coords = output[:, :ed]
        if args.cosine_norm:
            coords = F.normalize(coords, dim=-1)
        beta_val = torch.sigmoid(output[:, ed])

        if (mc_index_loss != 0).sum() < 4:
            loss = output.sum() * 0.0
            comp = None
        else:
            loss, comp = object_condensation_loss(
                coords=coords,
                beta=beta_val,
                mc_index=mc_index_loss.long(),
                batch=batch_ids.long(),
                qmin=args.qmin,
                attr_weight=args.attr_weight,
                repul_weight=args.repul_weight,
                fill_loss_weight=args.fill_loss_weight,
                use_average_cc_pos=args.use_average_cc_pos,
                beta_suppress_weight=args.beta_suppress_weight,
                var_weight=vw,
                return_components=True,
            )

        if torch.isnan(loss) or torch.isinf(loss):
            print_rank0(f"  Batch {batch_idx}: NaN/Inf loss, skipping grad")
            loss = output.sum() * 0.0
            comp = None

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if ema is not None:
            ema.update(raw_model)

        total_loss += loss.item()
        n_batches += 1
        if comp is not None:
            for k in comp_sums:
                comp_sums[k] += float(comp[k].item())
        if batch_idx % 50 == 0:
            elapsed = time.time() - t0
            avg = total_loss / max(n_batches, 1)
            lr = optimizer.param_groups[0]["lr"]
            lvar = float(comp["L_var"].item()) if comp is not None else 0.0
            latt = float(comp["L_V_att"].item()) if comp is not None else 0.0
            lrep = float(comp["L_V_rep"].item()) if comp is not None else 0.0
            print_rank0(
                f"  Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f} | "
                f"Avg {avg:.4f} | L_att {latt:.3f} L_rep {lrep:.3f} "
                f"L_var {lvar:.3f} (w={vw:.2f}) | LR {lr:.2e} | Time {elapsed:.1f}s"
            )

    n = max(n_batches, 1)
    print_rank0(
        f"  Epoch {epoch} component means: "
        f"L_att={comp_sums['L_V_att']/n:.3f} L_rep={comp_sums['L_V_rep']/n:.3f} "
        f"L_beta_sig={comp_sums['L_beta_sig']/n:.3f} "
        f"L_beta_noise={comp_sums['L_beta_noise']/n:.3f} "
        f"L_beta_suppress={comp_sums['L_beta_suppress']/n:.3f} "
        f"L_var={comp_sums['L_var']/n:.3f} (weight={vw:.2f})"
    )
    return total_loss / n


def _compute_batch_metrics_greedy(coords, beta_logits, mc_index, is_secondary, seq_lens, tbeta=0.5, td=0.5, cosine_norm=False):
    from collections import Counter

    coords = coords.detach().cpu().numpy()
    beta = torch.sigmoid(beta_logits.squeeze(-1)).detach().cpu().numpy()
    mc_index = mc_index.detach().cpu().numpy()
    is_secondary = is_secondary.detach().cpu().numpy()

    all_purities, all_effs, all_match = [], [], []
    noise_low_beta_counts, noise_total_counts = 0, 0
    offset = 0
    for n_hits in seq_lens:
        sl = slice(offset, offset + n_hits)
        offset += n_hits

        sec_mask = is_secondary[sl] | (mc_index[sl] == 0)
        if sec_mask.any():
            noise_total_counts += sec_mask.sum()
            noise_low_beta_counts += (beta[sl][sec_mask] < 0.1).sum()

        mask = (~is_secondary[sl]) & (mc_index[sl] != 0)
        c = coords[sl][mask]
        if cosine_norm:
            norms = np.linalg.norm(c, axis=1, keepdims=True)
            c = c / np.clip(norms, 1e-6, None)
        b = beta[sl][mask]
        mc = mc_index[sl][mask]
        if len(c) == 0:
            continue

        pred_labels = get_clustering_np(b, c, tbeta=tbeta, td=td)
        unique_pred = np.unique(pred_labels[pred_labels >= 0])
        unique_true = np.unique(mc[mc >= 0])
        if len(unique_pred) == 0:
            all_purities.append(0.0)
            all_effs.append(0.0)
            all_match.append(0.0)
            continue

        purities = []
        for pid in unique_pred:
            cluster_mc = mc[pred_labels == pid]
            if len(cluster_mc) > 0:
                purities.append(np.bincount(cluster_mc).max() / len(cluster_mc))
        if purities:
            all_purities.append(np.mean(purities))

        matched = 0
        n_matchable = 0
        effs = []
        for tid in unique_true:
            tmask = mc == tid
            n_true = tmask.sum()
            if n_true < 2:
                continue
            n_matchable += 1
            pred_for_track = pred_labels[tmask]
            pred_for_track = pred_for_track[pred_for_track >= 0]
            if len(pred_for_track) == 0:
                effs.append(0.0)
                continue
            best_label, best_match = Counter(pred_for_track).most_common(1)[0]
            eff = best_match / n_true
            pur = best_match / (pred_labels == best_label).sum()
            effs.append(eff)
            if pur > 0.75:
                matched += 1
        if effs:
            all_effs.append(np.mean(effs))
            all_match.append(matched / max(n_matchable, 1))

    noise_suppression = float(noise_low_beta_counts / max(noise_total_counts, 1))

    return {
        "purity": float(np.mean(all_purities)) if all_purities else 0.0,
        "efficiency": float(np.mean(all_effs)) if all_effs else 0.0,
        "match_rate": float(np.mean(all_match)) if all_match else 0.0,
        "noise_suppression": noise_suppression,
    }


@torch.no_grad()
def validate(model, loader, device, epoch, args, ema=None):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_metrics = []
    eval_model = model.module if isinstance(model, DDP) else model

    # Swap in EMA weights for evaluation
    saved_state = None
    if ema is not None:
        saved_state = copy.deepcopy(eval_model.state_dict())
        eval_model.load_state_dict(ema.state_dict())

    for batch in loader:
        if batch is None:
            continue
        features = batch["features"].to(device, non_blocking=True)
        mc_index = batch["mc_index"].to(device, non_blocking=True)
        is_secondary = batch["is_secondary"].to(device, non_blocking=True)
        seq_lens = batch["seq_lens"]
        batch_ids = _seq_lens_to_batch(seq_lens, device)

        output = eval_model(features, seq_lens)

        ed = args.embed_dim
        mc_index_loss = mc_index.clone()
        mc_index_loss[is_secondary] = 0

        coords = output[:, :ed]
        if args.cosine_norm:
            coords = F.normalize(coords, dim=-1)
        beta_val = torch.sigmoid(output[:, ed])

        if (mc_index_loss != 0).sum() >= 4:
            loss = object_condensation_loss(
                coords=coords,
                beta=beta_val,
                mc_index=mc_index_loss.long(),
                batch=batch_ids.long(),
                qmin=args.qmin,
                attr_weight=args.attr_weight,
                repul_weight=args.repul_weight,
                var_weight=float(args.var_weight),
            )
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                n_batches += 1

        all_metrics.append(
            _compute_batch_metrics_greedy(
                output[:, :ed], output[:, ed:ed+1], mc_index, is_secondary, seq_lens,
                tbeta=args.tbeta, td=args.td, cosine_norm=args.cosine_norm,
            )
        )

    # Restore original (non-EMA) weights for continued training
    if saved_state is not None:
        eval_model.load_state_dict(saved_state)

    avg_loss = total_loss / max(n_batches, 1)
    if all_metrics:
        avg_purity = np.mean([m["purity"] for m in all_metrics])
        avg_eff = np.mean([m["efficiency"] for m in all_metrics])
        avg_match = np.mean([m["match_rate"] for m in all_metrics])
        avg_noise_supp = np.mean([m["noise_suppression"] for m in all_metrics])
        print_rank0(
            f"  Epoch {epoch} | Val Loss: {avg_loss:.4f} | Purity: {avg_purity:.3f} | "
            f"Efficiency: {avg_eff:.3f} | Match Rate: {avg_match:.3f} | "
            f"Noise Supp: {avg_noise_supp:.3f} ({n_batches} batches)"
        )
    else:
        print_rank0(f"  Epoch {epoch} | Val Loss: {avg_loss:.4f} ({n_batches} batches)")
    return avg_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_seed_range(s):
    parts = s.split("-")
    return int(parts[0]), int(parts[1]) + 1


def main():
    parser = argparse.ArgumentParser(description="CGATr (Conformal GA Cl(4,1)) training")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--train_seeds", type=str, default="1-160")
    parser.add_argument("--val_seeds", type=str, default="101-110")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--start_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, default=2,
                        help="Number of epochs for linear LR warmup (default: 2)")
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="Override warmup_epochs with exact step count")
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="EMA decay rate (0 to disable EMA)")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_blocks", type=int, default=10)
    parser.add_argument("--hidden_mv_channels", type=int, default=16)
    parser.add_argument("--hidden_s_channels", type=int, default=64)
    parser.add_argument("--qmin", type=float, default=0.1)
    parser.add_argument("--attr_weight", type=float, default=1.0)
    parser.add_argument("--repul_weight", type=float, default=1.0)
    parser.add_argument("--fill_loss_weight", type=float, default=0.0)
    parser.add_argument("--use_average_cc_pos", type=float, default=0.0)
    parser.add_argument("--beta_suppress_weight", type=float, default=0.1,
                        help="Weight for penalizing high beta on non-alpha signal hits (0 = off)")
    parser.add_argument("--embed_dim", type=int, default=5,
                        help="Clustering embedding dimension (default 5; chosen from a PCA "
                             "analysis of trained CGATr embeddings showing the previous 6D "
                             "embedding was effectively 4D — see README_CGATr.md).")
    parser.add_argument("--var_weight", type=float, default=0.3,
                        help="Weight for within-cluster variance regularizer L_var (0 = off, "
                             "reduces to standard OC loss).")
    parser.add_argument("--var_warmup_epochs", type=int, default=2,
                        help="Linear warmup of var_weight from 0 over this many epochs "
                             "(lets clusters form before tightening them).")
    parser.add_argument("--beta_mlp", action="store_true", default=False,
                        help="Use 2-layer MLP for beta head")
    parser.add_argument("--cosine_norm", action="store_true", default=False,
                        help="L2-normalize clustering coords before loss")
    parser.add_argument("--tbeta", type=float, default=0.1)
    parser.add_argument("--td", type=float, default=0.2)
    parser.add_argument("--max_hits", type=int, default=None)
    parser.add_argument("--normalize_mv_inputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/cgatr")
    args = parser.parse_args()

    torch.manual_seed(42)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    use_ddp = local_rank >= 0
    if use_ddp:
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = (
            torch.device(f"cuda:{int(args.gpus.split(',')[0])}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    print_rank0(f"Using device: {device} | DDP: {use_ddp} | World size: {get_world_size()}")
    print_rank0(
        f"Model: CGATr (Cl(4,1) conformal GA, single MV channel, secondaries-as-noise) "
        f"| blocks={args.num_blocks} mv_ch={args.hidden_mv_channels} s_ch={args.hidden_s_channels}"
    )
    print_rank0(f"embed_dim={args.embed_dim} | beta_mlp={args.beta_mlp} | cosine_norm={args.cosine_norm}")
    print_rank0(
        f"OC loss: qmin={args.qmin} | beta_suppress={args.beta_suppress_weight} | "
        f"var_weight={args.var_weight} (warmup {args.var_warmup_epochs} epochs)"
    )
    print_rank0(f"EMA decay: {args.ema_decay}")

    train_start, train_end = parse_seed_range(args.train_seeds)
    val_start, val_end = parse_seed_range(args.val_seeds)
    print_rank0(f"Loading training data: seeds {train_start}-{train_end - 1}")
    train_dataset = IDEAParquetDataset(args.data_dir, seed_range=(train_start, train_end), max_hits_per_event=args.max_hits)
    print_rank0(f"Loading validation data: seeds {val_start}-{val_end - 1}")
    val_dataset = IDEAParquetDataset(args.data_dir, seed_range=(val_start, val_end), max_hits_per_event=args.max_hits)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if use_ddp else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=args.num_workers, collate_fn=collate_idea_events, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
        num_workers=args.num_workers, collate_fn=collate_idea_events, pin_memory=True,
    )

    model = CGATrParquetModel(args).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False)
    raw_model = model.module if use_ddp else model
    print_rank0(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # EMA
    ema = None
    if args.ema_decay > 0:
        ema = EMAModel(raw_model, decay=args.ema_decay)
        print_rank0(f"EMA enabled (decay={args.ema_decay})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.start_lr, weight_decay=1e-4)
    total_steps = args.num_epochs * max(len(train_loader), 1)

    # Warmup: epoch-based by default, --warmup_steps overrides
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else args.warmup_epochs * len(train_loader)
    print_rank0(f"Warmup: {warmup_steps} steps ({args.warmup_epochs} epochs x {len(train_loader)} batches)")

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(args.min_lr / args.start_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    for epoch in range(1, args.num_epochs + 1):
        if use_ddp:
            train_sampler.set_epoch(epoch)
        print_rank0(f"\n{'='*60}")
        print_rank0(f"Epoch {epoch}/{args.num_epochs} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        print_rank0(f"{'='*60}")

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, args, ema=ema)
        val_loss = validate(raw_model, val_loader, device, epoch, args, ema=ema)

        if get_rank() == 0:
            print(f"Epoch {epoch} Summary: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}", flush=True)
            # Save EMA weights if available, otherwise raw weights
            save_state = ema.state_dict() if ema is not None else raw_model.state_dict()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": save_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "ema_state_dict": ema.state_dict() if ema is not None else None,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                os.path.join(args.output_dir, f"cgatr_epoch{epoch}.pt"),
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(save_state, os.path.join(args.output_dir, "cgatr_best.pt"))
                print(f"  New best model: {os.path.join(args.output_dir, 'cgatr_best.pt')}", flush=True)

    print_rank0(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
