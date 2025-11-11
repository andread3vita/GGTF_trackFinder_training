from typing import Tuple, Union
import numpy as np
import torch
from torch_scatter import scatter_max, scatter_add, scatter_mean
from src.layers.object_cond import calc_LV_Lbeta

def object_condensation_loss_tracking(
    batch,
    pred,
    y,
    return_resolution=False,
    clust_loss_only=True,
    add_energy_loss=False,
    calc_e_frac_loss=False,
    q_min=0.1,
    frac_clustering_loss=0.1,
    attr_weight=1.0,
    repul_weight=1.0,
    fill_loss_weight=1.0,
    use_average_cc_pos=0.0,
    loss_type="hgcalimplementation",
    output_dim=4,
    clust_space_norm="none",
    tracking=False,
    CLD=False,
):

    _, S = pred.shape
    if clust_loss_only:
        clust_space_dim = output_dim - 1
    else:
        clust_space_dim = output_dim - 28

    xj = pred[:, 0:clust_space_dim]  # xj: cluster space coords
    bj = torch.sigmoid(torch.reshape(pred[:, clust_space_dim], [-1, 1]))  # 3: betas
    
    original_coords = batch.ndata["pos_hits_xyz"]  # [:, 0:clust_space_dim]
    

    dev = batch.device
    clustering_index_l = batch.ndata["particle_number"]

    len_batch = len(batch.batch_num_nodes())
    batch_numbers = torch.repeat_interleave(
        torch.arange(len_batch, device=dev),  # [0, 1, 2, ..., len_batch-1]
        batch.batch_num_nodes(),              # number of nodes per graph
    )

    a = calc_LV_Lbeta(
        original_coords,
        batch,
        y,
        None,
        None,
        # momentum=None,
        # predicted_pid=None,
        beta=bj.view(-1),
        cluster_space_coords=xj,  # Predicted by model
        cluster_index_per_event=clustering_index_l.view(-1).long(),  # Truth hit->cluster index
        batch=batch_numbers.long(),
        qmin=q_min,
        return_regression_resolution=return_resolution,
        post_pid_pool_module=None,
        clust_space_dim=clust_space_dim,
        frac_combinations=frac_clustering_loss,
        attr_weight=attr_weight,
        repul_weight=repul_weight,
        fill_loss_weight=fill_loss_weight,
        use_average_cc_pos=use_average_cc_pos,
        loss_type=loss_type,
        tracking=tracking,
        CLD=CLD,
    )
    
    

    loss = a[0] + a[1] 

    return loss, a