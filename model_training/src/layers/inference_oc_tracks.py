import dgl
import torch
import os
from sklearn.cluster import DBSCAN
from torch_scatter import scatter_max, scatter_add, scatter_mean
import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import pandas as pd
import wandb
from sklearn.cluster import DBSCAN, HDBSCAN
import sys
from collections import Counter

def hfdb_obtain_labels(X, device, eps=0.1):

    hdb = HDBSCAN(min_cluster_size=4, cluster_selection_epsilon=eps).fit(
        X.detach().cpu()
    )
    
    labels_hdb = hdb.labels_ + 1  # noise class goes to zero
    labels_hdb = np.reshape(labels_hdb, (-1))
    labels_hdb = torch.Tensor(labels_hdb).long().to(device)
    return labels_hdb

def evaluate_efficiency_tracks(
    batch_g,
    model_output,
    y,
    local_rank,
    step,
    epoch,
    path_save,
    store=False,
    predict=False,
    ct=False,
    clustering_mode="clustering_normal",
    tau=False
):
    number_of_showers_total = 0
    if not ct:
        batch_g.ndata["coords"] = model_output[:, 0:3]
        batch_g.ndata["beta"] = model_output[:, 3]
    else:
        batch_g.ndata["model_output"] = model_output
        
    graphs = dgl.unbatch(batch_g)
    
    batch_id = y[:, -1].view(-1)
    df_list = []
    df_hits = []
    for i, g in enumerate(graphs):
        
        mask = batch_id == i
        dic = {}
        dic["graph"] = g
        dic["part_true"] = y[mask]

        
        betas = torch.sigmoid(dic["graph"].ndata["beta"])
        X = dic["graph"].ndata["coords"]
        if ct:

            # labels can start at -1 in which case the 0 is the 'noise'
            labels_ = graphs[i].ndata["model_output"].long() + 1
            map_from = list(np.unique(labels_.detach().cpu()))
            labels = map(lambda x: map_from.index(x), labels_)
            labels = (
                torch.Tensor(list(labels))
                .long()
                .to(dic["graph"].ndata["coords"].device)
            )
            
        else:
            if clustering_mode == "clustering_normal":
                
                # def describe_tensor(name, t):
                #     print(f"Statistics for tensor '{name}':")
                #     print(f"  Shape: {tuple(t.shape)}")
                #     print(f"  Data type: {t.dtype}")
                #     print(f"  Device: {t.device}")
                #     print(f"  Number of elements: {t.numel()}")
                #     print(f"  Mean value: {t.mean().item():.6f}")
                #     print(f"  Standard deviation: {t.std().item():.6f}")
                #     print(f"  Minimum value: {t.min().item():.6f}")
                #     print(f"  Maximum value: {t.max().item():.6f}")
                #     print(f"  Median value: {t.median().item():.6f}")
                #     print(f"  First 5 elements: {t.flatten()[:5].tolist()}")
                #     print()

                # describe_tensor("betas", betas)
                # describe_tensor("X", X)
                
                clustering1 = get_clustering(betas, X, tbeta=0.2, td=0.15)
                map_from = list(np.unique(clustering1.detach().cpu()))
                cluster_id = map(lambda x: map_from.index(x), clustering1)
                clustering_ordered = (
                    torch.Tensor(list(cluster_id)).long().to(model_output.device)
                )
                
                # unique_vals, counts = torch.unique(clustering_ordered, return_counts=True)
                # for val, count in zip(unique_vals.tolist(), counts.tolist()):
                #     print(f"Value {val}: {count} occurrences")

                # print(f"\nTotal elements: {clustering_ordered.numel()}")
                
                # sys.exit()
                
                if torch.unique(clustering1)[0] != -1:
                    clustering = clustering_ordered + 1
                else:
                    clustering = clustering_ordered

                labels = clustering.view(-1).long()
            elif clustering_mode == "dbscan":
                labels = hfdb_obtain_labels(X, betas.device)
        
        particle_ids = torch.unique(dic["graph"].ndata["particle_number"])
        shower_p_unique = torch.unique(labels)
        
        pids, partIndices, deltaMCs, energies, pTs, thetas, genStatus, numSIhits, numCDChits, trackLabels, hitEfficiencies, hitPurities, fakeTrackIndices, siliconHits_fakeTracks, driftHits_fakeTracks, fileIDs, eventIDs, = match_tracks(labels, dic) 
        
        df_event = generate_tracks_dataframe(
            fileIDs, eventIDs, pids, partIndices, deltaMCs, energies, pTs, thetas, genStatus, numSIhits, numCDChits, trackLabels, hitEfficiencies, hitPurities, fakeTrackIndices, siliconHits_fakeTracks, driftHits_fakeTracks)
        
        df_list.append(df_event)
        
        
        df_hits_event = dataframe_position_labels(labels, dic) 
        df_hits.append(df_hits_event)
        
        if len(df_list) > 0:
            df_batch = pd.concat(df_list)
        else:
            df_batch = []
        if store:
            store_at_batch_end(
                path_save, df_batch, local_rank, step, epoch, predict=predict
            )
            
        if len(df_hits) > 0:
            df_batch_hits = pd.concat(df_hits)
        else:
            df_batch_hits = []
        if store:
            
            store_at_batch_end_hits(
                path_save, df_batch_hits, local_rank, step, epoch, predict=predict
            )
            
    
    return df_batch, df_batch_hits
        
def match_tracks(labels, dic):
    
    pids = []
    partIndices = []
    deltaMCs = []
    energies = []
    pTs = []
    thetas = []
    genStatus = []
    numSIhits = [] 
    numCDChits = []
    trackLabels = []
    hitEfficiencies = []
    hitPurities = []
    fakeTrackIndices = []
    fileIDs = []
    eventIDs = []
    
    part_true = dic["part_true"]
    graphInfo = dic["graph"]
    
    fileID = graphInfo.ndata["fileNumber"][0]
    eventID = graphInfo.ndata["eventNumber"][0]
    
    part_keys = [
        "part_theta",    # 0
        "part_phi",      # 1
        "part_m",        # 2
        "part_pid",      # 3
        "part_id",       # 4
        "part_p",        # 5
        "part_p_t",      # 6
        "gen_status",    # 7
        "part_parent",   # 8
        "batch_id"       # 9
    ]
    partInfo = {key: part_true[:, i] for i, key in enumerate(part_keys)}
    

    particle_number_nomap = graphInfo.ndata["particle_number_nomap"]  # particle index

    unique_labels, counts = torch.unique(labels, return_counts=True)
    numHits_tracks = {int(label): int(count) for label, count in sorted(zip(unique_labels, counts), key=lambda x: x[0])}
    
    unique_particles, counts = torch.unique(particle_number_nomap, return_counts=True)
    numHits_particle = {int(p): int(c) for p, c in zip(unique_particles, counts)}
    
    
    # number of silicon and drift hits per particle 
    hit_type = graphInfo.ndata["hit_type"]
    type_hits_particle = {}
    for particle in unique_particles:
        mask = particle_number_nomap == particle
        num_siliconHits = torch.sum(hit_type[mask] == 1).item()
        num_driftHits = torch.sum(hit_type[mask] == 0).item()

        type_hits_particle[int(particle.item())] = {
            "silicon_hits": num_siliconHits,
            "drift_hits": num_driftHits
        }
    
    # dictionary:
    # - each entry is a particle and the content is the number of hits that belong to that particle in each cluster
    particle_label_counts = {}
    for p in unique_particles:
        mask_p = particle_number_nomap == p
        counts_dict = {}
        for l in unique_labels:
            mask_label = labels == l
            count = torch.sum(mask_p & mask_label).item()
            counts_dict[int(l)] = count
        particle_label_counts[int(p)] = counts_dict        
    
    # efficiency and purity 
    efficiency = {}
    purity = {}
    for p in unique_particles:
        efficiency_p = {}
        purity_p = {}
        for l in unique_labels:
            hits_in_label = particle_label_counts[int(p)][int(l)]
            efficiency_p[int(l)] = hits_in_label / numHits_particle[int(p)] if numHits_particle[int(p)] > 0 else 0.0
            purity_p[int(l)] = hits_in_label / numHits_tracks[int(l)] if numHits_tracks[int(l)] > 0 else 0.0
        efficiency[int(p)] = efficiency_p
        purity[int(p)] = purity_p
    
    # check if particle matches the cluster and check which clusters are not assigned
    particle_matches = {}
    labels_matched_set = set()

    for p in unique_particles:
        matched = False
        matched_eff = []
        matched_purity = []
        matched_labels = []
        
        for l in unique_labels:
            eff = efficiency[int(p)][int(l)]
            pur = purity[int(p)][int(l)]
            
            # if eff > 0.5 and pur > 0.5:
            if pur > 0.75:
                matched = True
                matched_eff.append(eff)
                matched_purity.append(pur)
                matched_labels.append(int(l))
                labels_matched_set.add(int(l))
        
        particle_matches[int(p)] = {
            "matched": matched,
            "track": matched_labels,
            "efficiency": matched_eff,
            "purity": matched_purity
        }

        
    # fakeTracks
    labels_not_matched = [int(l) for l in unique_labels if int(l) not in labels_matched_set]
    fakeTrackIndices = labels_not_matched
    
    siliconHits_fakeTracks = []
    driftHits_fakeTracks = []
    for idx, fakeTrack in enumerate(fakeTrackIndices):
        
        mask = labels == fakeTrack
        num_siliconHits = torch.sum(hit_type[mask] == 1).item()
        num_driftHits = torch.sum(hit_type[mask] == 0).item()
        siliconHits_fakeTracks.append(num_siliconHits)
        driftHits_fakeTracks.append(num_driftHits)
    
    
    # particle info - tracks matching
    particle_id = partInfo["part_id"] 
    part_theta = partInfo["part_theta"]
    part_pt = partInfo["part_p_t"]
    part_m = partInfo["part_m"]
    gen_status = partInfo["gen_status"]
    part_p = partInfo["part_p"]
    part_pid = partInfo["part_pid"]
    
    unique_particles_true = torch.unique(particle_id)
    for particle in unique_particles_true:
        
        mask = particle_id == particle
        
        if torch.any(mask):
            
            idx = torch.nonzero(mask, as_tuple=False)[0].item()
            
            pid = part_pid[mask].item()
            theta = part_theta[mask].item()
            gen_Status = gen_status[mask].item()
            p_val = float(part_p[idx].item())   
            m_val = float(part_m[idx].item())  
            pt_val = float(part_pt[idx].item())  
            energy = (p_val**2 + m_val**2) ** 0.5

        pid_int = int(particle.item())

        if pid_int in type_hits_particle:
            numSiliconHits = type_hits_particle[pid_int]["silicon_hits"]
            numDriftHits = type_hits_particle[pid_int]["drift_hits"]

        if pid_int in particle_matches:
            trackLabel = particle_matches[pid_int]["track"]
            hitEfficiency = particle_matches[pid_int]["efficiency"]
            hitPurity = particle_matches[pid_int]["purity"]

        deltaMC = 0
        # deltaMC = ...
        
        
        pids.append(pid)
        partIndices.append(pid_int)
        energies.append(energy)
        deltaMCs.append(deltaMC)
        pTs.append(pt_val)
        thetas.append(theta)
        genStatus.append(gen_Status)
        numSIhits.append(numSiliconHits)
        numCDChits.append(numDriftHits)
        trackLabels.append(trackLabel)
        hitEfficiencies.append(hitEfficiency)
        hitPurities.append(hitPurity)
        fileIDs.append(fileID)
        eventIDs.append(eventID)
    
    if -1 in particle_number_nomap:
        pids.append(-1)
        partIndices.append(-1)
        energies.append(-1)
        deltaMCs.append(-1)
        pTs.append(-1)
        thetas.append(-1)
        genStatus.append(-1)
        numSIhits.append(type_hits_particle[-1]["silicon_hits"])
        numCDChits.append(type_hits_particle[-1]["drift_hits"])
        trackLabels.append(particle_matches[-1]["track"])
        hitEfficiencies.append(particle_matches[-1]["efficiency"])
        hitPurities.append(particle_matches[-1]["purity"])
        fileIDs.append(fileID)
        eventIDs.append(eventID)
    
    return pids, partIndices, deltaMCs, energies, pTs, thetas, genStatus, numSIhits, numCDChits, trackLabels, hitEfficiencies, hitPurities, fakeTrackIndices, siliconHits_fakeTracks, driftHits_fakeTracks, fileIDs, eventIDs

def generate_tracks_dataframe(
    fileIDs, 
    eventIDs, 
    pids,
    partIndices,
    deltaMCs,
    energies,
    pTs,
    thetas,
    genStatus,
    numSIhits,
    numCDChits,
    trackLabels,
    hitEfficiencies,
    hitPurities,
    fakeTrackIndices=None,
    siliconHits_fakeTracks=None,
    driftHits_fakeTracks=None
):
    """
    Create a pandas DataFrame from the outputs of match_tracks().
    Only uses the lists/tensors returned by match_tracks().

    Parameters
    ----------
    deltaMCs, energies, thetas, genStatus, numSIhits, numCDChits,
    trackLabels, hitEfficiencies, hitPurities : list or torch.Tensor
        Outputs from match_tracks().
    fakeTrackIndices : list or torch.Tensor, optional
        Track indices not matched to any particle.

    Returns
    -------
    pd.DataFrame
        Summary table of particles and tracks.
    """

    # Helper to convert tensors to numpy arrays
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x
    
    # Main DataFrame
    df_dict = {
        "fileID" : [to_numpy(fileIDs[0])] * len(eventIDs),
        "eventID" : [to_numpy(eventIDs[0])] * len(eventIDs),
        "partIndex" : to_numpy(partIndices),
        "pid" : to_numpy(pids),
        "energy": to_numpy(energies),
        "pT": to_numpy(pTs),
        "deltaMC": to_numpy(deltaMCs),
        "theta": to_numpy(thetas),
        "genStatus": to_numpy(genStatus),
        "numSIhits": to_numpy(numSIhits),
        "numCDChits": to_numpy(numCDChits),
        "trackLabel": to_numpy(trackLabels),
        "hitEfficiency": to_numpy(hitEfficiencies),
        "hitPurity": to_numpy(hitPurities),
    }

    df = pd.DataFrame(df_dict)

    # Append fake tracks if provided
    if fakeTrackIndices is not None and len(fakeTrackIndices) > 0:
        fake_df = pd.DataFrame({
            "fileID" : [to_numpy(fileIDs[0])] * len(fakeTrackIndices),
            "eventID" : [to_numpy(eventIDs[0])] * len(fakeTrackIndices),
            "partIndex" : [None] * len(fakeTrackIndices),
            "pid" : [None] * len(fakeTrackIndices),
            "energy": [None] * len(fakeTrackIndices),
            "pT": [None] * len(fakeTrackIndices),
            "deltaMC": [None] * len(fakeTrackIndices),
            "theta": [None] * len(fakeTrackIndices),
            "genStatus": [None] * len(fakeTrackIndices),
            "numSIhits": to_numpy(siliconHits_fakeTracks),
            "numCDChits": to_numpy(driftHits_fakeTracks),
            "trackLabel": to_numpy(fakeTrackIndices),
            "hitEfficiency": [0.0] * len(fakeTrackIndices),
            "hitPurity": [0.0] * len(fakeTrackIndices),
        })
        df = pd.concat([df, fake_df], ignore_index=True)

    return df

def dataframe_position_labels(labels,dic):
    
    graphInfo = dic["graph"]
    
    fileID = graphInfo.ndata["fileNumber"]
    eventID = graphInfo.ndata["eventNumber"]
    pos_x = graphInfo.ndata["pos_hits_xyz"][:,0]
    pos_y = graphInfo.ndata["pos_hits_xyz"][:,1]
    pos_z = graphInfo.ndata["pos_hits_xyz"][:,2]
    hit_type = graphInfo.ndata["hit_type"]
    originalParticle_afterLabel = graphInfo.ndata["particle_number_nomap"]
    originalParticle = graphInfo.ndata["particle_number_nomap_original"]


    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x
    
    df_dict = {
        "fileID" : [to_numpy(fileID)],
        "eventID" : [to_numpy(eventID)],
        "pos_x" :  [to_numpy(pos_x)],
        "pos_y" : [to_numpy(pos_y)],
        "pos_z" : [to_numpy(pos_z)],
        "hit_type" : [to_numpy(hit_type)],
        "clusterID" : [to_numpy(labels)],
        "particleLabel" : [to_numpy(originalParticle_afterLabel)],
        "originalClusterID" : [to_numpy(originalParticle)],
        
    }

    df = pd.DataFrame(df_dict)
    
    return df

def store_at_batch_end(
    path_save,
    df_batch,
    local_rank=0,
    step=0,
    epoch=None,
    predict=False,
):
    path_save_ = (
        path_save + "/" + str(local_rank) + "_" + str(step) + "_" + str(epoch) + "IDEAtracking.pt"
    )
    if predict:
        df_batch = pd.concat(df_batch)
        df_batch.to_pickle(path_save_)
    
def store_at_batch_end_hits(
    path_save,
    df_batch,
    local_rank=0,
    step=0,
    epoch=None,
    predict=False,
):
    path_save_ = (
        path_save + "/" + str(local_rank) + "_" + str(step) + "_" + str(epoch) + "IDEAtracking_hits.pt"
    )
    if predict:
        df_batch = pd.concat(df_batch)
        df_batch.to_pickle(path_save_)

def get_clustering(betas: torch.Tensor, X: torch.Tensor, tbeta=0.7, td=0.05):
    """
    Returns a clustering of hits -> cluster_index, based on the GravNet model
    output (predicted betas and cluster space coordinates) and the clustering
    parameters tbeta and td.
    Takes torch.Tensors as input.
    """
    n_points = betas.size(0)
    select_condpoints = betas > tbeta
    # Get indices passing the threshold
    indices_condpoints = select_condpoints.nonzero()
    # Order them by decreasing beta value
    indices_condpoints = indices_condpoints[(-betas[select_condpoints]).argsort()]
    # Assign points to condensation points
    # Only assign previously unassigned points (no overwriting)
    # Points unassigned at the end are bkg (-1)
    unassigned = torch.arange(n_points).to(betas.device)
    clustering = -1 * torch.ones(n_points, dtype=torch.long).to(betas.device)
    while len(indices_condpoints) > 0 and len(unassigned) > 0:
        index_condpoint = indices_condpoints[0]
        d = torch.norm(X[unassigned] - X[index_condpoint][0], dim=-1)
        assigned_to_this_condpoint = unassigned[d < td]
        clustering[assigned_to_this_condpoint] = index_condpoint[0]
        unassigned = unassigned[~(d < td)]
        # calculate indices_codpoints again
        indices_condpoints = find_condpoints(betas, unassigned, tbeta)
    return clustering

def find_condpoints(betas, unassigned, tbeta):
    n_points = betas.size(0)
    select_condpoints = betas > tbeta
    device = betas.device
    mask_unassigned = torch.zeros(n_points).to(device)
    mask_unassigned[unassigned] = True
    select_condpoints = mask_unassigned.to(bool) * select_condpoints
    # Get indices passing the threshold
    indices_condpoints = select_condpoints.nonzero()
    # Order them by decreasing beta value
    indices_condpoints = indices_condpoints[(-betas[select_condpoints]).argsort()]
    return indices_condpoints