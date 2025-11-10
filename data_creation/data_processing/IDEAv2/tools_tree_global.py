from podio import root_io
import edm4hep
import ROOT
from ROOT import TFile, TTree
from array import array
import math
import dd4hep as dd4hepModule
from ROOT import dd4hep
import numpy as np


def find_mother_particle_check_tau(j, gen_part_coll, tau_index0, tau_index1):
    parent_p = j
    counter = 0
    belongs_to_tau1 = False
    belongs_to_tau2 = False
    while len(np.reshape(np.array(parent_p), -1)) < 1.5:
        if type(parent_p) == list:
            if len(parent_p) > 0:
                parent_p = parent_p[0]
            else:
                break
        parent_p_r = get_genparticle_parents(
            parent_p,
            gen_part_coll,
        )
        if len(parent_p_r) == 0:
            break
        if parent_p_r[0] == tau_index0:
            belongs_to_tau1 = True
        if parent_p_r[0] == tau_index1:
            belongs_to_tau2 = True
        pp_old = parent_p
        counter = counter + 1
        parent_p = parent_p_r
        if belongs_to_tau1 or belongs_to_tau2:
            break

    return pp_old, belongs_to_tau1, belongs_to_tau2


def get_genparticle_daughters(i, mcparts):

    p = mcparts[i]
    daughters = p.getDaughters()
    daughter_positions = []
    # for j in range(p.daughters_begin, p.daughters_end):
    #     # print(j, daughters[j].index)
    #     daughter_positions.append(daughters[j].index)
    #     # break
    for daughter in daughters:
        daughter_positions.append(daughter.getObjectID().index)

    return daughter_positions


def get_genparticle_parents(i, mcparts):

    p = mcparts[i]
    parents = p.getParents()
    # print(p.parents_begin(), p.parents_end())
    parent_positions = []
    # for j in range(p.parents_begin(), p.parents_end()):
    #     # print(j, daughters[j].index)
    #     parent_positions.append(parents[j].index)
    #     # break
    for parent in parents:
        parent_positions.append(parent.getObjectID().index)

    return parent_positions


def gen_particles_find(event, debug, store_tau="False"):

    index_of_Z = 0
    index_of_tau1 = 0
    index_of_tau2 = 0
    genparts = "MCParticles"
    genparts_parents = "_MCParticles_parents"
    genparts_daughters = "_MCParticles_daughters"
    # gen_parent_link_indexmc = event.get(genparts_parents)
    # gen_daughter_link_indexmc = event.get(genparts_daughters)
    gen_part_coll = event.get(genparts)
    genpart_indexes_pre = (
        dict()
    )  ## key: index in gen particle collection, value: position in stored gen particle array
    indexes_genpart_pre = (
        dict()
    )  ## key: position in stored gen particle array, value: index in gen particle collection
    total_e = 0
    n_part_pre = 0
    e_pp = np.zeros(11)
    for j, part in enumerate(gen_part_coll):
        momentum = part.getMomentum()
        p = math.sqrt(momentum.x**2 + momentum.y**2 + momentum.z**2)
        if debug:
            if j < 11 and j > 1:
                e_pp[j] = p
                total_e = total_e + p
        theta = math.acos(momentum.z / p)
        phi = math.atan2(momentum.y, momentum.x)

        if store_tau == "True":
            if part.getPDG() == 23:
                daughters = get_genparticle_daughters(
                    j,
                    gen_part_coll,
                )
                if len(daughters) == 2:
                    index_of_Z = j
                    index_of_tau1 = daughters[0]
                    index_of_tau2 = daughters[1]
        if debug:
            if part.getGeneratorStatus() == 1:
                print(
                    "all genparts: N: {}, PID: {}, Q: {}, P: {:.2e}, Theta: {:.2e}, Phi: {:.2e}, M: {:.2e}, X(m): {:.3f}, Y(m): {:.3f}, R(m): {:.3f}, Z(m): {:.3f}, status: {}, parents: {}, daughters: {}, decayed_traacker: {}".format(
                        j,
                        part.getPDG(),
                        part.getCharge(),
                        p,
                        theta,
                        phi,
                        part.getMass(),
                        part.getVertex().x * 1e-03,
                        part.getVertex().y * 1e-03,
                        math.sqrt(part.getVertex().x ** 2 + part.getVertex().y ** 2)
                        * 1e-03,
                        part.getVertex().z * 1e-03,
                        part.getGeneratorStatus(),
                        get_genparticle_parents(
                            j,
                            gen_part_coll,
                        ),
                        get_genparticle_daughters(
                            j,
                            gen_part_coll,
                        ),
                        part.isDecayedInTracker() * 1,
                    )
                )

            # part.daughters_begin,  part.daughters_end, part.parents_begin,  part.parents_end, D1: {}, D2: {}, M1: {}, M2: {}

        ## store all gen parts for now
        genpart_indexes_pre[j] = n_part_pre
        indexes_genpart_pre[n_part_pre] = j
        n_part_pre += 1

        """
        # exclude neutrinos (and pi0 for now)
        if part.generatorStatus == 1 and abs(part.PDG) not in [12, 14, 16, 111]:

            genpart_indexes_pre[j] = n_part_pre
            indexes_genpart_pre[n_part_pre] = j
            n_part_pre += 1

        # extract the photons from the pi0
        elif part.generatorStatus == 1 and part.PDG == 111:

            daughters = get_genparticle_daughters(
                j, gen_part_coll, gen_daughter_link_indexmc
            )

            if len(daughters) != 2:
                print("STRANGE PI0 DECAY")

            for d in daughters:
                a = gen_part_coll[d]
                genpart_indexes_pre[d] = n_part_pre
                indexes_genpart_pre[n_part_pre] = d
                n_part_pre += 1
        """
    return (
        genpart_indexes_pre,
        indexes_genpart_pre,
        n_part_pre,
        total_e,
        e_pp,
        gen_part_coll,
        [index_of_tau1, index_of_tau2],
    )


def find_mother_particle(mc_particle):
    parent_p = mc_particle.getObjectID().index
    counter = 0
    decayed_in_tracker = 1
    while decayed_in_tracker == 1:
        if type(parent_p) == list:
            parent_p = parent_p[0]
        parents = mc_particle.getParents()
        parent_p_r = []
        for parent in parents:
            parent_p_r.append(parent.getObjectID().index)
            decayed_in_tracker = parent.isDecayedInTracker() * 1
        pp_old = parent_p
        counter = counter + 1
        parent_p = parent_p_r
        if len(np.reshape(np.array(parent_p), -1)) > 1.5:
            parent_p = pp_old
            decayed_in_tracker = 0
        # print("found parent", parent_p, pp_old)
    return parent_p


def initialize(t, store_tau):
    if store_tau == "True":
        hit_genlink_tau = ROOT.std.vector("float")()
        t.Branch("hit_genlink_tau", hit_genlink_tau)
    event_number = array("i", [0])
    n_hit = array("i", [0])
    n_part = array("i", [0])

    hit_EDep = ROOT.std.vector("float")()
    hit_time = ROOT.std.vector("float")()

    hit_pathLength = ROOT.std.vector("float")()
    hit_x = ROOT.std.vector("float")()
    hit_y = ROOT.std.vector("float")()
    hit_z = ROOT.std.vector("float")()
    hit_px = ROOT.std.vector("float")()
    hit_py = ROOT.std.vector("float")()
    hit_pz = ROOT.std.vector("float")()
    # for digitized hits
    leftPosition_x = ROOT.std.vector("float")()
    leftPosition_y = ROOT.std.vector("float")()
    leftPosition_z = ROOT.std.vector("float")()
    rightPosition_x = ROOT.std.vector("float")()
    rightPosition_y = ROOT.std.vector("float")()
    rightPosition_z = ROOT.std.vector("float")()
    cluster_count = ROOT.std.vector("float")()

    hit_type = ROOT.std.vector("float")()
    hit_genlink0 = ROOT.std.vector("float")()
    part_p = ROOT.std.vector("float")()
    part_p_t = ROOT.std.vector("float")()
    part_theta = ROOT.std.vector("float")()
    part_phi = ROOT.std.vector("float")()
    part_m = ROOT.std.vector("float")()
    part_id = ROOT.std.vector("float")()
    part_parent = ROOT.std.vector("float")()
    part_pid = ROOT.std.vector("float")()
    hit_cellID = ROOT.std.vector("int")()
    superLayer = ROOT.std.vector("float")()
    layer = ROOT.std.vector("float")()
    phi = ROOT.std.vector("float")()
    stereo = ROOT.std.vector("float")()
    gen_status = ROOT.std.vector("float")()
    # cov4 = ROOT.std.vector("float")()
    # cov5 = ROOT.std.vector("float")()
    produced_by_secondary = ROOT.std.vector("float")()

    t.Branch("event_number", event_number, "event_number/I")
    t.Branch("n_hit", n_hit, "n_hit/I")
    t.Branch("n_part", n_part, "n_part/I")
    t.Branch("gen_status", gen_status)
    t.Branch("hit_x", hit_x)
    t.Branch("hit_y", hit_y)
    t.Branch("hit_z", hit_z)
    t.Branch("hit_pathLength", hit_pathLength)
    t.Branch("hit_px", hit_px)
    t.Branch("hit_py", hit_py)
    t.Branch("hit_pz", hit_pz)

    t.Branch("leftPosition_x", leftPosition_x)
    t.Branch("leftPosition_y", leftPosition_y)
    t.Branch("leftPosition_z", leftPosition_z)
    t.Branch("rightPosition_x", rightPosition_x)
    t.Branch("rightPosition_y", rightPosition_y)
    t.Branch("rightPosition_z", rightPosition_z)
    t.Branch("cluster_count", cluster_count)
    t.Branch("hit_type", hit_type)
    t.Branch("hit_EDep", hit_EDep)
    t.Branch("hit_time", hit_time)
    t.Branch("hit_cellID", hit_cellID)

    # Create a branch for the hit_genlink_flat

    t.Branch("hit_genlink0", hit_genlink0)
    t.Branch("part_p", part_p)
    t.Branch("part_p_t", part_p_t)
    t.Branch("part_theta", part_theta)
    t.Branch("part_phi", part_phi)
    t.Branch("part_m", part_m)
    t.Branch("part_pid", part_pid)
    t.Branch("part_id", part_id)
    t.Branch("superLayer", superLayer)
    t.Branch("layer", layer)
    t.Branch("phi", phi)
    t.Branch("stereo", stereo)
    t.Branch("part_parent", part_parent)
    t.Branch("produced_by_secondary", produced_by_secondary)
    # t.Branch("cov5", cov5)

    dic = {
        "hit_x": hit_x,
        "hit_y": hit_y,
        "hit_z": hit_z,
        "hit_type": hit_type,
        "hit_EDep": hit_EDep,
        "hit_time": hit_time,
        "hit_pathLength": hit_pathLength,
        "hit_genlink0": hit_genlink0,
        "hit_px": hit_px,
        "hit_py": hit_py,
        "hit_pz": hit_pz,
        "part_p": part_p,
        "part_p_t": part_p_t,
        "part_theta": part_theta,
        "part_phi": part_phi,
        "part_m": part_m,
        "part_pid": part_pid,
        "part_id": part_id,
        "gen_status": gen_status,
        "hit_cellID": hit_cellID,
        "leftPosition_x": leftPosition_x,
        "leftPosition_y": leftPosition_y,
        "leftPosition_z": leftPosition_z,
        "rightPosition_x": rightPosition_x,
        "rightPosition_y": rightPosition_y,
        "rightPosition_z": rightPosition_z,
        "produced_by_secondary": produced_by_secondary,
        "cluster_count": cluster_count,
        "superLayer": superLayer,
        "layer": layer,
        "phi": phi,
        "stereo": stereo,
        "part_parent": part_parent,
    }
    if store_tau == "True":
        dic["hit_genlink_tau"] = hit_genlink_tau
    return (event_number, n_hit, n_part, dic, t)


def read_mc_collection(event, dic, n_part, debug, unique_MCS):
    mc_particles = event.get("MCParticles")
    for jj, mc_particle in enumerate(mc_particles):

        pdg = mc_particle.getPDG()
        m = mc_particle.getMass()
        p_ = mc_particle.getMomentum()
        p = math.sqrt(p_.x**2 + p_.y**2 + p_.z**2)
        p_t = math.sqrt(p_.x**2 + p_.y**2)
        object_id_particle = mc_particle.getObjectID()
        genlink0_particle = object_id_particle.index
        # only store particles that have hits
        if np.sum(unique_MCS == jj) > 0:
            if p > 0:
                theta = math.acos(p_.z / p)
                phi = math.atan2(p_.y, p_.x)
                dic["part_p"].push_back(p)
                dic["part_p_t"].push_back(p_t)
                dic["part_theta"].push_back(theta)
                dic["part_phi"].push_back(phi)
                dic["part_m"].push_back(m)
                dic["part_pid"].push_back(pdg)
                dic["part_id"].push_back(genlink0_particle)
                # jjz = jjz + 1
            else:
                theta = 0.0
                phi = 0.0
                dic["part_p"].push_back(p)
                dic["part_p_t"].push_back(p_t)
                dic["part_theta"].push_back(theta)
                dic["part_phi"].push_back(phi)
                dic["part_m"].push_back(m)
                dic["part_pid"].push_back(pdg)
                dic["part_id"].push_back(genlink0_particle)
            parents = mc_particle.getParents()
            dic["part_parent"].push_back(parents[0].getObjectID().index)
            dic["gen_status"].push_back(mc_particle.getGeneratorStatus())
        if debug and  np.sum(unique_MCS == jj) > 0:
            if mc_particle.getGeneratorStatus() ==1:
                print("gen status 1 part")
                print(
                    "all genparts: N: {}, PID: {}, Q: {}, P: {:.2e}, status: {}, parents: {}, daughters: {}, decayed_traacker: {}".format(
                        jj,
                        mc_particle.getPDG(),
                        mc_particle.getCharge(),
                        p,

                        mc_particle.getGeneratorStatus(),
                        get_genparticle_parents(
                            genlink0_particle,
                            mc_particles,
                        ),
                        get_genparticle_daughters(
                            genlink0_particle,
                            mc_particles,
                        ),
                        mc_particle.isDecayedInTracker() * 1,
                    )
                )
        if np.sum(unique_MCS == jj) > 0:
            n_part[0] += 1

    return n_part, dic


def clear_dic(dic):
    for key in dic:
        dic[key].clear()
    return dic


def store_hit_col_CDC(
    event,
    n_hit,
    dic,
    metadata,
    gen_part_coll,
    index_taus,
    store_tau="False",
):

    dc_hits_digi = event.get("CDCHDigis")

    dc_hits = event.get("CDCHHits")
    n_hit[0] = 0
    ii = 0
    number_of_hits_p_seatched = 0
    list_of_MCs1 = []
    for num_hit, dc_hit in enumerate(dc_hits):
        # if i > 2:
        #     break
        # print("   New hit: ", ii)
        cellID = dc_hit.getCellID()
        EDep = dc_hit.getEDep()
        time = dc_hit.getTime()
        dc_hit_digi = dc_hits_digi[num_hit]
        rightPosition = dc_hit_digi.getRightPosition()
        leftPosition = dc_hit_digi.getLeftPosition()
        produced_by_secondary = dc_hit.isProducedBySecondary()
        dic["leftPosition_x"].push_back(leftPosition.x)
        dic["leftPosition_y"].push_back(leftPosition.y)
        dic["leftPosition_z"].push_back(leftPosition.z)
        dic["rightPosition_x"].push_back(rightPosition.x)
        dic["rightPosition_y"].push_back(rightPosition.y)
        dic["rightPosition_z"].push_back(rightPosition.z)
        dic["produced_by_secondary"].push_back(1.0 * produced_by_secondary)
        cluster_count = dc_hit_digi.getClusterCount()
        dic["cluster_count"].push_back(cluster_count)

        pathLength = dc_hit.getPathLength()
        position = dc_hit.getPosition()
        x = position.x
        y = position.y
        z = position.z
        momentum = dc_hit.getMomentum()
        px = momentum.x
        py = momentum.y
        pz = momentum.z
        dic["hit_x"].push_back(x)
        dic["hit_y"].push_back(y)
        dic["hit_z"].push_back(z)
        p = math.sqrt(px * px + py * py + pz * pz)
        dic["hit_px"].push_back(px)
        dic["hit_py"].push_back(py)
        dic["hit_pz"].push_back(pz)
        htype = 0
        # dummy example, cellid_encoding = "foo:2,bar:3,baz:-4"
        cellid_encoding = metadata.get_parameter("CDCHHits__CellIDEncoding")
        # 2. Use DD4hep (Andre's magic) to create a decoder object, and then retrieve the field value by using the get method of the decoder object
        decoder = dd4hep.BitFieldCoder(cellid_encoding)
        superLayer = decoder.get(cellID, "superLayer")
        layer = decoder.get(cellID, "layer")
        phi = decoder.get(cellID, "phi")
        stereo = decoder.get(cellID, "stereo")

        dic["hit_cellID"].push_back(cellID)
        dic["hit_EDep"].push_back(EDep)
        dic["hit_time"].push_back(time)
        dic["hit_pathLength"].push_back(pathLength)
        dic["hit_type"].push_back(htype)
        dic["superLayer"].push_back(superLayer)
        dic["layer"].push_back(layer)
        dic["phi"].push_back(phi)
        dic["stereo"].push_back(stereo)

        mcParticle = dc_hit.getMCParticle()
        # print(dir(mcParticle))
        # this is to check that we are considering the correct particle
        # pdg_particle = mcParticle.getPDG()
        object_id = mcParticle.getObjectID()
        # # print(object_id.index)
        genlink0 = object_id.index
        if store_tau == "True":
            p, belongs_to_tau_1, belongs_to_tau_2 = find_mother_particle_check_tau(
                genlink0, gen_part_coll, index_taus[0], index_taus[1]
            )
            if belongs_to_tau_1:
                index_tau = index_taus[0]
            elif belongs_to_tau_2:
                index_tau = index_taus[1]
            else:
                index_tau = -1
            dic["hit_genlink_tau"].push_back(index_tau)

        if genlink0 == 74:
            number_of_hits_p_seatched = number_of_hits_p_seatched + 1
        # print(pdg_particle, genlink0, object_id.index)
        # mcParticle_mother_index = find_mother_particle(mcParticle)
        dic["hit_genlink0"].push_back(genlink0)
        list_of_MCs1.append(genlink0)
        # hit_pdg_particle.push_back(pdg_particle)
        ii += 1
        n_hit[0] += 1
    print("number_of_hits_p_seatched", number_of_hits_p_seatched)
    return n_hit, dic, list_of_MCs1


def store_hit_col_VTX(
    event,
    n_hit,
    dic,
    gen_part_coll,
    index_taus,
    store_tau="False",
):
    number_of_hits_p_seatched = 0
    vtx_hits = event.get("VTXDCollection")
    vtxi_hits = event.get("VTXIBCollection")
    vtxo_hits = event.get("VTXOBCollection")
    vtx_hits_digi = event.get("VTXDDigis")
    vtxi_hits_digi = event.get("VTXIBDigis")
    vtxo_hits_digi = event.get("VTXOBDigis")
    hit_collections = [vtx_hits, vtxi_hits, vtxo_hits]
    hit_collections_digi = [vtx_hits_digi, vtxi_hits_digi, vtxo_hits_digi]
    list_of_MCs1 = []
    for coll_number, coll in enumerate(hit_collections):
        ii = 0
        for dc_hit_index, dc_hit in enumerate(coll):
            dc_hit_digi = hit_collections_digi[coll_number][dc_hit_index]
            covMatrix = dc_hit_digi.getCovMatrix()
            cellID = dc_hit.getCellID()
            EDep = dc_hit_digi.getEDep()
            time = dc_hit_digi.getTime()
            pathLength = dc_hit.getPathLength()
            position = dc_hit_digi.getPosition()
            x = position.x
            y = position.y
            z = position.z
            momentum = dc_hit.getMomentum()
            px = momentum.x
            py = momentum.y
            pz = momentum.z
            htype = 1
            dic["hit_cellID"].push_back(cellID)
            dic["hit_EDep"].push_back(EDep)
            dic["hit_time"].push_back(time)
            dic["hit_pathLength"].push_back(pathLength)
            dic["hit_x"].push_back(x)
            dic["hit_y"].push_back(y)
            dic["hit_z"].push_back(z)
            dic["hit_px"].push_back(px)
            dic["hit_py"].push_back(py)
            dic["hit_pz"].push_back(pz)
            dic["hit_type"].push_back(htype)
            produced_by_secondary = dc_hit.isProducedBySecondary()
            # dic["cov0"].push_back(covMatrix[0])
            # dic["cov1"].push_back(covMatrix[1])
            # dic["cov2"].push_back(covMatrix[2])
            # dic["cov3"].push_back(covMatrix[3])
            # dic["cov4"].push_back(covMatrix[4])
            # dic["cov5"].push_back(covMatrix[5])
            dic["superLayer"].push_back(0)
            dic["layer"].push_back(0)
            dic["phi"].push_back(0)
            dic["stereo"].push_back(0)
            dic["leftPosition_x"].push_back(0)
            dic["leftPosition_y"].push_back(0)
            dic["leftPosition_z"].push_back(0)
            dic["rightPosition_x"].push_back(0)
            dic["rightPosition_y"].push_back(0)
            dic["rightPosition_z"].push_back(0)
            dic["cluster_count"].push_back(0)
            dic["produced_by_secondary"].push_back(1.0 * produced_by_secondary)
            mcParticle = dc_hit.getMCParticle()
            # # pdg_particle = mcParticle.getPDG()
            object_id = mcParticle.getObjectID()
            genlink0 = object_id.index
            if genlink0 == 74:
                number_of_hits_p_seatched = number_of_hits_p_seatched + 1
            if store_tau == "True":
                p, belongs_to_tau_1, belongs_to_tau_2 = find_mother_particle_check_tau(
                    genlink0, gen_part_coll, index_taus[0], index_taus[1]
                )
                if belongs_to_tau_1:
                    index_tau = index_taus[0]
                elif belongs_to_tau_2:
                    index_tau = index_taus[1]
                else:
                    index_tau = -1
                dic["hit_genlink_tau"].push_back(index_tau)

            # print("VTX")
            # mcParticle_mother_index = find_mother_particle(mcParticle)
            # if genlink0 == 69:
            #     print("genlink0", genlink0)
            #     print("position1", x, y, z)
            #     print(
            #         "position 2",
            #         dc_hit.getPosition().x,
            #         dc_hit.getPosition().y,
            #         dc_hit.getPosition().z,
            #     )
            #     print("1")
            dic["hit_genlink0"].push_back(genlink0)
            list_of_MCs1.append(genlink0)
            ii += 1
            n_hit[0] += 1
    print("number_of_hits_p_seatched   VTX",number_of_hits_p_seatched )
    return n_hit, dic, list_of_MCs1


def merge_list_MCS(list_1, list_2,store_tau, index_taus):
    list_1 = np.array(list_1)
    list_2 = np.array(list_2)
    unique_m1 = np.unique(list_1)
    unique_m2 = np.unique(list_2)
    if store_tau == "True":
        unique_mc = np.concatenate((unique_m1, unique_m2, index_taus), axis=0)
    else:
        unique_mc = np.concatenate((unique_m1, unique_m2), axis=0)
    
    print("unique_mc", unique_mc)
    return unique_mc
