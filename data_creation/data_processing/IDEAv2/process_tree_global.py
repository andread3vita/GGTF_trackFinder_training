from podio import root_io
import edm4hep
import sys
import ROOT
from ROOT import TFile, TTree

from array import array
import math
from tools_tree_global import (
    read_mc_collection,
    clear_dic,
    initialize,
    store_hit_col_CDC,
    store_hit_col_VTX,
    merge_list_MCS,
    gen_particles_find,
)

debug = False
rootfile = sys.argv[1]

reader = root_io.Reader(rootfile)
output_file = sys.argv[2]
store_tau = sys.argv[3]
# 1. Retrieve the cellid_encoding (a string) from the metadata tree in the ROOT file, by using podio
metadata = reader.get("metadata")[0]

out_root = TFile(output_file, "RECREATE")
t = TTree("events", "pf tree lar")
event_number, n_hit, n_part, dic, t = initialize(t, store_tau)


event_number[0] = 0
event_numbers = 0
i = 0
for event in reader.get("events"):
    (
        genpart_indexes_pre,
        indexes_genpart_pre,
        n_part_pre,
        total_e,
        e_pp,
        gen_part_coll,
        index_taus,
    ) = gen_particles_find(event, debug, store_tau)
    # if event_numbers == 8:
    clear_dic(dic)
    n_part[0] = 0
    n_hit, dic, list_of_MCs1 = store_hit_col_CDC(
        event,
        n_hit,
        dic,
        metadata,
        gen_part_coll,
        index_taus=index_taus,
        store_tau=store_tau,
    )
    n_hit, dic, list_of_MCs2 = store_hit_col_VTX(
        event,
        n_hit,
        dic,
        gen_part_coll,
        index_taus=index_taus,
        store_tau=store_tau,
    )
    unique_MCS = merge_list_MCS(list_of_MCs1, list_of_MCs2, store_tau, index_taus)
    n_part, dic = read_mc_collection(event, dic, n_part, debug, unique_MCS)

    event_number[0] += 1
    t.Fill()


t.SetDirectory(out_root)
t.Write()
