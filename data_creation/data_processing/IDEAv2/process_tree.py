from podio import root_io
import edm4hep
import sys
import ROOT
from ROOT import TFile, TTree

from array import array
import math
from tools_tree import (
    read_mc_collection,
    clear_dic,
    initialize,
    store_hit_col_CDC,
    store_hit_col_VTX,
)

debug = False
rootfile = sys.argv[1]

reader = root_io.Reader(rootfile)
output_file = sys.argv[2]

# 1. Retrieve the cellid_encoding (a string) from the metadata tree in the ROOT file, by using podio
metadata = reader.get("metadata")[0]

out_root = TFile(output_file, "RECREATE")
t = TTree("events", "pf tree lar")
event_number, n_hit, n_part, dic, t = initialize(t)


event_number[0] = 0
i = 0
for event in reader.get("events"):
    clear_dic(dic)
    n_part[0] = 0
    if debug:
        print("")
        print(" ----- new event: {} ----------".format(event_number[0]))
        print("")

    n_part, dic = read_mc_collection(event, dic, n_part)
    n_hit, dic = store_hit_col_CDC(event, n_hit, dic, metadata)
    n_hit, dic = store_hit_col_VTX(event, n_hit, dic)

    event_number[0] += 1
    t.Fill()

t.SetDirectory(out_root)
t.Write()
