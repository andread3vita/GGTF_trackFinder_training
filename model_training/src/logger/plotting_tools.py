import plotly.express as px
import dgl
import torch
import pandas as pd
import numpy as np
import os
import wandb

def PlotCoordinates(
    g,
    path,
    outdir,
    num_layer=0,
    predict=False,
    egnn=False,
    features_type="ones",
    epoch="",
    step_count=0,
):
    if predict:
        outdir = outdir + "/figures_evaluation"
    else:
        outdir = outdir + "/figures"
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    name = path
    graphs = dgl.unbatch(g)
    for i in range(0, 1):
        graph_i = graphs[i]
        if path == "input_coords":
            coords = graph_i.ndata["original_coords"]
            if egnn:
                features = graph_i.ndata["h"][:, 4]
            elif features_type == "ones":
                features = torch.ones_like(coords[:, 0]).view(-1, 1)
            else:
                features = graph_i.ndata["h"][:, -2]  # consider energy for size

        if path == "gravnet_coord":
            coords = graph_i.ndata["gncoords"]
            if egnn:
                features = graph_i.ndata["h"][:, 4]
            else:
                features = graph_i.ndata["h"][:, -2]
        if path == "final_clustering":
            coords = graph_i.ndata["final_cluster"]
            features = torch.sigmoid(graph_i.ndata["beta"])

        tidx = graph_i.ndata["particle_number"]
        data = {
            "X": coords[:, 0].view(-1, 1).detach().cpu().numpy(),
            "Y": coords[:, 1].view(-1, 1).detach().cpu().numpy(),
            "Z": coords[:, 2].view(-1, 1).detach().cpu().numpy(),
            "tIdx": tidx.view(-1, 1).detach().cpu().numpy(),
            "features": features.view(-1, 1).detach().cpu().numpy(),
        }
        hoverdict = {}
        # if hoverfeat is not None:
        #     for j in range(hoverfeat.shape[1]):
        #         hoverdict["f_" + str(j)] = hoverfeat[:, j : j + 1]
        #     data.update(hoverdict)

        # if nidx is not None:
        #     data.update({"av_same": av_same})

        df = pd.DataFrame(
            np.concatenate([data[k] for k in data], axis=1),
            columns=[k for k in data],
        )
        df["orig_tIdx"] = df["tIdx"]
        # rdst = np.random.RandomState(1234567890)  # all the same
        # shuffle_truth_colors(df, "tIdx", rdst)

        # hover_data = ["orig_tIdx", "idx"] + [k for k in hoverdict.keys()]
        # if nidx is not None:
        #     hover_data.append("av_same")
        fig = px.scatter_3d(
            df,
            x="X",
            y="Y",
            z="Z",
            color="tIdx",
            size="features",
            # hover_data=hover_data,
            template="plotly_dark",
            color_continuous_scale=px.colors.sequential.Rainbow,
        )
        fig.update_traces(marker=dict(line=dict(width=0)))
        if path == "gravnet_coord":
            fig.write_html(
                outdir + "/" + name + "_" + num_layer + "_" + str(i) + epoch + ".html"
            )
        else:
            print(
                outdir
                + "/"
                + name
                + "_"
                + epoch
                + "_"
                + str(step_count)
                + "_"
                + str(i)
                + ".html"
            )
            fig.write_html(
                outdir
                + "/"
                + name
                + "_"
                + epoch
                + "_"
                + str(step_count)
                + "_"
                + str(i)
                + ".html"
            )


def shuffle_truth_colors(df, qualifier="truthHitAssignementIdx", rdst=None):
    ta = df[qualifier]
    unta = np.unique(ta)
    unta = unta[unta > -0.1]
    if rdst is None:
        np.random.shuffle(unta)
    else:
        rdst.shuffle(unta)
    out = ta.copy()
    for i in range(len(unta)):
        out[ta == unta[i]] = i
    df[qualifier] = out
