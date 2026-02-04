import torch
from sklearn.metrics import adjusted_rand_score

def find_condpoints(betas, unassigned, tbeta):
    n_points = betas.size(0)
    device = betas.device
    mask_unassigned = torch.zeros(n_points, device=device)
    mask_unassigned[unassigned] = 1
    select_condpoints = (mask_unassigned.bool()) & (betas > tbeta)
    indices_condpoints = select_condpoints.nonzero(as_tuple=False)
    if len(indices_condpoints) == 0:
        return indices_condpoints
    indices_condpoints = indices_condpoints[(-betas[select_condpoints]).argsort()]
    return indices_condpoints

def get_clustering(betas: torch.Tensor, X: torch.Tensor, tbeta=0.7, td=0.05):
    n_points = betas.size(0)
    select_condpoints = betas > tbeta
    indices_condpoints = select_condpoints.nonzero(as_tuple=False)
    if len(indices_condpoints) == 0:
        return -1*torch.ones(n_points, dtype=torch.long)
    indices_condpoints = indices_condpoints[(-betas[select_condpoints]).argsort()]

    unassigned = torch.arange(n_points)
    clustering = -1*torch.ones(n_points, dtype=torch.long)

    while len(indices_condpoints) > 0 and len(unassigned) > 0:
        index_condpoint = indices_condpoints[0]
        d = torch.norm(X[unassigned] - X[index_condpoint][0], dim=-1)
        assigned_to_this_condpoint = unassigned[d < td]
        clustering[assigned_to_this_condpoint] = index_condpoint[0]
        unassigned = unassigned[~(d < td)]

        indices_condpoints = find_condpoints(betas, unassigned, tbeta)

    return clustering

def read_file_as_tensor(filename):
    X_list = []
    beta_list = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Element"):
                vals = line.split('=')[1].strip().strip('[]')
                vals = [float(v) for v in vals.split(',')]
                X_list.append(vals[:3])
                beta_list.append(vals[3])
    X = torch.tensor(X_list, dtype=torch.float)
    betas = torch.tensor(beta_list, dtype=torch.float)
    return X, betas

file1 = "/afs/cern.ch/work/a/adevita/public/GGTF_trackFinder_training/conversion_to_onnx/output_dump_cluster.txt"
file2 = "/afs/cern.ch/work/a/adevita/public/GGTF_trackFinder_training/conversion_to_onnx/output_dump.txt"

X1, betas1 = read_file_as_tensor(file1)
X2, betas2 = read_file_as_tensor(file2)

clustering1 = get_clustering(betas1, X1, tbeta=0.005, td=0.005)
clustering2 = get_clustering(betas2, X2, tbeta=0.005, td=0.005)

ari = adjusted_rand_score(clustering1.tolist(), clustering2.tolist())
print(f"Adjusted Rand Index between the two clusterings: {ari:.4f}")

same_cluster = (clustering1 == clustering2).sum().item()
total_points = len(clustering1)
print(f"Points with identical cluster assignment: {same_cluster}/{total_points} ({same_cluster/total_points*100:.2f}%)")
