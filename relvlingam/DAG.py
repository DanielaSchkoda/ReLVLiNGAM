from scipy.linalg import inv
import numpy as np
import networkx as nx
import numpy.linalg as LA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def path_matrix_to_adjacency(path_matrix, tol=1e-10):
    try:
        p = path_matrix.shape[0]
        Lambda = -inv(path_matrix[:p, :p]) + np.eye(p)
        Gamma = (np.eye(p)-Lambda) @ path_matrix[:p, p:]
        adj = np.hstack((Lambda, Gamma))
        adj[np.isclose(adj, 0, atol=tol)] = 0
    except Exception:
        adj = np.full(path_matrix.shape[1], np.nan)
    return adj

def adjacency_to_path_matrix(adjacency):
    p = adjacency.shape[0]
    Lambda = adjacency[:p, :p]
    Gamma = adjacency[:p, p:]
    return np.linalg.inv(np.eye(p)-Lambda) @ np.hstack((np.eye(p), Gamma))

def confounders_btw_pairs(Lambda, Gamma, tol=1e-10):
    p = Lambda.shape[0]
    G = _adjacency_to_nx_graph(np.hstack((Lambda, Gamma)), tol)
    confounders_btw_pairs = np.array([[len(comon_latent_confounders(w, v, G, p, tol)) for w in range(p)] for v in range(p)])
    np.fill_diagonal(confounders_btw_pairs, -1)
    return confounders_btw_pairs

def _adjacency_to_nx_graph(adjacency, tol=1e-10):
    p = adjacency.shape[0]
    q = adjacency.shape[1]
    extended_adjacency = np.vstack((adjacency, np.zeros((q-p, q))), dtype=float)
    extended_adjacency[np.abs(extended_adjacency) < tol] = 0
    # networkx uses transposed adjacency matrix compared to my setup
    G = nx.DiGraph(np.transpose(extended_adjacency))
    return G 

def comon_latent_confounders(v, w, adjacency, p, tol=1e-10):
    if v > w:
        v, w = w, v
    G = _adjacency_to_nx_graph(adjacency, tol)
    return [q for q in range(p, G.number_of_nodes()) if nx.has_path(G, q, v) and len([path for path in nx.all_simple_paths(G, source=q, target=w) if v not in path]) > 0]

def confounders_btw_pairs_during_iteration(Lambda, Gamma, tol=1e-10):
    upper_triangle = np.triu(Lambda, k=1)
    if not np.all(np.isclose(upper_triangle, 0, atol=tol)):
        raise ValueError("This function requires the nodes to be in topological order, i.e. Lambda to be upper triangular.")
    p = Lambda.shape[0]
    adjacency = np.hstack((Lambda, Gamma))
    # Additiional constraint that v or w is olderst child of q
    confounders_btw_pairs = np.array([[len([q for q in comon_latent_confounders(w, v, adjacency, p, tol) if np.all(np.abs(Gamma[:v, q-p]) < tol) or np.all(np.abs(Gamma[:w, q-p]) < tol)]) for w in range(p)] for v in range(p)])
    np.fill_diagonal(confounders_btw_pairs, -1)
    return confounders_btw_pairs

def plot_dag(adjacency, tol=1e-10):
    p = adjacency.shape[0]
    q = adjacency.shape[1]
    G = _adjacency_to_nx_graph(adjacency, tol)
    plt.figure(figsize=(4, 3)) 
    colors = {**{observed_node: '#1f78b4' for observed_node in range(p)}, **{latent_node: 'gray' for latent_node in range(p, q)}}
    nx.draw(G, with_labels=True, font_weight='bold', node_color=[colors[node] for node in G.nodes])
    plt.show()

def topological_order(path_matrix, tol=1e-10):
    p = path_matrix.shape[0]
    # First check for nans to check whether no next source was found in some iteration
    positions_nans = np.where(np.any(np.isnan(path_matrix[:,:p]), axis=1))[0]
    nr_parents = np.sum(np.abs(path_matrix[:,:p])>tol, axis=1) # comparison yields false for nans so they (wrongly) will be in beginning of top order. Therfore they a re removed again later
    topological_order = np.argsort(nr_parents)
    # remove positions_nans
    topological_order = np.delete(topological_order, positions_nans)
    topological_order = np.hstack((topological_order, [np.nan]*len(positions_nans)))
    return topological_order

def causal_paths(path_matrix, tol=1e-10):
    return np.abs(path_matrix) > tol

def is_compatible_topological_order(topological_order, path_matrix, tol=1e-10):
    if np.any(np.isnan(topological_order)):
        return False
    topological_order_int = [int(x) for x in topological_order]
    permuted_path_matrix = path_matrix[topological_order_int][:, topological_order_int]
    return np.allclose(permuted_path_matrix, np.tril(permuted_path_matrix), atol=tol)

def is_source(node, path_matrix, tol=1e-2):
    p = path_matrix.shape[0]
    other_nodes = [v for v in range(p) if v != node]
    return np.all(np.logical_or(np.isnan(path_matrix[node, other_nodes]), np.abs(path_matrix[node, other_nodes])<tol))

# This gives sparsest B
def get_sparsest_B(path_matrix, possible_permutations, tol=1e-2):
    argmin_index = np.argmin([np.sum(np.abs(path_matrix[:,perm]) < tol) for perm in possible_permutations])
    all_argmin_indices = np.where(np.array([np.sum(np.abs(path_matrix[:,perm]) < tol) for perm in possible_permutations]) == np.sum(np.abs(path_matrix[:,possible_permutations[argmin_index]]) < tol))[0]
    # If there are multiple Bs with the same sparsity, choose the one with the smallest sum of absolute values of the close to zero entries
    if len(all_argmin_indices) > 1:
        new_argmin = np.argmin([absolute_sum_of_close_to_zero_entries(path_matrix[:,possible_permutations[ind]], tol) for ind in all_argmin_indices])
        argmin_index = all_argmin_indices[new_argmin]
    perm = possible_permutations[argmin_index]
    return path_matrix[:,perm]

def absolute_sum_of_close_to_zero_entries(B, tol=1e-2):
    return np.sum(np.abs(np.where(np.abs(B) < tol, B, 0)))

def scale_columns(B):
    max_entries = B[np.abs(B).argmax(axis=0), np.arange(B.shape[1])]
    max_entries_B = np.repeat(max_entries[np.newaxis, :], B.shape[0], axis=0)
    B = B / max_entries_B
    return B

def get_closest_B(B_hat, true_B, possible_permutations):
    if np.all(np.isnan(B_hat)):
        return B_hat
    # Padding
    p = B_hat.shape[0]
    if B_hat.shape[1] < true_B.shape[1]:
        additional_columns = true_B.shape[1] - B_hat.shape[1]
        all_Bs = [np.hstack((B_hat[:,perm], np.zeros((p, additional_columns)))) for perm in possible_permutations]
    elif true_B.shape[1] < B_hat.shape[1]:
        true_B = np.hstack((true_B, np.zeros((p, B_hat.shape[1] - true_B.shape[1]))))
        all_Bs = [B_hat[:,perm] for perm in possible_permutations]
    else:
        all_Bs = [B_hat[:,perm] for perm in possible_permutations]
    # Search for the best B_hat ignoring nans
    min_q = min(B_hat.shape[1], true_B.shape[1])
    argmin_index = np.argmin([LA.norm(np.nan_to_num(B[:,:min_q] - true_B[:,:min_q])) for B in all_Bs])
    return all_Bs[argmin_index]

def precision(B_hat, True_B, tol_true=1e-10, tol_hat=1e-2):
    min_q = min(B_hat.shape[1], True_B.shape[1])
    estimated_causal_paths = np.abs(B_hat) > tol_hat
    true_causal_paths = np.abs(True_B) > tol_true
    true_recovered_paths = np.sum(estimated_causal_paths[:,:min_q] & true_causal_paths[:,:min_q])
    recovered_paths = np.sum(estimated_causal_paths)
    return true_recovered_paths/recovered_paths

def recall(B_hat, True_B, tol_true=1e-10, tol_hat=1e-2):
    min_q = min(B_hat.shape[1], True_B.shape[1])
    estimated_causal_paths = np.abs(B_hat) > tol_hat 
    true_causal_paths = np.abs(True_B) > tol_true
    true_recovered_paths = np.sum(estimated_causal_paths[:,:min_q] & true_causal_paths[:,:min_q])
    nr_true_causal_paths = np.sum(true_causal_paths)
    return true_recovered_paths/nr_true_causal_paths

def RMSE(B_hat, True_B):
    if np.any(np.isnan(B_hat)):
        return np.nan
    if B_hat.shape[1] < True_B.shape[1]:
        B_hat = np.pad(B_hat, ((0, 0), (0, True_B.shape[1] - B_hat.shape[1])), 'constant', constant_values=0)
    elif True_B.shape[1] < B_hat.shape[1]:
        True_B = np.pad(True_B, ((0, 0), (0, B_hat.shape[1] - True_B.shape[1])), 'constant', constant_values=0)
    return np.sqrt(mean_squared_error(B_hat, True_B)) 