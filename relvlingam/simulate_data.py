import numpy as np
from relvlingam.DAG import adjacency_to_path_matrix

def get_Lambda(edges, p):
    return np.array([[1 if (v, w) in edges else 0 for w in range(p)] for v in range(p)])

def get_Gamma(edges, p, q):
    return np.array([[1 if (v, w) in edges else 0 for w in range(p, q)] for v in range(p)])

all_settings = {
    "a": {'Lambda': np.tril(np.ones((2, 2)), k=-1), 'Gamma': np.ones((2, 1)), 'highest_l': 1},
    "b": {'Lambda': get_Lambda([(1,0), (2, 1)], 3), 'Gamma': np.array([[0], [1], [1]]), 'highest_l': 1},
    "c": {'Lambda': get_Lambda([(1,0), (2,1)], 3), 'Gamma': np.array([[1, 0], [1, 1], [0, 1]]), 'highest_l': 1},
    "d": {'Lambda': np.tril(np.ones((3, 3)), k=-1), 'Gamma': np.ones((3, 1)), 'highest_l': 1},    
    "e": {'Lambda': np.tril(np.ones((3, 3)), k=-1), 'Gamma':np.ones((3, 2)), 'highest_l': 2},
    'f': {'Lambda': get_Lambda([(1,0), (2,0), (3,1), (4,1)], 5), 
          'Gamma': np.array([[1, 0], [1, 1], [0, 0], [1, 0], [0, 1]]), 'highest_l': 1}
}

def simulate_data(n, noise_distribution, Lambda, Gamma, permute_order=True):
    """
    Simulate data according to a linear non-Gaussian acyclic model with latent confounfing.

    Parameters:
    - n (int): Sample size.
    - noise_distribution (str): Distribution for exogeneous sources epsilon, L. Options are "gamma", "beta", or "lognormal".
    - Lambda (ndarray): Coefficient matrix for the direct effects among observed variables.
    - Gamma (ndarray): Coefficient matrix for the direct effects from latent on observed variables.
    - permute_order (bool, optional): Whether to randonmly permute the order of variables. Otherwise X is in topological order. Defaults to True.

    Returns:
    - X (ndarray): Generated data with shape (n, p), where p is the number of variables.
    - B (ndarray): Path matrix with shape (q, p), where q is the total number of variables including noise variables.
    """
    adjacency = np.hstack((Lambda, Gamma))
    p, q = adjacency.shape

    adjacency = np.where(adjacency == 1, np.random.choice([-1, 1], adjacency.shape) * np.random.uniform(0.5, 0.9, adjacency.shape), 0)
    oldest_children = np.argmax(adjacency[:, p:] != 0, axis=0)
    adjacency[oldest_children, range(p, q)] = 1

    if permute_order:
        permutation = np.random.permutation(p)
        adjacency = adjacency[permutation]
        adjacency[:, :p] = adjacency[:, permutation]
    
    B = adjacency_to_path_matrix(adjacency)
    eta = sample_eta(n, q, noise_distribution)
    # Fortran array as required by moment estimation function
    X = np.asfortranarray(eta @ np.transpose(B))
    return X, B

def sample_eta(n, q, noise_distribution):
    """
    Generate samples for eta from a specified distribution.

    Parameters:
    - n (int): Sample size.
    - q (int): Number of exogeneous sources, i.e. l+p.
    - noise_distribution (str): Distribution. Options are "gamma", "beta", or "lognormal".

    Returns:
    - ndarray: Generated samples for eta with shape (n, q).
    """
    if noise_distribution == "gamma":
        shapes = np.random.uniform(0.1, 1, q)
        scales = np.random.uniform(0.1, 0.5, q)
        eta = np.array([np.random.gamma(shape, scale, n) - shape * scale for shape, scale in zip(shapes, scales)])
    elif noise_distribution == "beta":
        alphas = np.random.uniform(1.5, 2, q)
        betas = np.random.uniform(2, 10, q)
        eta = np.array([np.random.beta(alpha, beta, n) - alpha / (alpha + beta) for alpha, beta in zip(alphas, betas)])
    elif noise_distribution == "lognormal":
        mus = np.random.uniform(-2, -0.5, q)
        sigmas = np.random.uniform(0.1, 0.4, q)
        eta = np.array([np.random.lognormal(mu, sigma, n) - np.exp(mu + sigma ** 2 / 2) for mu, sigma in zip(mus, sigmas)])
    
    return np.transpose(eta)