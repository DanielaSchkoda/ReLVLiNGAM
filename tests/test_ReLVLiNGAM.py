import unittest
import numpy as np
import itertools as it
from sympy import Array, simplify
import random 
import numpy.linalg as LA

from relvlingam.ReLVLiNGAM import ReLVLiNGAM

class MockReLVLiNGAM(ReLVLiNGAM):
    """ReLVLiNGAM class to use cumulants under infinite sample size rather than estimating them from a finite sample."""
    def __init__(self, X, highest_l, true_B, true_omegas):
        super().__init__(X, highest_l=highest_l, threshold_power=0, thresholds=[1e-5, 2e-5])
        self.n = np.inf
        self.true_B = true_B
        self.true_omegas = true_omegas
        
    def _estimate_cumulants(self, X, *args):
        k2 = self.highest_order
        return {k: np.array(self._calculate_true_cumulant(k), dtype=float) for k in range(2, k2+1)}

    def _calculate_true_cumulant(self, k):
        p, q = self.true_B.shape
        cumulant = Array([simplify(sum([ self.true_omegas[a, k-2]*np.prod(self.true_B[ind,a]) for a in range(q)])) for ind in it.product(range(p), repeat=k)], (p,)*k)#.reshape()
        return cumulant

    
class TestReLVLiNGAM(unittest.TestCase):
    def setUp(self):
        random.seed(10)

    def test_fit(self):
        p = 3

        # Test the fit method
        # Need X since model initilizes self.p with X
        X = np.zeros((0, p))
        true_B = np.array([[ 1.,  0.,  0.,  1. ],
                            [-1.3142598,  1.,  0.,  0.47955378],
                            [ 1.68266478, -1.7050696,  1.,  0.10347315]])
        true_omegas = np.array([[ 1.50924438, -1.40294611,  0.83518742],
                                [ 0.83916154,  1.19641473,  1.43729886],
                                [ 1.81069409,  1.41449443,  0.59139172],
                                [ 1.08915941,  1.72865295,  1.85664261]])
        model = MockReLVLiNGAM(X, highest_l=1, true_B=true_B, true_omegas=true_omegas)
        model.fit(X)
        all_possible_permutations = model.get_all_possible_permutations()
        best_permutation = all_possible_permutations[np.argmin([LA.norm(np.nan_to_num(model.B[:,perm] - true_B)) for perm in all_possible_permutations])]
        B_hat = model.B[:,best_permutation]
        omega_hat = model.omegas[best_permutation]

        self.assertTrue(np.allclose(B_hat, true_B))
        self.assertTrue(np.allclose(omega_hat, true_omegas))

        # Need X since model initilizes self.p with X
        X = np.zeros((0, p))
        true_B = np.array([[ 1.        ,  0.        ,  0.        ,  1.        ,  1.        ],
                            [ 0.91566425,  1.        ,  0.        , -0.74871774,  1.8378342 ],
                            [ 2.47411986,  1.6583288 ,  1.        , -1.99660408,  2.1862876 ]])
        true_omegas = np.array([[ 1.49183705, -0.65572101,  0.56157814,  1.66630208,  1.04755528],
                                [ 1.78648017,  0.79282877,  1.57507352,  1.17019068,  0.6244907 ],
                                [ 0.95909245,  1.41018651,  0.97093479,  1.67533183,  1.13705284],
                                [ 0.79217525,  1.24837243,  1.52295585, -0.66135885,  1.84713367],
                                [ 0.86232169,  1.64624653,  1.38307356,  1.13445211,  0.94921284]])
        
        
        model = MockReLVLiNGAM(X, highest_l=2, true_B=true_B, true_omegas=true_omegas)
        model.fit(X)
        all_possible_permutations = model.get_all_possible_permutations()
        best_permutation = all_possible_permutations[np.argmin([LA.norm(np.nan_to_num(model.B[:,perm] - true_B)) for perm in all_possible_permutations])]
        B_hat = model.B[:,best_permutation]
        omega_hat = model.omegas[best_permutation]

        self.assertTrue(np.allclose(B_hat, true_B))
        self.assertTrue(np.allclose(omega_hat, true_omegas))

    def test_match_etas(self):
        # Test if epsilon source is correctly identified as latent with the largest error if there is no latent l with ch(l) = node_group
        p = 3
        X = np.zeros((0, p))
        model = ReLVLiNGAM(X, highest_l=1)
        model.pairwise_confounders[1,0] = 2
        model.pairwise_confounders[2,0] = 1
        descendants_latents, matches_latents, matches_noise = model._match_etas(remaining_nodes=[1,2], source=0, marginal_omegas=np.array([[[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]], [[1,1], [2,2], [3,3]], [[1,1], [2.02,2.04], [np.nan, np.nan]]]))
        self.assertEqual(descendants_latents, {0: [1, 2], 1: [1]})
        self.assertEqual(matches_latents, {(1, 0): 0, (1, 1): 1, (1, 2): 2, (2, 0): 0, (2, 1): 1})
        self.assertEqual(matches_noise, {1: 2, 2: 1})

if __name__=='__main__':
	unittest.main()


