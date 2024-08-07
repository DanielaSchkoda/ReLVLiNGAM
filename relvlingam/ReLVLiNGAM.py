import numpy as np
from numpy.linalg import svd

from sympy import symbols, Number
from relvlingam.moment_estimation_c import estimate_moment
from relvlingam.constraints_to_test import get_constraints_for_l_latents, get_cumulant_formula, calculate_orders_needed
import itertools as it
from functools import reduce

class NoSourceFoundError(RuntimeError):
    pass

class MathError(RuntimeError):
    pass

class ReLVLiNGAM():
    def __init__(self, X, highest_l, verbose=False, only_lowest_order_equations=True, threshold_power=1/8, thresholds=[0.008, 0.8], scale_partly=True):
        self.X = X
        self.n, self.p = X.shape[0], X.shape[1]
        self.highest_l = highest_l
        self.verbose = verbose
        self.only_lowest_order_equations = only_lowest_order_equations
        self.scale_partly = scale_partly
        self.threshold_power = threshold_power
        self.thresholds = thresholds

        self.constraints = {l: get_constraints_for_l_latents(l) for l in range(self.highest_l+1)}
        self.highest_order = calculate_orders_needed(highest_l)[1]

        self.orders = range(2, self.highest_order+1)
        self.omegas = np.full((self.p, len(self.orders)), np.nan)
        self.pairwise_confounders = np.full((self.p, self.p), -1, dtype=int)
        self.upper_bounds_confounders = np.full((self.p, self.p), np.inf)
        self.topological_order = []
        self.B = np.full((self.p, self.p), np.nan)
        self.descendants_latents = []
        self.fitted = False

    def fit(self, X):
        """
        Fit a LiNGAM model to X.

        Parameters:
        X (numpy.ndarray): Input data matrix.

        Returns:
        tuple: A tuple containing the topological order and the matrix B.
        """
        remaining_nodes = range(X.shape[1])
        cumulants = self._estimate_cumulants(X)

        while len(remaining_nodes) > 1:
            source = self._find_source(remaining_nodes, cumulants)
            remaining_nodes = [i for i in remaining_nodes if i != source]
            num_latents_before = len(self.descendants_latents)
            try:
                self._estimate_bs(source, remaining_nodes, cumulants)
                new_latents = list(range(num_latents_before+self.p, len(self.descendants_latents)+self.p))
                self.upper_bounds_confounders[remaining_nodes, :][:, remaining_nodes] -= np.array([[self._new_confounders(v, w, new_latents) for w in remaining_nodes] for v in remaining_nodes])
                self._estimate_omegas_overall_model(source, [source] + new_latents, remaining_nodes, cumulants)
                cumulants = self._update_cumulants(cumulants, source, new_latents, remaining_nodes)
            except MathError as e:
                if self.verbose:
                    print(f'Error in iteration {len(self.topological_order)}: {e}')
                # fill in remaining nodes with nans
                self.topological_order = self.topological_order + [np.nan for _ in range(len(remaining_nodes))]
                self.fitted = True
                return self.topological_order, self.B
        
        self._process_last_node(remaining_nodes[0], cumulants)
        self.fitted = True
        return self.topological_order, self.B
    
    def _estimate_cumulants(self, X):
        """Estimate all cumulants that are relevant for ReLVLiNGAM, i.e. all cumulants with up to two distinct indices."""
        # For efficiency reasons, first estimate all moments and then plug them into the cumulant formulas instead of estimating each cumulant separately.
        moment_dict = self._estimate_moments(X)
        all_cumulants = {}
        for k in range(2, self.highest_order+1):
            kth_cumulant = np.array([get_cumulant_formula(ind).subs(moment_dict) if len(set(ind)) <= 2 else np.nan for ind in it.product(range(self.p), repeat = k)], dtype=float).reshape((self.p,)*k)
            all_cumulants.update({k: kth_cumulant})
        return all_cumulants

    def _estimate_moments(self, X):
        """Estimate all moments that are relevant for ReLVLiNGAM, i.e. all moments with up to two distinct indices."""
        nodes = range(self.p)
        nodes = sorted(nodes)
        moment_dict = {}
        for k in range(2, self.highest_order+1):
            moment_dict.update({symbols(f"m_{''.join(map(str, ind))}"): estimate_moment(np.array(ind), X) for ind in it.combinations_with_replacement(nodes, k) if len(set(ind)) <= 2})
        return moment_dict
    
    def _process_last_node(self, last_node, cumulants):
        self.topological_order.append(last_node)
        self.B[:,last_node] = np.zeros(self.p)
        self.B[last_node,last_node] = 1
        self.omegas[last_node,:] = [cumulants[k][(last_node,)*k] for k in self.orders]

    def _form_symbol_to_cumulant_dict(self, cumulants, nodes, scale_partly):
        nodes = sorted(nodes)
        highest_k = len(cumulants) + 1
        cumulant_dict = {}
        # scaling
        if scale_partly:
            scales = np.array([cumulants[2][i,i]**(1/2) if i in nodes else np.nan for i in range(self.p)])
            for k in range(2, highest_k+1):
                cumulant_dict.update({symbols(f"c_{''.join(map(str, ind))}"): cumulants[k][ind]/np.prod(scales[list(ind)]) for ind in it.combinations_with_replacement(nodes, k) if len(set(ind)) <= 2})
        else:
            for k in range(2, highest_k+1):
                cumulant_dict.update({symbols(f"c_{''.join(map(str, ind))}"): cumulants[k][ind] for ind in it.combinations_with_replacement(nodes, k) if len(set(ind)) <= 2})
        return cumulant_dict

        
    def _find_source(self, remaining_nodes, cumulants):
        """Find source among remaining nodes."""
        all_singular_values = self._calculate_all_singular_values(remaining_nodes, cumulants)
        confounders = np.array([[self._estimate_num_confounders(potential_source, other_node, all_singular_values) if potential_source != other_node else 0 for potential_source in remaining_nodes] for other_node in remaining_nodes])
        # Save num confounders found to use as upper bounds in later iterations
        self.upper_bounds_confounders[remaining_nodes, :][:, remaining_nodes] = np.minimum(confounders, self.upper_bounds_confounders[remaining_nodes, :][:, remaining_nodes])
        # if all(np.sum(confounders, axis=0) == np.inf):
        #     if self.highest_l == np.inf:
        #         raise NoSourceFoundError(f"No source found in iteration {len(self.topological_order)}.")
        #     else:
        #         # find the node with lowest ratios
        #         sum_ratios = [np.sum([self._ratio_highest_l(potential_source, other_node, all_singular_values) for other_node in remaining_nodes if other_node != potential_source]) for potential_source in remaining_nodes]
        #         source = remaining_nodes[np.argmin(sum_ratios)]
        #         self.pairwise_confounders[remaining_nodes, source] = [self.highest_l if potential_source != source else 0 for potential_source in remaining_nodes]
        #         self.topological_order.append(source)
        #         return source
        
        argminima = get_all_argminima(np.sum(confounders, axis=0))
        if len(argminima) == 1:
            argmin = argminima[0]
        else:
            corresponding_ratios = [[self._corresponding_ratio(potential_source, other_node, all_singular_values, confounders, remaining_nodes) if potential_source != other_node else 0.0 for potential_source in argminima] for other_node in range(len(remaining_nodes))]
            # Argmin should be unique since matrix takes values in R
            argmin = argminima[np.argmin(np.sum(corresponding_ratios, axis=0))]
        # Go back to the original indices
        source = remaining_nodes[argmin]
        confounders[:,argmin] = np.where(confounders[:,argmin] == np.inf, self.highest_l, confounders[:,argmin])
        self.pairwise_confounders[remaining_nodes, source] = confounders[:,argmin]
        self.topological_order.append(source)
        return source

    def _corresponding_ratio(self, potential_source, other_node, all_singular_values, confounders, remaining_nodes):
        """Return the ratio sigma_[l+1]/sigma[0] where l is the estimated number of confounders."""
        true_l = self.highest_l # int(min(confounders[other_node, potential_source], self.highest_l))
        sigmas = all_singular_values[f"{remaining_nodes[potential_source]}{remaining_nodes[other_node]}{true_l}"]
        return sigmas[true_l+1]/sigmas[0]
    
    def _ratio_highest_l(self, potential_source, other_node, all_singular_values):
        sigmas = all_singular_values[f"{potential_source}{other_node}{self.highest_l}"]
        return sigmas[self.highest_l+1]/sigmas[0]

    def _calculate_all_singular_values(self, remaining_nodes, cumulants):
        """Calculate all singular values for all pairs of remaining nodes."""
        cumulant_dict = self._form_symbol_to_cumulant_dict(cumulants, remaining_nodes, self.scale_partly)
        sigmas = {}
        for (potential_source, other_node) in it.combinations(remaining_nodes, 2):
            for l in range(self.highest_l+1):
                r = self.constraints[l]["r"]
                A, A_rev = self.constraints[l]["A"], self.constraints[l]["A_rev"]
                specify_nodes = {sym: symbols("c_" + "".join(sorted(sym.name[2:].replace("j", str(potential_source)).replace("i", str(other_node))))) for sym in A.free_symbols | A_rev.free_symbols}
                A_hat = np.array(A.subs(specify_nodes).subs(cumulant_dict), dtype=float)
                A_rev_hat = np.array(A_rev.subs(specify_nodes).subs(cumulant_dict), dtype=float)
                sigma = svd(A_hat, compute_uv=False)
                sigma_rev = svd(A_rev_hat, compute_uv=False)
                sigmas[f"{potential_source}{other_node}{l}"] = sigma.tolist()
                sigmas[f"{other_node}{potential_source}{l}"] = sigma_rev.tolist()
        return sigmas
        
    def _estimate_num_confounders(self, potential_source, other_node, all_singular_values):
        """Estimate the number of confounders between two nodes."""
        iteration = len(self.topological_order)
        threshold = self.thresholds[0]/self.n**self.threshold_power if iteration == 0 else self.thresholds[1]*iteration/self.n**self.threshold_power
            
        highest_l = min(self.upper_bounds_confounders[other_node, potential_source], self.highest_l)
        for l in range(highest_l+1):
            r = self.constraints[l]["r"]
            sigma = all_singular_values[f"{potential_source}{other_node}{l}"]
            if (sigma[r]/sigma[0] < threshold):
                return l
        return self.upper_bounds_confounders[other_node, potential_source]
        
    def _estimate_bs(self, source, remaining_nodes, cumulants):
        """Estimates the causal effects from the source and its latent parents on the remaining nodes."""
        cumulant_dict = self._form_symbol_to_cumulant_dict(cumulants, [source] + remaining_nodes, scale_partly=False)

        highest_l = max([self.pairwise_confounders[other_node, source] for other_node in remaining_nodes])+1
        bs_unmatched = np.full((self.p, highest_l), np.nan)
        # len(remaining_nodes) x highest_l+1 x highest_order-1 omegas will be estimated.
        marginal_omegas = np.full((self.p, self.highest_l+1, self.highest_order-1), np.nan) 
        bs_unmatched[source,:] = np.ones(highest_l)
        for other_node in remaining_nodes:
            l = self.pairwise_confounders[other_node, source]
            bs_unmatched[other_node,:(l+1)] = self._estimate_bij(source, other_node, cumulant_dict)
            if l > 0:
                marginal_omegas[other_node,:(l+1),:] = self._estimate_marginal_omegas(source, other_node, bs_unmatched[other_node,:(l+1)], cumulant_dict)
        
        # All edges remaining_nodes -> source are set to 0
        self.B[source, remaining_nodes] = np.zeros(len(remaining_nodes))

        if highest_l > 1:
            descendants_new_latents, matches_latents, matches_noise = self._match_etas(remaining_nodes, source, marginal_omegas)
            self.B[remaining_nodes, source] = [bs_unmatched[v, matches_noise[v]] for v in remaining_nodes]
            self.B[source, source] = 1
            # For all nodes not descendant of latent l, b_jl = b_js
            # Therefore, initialize additional_columns_B with b_js
            additional_columns_B = np.tile(self.B[:,source], (len(descendants_new_latents), 1)).T
            for i, latent in enumerate(descendants_new_latents):
                descendants = descendants_new_latents[latent]
                old_latents = [matches_latents[(v, latent)] for v in descendants]
                self.descendants_latents.append(descendants + [source])
                additional_columns_B[descendants, i] = bs_unmatched[descendants, old_latents]

            self.B = np.hstack((self.B, additional_columns_B))

        else:
            self.B[source, source] = 1
            self.B[remaining_nodes, source] = bs_unmatched.reshape(-1)[remaining_nodes]
        
    def _estimate_bij(self, j, i, cumulant_dict):
        """Estimate the coefficients for the causal effect from j on i. Returns a np.ndarray with the (l+1) options for b_ij."""
        l = self.pairwise_confounders[i, j]
        equations_bij = self.constraints[l]["equations_bij"]
        if self.only_lowest_order_equations:
            equations_bij = [equations_bij[0]] if l in [0,2] else equations_bij[:2]
        specify_nodes = {sym: symbols(sym.name[:2] + "".join(sorted(sym.name[2:].replace("j", str(j)).replace("i", str(i))))) for sym in reduce(set.union, [eq.free_symbols for eq in equations_bij]) if str(sym) != "b_ij"}
        all_roots = np.full((l+1,len(equations_bij)), np.nan)
        for e in range(len(equations_bij)):
            eq = equations_bij[e]
            # Need type conversion for numpy root function to work
            estimated_coeffs = [float(coeff.subs(specify_nodes).subs(cumulant_dict)) for coeff in eq.all_coeffs()]
            # A numpy polynomial has the opposite order of coefficients to sympy: Numpy starts with the lowest power, 
            # Sympy with the highest. Therefore, reverse the coefficients.
            roots = np.polynomial.Polynomial(estimated_coeffs[::-1]).roots()
            if len(roots) < l+1:
                print(f"Warning: {l} confounders were estimated but corresponding equation does only have {len(roots)} roots. Roots are {roots}.")
                missing = l+1 - len(roots)
                roots = np.append(roots, [np.nan]*missing)
            roots = np.sort(np.real(roots))
            all_roots[:,e] = roots

        # Return the mean of the roots of all the theoretically equivalent equations  
        all_roots[np.isinf(all_roots)] = np.nan
        mean_roots = np.nanmean(all_roots, axis=1)
        if not np.all(np.isfinite(mean_roots)):
            raise MathError(f"Estimated b is NaN/Inf for source {j} and test node {i}, confs =1, all roots are {all_roots}.")
        return mean_roots
             
    def _estimate_marginal_omegas(self, source, other_node, bs_unmatched, cumulant_dict):
        """Estimate the marginal omegas for a pair of nodes.

        Args:
            source (int): The source.
            other_node (int): The other node.
            bs_unmatched (np.ndarray): Estimated causal effects from source, its latent parents -> other_node.
            cumulant_dict (dict): Dictionary of cumulants.

        Returns:
            np.ndarray: Estimated marginal omegas, omega[0,k] contains the (k+2)th order omega for epsilon_source,
                        omega[1,k] for the first latent and so on.
        """
        l = self.pairwise_confounders[other_node, source]
        marginal_omegas = np.full((l+1, self.highest_order-1), np.nan)
        # For k < l+1, the marginal omega cannot be infered.
        for k in range(l+1, self.highest_order+1):
            B_tilde = [bs_unmatched**i for i in range(k)]
            y = np.array([float(cumulant_dict[symbols(f"c_{''.join(sorted((str(source),)*(k - i) + (str(other_node),)*i))}")]) for i in range(k)])
            try:
                marginal_omegas[:,k-2] = np.linalg.lstsq(B_tilde, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                raise MathError(f"Linear system for {k}th order omega for source {source} and test node {other_node} is singular.")
        return marginal_omegas
    
    def _match_etas(self, remaining_nodes, source, marginal_omegas):
        """
        Match the latents in the marginal models to infer the latents in the overall graph.
        
        Args:
            remaining_nodes (list): List of remaining nodes.
            source (int): The source.
            marginal_omegas (np.ndarray): Estimated omegas in all marginal models.

        Returns:
            descendants_new_latents (dict): Descendants of each latent in the overall model.
            matches_latents (dict): Matching marginal latents of each overall latent.
            matches_noise (dict): Matching marginal latents of the noise.
        """
        threshold = 0.1

        # Dicitionaries to store the matches
        descendants_new_latents = {}
        matches_latents = {}

        # Group exogeneous sources of all confounded nodes
        for other_node in [other_node for other_node in remaining_nodes if self.pairwise_confounders[other_node, source] > 0]:
            # Latents L -> other_node can only be matched with latents L -> w for w != other_node. 
            # Hence, store the number of groups found for earlier nodes.
            count_groups_found_so_far = len(descendants_new_latents)
            for latent_to_match in range(self.pairwise_confounders[other_node, source]+1):
                candidate = 0
                group_found = False
                while not group_found and candidate < count_groups_found_so_far:
                    descendants = descendants_new_latents[candidate]
                    old_latents = [matches_latents[(d, candidate)] for d in descendants]                          
                    mean_omegas = np.mean([marginal_omegas[descendants, old_latents, :]], axis=0)
                    if other_node not in descendants and np.all(np.abs(np.nan_to_num(marginal_omegas[other_node, latent_to_match, :] - mean_omegas)) < threshold):
                        descendants_new_latents[candidate].append(other_node)
                        matches_latents[(other_node, candidate)] = latent_to_match
                        group_found = True
                    else:
                        candidate += 1
                if not group_found:
                    # If no group was found, create a new one
                    descendants_new_latents[len(descendants_new_latents)] = [other_node]
                    matches_latents[(other_node, len(descendants_new_latents)-1)] = latent_to_match
                    group_found = True

        # Process unconfounded nodes
        for other_node in [other_node for other_node in remaining_nodes if self.pairwise_confounders[other_node, source] == 0]:
            descendants_new_latents[len(descendants_new_latents)] = [other_node]
            matches_latents[(other_node, len(descendants_new_latents)-1)] = 0

        descendants_new_latents, matches_latents, matches_noise = self._disentangle_noise(descendants_new_latents, matches_latents, marginal_omegas, remaining_nodes)
        return descendants_new_latents, matches_latents, matches_noise
    
    def _disentangle_noise(self, descendants_new_latents, matches_latents, marginal_omegas, remaining_nodes):
        """
        Find the eta_j corresponding to the noise epsilon_s. Epsilon_s should show up in the marginal model for each other node. However, its
        estimated cumulants in the marginal model generally differ whenever two nodes do not have precisely the same latent ancestors. Thus,
        for each set of nodes that have coinciding latent ancestors, there should be one eta_j that has precisely this set as its descendants.
        If not, the latent with the largest error is split up again to satisfy this condition. 
        """
        nodes_with_same_latents = self._identify_nodes_with_same_latents(descendants_new_latents, remaining_nodes)
        matches_noise = {}
        for latent_parents, node_group in nodes_with_same_latents.items():
            try:
                noise = next(k for k, v in descendants_new_latents.items() if v == node_group)
                # Remove from matching_dict since we need all remaining latents later on
                descendants = descendants_new_latents.pop(noise)
                matches_noise.update({v: matches_latents[(v, noise)] for v in descendants})
            except StopIteration:
                # means_node_group has shape latent_parents x orders
                means_node_group = np.array([np.mean(marginal_omegas[node_group, [matches_latents[(v, l)] for v in node_group], :], axis=0) for l in latent_parents])
                other_descendants = {l: [v for v in descendants_new_latents[l] if v not in node_group] for l in latent_parents}
                means_other_descendants = np.array([np.mean(marginal_omegas[other_descendants[l], [matches_latents[(w, l)] for w in other_descendants[l]], :], axis=0) for l in latent_parents])
                errors = np.mean(np.abs(means_node_group - means_other_descendants), axis=1)
                latent_with_largest_error = latent_parents[np.argmax(errors)]
                # Split this group up again into being source for node_group and some latent for the other nodes
                matches_noise.update({v: matches_latents[(v, latent_with_largest_error)] for v in node_group})
                descendants_new_latents[latent_with_largest_error] = other_descendants[latent_with_largest_error]

        return descendants_new_latents, matches_latents, matches_noise

    def _identify_nodes_with_same_latents(self, descendants_new_latents, remaining_nodes):
        """Group observed nodes according to which have the same latent ancestors."""
        descendants_to_latents_dict = {v: [l for l in descendants_new_latents if v in descendants_new_latents[l]] for v in remaining_nodes}
        all_values_in_dict = set([tuple(value) for value in descendants_to_latents_dict.values()])
        nodes_with_same_latents = {value: [] for value in all_values_in_dict}
        for key, value in descendants_to_latents_dict.items():
            nodes_with_same_latents[tuple(value)].append(key)
        return nodes_with_same_latents
    
    def _estimate_omegas_overall_model(self, source, parents_source, remaining_nodes, cumulants):
        """Estimate the omegas of exog(source) in the whole graph."""
        if len(parents_source) == 1:
            self.omegas[source,:] = [cumulants[k][(source,)*k] for k in self.orders]
        else:
            omegas = np.full((len(parents_source), self.highest_order-1), np.nan)
            for k in range(2, self.highest_order+1):
                B_tilde = np.vstack((np.ones(len(parents_source)), np.array([self.B[v, parents_source]**i for i in range(1,k) for v in remaining_nodes])))
                y = np.array([float(cumulants[k][(source,)*k])] + [float(cumulants[k][(source,)*(k - i) + (v,)*i]) for i in range(1, k) for v in remaining_nodes])
                if B_tilde.shape[0] < B_tilde.shape[1]:
                    raise MathError(f"Linear system for {k}th order omega for source {source} and all remaining nodes is underdetermined.")
                try:
                    omegas[:, k-2] = np.linalg.lstsq(B_tilde, y, rcond=None)[0]
                except np.linalg.LinAlgError:
                    raise MathError(f"Linear system for {k}th order omega for source {source} and all remaining nodes is singular.")
            self.omegas[source,:] = omegas[0,:]
            self.omegas = np.vstack((self.omegas, omegas[1:,:]))

    def _new_confounders(self, v, w, new_latents):
        confounders_found_so_far = len([l for l in new_latents if v in self.descendants_latents[l-self.p] and w in self.descendants_latents[l-self.p]])
        return confounders_found_so_far

    def _update_cumulants(self, cumulants, source, new_latents, remaining_nodes):
        """Calculate the cumulants of the distribution with the sourcea and the new latents removed."""
        for k in self.orders:
            cumulants[k] = cumulants[k] - self.omegas[source, k-2] * tensor_outer_product(self.B[:,source], k) \
                - sum(self.omegas[l, k-2] * tensor_outer_product(self.B[:,l], k) for l in new_latents)
            
        # Don't abort if negative variances are found, but store iteration of this anomaly
        if any(cumulants[2][remaining_nodes, remaining_nodes] < 0):
            # Continue without scaling. For schaling, I need to calculate the root of the variances.
            self.scale_partly = False
    
        return cumulants

    # This gives all Bs compatible with distribution, not only the sparsest ones
    def get_all_possible_permutations(self):
        """
        Return all possible path matrices compatible with the distribution. 
        The function does not restrict to the path matrices corresponding to the sparsest graph.
        """
        all_possible_permutations = [list(range(self.B.shape[1]))]

        # For each latent, swap it with its oldest child
        for latent, descendants in enumerate(self.descendants_latents):
            oldest_child = next(element for element in self.topological_order if element in descendants)
            new_permutations = []
            for permutation in all_possible_permutations:
                additional_permutation = permutation.copy()
                additional_permutation[oldest_child], additional_permutation[latent+self.p] = additional_permutation[latent+self.p], additional_permutation[oldest_child]
                new_permutations.append(additional_permutation)
            all_possible_permutations += new_permutations
        
        return all_possible_permutations

def tensor_outer_product(v, k):
    result = v
    for _ in range(k-1):
        result = np.tensordot(result, v, axes=0)
    return result

def get_all_argminima(A):
    return np.where(A == np.min(A))[0]
