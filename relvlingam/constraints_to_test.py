import itertools as it
from math import sqrt, ceil
from sympy import factorial, parse_expr, Matrix, symbols, BlockMatrix, poly
from more_itertools import set_partitions

def calculate_orders_needed(l):
    return l+2, ceil(1/2 * (sqrt(8*l + 17) - 3)) + l+2

def get_constraints_for_l_latents(l):
    b = symbols("b_ij")
    k1, k2 = calculate_orders_needed(l)
    col_indices = list(it.combinations_with_replacement(['j', 'i'], k1))
    row_indices = [tuple([i for i in ind]) for ind in it.combinations_with_replacement(['', 'j', 'i'], k2-k1)]
    A = Matrix([[symbols(f"c_{''.join(map(str, sorted(col_ind + row_ind)))}") for col_ind in col_indices[:-1]] for row_ind in row_indices])
    A_rev = Matrix([[symbols(f"c_{''.join(map(str, sorted(col_ind + row_ind)))}") for col_ind in col_indices[1:]] for row_ind in row_indices])
    
    A_tilde = BlockMatrix([[Matrix([b**l for l in range(A.shape[1])]).T], [A]]).as_explicit()
    rows_for_minors = list(it.combinations(range(1, A_tilde.shape[0]), A_tilde.shape[1]-1))
    minors = [A_tilde[(0,)+row_selection, :] for row_selection in rows_for_minors]
    equations_bij = [poly(minor.det(), b) for minor in minors]

    return {"A": A, "A_rev": A_rev, "r": l+1, "highest_order": k2, "equations_bij": equations_bij}

def get_cumulant_formula(ind):
    k = len(ind)
    ind = sorted(ind)
    if k <= 3:
        return symbols(f"m_{''.join(map(str, ind))}")
    
    result = ""
    for partition in set_partitions(ind):
        if all(len(set) > 1 for set in partition):
            l = len(partition)
            coeff = (-1) ** (l - 1) * factorial(l - 1)
            moments = ["".join(map(str, sorted(set))) for set in partition]
            result += f" + ({coeff})  * m_{' * m_'.join(moments)}"
    return parse_expr(result)

def calculate_cov(mom0, mom1):
    indices = [char for char in mom0.name[2:]] + [char for char in mom1.name[2:]]
    indices.sort()
    return symbols(f"m_{''.join(indices)}") - mom0*mom1