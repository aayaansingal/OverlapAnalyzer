import json
import numpy as np
import scipy as sp
import copy
import pandas as pd
from numpy.polynomial import chebyshev
from numpy.polynomial import Polynomial
from scipy.optimize import minimize, NonlinearConstraint, linprog
from scipy.special import eval_chebyt
import os
import matplotlib.pyplot as plt
# Uncomment for production plotting with LaTeX (slower)
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.sans-serif": "Computer Modern Roman",
#     "font.size": 17,
# })
from overlapanalyzer.utils import exp_val_higher_moment
from overlapanalyzer.contour_integration import getContourDataFromEndPoints
from overlapanalyzer.alg_LinearOp import linear_solver
from overlapanalyzer.polynomial_estimates import rescale_E
import overlapanalyzer.visualization as viz
from scipy.special import comb

def f_piecewise(x, x0, x1):
    return np.piecewise(x, 
                        [x < x0, (x0 <= x) & (x <= x1), x > x1],
                        [0, lambda x: (x-x0), (x1-x0)])

def f_logistic(x, x_c, w):
    return 1/(1+np.exp(-w*(x-x_c)))

def find_logistic_w(x0, x1, threshold=0.95):
    """
    Find the omega parameter for the logistics function f(x)=1/(1+exp(-omega (x-x_c))) based on the given x0 and x1.
    Threshold is the value of f at x1.
    """
    return -2*np.log(1/threshold - 1)/(x1-x0)

# Constraint function: h(k, c_i, x)
def constraint_function_piecewise(c, x, x0, x1):
    if x < x0:
        return np.sum(c * x ** np.arange(len(c)))
    elif x0 <= x < x1:
        return np.sum(c * x ** np.arange(len(c))) - (x - x0)
    elif x1 <= x:
        return np.sum(c * x ** np.arange(len(c))) - (x1 - x0)


def vectorized_constraint_piecewise(params, x_points, x0, x1):
    return np.array([constraint_function_piecewise(params, x, x0, x1) for x in x_points])

def constraint_function_logistic(params, x, x_c, w):
    c = params
    return np.sum(c * x ** np.arange(len(c))) - f_logistic(x, x_c, w)

def chebyshev_approximation(f, a, b, degree):
    """
    Approximate the function f over the interval [a, b] using Chebyshev polynomials of a given degree.
    
    Args:
    f: function to approximate
    a: start of the interval
    b: end of the interval
    degree: degree of the Chebyshev polynomial
    
    Returns:
    Approximation function that evaluates the Chebyshev approximation on any input.
    """
    
    # Rescale x from [a, b] to [-1, 1]
    def rescale_x(x):
        return 2 * (x - a) / (b - a) - 1

    # Rescale x from [-1, 1] back to [a, b]
    def rescale_back(t):
        return 0.5 * (t + 1) * (b - a) + a

    # Generate Chebyshev nodes in the interval [-1, 1]
    nodes = np.cos(np.pi * (np.arange(degree + 1) + 0.5) / (degree + 1))
    
    # Map nodes to the interval [a, b]
    nodes_mapped = rescale_back(nodes)
    
    # Evaluate the function at these nodes
    f_values = f(nodes_mapped)
    
    # Get the Chebyshev coefficients by fitting to the values
    coeffs = chebyshev.chebfit(nodes, f_values, degree)
    
    # Return the approximation function
    def chebyshev_approx(x):
        # Rescale x to [-1, 1]
        t = rescale_x(x)
        # Evaluate the Chebyshev series
        return chebyshev.chebval(t, coeffs)

    return chebyshev_approx, coeffs

def poly_form_of_chebyshev_approximation(f, a, b, degree):
    """
    Compute the polynomial in standard form of the Chebyshev approximation.
    Return: Numpy Polynomial object in standard form.
    """
    # First, convert the Chebyshev coefficients (in t) to the standard polynomial in t
    cheb_approx, cheb_coeffs = chebyshev_approximation(f, a, b, degree)
    poly_coeffs_t = chebyshev.cheb2poly(cheb_coeffs)
    
    # The transformation is t = (2*(x - a))/(b - a) - 1
    # So we need to substitute t = (2*(x - a))/(b - a) - 1 into the polynomial

    # Create the polynomial P(t) in terms of t
    P_t = Polynomial(poly_coeffs_t)
    
    # Define the linear transformation: t(x) = (2*(x - a)) / (b - a) - 1
    T_x = Polynomial([-(a + b) / (b - a), 2 / (b - a)])  # -1 + (2/(b - a)) * (x - a)
    
    # Compose P(t) with T_x, i.e., P(T_x(x))
    P_x = P_t(T_x)
    
    # Return the resulting polynomial in x
    return P_x

def chebyshev_matrix_from_degree(degree):
    # Initialize the transformation matrix
    transform_matrix = np.zeros((degree + 1, degree + 1))
    
    # Populate the matrix by creating each Chebyshev polynomial in the power basis
    for i in range(degree + 1):
        # Create the i-th Chebyshev polynomial (T_i)
        cheb_poly = chebyshev.Chebyshev.basis(i).convert(kind=np.polynomial.Polynomial)
        # Get the coefficients for T_i in terms of power basis (1, a, a^2, ...)
        transform_matrix[i, :len(cheb_poly.coef)] = cheb_poly.coef
        
    return transform_matrix

def f_piecewise_concave(x, left_low, left_up, right_low, right_up, v2, v1):
    return np.piecewise(x, 
                    [x < left_low, (left_low <= x) & (x <= left_up), (left_up < x) & (x <= right_low), (right_low < x) & (x <= right_up), x > right_up],
                    [v2, lambda x: x*(v1-v2)/(left_up - left_low) + v2-left_low*(v1-v2)/(left_up - left_low), v1, lambda x: x*(v2-v1)/(right_up - right_low) + v1-right_low*(v2-v1)/(right_up - right_low), v2])

# def f_function(x, v1_range, v2, v1, left_v2=None, right_v2=None):
#     """
#     A piecewise function with optional linear interpolation at the boundaries.
#     Supports scalar or array inputs for x.

#     Parameters:
#     - x: float or array-like, the input values at which the function is evaluated.
#     - v1_range: tuple, (v1_start, v1_end) defining where the function evaluates to v1.
#     - v2: float, the value of the function outside of v1_range.
#     - v1: float, the value of the function within v1_range.
#     - left_v2: float, optional, the value at x < v1_range[0] to create a linear interpolation.
#     - right_v2: float, optional, the value at x > v1_range[1] to create a linear interpolation.

#     Returns:
#     - np.ndarray, the values of the function at x.
#     """
#     x = np.asarray(x)  # Ensure x is a NumPy array for vectorized operations
#     v1_start, v1_end = v1_range

#     # Initialize the result with v2 values
#     result = np.full_like(x, v2, dtype=np.float64)

#     # Region: Linear interpolation before v1_start
#     if left_v2 is not None:
#         left_mask = (x >= left_v2) & (x < v1_start)
#         result[left_mask] = v2 + (v1 - v2) * (x[left_mask] - left_v2) / (v1_start - left_v2)

#     # Region: v1 within v1_range
#     v1_mask = (x >= v1_start) & (x <= v1_end)
#     result[v1_mask] = v1

#     # Region: Linear interpolation after v1_end
#     if right_v2 is not None:
#         right_mask = (x > v1_end) & (x <= right_v2)
#         result[right_mask] = v1 + (v2 - v1) * (x[right_mask] - v1_end) / (right_v2 - v1_end)

#     return result

def ITE_UB(E0, exp_H, exp_H2):
    return np.exp(-(exp_H - E0)**2 / (2*(exp_H2 - exp_H**2)))

def gen_b_ub_from_length(target_indices, b_values, L):
    """
    Generate the b_ub vector for linprog constraints from exact constraints.
    Each b value only corresponds to a single target index.
    :param target_indices: list of target indices.
    :param b_values: list of b values for the constraints. Must match the length of target_indices.
    :param L: length of the vector. Must be greater than the maximum index in target_indices.
    """
    b_ub = np.zeros(L)
    for i, index in enumerate(target_indices):
        b_ub[index] = b_values[i]
    return b_ub

def gen_gaps_from_values(value_list):
    return [(value_list[i] - value_list[i-1]) for i in range(1, len(value_list))]

def gen_bound_pairs(value_list, bound_extension, E0_LB=None, Emax_UB = None):
    """
    Generate a list of (lower, upper) bound pairs from a list of values.
    If E0_LB or Emax_UB is provided, they will be used as LB/UB for the first/last value.
    Otherwise, the E0-E1 and Emax-(Emax-1) gaps will be used.
    :param value_list: list of values, assumed to be sorted from lowest to highest.
    :param bound_extension: the extension factor for the bounds. Must be between 0 and 1/2.
    :param E0_LB: lower bound for the first value. If None, the E0-E1 gap will be used.
    :param Emax_UB: upper bound for the last value. If None, the Emax-(Emax-1) gap will be used.
    """
    gaps = np.array(gen_gaps_from_values(value_list)) * bound_extension

    if E0_LB is None:
        E0_LB = value_list[0] - gaps[0]
    elif E0_LB >= value_list[0]:
        raise ValueError("The provided E0_LB must be less than the first value.")
    if Emax_UB is None:
        Emax_UB = value_list[-1] + gaps[-1]
    elif Emax_UB <= value_list[-1]:
        raise ValueError("The provided Emax_UB must be greater than the last value.")
    return np.array([(E0_LB, value_list[0]+gaps[0])] + [(value_list[i]-gaps[i-1], value_list[i]+gaps[i]) for i in range(1, len(value_list) - 1)] + [(value_list[-1]-gaps[-1], Emax_UB)])

def compute_boundaries(target_indices, L, merge_consec_indices=True):
    boundaries = []
    target_indices = sorted(target_indices)  # Ensure indices are sorted
    previous_index = 0

    for index in target_indices:
        if previous_index < index:  # Handle ranges before the target index
            boundaries.append((previous_index, index - 1))
        boundaries.append((index, index))  # Add the target index as its own boundary
        previous_index = index + 1

    if previous_index < L:  # Handle the final range after the last target index
        boundaries.append((previous_index, L - 1))

    return boundaries

def process_bound_pairs(bound_pairs, target_indices, dense_info=(None, None)):
    """
    Keep the bound pairs corresponding to the target indices, and merge the rest that are neighbours to each other.
    If dense_info is provided, and they are at the correct locations, then they will be added as additional separation points.
    :param bound_pairs: numpy array of (lower, upper) bound pairs.
    :param target_indices: list of target indices.
    :param dense_info: tuple (dense_lb, dense_ub) defining the dense region. Either entry can be None.
    """
    if 0 in target_indices and dense_info[0] is not None:
        raise ValueError("dense_info[0] cannot be used since 0 is in target_indices.")
    nonzero_b_indices = target_indices.copy() if dense_info[0] is None else np.array(target_indices)+1
    def _insert_lower_bound(pair_list, bound):
        pair_list.insert(1,(bound, pair_list[0][1]))
        pair_list[0] = (pair_list[0][0], bound)
    def _insert_upper_bound(pair_list, bound):
        pair_list.insert(-1, (pair_list[-1][0], bound))
        pair_list[-1] = (bound, pair_list[-1][1])
        
    boundaries = compute_boundaries(target_indices, len(bound_pairs))
    new_bound_pairs = [(bound_pairs[lower][0], bound_pairs[upper][1]) for lower, upper in boundaries]
    incl_endpoints = [True for _ in range(len(new_bound_pairs))]
    # The following lines define what "correct locations" mean for the dense region
    if dense_info[0] is not None and new_bound_pairs[0][0] < dense_info[0] < new_bound_pairs[target_indices[0]][1]:
        _insert_lower_bound(new_bound_pairs, dense_info[0])
        incl_endpoints.insert(0, False)
    if dense_info[1] is not None and new_bound_pairs[target_indices[-1]][0] < dense_info[1] < new_bound_pairs[-1][1]:
        _insert_upper_bound(new_bound_pairs, dense_info[1])
        incl_endpoints.insert(-1, False)
    return new_bound_pairs, incl_endpoints, nonzero_b_indices

def gen_constraints_recipe(bound_pairs, target_b_values, incl_endpoints_list, num_x_points):
    """
    Generate the constraint recipe table for the linprog constraints.
    :param bound_pairs: list of (lower, upper) bound pairs.
    :param target_b_values: list of target b values for the constraints. Must match the length of target_indices.
    :param incl_endpoints_list: list of booleans indicating whether the endpoints are included in the region.
    :param num_x_points: tuple of (num_0, num_1, ...) for the number of points in each region. Must match the number of regions.
    Output: a dictionary with the following entries:
    - Region: (lower bound, upper bound) of the region
    - b_values: b values for the region
    - num_points: number of discretization points in the region
    - include_endpoints: whether the endpoints are included in the region
    """
    return {"Region": bound_pairs, "b_values": target_b_values, "num_points": num_x_points, "include_endpoints": incl_endpoints_list}

def save_constraints_dict_as_csv(constraints_dict, filename):
    """
    Save the constraints dictionary as a CSV file.
    """
    df = pd.DataFrame(constraints_dict)
    df.to_csv(filename, index=False)

def load_constraints_dict_from_csv(filename):
    """
    Load the constraints dictionary from a CSV file.
    """
    df = pd.read_csv(filename)
    return df.to_dict()

class PB_linprog_params_from_eig:
    """
    Prepares the parameters for linprog based on the eigenvalue results.
    :param eig_results: dictionary of eigenvalue results.
    :param target_indices: list of target indices.
    :param b_values: list of b values for the constraints. Must match the length of target_indices.
    :param poly_degree: degree of the polynomial.
    :param use_exact_energies: whether to use exact energies as constraints.
    :param rescale: whether to rescale the energies.
    :param bound_extension: extension factor for the bounds. Only used when use_exact_energies=False.
    :param dense_info: tuple (dense_lb, dense_ub) defining the dense region. Only used when use_exact_energies=False.
    :param num_x_points: tuple of (num_0, num_1, ...) for the number of points in each region. Only used when use_exact_energies=False, and len must match definitions target_indices & dense_info.
    :param Energy_constraint_limits: tuple (Emin, Emax) defining the energy constraint limits; energies outside the range will not be used as constraints.
    """
    def __init__(self, eig_results, target_indices, use_exact_energies, use_all_evals=False, custom_constraints=None,
                 use_cheb=False, rescale=True, bound_extension=None, dense_info=(None, None), num_x_points=None, Energy_constraint_limits=(float('-inf'), float('inf'))):
        # Store inputs as attributes
        local_vars = locals()
        for name, value in local_vars.items():
            if name != 'self':
                setattr(self, name, value)

        # Rescale parameters and overlaps
        self.E0_rescale, self.Emax_rescale = eig_results['rescale_values']
        self.exact_overlaps = eig_results['overlaps'] if use_all_evals else eig_results['overlaps_truncated']
        self.unscaled_energies = eig_results['exact_energies_no_degen'] if use_all_evals else eig_results['exact_energies_truncated']
        self.exact_mask = (np.array(self.unscaled_energies) >= Energy_constraint_limits[0]) & (np.array(self.unscaled_energies) <= Energy_constraint_limits[1])
        self.exact_energies = self._rescale(self.unscaled_energies)
        self.moments = np.array(eig_results['moments_rescaled']) if rescale else np.array(eig_results['moments'])
        self.max_poly_degree = len(self.moments) - 1

        if custom_constraints is not None:
            self.constraint_pts = custom_constraints
        else:
            # Below, we use a symmetric bound extension for the ground and highest excited states in gen_bound_pairs; can be changed in future
            if bound_extension is not None:
                self.bound_pairs, self.incl_endpoints, self.nonzero_b_indices = process_bound_pairs(gen_bound_pairs(self.unscaled_energies, bound_extension), target_indices, dense_info)

            self.constraint_pts = self._get_exact_constraints() if use_exact_energies else self._get_dense_constraints()
        self.A_ub_full = np.polynomial.chebyshev.chebvander(self.constraint_pts, self.max_poly_degree) if use_cheb else np.vander(self.constraint_pts, self.max_poly_degree + 1, increasing=True)

    def _rescale(self, E):
        """Rescale energies if rescale=True."""
        if self.rescale:
            return rescale_E(np.array(E), self.E0_rescale, self.Emax_rescale)
        else:
            return np.array(E)

    def _get_exact_constraints(self):
        """
        Get exact energies within limits as constraint points.
        """
        exact_energies = np.array(self.unscaled_energies)
        energies_within_limits = exact_energies[self.exact_mask]
        return self._rescale(energies_within_limits)

    def _get_dense_constraints(self):
        """
        Generate dense constraints based on bound pairs and number of discretization points.
        """
        if self.num_x_points is None:
            raise ValueError("num_x_points must be provided when use_exact_energies is False.")
        if len(self.num_x_points) != len(self.bound_pairs):
            raise ValueError("num_x_points must match the number of bound pairs! Check the number of constraint regions.")
        energy_constraints = []
        for i, (lower, upper) in enumerate(self.bound_pairs):
            if lower > self.Energy_constraint_limits[0] and upper < self.Energy_constraint_limits[1]:
                energy_constraints.extend(np.linspace(lower, upper, self.num_x_points[i], endpoint=self.incl_endpoints[i]))
        # energy_constraints = np.concatenate(energy_constraints)
        return self._rescale(energy_constraints)
    
    def get_approx_intervals(self):
        """
        Get constraint regions for approx mode.
        """
        return self._rescale(self.bound_pairs)

    def get_A_ub(self, poly_degree):
        """Return A_ub."""
        return self.A_ub_full[:,:poly_degree + 1]

    def get_moment_values(self, poly_degree):
        """
        Return computed moment values.
        Note: need to compute the chebyshev moments by applying the transformation matrix.
        """
        if self.use_cheb:
            return (chebyshev_matrix_from_degree(self.max_poly_degree) @ self.moments)[:poly_degree + 1]
        else:
            return self.moments[:poly_degree + 1]
    
    def get_b_ub(self, b_list):
        """Return b_ub values."""
        if self.custom_constraints is not None:
            if len(self.custom_constraints) != len(b_list):
                raise ValueError("Length of b_list must match the number of custom constraints.")
            return b_list
        elif self.use_exact_energies:        
            full_b = gen_b_ub_from_length(self.target_indices, b_list, len(self.unscaled_energies))
            return full_b[self.exact_mask]
        else:
            b_recipe = np.zeros(len(self.bound_pairs))
            for i, index in enumerate(self.nonzero_b_indices):
                b_recipe[index] = b_list[i]
            # Repeat the b values for the number of points in each region
            return np.repeat(b_recipe, self.num_x_points)
    
    def get_constraint_pts(self):
        return self.constraint_pts
    
    def get_all_for_PolyBounds(self, poly_degree, b_list):
        values = self.get_moment_values(poly_degree)
        A_ub = self.get_A_ub(poly_degree)
        b_ub = self.get_b_ub(b_list)
        return values, A_ub, b_ub
    def get_exact_ovlp(self, b_list, b_list_len_eig=None):
        """Return the exact (linear combination of) overlap defined by the b_list"""
        if self.custom_constraints is not None:
            return np.dot(self.exact_overlaps, b_list_len_eig)
        else:
            full_b = gen_b_ub_from_length(self.target_indices, b_list, len(self.unscaled_energies))
            return np.dot(self.exact_overlaps, full_b)

def _build_binomial_transformation(E_max, E_min, N):
    """
    Build the binomial transformation matrix T such that:
        tilde_mu = T @ mu
    where:
        tilde_mu_n = <((2H - (E_max + E_min)I)/(E_max - E_min))^n>
        mu_k = <H^k>

    Parameters:
        E_max (float): Maximum eigenvalue of H
        E_min (float): Minimum eigenvalue of H
        N (int): Maximum moment index

    Returns:
        T (ndarray): (N+1)x(N+1) transformation matrix
    """
    a = E_max + E_min
    b = E_max - E_min
    T = np.zeros((N+1, N+1))
    for n in range(N+1):
        for k in range(n+1):
            T[n, k] = comb(n, k) * (2 ** k) * ((-a) ** (n - k)) / (b ** n)
    return T

def rescale_moments(original_moments, E_max, E_min) -> list:
    """
    Rescale the moments of H to moments of (2H - aI)/b,
    where a = E_max + E_min and b = E_max - E_min.

    Parameters:
        original_moments (array-like): Moments [<H^0>, ..., <H^N>]
        E_max (float): Maximum eigenvalue of H
        E_min (float): Minimum eigenvalue of H

    Returns:
        rescaled_moments (ndarray): [<tilde{H}^0>, ..., <tilde{H}^N>]
    """
    original_moments = np.asarray(original_moments)
    N = len(original_moments) - 1
    T = _build_binomial_transformation(E_max, E_min, N)
    return (T @ original_moments).tolist()

def rescale_number(num, E_max, E_min) -> float:
    """
    Rescale a number to the new scale defined by E_max and E_min.

    Parameters:
        num (float): The number to rescale.
        E_max (float): Maximum eigenvalue of H.
        E_min (float): Minimum eigenvalue of H.

    Returns:
        float: The rescaled number.
    """
    a = E_max + E_min
    b = E_max - E_min
    return (2 * num - a) / b

def PBinputs_from_eig_agnostic_recipe(recipe: dict, rescale=True, use_cheb=False):
    """
    Prepare the parameters for linprog in eigenvalue-agnostic mode.
    :param recipe: a dictionary with the following entries:
    - moments: list of moments.
    - intervals: a list of 4-tuples (lower_bound, upper_bound, b_value, num_points) defining the intervals.
    : param use_rescaled: boolean indicating whether to rescale moments and intervals.
    : param use_cheb: boolean indicating whether to use Chebyshev polynomials as basis functions.
    (20250717: use_cheb=True is causing issues when interval is not between [-1,1]. Defaulting to False.)
    """
    intervals = recipe.get("intervals", [])
    if rescale:
        # Only works for num_intervals=2 at this point
        if len(intervals) != 2:
            raise ValueError("Rescaling only works for two intervals at the moment.")
        E_lb = intervals[0][0]
        E_ub = intervals[1][1]
        rescaled_endpoints = [rescale_number(intervals[0][1], E_ub, E_lb),
                              rescale_number(intervals[1][0], E_ub, E_lb)]
        intervals[0][0] = -1
        intervals[0][1] = rescaled_endpoints[0]
        intervals[1][0] = rescaled_endpoints[1]
        intervals[1][1] = 1
    # Generate constraint_pts and b_ub together
    pts_and_b = [
        (np.linspace(low, up, num_points, endpoint=incl), np.full(num_points, b), num_points)
        for (low, up, b, num_points, incl) in intervals
    ]
    constraint_pts = np.concatenate([x for x, _, _ in pts_and_b])
    b_ub = np.concatenate([y for _, y, _ in pts_and_b])
    num_x_points = [n for _, _, n in pts_and_b]

    moments = recipe.get("moments", [])
    if rescale:
        moments = rescale_moments(moments, E_ub, E_lb)
    if use_cheb:
        moments = chebyshev_matrix_from_degree(len(moments) - 1) @ moments


    # A_ub is constructed from the constraint points
    A_ub = np.polynomial.chebyshev.chebvander(constraint_pts, len(moments) - 1) if use_cheb else np.vander(constraint_pts, len(moments), increasing=True)
    return moments, constraint_pts, b_ub, A_ub, num_x_points, (E_lb, E_ub) if rescale else (None, None)

class PolyBounds_new():
    """
    Obtaining lower and upper bounds on the overlap using polynomial majorisation/minorisation.
    :param values: list of values defining the objective function
    :param A_ub: matrix A_ub for linprog
    :param b_ub: vector b_ub for linprog
    :param constraint_pts: x-values corresponding to b_ub
    :param P_exact: exact overlap value
    """
    def __init__(self, values, A_ub, b_ub, constraint_pts, P_exact=None):
        self.values = np.array(values)
        self.A_ub = A_ub
        self.b_ub = b_ub
        self.constraint_pts = constraint_pts
        self.max_poly_degree = len(values) - 1
        self.poly_degree = self.max_poly_degree
        self.multi_degree_results = None
        self.O_exact = P_exact
        if P_exact is not None:
            print("Exact overlap: ", P_exact)
    
    def compute_chebyshev_coeffs(self, n):
        """Computes Chebyshev approximation coefficients."""
        x_min, x_max = self.constraint_pts[0], self.constraint_pts[-1]
        chebyshev_nodes = np.cos((2*np.arange(1, n+2)-1) / (2*(n+1)) * np.pi)  # In [-1,1]
        x_nodes = 0.5 * (x_max - x_min) * chebyshev_nodes + 0.5 * (x_max + x_min)  # Map to [x_min, x_max]
        y_nodes = np.interp(x_nodes, self.constraint_pts, self.b_ub)
        coeffs = chebyshev.chebfit(chebyshev_nodes, y_nodes, n)
        return coeffs
    
    def optimize(self, poly_degrees=None, bound=1e20, method='highs', linprog_options=None, visualize=False, **kwargs):
        if linprog_options is None:
            linprog_options = {}
        if poly_degrees is None:
            poly_degrees = [self.poly_degree]
        poly_degrees = sorted({int(deg) for deg in poly_degrees})
        if not poly_degrees:
            raise ValueError("poly_degrees must contain at least one degree.")
        summaries = []
        for deg in poly_degrees:
            if deg < 0 or deg > self.max_poly_degree:
                raise ValueError(f"Requested poly_degree {deg} is outside [0, {self.max_poly_degree}].")
            current_values = self.values[:deg + 1]
            current_A = self.A_ub[:, :deg + 1]
            # print("Condition number for A: ", np.linalg.cond(current_A))
            LB_result = linprog(-current_values, A_ub=current_A, b_ub=self.b_ub, bounds=(-bound, bound), method=method, options=linprog_options)
            UB_result = linprog(current_values, A_ub=-current_A, b_ub=-self.b_ub, bounds=(-bound, bound), method=method, options=linprog_options)
            degree_bounds = []
            for bound_name, result in (('LB', LB_result), ('UB', UB_result)):
                if result.success:
                    print(f"Poly degree {deg} {bound_name} optimal parameters found:")
                    print("c_i =", result.x)
                    bound_val = np.dot(current_values, result.x)
                    degree_bounds.append(bound_val)
                    print(f"Overlap {bound_name}: ", bound_val)
                else:
                    print(f"Poly degree {deg} {bound_name} optimization failed: {result.message}")
                    degree_bounds.append(np.nan)
            cheb_coeffs = self.compute_chebyshev_coeffs(n=deg) if deg >= 0 else np.array([0.0])
            save_plots = bool(kwargs.get('save_plots', False))
            save_dir = kwargs.get('save_dir', None)
            mode_name = kwargs.get('mode', None)
            show_plot = bool(kwargs.get('show', visualize))
            if (visualize or save_plots) and LB_result.success and UB_result.success:
                plot_kwargs = {
                    "LB_result": LB_result,
                    "UB_result": UB_result,
                    "poly_degree": deg,
                    "bounds": degree_bounds,
                    "used_cheb": kwargs.get("used_cheb", False),
                    "legend_anchor": kwargs.get("legend_anchor", (1,0.2)),
                    "show_constraint_points": kwargs.get("show_constraint_points", True),
                    "linewidth": kwargs.get("linewidth", 1),
                    "show_spectrum": kwargs.get("show_spectrum", False),
                    "show_lowest_highest": kwargs.get("show_lowest_highest", False),
                    "constraint_recipe_obj": kwargs.get("constraint_recipe_obj"),
                    "plot_title": (kwargs.get("plot_title","")),
                    "plot_chebyshev": kwargs.get("plot_chebyshev", False),
                    "truncate_lower": kwargs.get("truncate_lower"),
                }
                # prepare save path if requested
                save_path = None
                if save_plots and save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    title_safe = (plot_kwargs.get('plot_title') or '').replace(' ', '_') or 'bounds'
                    mode_safe = (mode_name or '').replace(' ', '_')
                    fname = f"{title_safe}_deg{deg}.pdf"
                    save_path = os.path.join(save_dir, fname)
                plot_kwargs.update({
                    'show': show_plot,
                    'save_path': save_path
                })
                self.plot_results(
                    constraint_pts=self.constraint_pts,
                    **plot_kwargs,
                )
            cheb_value = np.dot(current_values, cheb_coeffs[:deg + 1]) if deg >= 0 else 0.0
            summaries.append({
                "degree": deg,
                "LB_result": LB_result,
                "UB_result": UB_result,
                "bounds": degree_bounds,
                "chebyshev_coeffs": cheb_coeffs,
                "chebyshev_value": cheb_value
            })
        return summaries

    def plot_results(self, constraint_pts, n_plot_points=1000, used_cheb=False, legend_anchor=(1,0.2), show_constraint_points=False, limit_adjust=(0,0), linewidth=1, show_spectrum=False, show_lowest_highest=False, constraint_recipe_obj=None, plot_title=None, plot_chebyshev=False, truncate_lower=None, show=False, save_path=None, **kwargs):
        """
        constraint_pts: the constraint points used in the optimization.
        """
        poly_plot_points = np.linspace(constraint_pts[0], constraint_pts[-1], n_plot_points)
        if truncate_lower is not None:
            # compute truncated lower interval correctly
            span = constraint_pts[-1] - constraint_pts[0]
            poly_plot_points_lower = np.linspace(constraint_pts[0], constraint_pts[-1] - span * truncate_lower, n_plot_points)
        LB_polynomial = chebyshev.Chebyshev(kwargs.get('LB_result').x) if used_cheb else Polynomial(kwargs.get('LB_result').x)
        LB_polynomial_values = LB_polynomial(poly_plot_points) if truncate_lower is None else LB_polynomial(poly_plot_points_lower)
        UB_polynomial = chebyshev.Chebyshev(kwargs.get('UB_result').x) if used_cheb else Polynomial(kwargs.get('UB_result').x)
        UB_polynomial_values = UB_polynomial(poly_plot_points)
        f_values = self.b_ub
        # ax1=plt.gca()
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        # Plot the polynomial
        ax1.plot(poly_plot_points if truncate_lower is None else poly_plot_points_lower, LB_polynomial_values, label='Minorizing Poly', color='blue', linewidth=linewidth)
        ax1.plot(poly_plot_points, UB_polynomial_values, label='Majorizing Poly', color='green', linewidth=linewidth)

        # Plot the exact piecewise linear function
        ax1.plot(constraint_pts, f_values, label=r'$f_0(E)$', linestyle='--', color='red', linewidth=linewidth)
        # Overlay the spectrum information
        if plot_chebyshev:
            t_vals = 2 * (poly_plot_points - constraint_pts[0]) / (constraint_pts[-1] - constraint_pts[0]) - 1  # Map x to [-1,1]
            y_cheb = chebyshev.chebval(t_vals, self.chebyshev_coeffs)
            ax1.plot(poly_plot_points, y_cheb, label=f"Chebyshev Approximation", color='magenta', linewidth=linewidth)
        if show_spectrum:
            ax2 = ax1.twinx()
            exact_energies = constraint_recipe_obj.exact_energies
            exact_overlaps = constraint_recipe_obj.exact_overlaps
            # non_degen_evals = non_degenerate_values(eig_results['exact_energies'], degens)
            ax2.stem(exact_energies, exact_overlaps, linefmt='grey', markerfmt=' ', basefmt='C0', label='Exact spectrum')
            ax2.set_ylabel('Overlap')
        # Mark E0_LB, Emax_UB on the x-axis
        if show_lowest_highest:
            ax1.axvline(x=constraint_recipe_obj.exact_energies[0], color='gray', linestyle=':')
            ax1.axvline(x=constraint_recipe_obj.exact_energies[-1], color='gray', linestyle=':')
        if show_constraint_points:
            ax1.scatter(constraint_pts, [0 for _ in range(len(constraint_pts))], color='pink', marker='|', zorder=4)
            
        # Add labels and legend
        if plot_title is not None:
            # ax1.set_title(plot_title+ f', Degree={kwargs.get("poly_degree")}' + r', $O_{exact}=$' + f'{self.O_exact:.3f}' + r', $O_{LB}=$' + f'{kwargs.get("bounds")[0]:.3f}' + r', $O_{UB}=$' + f'{kwargs.get("bounds")[1]:.3f}')
            ax1.set_title(plot_title+ f', Degree={kwargs.get("poly_degree")}' + f', [{kwargs.get("bounds")[0]:.3f}, {kwargs.get("bounds")[1]:.3f}] ({self.O_exact:.3f})')
            ax1.set_xlabel('Energy')
        ax1.set_ylabel('Function Value')
        # fig.legend(loc='lower right', bbox_to_anchor=legend_anchor, bbox_transform=ax1.transAxes)
        plt.tight_layout()
        # Save if requested
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            except Exception:
                pass
            try:
                fig.savefig(save_path, format='pdf')
            except Exception:
                fig.savefig(save_path)
        # Show if requested
        if show:
            plt.show()
        plt.close(fig)

# helper: turn Gram matrix into coefficients of v^T Q v
def gram_to_coefs(Q, dim):
    coefs = []
    for k in range(2*dim-1):
        expr = 0
        for i in range(dim-1):
            j = k-i
            if 0 <= j < dim:
                expr += Q[i,j]
        coefs.append(expr)
    return coefs

class PolyBounds_SIP:
    """
    Semi-Infinite Programming approaches for polynomial bounds:
    - Exchange method (adaptive LP)
    - SOS/SDP method (via cvxpy)
    """

    def __init__(self, values, poly_degree=None, O_exact=None):
        self.values = np.array(values)
        self.poly_degree = poly_degree if poly_degree is not None else len(values)-1
        self.bounds = []
        self.multi_degree_results = None
        self.O_exact = O_exact
        if O_exact is not None:
            print("Exact overlap:", O_exact)

    def optimize(self, intervals, lower_coefs_list, solver="SCS", verbose=False, global_bnd=None, solver_max_iters=None, poly_degrees=None, visualize=False, **kwargs):
        """
        SOS/SDP (power basis) for piecewise polynomial constraints on intervals.
        Two SDPs:
        1) Minorizer p_l:  l_j(x) - p_l(x) is SOS on I_j, maximize M·c
        2) Majorizer p_u:  p_u(x) - l_j(x) is SOS on I_j, minimize M·c

        intervals: list[(a,b)]
        lower_coefs_list: list of lists/arrays; coefficients of l_j(x) in power basis (lowest degree first)
        solver: CVXPY solver name (default "SCS")
        solver_max_iters: optional int, forwarded to cp.Problem.solve(...) as max_iters when provided
        """
        import cvxpy as cp
        if len(lower_coefs_list) != len(intervals):
            raise ValueError("lower_coefs_list must have the same length as intervals.")
        if poly_degrees is None:
            poly_degrees = [self.poly_degree]
        poly_degrees = sorted({int(deg) for deg in poly_degrees})
        if not poly_degrees:
            raise ValueError("poly_degrees must contain at least one degree.")

        max_available = len(self.values) - 1
        for deg in poly_degrees:
            if deg < 0 or deg > max_available:
                raise ValueError(f"Requested poly_degree {deg} is outside [0, {max_available}].")

        solve_kwargs = {}
        if solver_max_iters is not None:
            solve_kwargs['max_iters'] = int(solver_max_iters)

        def solve_for_degree(degree, init_c_LB=None, init_c_UB=None):
            M = self.values[:degree + 1]

            def interval_constraints(cvar, sign):
                def pad(lst, max_L):
                    return list(lst) + [0] * (max_L - len(lst))
                cons = []
                for (l, r), lcoef in zip(intervals, lower_coefs_list):
                    deg_r = int(max(1, degree))
                    if deg_r % 2 == 0:
                        d_sa = deg_r // 2 + 1
                        d_sb = d_sa - 1
                        Qa = cp.Variable((d_sa, d_sa), PSD=True) # a(x)
                        Qb = cp.Variable((d_sb, d_sb), PSD=True) # b(x)
                        sa = gram_to_coefs(Qa, d_sa)
                        sb = gram_to_coefs(Qb, d_sb)
                        quad = [-l * r, (l + r), -1.0]
                        conv = [0] * (deg_r + 1)
                        for i in range(len(sb)):
                            for j in range(len(quad)):
                                conv[i + j] += sb[i] * quad[j] # Multiply b(x) by ( -l r + (l + r) x - x^2 )
                        cp_poly = [cvar[i] if i <= degree else 0 for i in range(deg_r + 1)]
                        lp = pad([lcoef], deg_r + 1)
                        rcoefs = [sign * (cp_poly[k] - lp[k]) for k in range(deg_r + 1)]
                        L = max(len(rcoefs), len(sa), len(conv))
                        cons_a, cons_b, cons_f = pad(sa, L), pad(conv, L), pad(rcoefs, L)
                        for k in range(deg_r + 1):
                            cons.append(cons_f[k] == cons_a[k] + cons_b[k])
                        cons += [Qa >> 0, Qb >> 0]
                    else:
                        d_sc = (deg_r + 1) // 2
                        d_sd = (deg_r + 1) // 2
                        Qc = cp.Variable((d_sc, d_sc), PSD=True)
                        Qd = cp.Variable((d_sd, d_sd), PSD=True)
                        sc = gram_to_coefs(Qc, d_sc)
                        sd = gram_to_coefs(Qd, d_sd)
                        term_c = pad([0], len(sc) + 1)
                        for i in range(len(sc)):
                            term_c[i] += -l * sc[i] # Multiply by ( -l )
                            term_c[i + 1] += sc[i] # Multiply by ( x )
                        term_d = pad([0], len(sd) + 1)
                        for i in range(len(sd)):
                            term_d[i] += r * sd[i] # Multiply by ( r )
                            term_d[i + 1] -= sd[i] # Multiply by ( -x )
                        cp_poly = [cvar[i] if i <= degree else 0 for i in range(deg_r + 1)]
                        lp = pad([lcoef], deg_r + 1)
                        rcoefs = [sign * (cp_poly[k] - lp[k]) for k in range(deg_r + 1)]
                        L = max(len(rcoefs), len(term_c), len(term_d))
                        cons_c, cons_d, cons_f = pad(term_c, L), pad(term_d, L), pad(rcoefs, L)
                        for k in range(deg_r + 1):
                            cons.append(cons_f[k] == cons_c[k] + cons_d[k])
                        cons += [Qc >> 0, Qd >> 0]
                return cons

            c_LB = cp.Variable(degree + 1)
            if init_c_LB is not None:
                try:
                    c_LB.value = np.asarray(init_c_LB, dtype=float)
                except Exception:
                    c_LB.value = None
            cons_LB = interval_constraints(c_LB, sign=-1)
            if global_bnd is not None:
                cons_LB.append(c_LB <= global_bnd)
                cons_LB.append(-global_bnd <= c_LB)
            prob_LB = cp.Problem(cp.Maximize(M @ c_LB), cons_LB)
            prob_LB.solve(solver=solver, verbose=verbose, warm_start=True, **solve_kwargs)
            LB_val = float(M @ c_LB.value) if c_LB.value is not None else None

            c_UB = cp.Variable(degree + 1)
            if init_c_UB is not None:
                try:
                    c_UB.value = np.asarray(init_c_UB, dtype=float)
                except Exception:
                    c_UB.value = None
            cons_UB = interval_constraints(c_UB, sign=+1)
            if global_bnd is not None:
                cons_UB.append(c_UB <= global_bnd)
                cons_UB.append(c_UB >= -global_bnd)
            prob_UB = cp.Problem(cp.Minimize(M @ c_UB), cons_UB)
            prob_UB.solve(solver=solver, verbose=verbose, warm_start=True, **solve_kwargs)
            UB_val = float(M @ c_UB.value) if c_UB.value is not None else None

            lb_dummy = type("obj", (), {"x": c_LB.value})
            ub_dummy = type("obj", (), {"x": c_UB.value})
            bounds = [LB_val, UB_val]
            print(f"SOS bounds (degree {degree}):", bounds)
            return {
                "degree": degree,
                "bounds": bounds,
                "LB_problem": prob_LB,
                "UB_problem": prob_UB,
                "c_LB": c_LB.value,
                "c_UB": c_UB.value,
                "LB_result": lb_dummy,
                "UB_result": ub_dummy
            }

        summaries = []
        prev_c_LB = None
        prev_c_UB = None
        for deg in poly_degrees:
            # build warm starts by padding/truncating as needed
            def _expand(guess, new_len):
                if guess is None:
                    return None
                g = np.asarray(guess, dtype=float)
                if g.shape[0] == new_len:
                    return g
                if g.shape[0] < new_len:
                    return np.concatenate([g, np.zeros(new_len - g.shape[0])])
                return g[:new_len]

            init_lb = _expand(prev_c_LB, deg + 1)
            init_ub = _expand(prev_c_UB, deg + 1)
            summary = solve_for_degree(deg, init_c_LB=init_lb, init_c_UB=init_ub)
            summaries.append(summary)
            prev_c_LB = summary.get("c_LB")
            prev_c_UB = summary.get("c_UB")
            # handle visualization and optional saving
            save_plots = bool(kwargs.get('save_plots', False))
            save_dir = kwargs.get('save_dir', None)
            mode_name = kwargs.get('mode', None)
            show_plot = bool(kwargs.get('show', visualize))
            if visualize or save_plots:
                base_title = kwargs.get("plot_title", "")
                title = (base_title + " ").rstrip()
                save_path = None
                if save_plots and save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    title_safe = title.replace(' ', '_')
                    fname = f"{title_safe}_deg{deg}.pdf"
                    save_path = os.path.join(save_dir, fname)
                self.plot_results(
                    intervals,
                    deg,
                    lower_coefs_list,
                    n_plot_points=kwargs.get("n_plot_points", 1000),
                    title=title,
                    show_spectrum=kwargs.get("show_spectrum", False),
                    linewidth=kwargs.get("linewidth", 1),
                    exact_energies=kwargs.get("exact_energies"),
                    exact_overlaps=kwargs.get("exact_overlaps"),
                    c_LB=summary.get("c_LB"),
                    c_UB=summary.get("c_UB"),
                    bounds=summary.get("bounds"),
                    method="sos",
                    show=show_plot,
                    save_path=save_path,
                    mode=mode_name,
                )
        return summaries

    def optimize_2(self, intervals, lower_coefs_list, solver="SCS", verbose=False,
                global_bnd=None, solver_max_iters=None, poly_degrees=None,
                visualize=False, **kwargs):
        """
        Unified SOS/SDP solver (fixed, robust): replaces your previous optimize.
        - intervals: list[(a,b)]
        - lower_coefs_list: list of coefficient-lists for l_j(x) (power-basis, low-first)
        - poly_degrees: iterable of degrees to sweep (each degree solved in sequence,
        warm-starting from previous solution)
        """
        import cvxpy as cp

        if len(lower_coefs_list) != len(intervals):
            raise ValueError("lower_coefs_list must have the same length as intervals.")

        # degrees to sweep
        if poly_degrees is None:
            poly_degrees = [self.poly_degree]
        poly_degrees = sorted({int(d) for d in poly_degrees})
        if not poly_degrees:
            raise ValueError("poly_degrees must contain at least one degree.")

        max_available = len(self.values) - 1
        for deg in poly_degrees:
            if deg < 0 or deg > max_available:
                raise ValueError(f"Requested poly_degree {deg} is outside [0, {max_available}].")

        solve_kwargs = {}
        if solver_max_iters is not None:
            solve_kwargs['max_iters'] = int(solver_max_iters)

        # helper: Gram matrix -> power-basis coefficients
        def gram_to_coefs(Q, deg):
            # returns list length 2*deg+1
            coefs = []
            for k in range(2 * deg + 1):
                expr = 0
                for i in range(deg + 1):
                    j = k - i
                    if 0 <= j <= deg:
                        expr = expr + Q[i, j]
                coefs.append(expr)
            return coefs

        # Build interval constraints for a given polynomial variable cvar and sign
        # sign = +1 -> enforce p - l = SOS (majorizer)
        # sign = -1 -> enforce l - p = SOS (minorizer)
        def interval_constraints(cvar, sign, target_degree):
            """
            Build SOS constraints for:
                sign * (p(x) - l(x)) >= 0 on each interval [a, b]
            using the Markov–Lukács decomposition:
                sign*(p-l) = s_L(x)*(x-a) + s_R(x)*(b-x)
            where s_L and s_R are SOS polynomials.

            This works for ALL degrees (even or odd).
            """
            import cvxpy as cp

            cons = []
            n = target_degree  # degree of p(x)

            for (a, b), lcoef in zip(intervals, lower_coefs_list):
                # Ensure lower function is a coefficient list
                if np.isscalar(lcoef):
                    l_list = [float(lcoef)]
                else:
                    l_list = list(lcoef)

                # Residual degree = max(deg(p), deg(l))
                deg_r = max(n, len(l_list) - 1)
                deg_r = int(deg_r)

                # Degree of s_L and s_R:
                # (deg(s_L) + 1) + 1 = deg_r+1  => deg(s_L) = deg_r - 1
                # Same for s_R.
                # If deg_r == 0 (constant residual), deg_s = 0 is OK.
                deg_s = max(0, deg_r - 1)

                # Gram matrices for SOS polynomials
                QL = cp.Variable((deg_s + 1, deg_s + 1), PSD=True)
                QR = cp.Variable((deg_s + 1, deg_s + 1), PSD=True)

                # Convert Gram matrices to coefficient lists
                sL = gram_to_coefs(QL, deg_s)  # SOS => degree 2*deg_s
                sR = gram_to_coefs(QR, deg_s)

                # Build (x-a) and (b-x) linear factors in power basis
                # (x - a) = [-a, 1]
                # (b - x) = [b, -1]
                xa = [-a, 1.0]
                bx = [b, -1.0]

                # Convolve SOS coefs with linear factors
                conv_L = [0] * (len(sL) + 1)
                for i in range(len(sL)):
                    conv_L[i]     += sL[i] * xa[0]
                    conv_L[i + 1] += sL[i] * xa[1]

                conv_R = [0] * (len(sR) + 1)
                for i in range(len(sR)):
                    conv_R[i]     += sR[i] * bx[0]
                    conv_R[i + 1] += sR[i] * bx[1]

                # Build p(x) coefficients up to deg_r
                pcoefs = [cvar[i] if i <= n else 0 for i in range(deg_r + 1)]

                # Build l(x) coefficients up to deg_r
                lpad = list(l_list[:deg_r + 1]) + [0] * max(0, (deg_r + 1) - len(l_list))

                # Residual r(x) = sign*(p - l)
                rcoefs = [sign * (pcoefs[k] - lpad[k]) for k in range(deg_r + 1)]

                # Pad SOS sides
                L = max(len(rcoefs), len(conv_L), len(conv_R))
                def pad(v): return list(v) + [0] * (L - len(v))
                rpad = pad(rcoefs)
                Lpad = pad(conv_L)
                Rpad = pad(conv_R)

                # Coefficient matching
                for k in range(deg_r + 1):
                    cons.append(rpad[k] == Lpad[k] + Rpad[k])

                # PSD constraints explicitly added (redundant but stabilizing)
                cons.append(QL >> 0)
                cons.append(QR >> 0)

            return cons

        # solve for a specific degree (with optional warm-starts provided)
        def solve_for_degree(degree, init_c_LB=None, init_c_UB=None):
            n = degree
            M = self.values[:n + 1]

            # ----------------- Minorizer (maximize M·c subject to l - p SOS) -----------------
            c_LB = cp.Variable(n + 1)
            if init_c_LB is not None:
                try:
                    arr = np.asarray(init_c_LB, dtype=float)
                    if arr.size == n + 1:
                        c_LB.value = arr
                    elif arr.size < n + 1:
                        c_LB.value = np.concatenate([arr, np.zeros(n + 1 - arr.size)])
                    else:
                        c_LB.value = arr[:n + 1]
                except Exception:
                    c_LB.value = None

            cons_LB = interval_constraints(c_LB, sign=-1, target_degree=n)
            if global_bnd is not None:
                # elementwise box on coefficients
                cons_LB += [c_LB <= float(global_bnd), c_LB >= -float(global_bnd)]

            prob_LB = cp.Problem(cp.Maximize(M @ c_LB), cons_LB)
            prob_LB.solve(solver=solver, verbose=verbose, warm_start=True, **solve_kwargs)

            LB_val = float(M @ c_LB.value) if c_LB.value is not None else None

            # ----------------- Majorizer (minimize M·c subject to p - l SOS) -----------------
            c_UB = cp.Variable(n + 1)
            if init_c_UB is not None:
                try:
                    arr = np.asarray(init_c_UB, dtype=float)
                    if arr.size == n + 1:
                        c_UB.value = arr
                    elif arr.size < n + 1:
                        c_UB.value = np.concatenate([arr, np.zeros(n + 1 - arr.size)])
                    else:
                        c_UB.value = arr[:n + 1]
                except Exception:
                    c_UB.value = None

            cons_UB = interval_constraints(c_UB, sign=+1, target_degree=n)
            if global_bnd is not None:
                cons_UB += [c_UB <= float(global_bnd), c_UB >= -float(global_bnd)]

            prob_UB = cp.Problem(cp.Minimize(M @ c_UB), cons_UB)
            prob_UB.solve(solver=solver, verbose=verbose, warm_start=True, **solve_kwargs)

            UB_val = float(M @ c_UB.value) if c_UB.value is not None else None

            lb_dummy = type("obj", (), {"x": c_LB.value})
            ub_dummy = type("obj", (), {"x": c_UB.value})
            bounds = [LB_val, UB_val]
            print(f"SOS bounds (degree {degree}):", bounds)
            return {
                "degree": degree,
                "bounds": bounds,
                "LB_problem": prob_LB,
                "UB_problem": prob_UB,
                "c_LB": c_LB.value,
                "c_UB": c_UB.value,
                "LB_result": lb_dummy,
                "UB_result": ub_dummy
            }

        # Sweep degrees with warm-starting of c only (Gram warm-start optional)
        summaries = []
        prev_c_LB = None
        prev_c_UB = None

        for deg in poly_degrees:
            def _expand(guess, new_len):
                if guess is None:
                    return None
                g = np.asarray(guess, dtype=float)
                if g.shape[0] == new_len:
                    return g
                if g.shape[0] < new_len:
                    return np.concatenate([g, np.zeros(new_len - g.shape[0])])
                return g[:new_len]

            init_lb = _expand(prev_c_LB, deg + 1)
            init_ub = _expand(prev_c_UB, deg + 1)
            summary = solve_for_degree(deg, init_c_LB=init_lb, init_c_UB=init_ub)
            summaries.append(summary)
            prev_c_LB = summary.get("c_LB")
            prev_c_UB = summary.get("c_UB")

            # plotting / saving (same API as your original)
            save_plots = bool(kwargs.get('save_plots', False))
            save_dir = kwargs.get('save_dir', None)
            mode_name = kwargs.get('mode', None)
            show_plot = bool(kwargs.get('show', visualize))
            if visualize or save_plots:
                base_title = kwargs.get("plot_title", "")
                title = (base_title + " ").rstrip()
                save_path = None
                if save_plots and save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    title_safe = title.replace(' ', '_')
                    fname = f"{title_safe}_deg{deg}.pdf"
                    save_path = os.path.join(save_dir, fname)
                self.plot_results(
                    intervals,
                    deg,
                    lower_coefs_list,
                    n_plot_points=kwargs.get("n_plot_points", 1000),
                    title=title,
                    show_spectrum=kwargs.get("show_spectrum", False),
                    linewidth=kwargs.get("linewidth", 1),
                    exact_energies=kwargs.get("exact_energies"),
                    exact_overlaps=kwargs.get("exact_overlaps"),
                    c_LB=summary.get("c_LB"),
                    c_UB=summary.get("c_UB"),
                    bounds=summary.get("bounds"),
                    method="sos",
                    show=show_plot,
                    save_path=save_path,
                    mode=mode_name,
                )

        return summaries

    def plot_results(self, intervals, poly_degree=None, lower_coefs_list=None, n_plot_points=1000, linewidth=2, title=None, show_spectrum=False, show=True, save_path=None, legend_anchor=(1,0.2), show_constraint_points=False, plot_chebyshev=False, constraint_recipe_obj=None, truncate_lower=None, **kwargs):
        """
        Plot optimized polynomials and constraint functions using the same style as PolyBounds_new.plot_results.
        Accepts `show` and `save_path` to control display and saving.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # global xs across all intervals
        xs = np.linspace(intervals[0][0], intervals[-1][-1], n_plot_points)

        # determine polynomial coefficients
        c_LB = kwargs.get('c_LB', getattr(self, 'c_LB', None))
        c_UB = kwargs.get('c_UB', getattr(self, 'c_UB', None))
        if c_LB is not None:
            LB_poly = Polynomial(c_LB)
            LB_vals = LB_poly(xs)
        else:
            LB_vals = None
        if c_UB is not None:
            UB_poly = Polynomial(c_UB)
            UB_vals = UB_poly(xs)
        else:
            UB_vals = None

        # Plot lower and upper polynomials if available
        if LB_vals is not None:
            ax.plot(xs, LB_vals, color='blue', label='Lower Bound', linewidth=linewidth)
        if UB_vals is not None:
            ax.plot(xs, UB_vals, color='green', label='Upper Bound', linewidth=linewidth)

        # Plot piecewise constraint functions on each interval
        for (a, b), lcoefs in zip(intervals, lower_coefs_list if lower_coefs_list is not None else []):
            xs_piece = np.linspace(a, b, max(3, n_plot_points // max(1, len(intervals))))
            f_poly = Polynomial(lcoefs)
            ax.plot(xs_piece, f_poly(xs_piece), linestyle='--', color='red', linewidth=linewidth)

        # Optionally show spectrum on twin axis
        if show_spectrum and constraint_recipe_obj is not None:
            ax2 = ax.twinx()
            exact_energies = getattr(constraint_recipe_obj, 'exact_energies', kwargs.get('exact_energies'))
            exact_overlaps = getattr(constraint_recipe_obj, 'exact_overlaps', kwargs.get('exact_overlaps'))
            if exact_energies is not None and exact_overlaps is not None:
                ax2.stem(exact_energies, exact_overlaps, linefmt='grey', markerfmt=' ', basefmt='C0', label='Exact spectrum')
                ax2.set_ylabel('Overlap')

        # Constraint points marker
        if show_constraint_points:
            # plot interval endpoints as markers
            pts = []
            for a, b in intervals:
                pts.extend([a, b])
            pts = np.array(pts)
            ax.scatter(pts, np.zeros_like(pts), color='pink', marker='|', zorder=4)

        # Title and legend
        if title:
            # try to include bounds info if provided
            bounds = kwargs.get('bounds')
            deg_for_title = poly_degree if poly_degree is not None else getattr(self, 'poly_degree', None)
            if bounds is not None:
                oex = getattr(self, "O_exact", None)
                # format bounds safely if any entry is None
                try:
                    b0, b1 = bounds[0], bounds[1]
                except Exception:
                    b0, b1 = None, None
                if b0 is not None and b1 is not None:
                    bounds_fmt = f'[{b0:.3f}, {b1:.3f}]'
                else:
                    bounds_fmt = f'[{b0}, {b1}]'
                if oex is not None:
                    try:
                        ax.set_title(title + f', Degree={deg_for_title}' + f', {bounds_fmt} ({oex:.3f})')
                    except Exception:
                        ax.set_title(title + f', Degree={deg_for_title}' + f', {bounds_fmt} ({oex})')
                else:
                    ax.set_title(title + f', Degree={deg_for_title}' + f', {bounds_fmt}')
            else:
                ax.set_title(title + f' (deg={deg_for_title})')
        ax.set_ylabel('Function Value')
        fig.legend(loc='lower right', bbox_to_anchor=legend_anchor, bbox_transform=ax.transAxes)
        plt.tight_layout()

        # Save if requested
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            except Exception:
                pass
            try:
                fig.savefig(save_path, format='pdf')
            except Exception:
                fig.savefig(save_path)

        # Show if requested
        if show:
            plt.show()
        plt.close(fig)
    
def intervals_from_eigen(eig_results, target_index, b_values):
    """
    Generate intervals and lower polynomial coefficients from eigenvalue results.
    :param eig_results: dictionary of eigenvalue results.
    :param target_index: index of the target state.
    :param b_values: list of b values for the constraints. Must match the length of target_indices.
    """
    exact_energies = eig_results['exact_energies_no_degen']
    if target_index < 0 or target_index >= len(exact_energies):
        raise ValueError("target_index is out of bounds.")
    if len(b_values) != 1:
        raise ValueError("Currently only supporting b_values with one element.")
    E0 = exact_energies[target_index]
    Emax = exact_energies[-1]
    intervals = [(exact_energies[0], E0), (E0, Emax)]
    lower_coefs_list = []
    for (a, b), b_val in zip(intervals, b_values):
        lower_coefs_list.append([b_val * (b - x) / (b - a) for x in [a, b]])
    return intervals, lower_coefs_list
def moments_from_distr(evals, overlaps, degree):
    """
    Compute the first n moments of a Hamiltonian given its eigenvalues and overlaps.
    Formula: moments = sum(overlap * eval^i) / sum(overlap)
    """
    moments = np.array([np.dot(evals**i, overlaps) for i in range(degree+1)])
    # if return_cheb:
    #     moments = chebyshev_matrix_from_degree(degree) @ moments
    return moments

def random_distr(num_evals, major_ovlp=0.6, max_ovlp_index=1, energy_bounds=(-1,1), seed=None):
    """
    Generate a random distribution of eigenvalues and overlaps.
    There should be one dominant overlap equal to max_ovlp, and the rest are random.
    The overlap values should sum to 1.
    """
    if seed is not None:
        np.random.seed(seed)
    evals = np.sort(np.random.uniform(energy_bounds[0], energy_bounds[1], num_evals))
    small_probs = np.random.dirichlet([1] * (num_evals - 1)) * (1-major_ovlp)
    overlaps = np.insert(small_probs, max_ovlp_index, major_ovlp)
    return (evals, overlaps)

class OverlapEstimator:
    """
    Estimate overlaps via polynomial bounds or SIP configurations.
    """

    def __init__(self, moments=None, exact_overlaps=None, exact_energies=None):
        self.moments = moments
        self.exact_overlaps = exact_overlaps
        self.exact_energies = exact_energies
        self.last_results = None

    def _normalize_modes(self, recipe: dict):
        modes = recipe.get("mode")
        if modes is None:
            raise ValueError("recipe must include a 'mode' entry.")
        if isinstance(modes, str):
            modes = [modes]
        elif isinstance(modes, (list, tuple)):
            modes = list(modes)
        else:
            raise ValueError("mode must be a string or list/tuple of strings.")
        normalized = []
        for mode in modes:
            if mode is None:
                continue
            normalized.append(str(mode).lower())
        if not normalized:
            raise ValueError("mode list cannot be empty.")
        unsupported = set(normalized) - {"exact", "approx", "manual"}
        if unsupported:
            raise ValueError(f"Unsupported mode(s): {unsupported}.")
        return normalized

    def _build_mode_recipe(self, recipe: dict, mode: str):
        mode_overrides = recipe.get(mode, {})
        if mode_overrides and not isinstance(mode_overrides, dict):
            raise ValueError(f"Overrides for mode '{mode}' must be provided as a dict.")
        # Copy base recipe excluding explicit per-mode override blocks
        base = {k: v for k, v in recipe.items() if k not in {"mode", "exact", "approx", "manual"}}
        merged = {**base, **(mode_overrides or {})}
        merged["mode"] = mode
        return merged

    def _validate_single_mode(self, recipe: dict):
        mode = recipe["mode"].lower()
        if mode in {"exact", "approx"}:
            if "eig_results" not in recipe:
                raise ValueError(f"eig_results are required for '{mode}' mode.")
            if "target_indices" not in recipe or "b_values" not in recipe:
                raise ValueError(f"target_indices and b_values must be provided for '{mode}' mode.")
            if len(recipe["target_indices"]) != len(recipe["b_values"]):
                raise ValueError("target_indices and b_values must have the same length.")
        if mode == "manual":
            if recipe.get("intervals") is None:
                raise ValueError("intervals must be specified for manual mode.")
            has_moments = any([
                recipe.get("moments") is not None,
                recipe.get("moments_file") is not None,
                self.moments is not None
            ])
            if not has_moments:
                raise ValueError("moments must be provided inline, via moments_file, or at initialization for manual mode.")
        if recipe.get("poly_degrees") is None and recipe.get("poly_degree") is None:
            raise ValueError("Specify poly_degrees or poly_degree in the recipe.")

    def compute_bounds(self, recipe: dict, visualize:bool=False, save_plots:bool=False, save_dir: str=None):
        modes = self._normalize_modes(recipe)
        all_payloads = []
        for mode in modes:
            mode_recipe = self._build_mode_recipe(recipe, mode)
            self._validate_single_mode(mode_recipe)
            poly_degrees = self._normalize_poly_degrees(mode_recipe)
            if mode == "exact":
                payload = self._compute_exact_mode(mode_recipe, poly_degrees, visualize=visualize, save_plots=save_plots, save_dir=save_dir, mode=mode)
            elif mode == "approx":
                payload = self._compute_approx_mode(mode_recipe, poly_degrees, visualize=visualize, save_plots=save_plots, save_dir=save_dir, mode=mode)
            else:
                payload = self._compute_manual_mode(mode_recipe, poly_degrees, visualize=visualize, save_plots=save_plots, save_dir=save_dir, mode=mode)
            all_payloads.append(payload)
        self.last_results = all_payloads[-1] if all_payloads else None
        return all_payloads if len(all_payloads) > 1 else all_payloads[0]

    def save_results(self, results, output_file_path, fmt='json'):
        """Save compute_bounds results to a local file.

        Supported formats: 'json', 'csv', 'pickle'.
        """
        import json as _json
        import pickle as _pickle
        import csv as _csv

        if results is None:
            raise ValueError("No results to save")
        # Normalize to list
        payloads = results if isinstance(results, (list, tuple)) else [results]
        def _to_basic(o):
            # Convert payloads into JSON/pickle-safe basic Python structures
            if isinstance(o, dict):
                new = {}
                for k, v in o.items():
                    # ensure key is a string
                    try:
                        key = k if isinstance(k, (str, int, float, bool)) or k is None else str(k)
                    except Exception:
                        key = str(k)
                    new[str(key)] = _to_basic(v)
                return new

            if isinstance(o, (list, tuple, set)):
                return [_to_basic(x) for x in o]

            # numpy arrays / scalars -> python types via tolist()
            try:
                if hasattr(o, 'tolist') and not isinstance(o, (str, bytes)):
                    return _to_basic(o.tolist())
            except Exception:
                pass

            if isinstance(o, (str, int, float, bool)) or o is None:
                return o

            return str(o)

        payloads_clean = _to_basic(payloads)

        if fmt == 'json':
            with open(output_file_path, 'w', encoding='utf-8') as fh:
                _json.dump(payloads_clean, fh, indent=2)
            return output_file_path

        if fmt == 'pickle':
            with open(output_file_path, 'wb') as fh:
                _pickle.dump(payloads_clean, fh)
            return output_file_path

        if fmt == 'csv':
            # Flatten degree_results rows: mode, degree, lower, upper
            with open(output_file_path, 'w', newline='', encoding='utf-8') as fh:
                writer = _csv.writer(fh)
                writer.writerow(['mode', 'degree', 'lower', 'upper'])
                for p in payloads_clean:
                    mode = p.get('mode')
                    for entry in p.get('degree_results', []):
                        writer.writerow([mode, entry.get('degree'), entry.get('lower'), entry.get('upper')])
            return output_file_path

        raise ValueError(f"Unsupported format: {fmt}")

    def plot_bounds(self, results=None, show=True, output_file_path=None, **kwargs):
        """Visualize bounds computed by compute_bounds().

        If results is None, uses `self.last_results`.
        """
        payloads = results if results is not None else (self.last_results if self.last_results is not None else None)
        if payloads is None:
            raise ValueError("No results available to plot. Run compute_bounds() first or pass results explicitly.")
        viz.plot_bounds(payloads, show=show, save_path=output_file_path, **kwargs)

    def _normalize_poly_degrees(self, recipe: dict):
        degrees = recipe.get("poly_degrees")
        if degrees is None:
            degrees = [recipe.get("poly_degree")]
        if degrees is None:
            raise ValueError("poly_degrees could not be determined from the recipe.")
        if isinstance(degrees, int):
            degrees = [degrees]
        normalized = sorted({int(deg) for deg in degrees})
        if any(deg < 0 for deg in normalized):
            raise ValueError("poly_degrees must be non-negative.")
        return normalized

    def _format_solver_output(self, solver_output, degrees):
        if isinstance(solver_output, list):
            formatted = []
            for entry in solver_output:
                bounds = entry.get("bounds", [np.nan, np.nan])
                formatted.append({
                    "degree": entry.get("degree"),
                    "lower": bounds[0],
                    "upper": bounds[1]
                })
            return formatted
        if isinstance(solver_output, tuple) and len(solver_output) == 3:
            bounds = solver_output[2]
            return [{"degree": degrees[-1], "lower": bounds[0], "upper": bounds[1]}]
        raise ValueError("Unexpected solver output format.")

    def _load_eig_results(self, recipe: dict):
        eig_results = recipe.get("eig_results")
        if eig_results is None:
            raise ValueError("eig_results must be provided for exact/approx modes.")
        if isinstance(eig_results, str):
            with open(eig_results, "r", encoding="utf-8") as handle:
                eig_results = json.load(handle)
        elif isinstance(eig_results, dict):
            eig_results = copy.copy(eig_results)
        else:
            raise ValueError("eig_results must be a path or a dictionary.")
        return eig_results

    def _load_manual_moments(self, recipe: dict):
        manual_values = recipe.get("moments")
        if manual_values is not None:
            return np.array(manual_values, dtype=float)
        moments_file = recipe.get("moments_file")
        if moments_file:
            try:
                with open(moments_file, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"Moments file not found: {moments_file}") from exc
            except Exception as exc:
                raise ValueError(f"Failed to load moments file '{moments_file}': {exc}") from exc
            if isinstance(payload, dict):
                manual_values = payload.get("moments")
            else:
                manual_values = payload
            if manual_values is None:
                raise ValueError(f"Moments file '{moments_file}' does not contain 'moments'.")
            return np.array(manual_values, dtype=float)
        if self.moments is not None:
            return np.array(self.moments, dtype=float)
        raise ValueError("moments must be provided either directly, via moments_file, or during initialization.")

    def _compute_exact_mode(self, recipe: dict, poly_degrees, visualize:bool=False, **kwargs):
        eig_results = self._load_eig_results(recipe)
        target_indices = recipe["target_indices"]
        b_values = recipe["b_values"]
        constraint_info = PB_linprog_params_from_eig(
            eig_results,
            target_indices=target_indices,
            use_exact_energies=True,
            use_all_evals=recipe.get("use_all_evals", False),
            custom_constraints=recipe.get("custom_constraints"),
            use_cheb=recipe.get("use_cheb", True),
            rescale=recipe.get("rescale", True),
            Energy_constraint_limits=recipe.get("energy_constraint_limits", (float('-inf'), float('inf')))
        )
        max_degree = poly_degrees[-1]
        if max_degree > constraint_info.max_poly_degree:
            raise ValueError("Requested polynomial degree exceeds available moments.")
        values = constraint_info.get_moment_values(max_degree)
        A_full = constraint_info.get_A_ub(max_degree)
        b_ub = constraint_info.get_b_ub(b_values)
        exact_overlap = constraint_info.get_exact_ovlp(b_values)
        linprog_options = recipe.get("linprog_options", {})
        poly = PolyBounds_new(values, A_full, b_ub, constraint_info.get_constraint_pts(), P_exact=exact_overlap)
        solver_output = poly.optimize(
            poly_degrees=poly_degrees,
            bound=recipe.get("linprog_bound", 1e20),
            method=recipe.get("linprog_method", "highs"),
            linprog_options=linprog_options,
            visualize=visualize,
            plot_title = "Exact Mode",
            save_plots=kwargs.get('save_plots', False),
            save_dir=kwargs.get('save_dir', None),
            mode=kwargs.get('mode', None),
        )
        summary = self._format_solver_output(solver_output, poly_degrees)
        return {
            "mode": "exact",
            "degree_results": summary,
            "exact_overlap": exact_overlap,
            "raw_solver_output": solver_output,
            "poly_degrees": list(poly_degrees)
        }

    def _compute_approx_mode(self, recipe: dict, poly_degrees, visualize:bool=False, **kwargs):
        eig_results = self._load_eig_results(recipe)
        target_indices = recipe["target_indices"]
        b_values = recipe["b_values"]
        bound_extension = recipe.get("bound_extension")
        if bound_extension is None:
            raise ValueError("bound_extension must be provided for approx mode.")
        dense_info = tuple(recipe.get("dense_info", (None, None)))
        constraint_info = PB_linprog_params_from_eig(
            eig_results,
            target_indices=target_indices,
            use_exact_energies=True,
            use_all_evals=recipe.get("use_all_evals", False),
            custom_constraints=recipe.get("custom_constraints"),
            use_cheb=recipe.get("use_cheb", False),
            rescale=recipe.get("rescale", True),
            bound_extension=bound_extension,
            dense_info=dense_info,
            Energy_constraint_limits=recipe.get("energy_constraint_limits", (float('-inf'), float('inf')))
        )
        max_degree = poly_degrees[-1]
        if max_degree > constraint_info.max_poly_degree:
            raise ValueError("Requested polynomial degree exceeds available moments.")
        intervals = constraint_info.get_approx_intervals()
        region_b_list = np.zeros(len(intervals))
        for idx, b_val in zip(np.atleast_1d(constraint_info.nonzero_b_indices), b_values):
            region_b_list[int(idx)] = b_val
        values = constraint_info.get_moment_values(max_degree)
        sip = PolyBounds_SIP(values, poly_degree=max_degree, O_exact=constraint_info.get_exact_ovlp(b_values))
        solver_output = sip.optimize(
            intervals,
            region_b_list,
            solver=recipe.get("solver", "SCS"),
            verbose=recipe.get("verbose", False),
            global_bnd=recipe.get("global_bound"),
            solver_max_iters=recipe.get("solver_max_iters"),
            poly_degrees=poly_degrees,
            visualize=visualize,
            plot_title = "Approx Mode",
            save_plots=kwargs.get('save_plots', False),
            save_dir=kwargs.get('save_dir', None),
            mode=kwargs.get('mode', None),
        )
        summary = self._format_solver_output(solver_output, poly_degrees)
        return {
            "mode": "approx",
            "degree_results": summary,
            "intervals": intervals,
            "region_b_values": region_b_list.tolist(),
            "raw_solver_output": solver_output,
            "poly_degrees": list(poly_degrees)
        }

    def _compute_manual_mode(self, recipe: dict, poly_degrees, visualize:bool=False, **kwargs):
        manual_moments = self._load_manual_moments(recipe)
        max_available = len(manual_moments) - 1
        max_degree = poly_degrees[-1]
        if max_degree > max_available:
            raise ValueError("Requested polynomial degree exceeds available manual moments.")
        intervals_cfg = recipe.get("intervals", [])
        intervals = []
        lower_coefs_list = []
        for entry in intervals_cfg:
            if len(entry) < 3:
                raise ValueError("Each interval entry must include at least (low, high, b_value).")
            low, high = entry[0], entry[1]
            b_val = entry[2]
            intervals.append((low, high))
            if isinstance(b_val, (list, tuple, np.ndarray)):
                lower_coefs_list.append(list(b_val))
            else:
                lower_coefs_list.append([b_val])
        sip = PolyBounds_SIP(manual_moments[:max_degree + 1], poly_degree=max_degree)
        solver_output = sip.optimize(
            intervals,
            lower_coefs_list,
            solver=recipe.get("solver", "SCS"),
            verbose=recipe.get("verbose", False),
            global_bnd=recipe.get("global_bound"),
            solver_max_iters=recipe.get("solver_max_iters"),
            poly_degrees=poly_degrees,
            visualize=visualize,
            plot_title = "Manual Mode",
            save_plots=kwargs.get('save_plots', False),
            save_dir=kwargs.get('save_dir', None),
            mode=kwargs.get('mode', None),
        )
        summary = self._format_solver_output(solver_output, poly_degrees)
        return {
            "mode": "manual",
            "degree_results": summary,
            "intervals": intervals,
            "raw_solver_output": solver_output,
            "poly_degrees": list(poly_degrees)
        }
    
class MultStateBounds:
    """
    Resolves the bounds for multiple states using an "outer" linear program loop.
    Junan 20251115: UNFINISHED
    """
    def __init__(self, F, l, u):
        if F.shape[0] != len(l) or F.shape[0] != len(u):
            raise ValueError(f"# rows in F: {F.shape[0]} must match the length of l: {len(l)} and u: {len(u)}.")
        if F.shape[0] < F.shape[1]:
            raise ValueError(f"F must have more rows {F.shape[0]} than columns {F.shape[1]}.")
        self.F = F
        self.l = l
        self.u = u
    def solve(self):
        self.lb_P_list = []
        self.ub_P_list = []
        A_ub = np.vstack((self.F, -self.F))
        b_ub = np.hstack((self.u, -self.l))
        for i in range(self.F.shape[1]+1):
            if i < self.F.shape[1]:
                specification_vector = np.zeros(self.F.shape[1])
                specification_vector[i] = 1
            else:
                specification_vector = np.ones(i) # Add a final entry: the best combined overlap
            ms_lb = linprog(specification_vector, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1))
            ms_ub = linprog(-specification_vector, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1))
            self.lb_P_list.append(np.dot(specification_vector, ms_lb.x))
            self.ub_P_list.append(np.dot(specification_vector, ms_ub.x)) # Note the cancellation of negative sign
        return self.lb_P_list, self.ub_P_list

def get_MS_l_u(constraint_recipe:PB_linprog_params_from_eig, F:np.array, poly_degree:int, visualize=False, used_cheb=False):
    """
    Generate the l and u vectors for the MultStateBounds class.
    """
    l = []
    u = []
    for b_values in F:
        values = constraint_recipe.get_moment_values(poly_degree)
        A_ub = constraint_recipe.get_A_ub(poly_degree)
        b_ub = constraint_recipe.get_b_ub(b_values)
        poly_bounds = PolyBounds_new(values, A_ub, b_ub, constraint_pts=constraint_recipe.get_constraint_pts())
        poly_bounds.optimize()
        if visualize:
            poly_bounds.plot_results(constraint_pts=constraint_recipe.get_constraint_pts(), constraint_recipe_obj=constraint_recipe, show_constraint_points=True, show_spectrum=True, used_cheb=used_cheb)
        l.append(poly_bounds.bounds[0])
        u.append(poly_bounds.bounds[1])
    return np.array(l), np.array(u)

class CauchyLB():
    """
    Using the Cauchy integral formula to directly compute a lower bound for the overlap,
    by calculating the expectation of <f(H)>.
    Junan 20241017: UNFINISHED
    """
    def __init__(self, H, v, E0, E1, Emax, num_z_points, plot_title, O_exact, print_iter=False, visualize=True):
        local_vars = locals()  # Dictionary of local variables
        for name, value in local_vars.items():
            if name != 'self':  # Exclude 'self' from being set as an attribute
                setattr(self, name, value)
        gap = E1 - E0
        self.contour = getContourDataFromEndPoints(E0-gap/2, E0+gap/2, num_z_points)
    def run(self):
        # Junan 20240929: UNFINISHED
        computed_QF = []
        for z in self.contour['Points']:
            MTI = copy.copy(-self.H) # This is the matrix to be inverted
            identity = sp.sparse.diags([z for _ in range(MTI.shape[0])])
            MTI += identity
            my_solver = linear_solver(MTI, self.v, self.v)
            state, n_iter = my_solver.gmres_solve(tol=1e-7, restart=30, maxiter=100)
            computed_QF.append(np.real_if_close(np.vdot(self.v, state)))
        
def stddev_and_var(H, vs, threshold=1e-10):
    """
    Returns the standard deviation and variance of the Hamiltonian H wrt vectors vs.
    vs is a collection of column vectors.
    """
    variance = []
    for i in range(vs.shape[1]):
        first_two_moments = exp_val_higher_moment(H, vs[:, i:i+1], 2, return_all=True)
        this_var = first_two_moments[2] - first_two_moments[1]**2
        if this_var < 0:
            if -this_var < threshold:
                print("Small negative variance, casting to 0.")
                this_var = 0
            else:
                print("Negative variance detected.")
                print("Variance is", this_var)
                print("Threshold is", threshold)
                print("Returning 0 variance, proceed with caution.")
                this_var = 0
        variance.append(this_var)
    
    variance = np.array(variance)
    return np.sqrt(variance), variance
    
def K_matrix(eps, ritz:np.array, std_devs:np.array):
    """
    Return the matrix used in Pollak−Martinazzo theory for obtaining energy lower bounds.
    """
    Ritz_vals = np.append(ritz, eps + np.sum(std_devs**2/(ritz-eps)))
    L = len(ritz)
    K = np.diag(Ritz_vals)
    K[:-1, L] = std_devs.T
    K[L, :-1] = std_devs
    return K

def visualize_LBs(exact_energies, Ritz_energies, K_evals, epsilon):
    plt.figure()
    # Plot the individual exact energies as points on the x-axis
    plt.scatter(exact_energies, np.zeros_like(exact_energies), marker='|', color='blue', label='Exact energies')
    # Plot the Ritz energies as points on the x-axis
    plt.scatter(Ritz_energies, np.zeros_like(Ritz_energies), marker='|', color='red', label='Ritz energies')
    # Plot the K eigenvalues as points on the x-axis
    plt.scatter(K_evals, np.zeros_like(K_evals), marker='|', color='purple', label='K eigenvalues')
    # Plot the value of epsilon as a point on the x-axis
    plt.scatter(epsilon, 0, marker='|', color='green', label='Value of epsilon')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    recipe = {"moments": [1, -2, 4, 8, -16, 32],
              "regions": [(-3, -2, 1, 10, True), (-1.95, 2, 0, 20, True)],
              "use_cheb": True}
    constraint_pts, b_ub, A_ub = PBinputs_from_eig_agnostic_recipe(recipe)
    print("Constraint points:", constraint_pts)
    print("b_ub:", b_ub)
    print("A_ub shape:", A_ub.shape)
    print("b_ub shape:", b_ub.shape)