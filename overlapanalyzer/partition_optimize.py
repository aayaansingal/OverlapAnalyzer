from scipy.optimize import minimize
import numpy as np
from scipy.sparse import identity, diags
from scipy.sparse.linalg import norm
from openfermion import load_operator
from openfermion.linalg import get_sparse_operator
from overlapanalyzer.read_ham import find_files, quick_load


def objective_function_real_imag(params, H, H0):
    """Objective function to minimize with real and imaginary parts of c."""
    a, b = params  # Real and imaginary parts of c
    c = complex(a, b)
    I = identity(H.shape[0], dtype=complex)  # Identity matrix with complex dtype
    difference = H - (H0 + c * I)
    return norm(difference)

def minimize_norm_real_imag(H, H0):
    """Find the optimal real and imaginary parts of c that minimizes the norm."""
    # Initial guesses for real and imaginary parts of c
    initial_guess = [0, 0]

    # Minimize the objective function
    result = minimize(objective_function_real_imag, initial_guess, args=(H, H0), method='Nelder-Mead')

    a_optimal, b_optimal = result.x  # Optimal values of real and imaginary parts of c
    c_optimal = complex(a_optimal, b_optimal)
    minimized_norm = result.fun  # Minimized norm

    # Calculate the operator H0 + c_optimal * I
    I = identity(H.shape[0], dtype=complex)
    operator = H0 + c_optimal * I

    return c_optimal, minimized_norm, operator

if __name__ == "__main__":
    dir = 'hamiltonian_gen_test/h2o/4e4o'
    original_hams = find_files(dir+'/ham_fer',".data")

    for i, filename in enumerate(original_hams):
        H = get_sparse_operator(load_operator(data_directory=dir+'/ham_fer', file_name=filename, plain_text=True))
        H_0_info = quick_load(dir+'/ham_frag',filename[:-5] + "_frags.pkl")
        for method in ["DF", "DF-boost", "CSA"]:
            H_0 = H_0_info.H0[method]
            c_optimal, minimized_norm, operator = minimize_norm_real_imag(H, H_0)
            print(f"Optimal c with {method}: ", c_optimal)
            print("Minimized norm: ", minimized_norm)
            # print("New operator: ", operator.toarray())
        H_0 = diags(H.diagonal())
        c_optimal, minimized_norm, operator = minimize_norm_real_imag(H, H_0)
        print(f"Optimal c with Jacobi: ", c_optimal)
        print("Minimized norm: ", minimized_norm)

    