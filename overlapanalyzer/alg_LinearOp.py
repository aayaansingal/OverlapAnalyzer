# Import necessary packages
import numpy as np
import json
import os
import copy
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.sparse.linalg import gmres, inv
import time
from openfermion import QubitOperator, get_sparse_operator
from openfermion.linalg import get_linear_qubit_operator_diagonal

from overlapanalyzer.eigen import sort_eigen
from overlapanalyzer.contour_integration import sum_gauss_points, getContourDataFromEigsh, getContourDataFromEndPoints

def select_values_and_vectors(values, vectors, low, high):
    # Create a mask for values in the range [low, high]
    mask = (low <= values) & (values <= high)

    # Select the values and vectors using the mask
    selected_values = values[mask]
    selected_vectors = vectors[:, mask] # Select columns

    return selected_values, selected_vectors

def select_lowest_value_and_vectors(values, c):
    lowest_value = values[0]
    mask = np.isclose(values, lowest_value)
    selected_values = values[mask]
    selected_vectors = c[:, mask]
    return selected_values, selected_vectors

def total_overlap_with_vectors(v, vectors):
    return np.linalg.norm(v.conj().T @ vectors)**2

def overlaps_with_vector_collection(v, vector_collection):
    return [total_overlap_with_vectors(v, vectors) for vectors in vector_collection]

banner = "="*70
class gmres_counter():
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))

class linear_solver():
    def __init__(self, A_sparse, b, x_guess):
        self.A_sparse = A_sparse
        self.b = b
        self.x_guess = x_guess
        self.matvec_count = 0

    def gmres_solve(self, tol=1e-4, restart=10, maxiter=1, M=None):
        def matvec_counted(x):
            self.matvec_count += 1
            return self.A_sparse @ x
        
        counter = gmres_counter()
        self.matvec_count = 0
        A_operator = LinearOperator(self.A_sparse.shape, matvec=matvec_counted)
        x, info = gmres(A_operator, self.b, x0 = self.x_guess, M=M, atol=tol, restart=restart, maxiter=maxiter, callback=counter, callback_type='pr_norm')
        return x, counter.niter


# def get_diagonal(A):
#     n = A.shape[0]
#     diag = np.zeros(n)
#     for i in range(n):
#         e = np.zeros(n)
#         e[i] = 1
#         diag[i] = A.matvec(e)[i]
#     return diag

def get_diag_part_QubitOperator(H: QubitOperator):
    original_H = copy.deepcopy(H)
    # Diagonal part of a QubitOperator is the collection of terms which only contains PauliZ operators, plus the identity
    diag_operator = QubitOperator()
    for term, coeff in original_H.terms.items():
        if all(map(lambda x: x[1] == 'Z', term)):
            if np.isclose(coeff.imag, 0):
                coeff = coeff.real
            diag_operator += QubitOperator(term, coeff)
    return diag_operator

def rescale_QubitOperator(H: QubitOperator, L: float, U: float):
    """
    Shift and rescale a QubitOperator according to the formula:
    H -> (2 * H - (U+L))/(U-L)
    If L is less than the minimum eigenvalue and U is greater than the maximum eigenvalue, 
    then the result will be a normalized QubitOperator with eigenvalues between -1 and 1.
    """
    temp = 2/(U-L) * H
    temp.constant -= (U+L)/(U-L)
    return temp

def get_id_coeff_QubitOperator(H: QubitOperator):
    """
    Returns the coefficient of the identity part of a QubitOperator
    """
    return H.terms.get((), 0)

def compute_inv_applied_to_state(E_Ritz, H_diag, phi, n_qubits, tol = 1e-2):
    """
    Apply the inverse of H_diag onto phi by solving a linear system: (E_Ritz - H_diag) x = phi
    H_diag is a diagonal QubitOperator
    phi is a numpy array
    Returns (x, niter) where x is a numpy array and niter is the number of iterations
    """
    E_Ritz_QubitOperator = QubitOperator('', E_Ritz)
    # Find identity part ([]) of H_diag, extract the coefficient
    identity_coeff = get_id_coeff_QubitOperator(H_diag)

    precond = get_sparse_operator(E_Ritz_QubitOperator - H_diag, n_qubits=n_qubits)
    # Solve the linear system
    solver = linear_solver(precond, phi, 1/(E_Ritz - identity_coeff) * phi) # Here a preconditioner which only involves the identity part is used
    x, niter = solver.gmres_solve(tol = tol, restart=10)
    return x, niter


def lanczos_lowest_eigenvalue(A, v, low, high, tol=1e-8, max_iter=100):
    if not isinstance(A, LinearOperator):
        raise ValueError("A must be a scipy.sparse.linalg.LinearOperator")
    matvec_count = 0
    vecvec_count = 0
    v1 = v / np.linalg.norm(v)
    V = v1.copy()
    vecvec_count += 1
    w1_prime = A @ v1
    matvec_count += 1
    time_now = time.time()
    alpha_1 = np.real(np.vdot(w1_prime, v1))
    # Raise warning if imaginary part of alpha_1 is not negligible
    if abs(np.imag(alpha_1)) > 1e-14:
        print(f"Imaginary part of alpha_1 is {np.imag(alpha_1)}, casting to real; results may not be accurate.")
    vecvec_count += 1
    w1 = w1_prime - alpha_1 * v1
    vecvec_count += 1

    vk_old = v1
    wk_old = w1

    eigenvalues = []
    eigenvectors = []
    iter_number = []
    matvec_list = []
    vecvec_list = []
    method_list = []
    runtime_list = []

    alphas = [alpha_1]
    betas = []
    time_init = time.time()
    for i in range(max_iter):
        beta = np.linalg.norm(wk_old)
        if beta > 0:
            betas.append(beta)
            vk = wk_old / beta
            V = np.column_stack((V, vk))

        wk_prime = A @ vk
        matvec_count += 1
        alpha = np.real(np.vdot(wk_prime, vk))
        if abs(np.imag(alpha)) > 1e-14:
            print(f"Imaginary part of alpha_{i+1} is {np.imag(alpha)}, casting to real; results may not be accurate.")
        vecvec_count += 1

        alphas.append(alpha)
        wk = wk_prime - alpha * vk - beta * vk_old
        vecvec_count += 2

        eig_val, eig_vec = eigh_tridiagonal(alphas, betas)
        time_now = time.time()
        eig_val, eig_vec = sort_eigen(eig_val, eig_vec)
        print("Lowest eigenvalue from Lanczos: ", eig_val[0])
        eval_selected, evec_selected = select_values_and_vectors(eig_val, eig_vec, low, high)
        if len(eval_selected) == 0:
            eigenvalues.append(eig_val[i])
            eigenvectors.append(V @ eig_vec[:, i:i+1])
        else:
            eigenvalues.append(eval_selected)
            eigenvectors.append(V @ evec_selected)
        iter_number.append(i+1)
        matvec_list.append(matvec_count)
        vecvec_list.append(vecvec_count)
        method_list.append("Lanczos")
        runtime_list.append(time_now - time_init)

        # Use the ground state energy change as a convergence criterion, removing for now for testing
        # if len(eigenvalues) >= 2 and abs(eigenvalues[i] - eigenvalues[i-1]) < tol:
        #     break

        vk_old = vk
        wk_old = wk

    return {"method": method_list, "eigenvalues": eigenvalues, "eigenvectors": eigenvectors, "iterations": iter_number, "matvec_list": matvec_list, "vecvec_list": vecvec_list, "runtime_list": runtime_list}

class CustomLinearSolver:
    """
    A class for manipulating and saving results of custom linear solvers.
    """
    def __init__(self):
        pass
    def save_results_to_json(self, vec_to_comp_overlap_with, exact_overlap, savedir, filename):
        if isinstance(self, CustomDavidson):
            dict_save = {"method": "Davidson"}
            attribute_list = ["tol", "inner_tol", "max_iter", "inner_matvec", "outer_matvec", "eigenvalues"]
            for attr in attribute_list:
                dict_save[attr] = getattr(self, attr)
            dict_save["overlaps"] = overlaps_with_vector_collection(vec_to_comp_overlap_with, self.RitzVectors)
            dict_save["exact_overlap"] = exact_overlap
            dict_save['absolute_overlap_error'] = (np.array(dict_save["overlaps"]) - exact_overlap).tolist()
            dict_save["relative_overlap_error"] = (np.abs(dict_save['absolute_overlap_error'])/exact_overlap).tolist()
            with open(os.path.join(savedir, filename+'_Davidson.json'), 'w') as f:
                json.dump(dict_save, f)
        elif isinstance(self, CustomShiftedParallelLanczos):
            dict_save = {"method": self.method}
            attribute_list = ["tol", "max_iter", "overlaps", "matvec", "vecvec"]
            for attr in attribute_list:
                dict_save[attr] = getattr(self, attr)
            dict_save["exact_overlap"] = exact_overlap
            dict_save['absolute_overlap_error'] = (np.array(self.overlaps) - exact_overlap).tolist()
            dict_save["relative_overlap_error"] = (np.abs(dict_save['absolute_overlap_error'])/exact_overlap).tolist()
            with open(os.path.join(savedir, filename+f'_{self.method}.json'), 'w') as f:
                json.dump(dict_save, f)
        elif isinstance(self, CustomGMRES):
            dict_save = {"method": self.method}
            attribute_list = ["tol", "max_iter", "overlaps", "matvec"]
            for attr in attribute_list:
                dict_save[attr] = getattr(self, attr)
            dict_save["exact_overlap"] = exact_overlap
            dict_save['absolute_overlap_error'] = {i: (self.overlaps[i] - exact_overlap).tolist() for i in self.overlaps} # i's label the preconditioners
            dict_save["relative_overlap_error"] = {i: (np.abs(self.overlaps[i] - exact_overlap)/exact_overlap).tolist() for i in self.overlaps}
            with open(os.path.join(savedir, filename+f'_{self.method}.json'), 'w') as f:
                json.dump(dict_save, f)
        
class CustomDavidson(CustomLinearSolver):
    def __init__(self, A, v, n_qubits, diag=None, use_range=False, low=None, high=None, tol=1e-8, inner_tol = 1e-3, max_iter=20, use_inv_diag=False):
        self.method = "Davidson"
        self.A = get_sparse_operator(A) if isinstance(A, QubitOperator) else A
        self.v = v
        self.n_qubits = n_qubits
        self.diag = diag if diag is not None else get_diag_part_QubitOperator(A) if isinstance(A, QubitOperator) else A.diagonal()
        self.use_range = use_range
        self.low = low
        self.high = high
        self.tol = tol
        self.inner_tol = inner_tol
        self.max_iter = max_iter
        self.use_inv_diag = use_inv_diag
        self.RitzValues = []
        self.RitzVectors = []
        self.outer_matvec = []
        self.inner_matvec = []
        self.V = None

    def run(self):
        matvec_count = 0
        vecvec_count = 0
        
        # Ensure v is treated as a column vector
        v = self.v if self.v.ndim == 2 else self.v.reshape(-1, 1)
        
        v1 = v / np.linalg.norm(v)
        vecvec_count += 1
        
        V = v1.copy()  # V will now be a matrix with v1 as its first column
        w = self.A @ v1
        matvec_count += 1
        
        W = w.copy()  # W will now be a matrix with w as its first column
        
        H = np.zeros((1, 1))
        E_save = np.real(np.vdot(v1[:, 0], w[:, 0]))
        H[0, 0] = E_save
        vecvec_count += 1

        time_init = time.time()
        if self.use_inv_diag:
            self.inv_diag = np.expand_dims(1/self.diag, axis=1)
        for i in range(self.max_iter):
            if i > 0:
                H = np.pad(H, ((0, 1), (0, 1)), 'constant')
                for j in range(i):
                    H[i, j] = np.real(np.vdot(V[:, j], W[:, i]))
                    H[j, i] = np.real(np.vdot(V[:, i], W[:, j]))
                    vecvec_count += 2
                H[i, i] = np.real(np.vdot(V[:, i], W[:, i]))
                vecvec_count += 1
            
            E_Ritz, c = eigh(H)
            time_now = time.time()
            E_Ritz, c = sort_eigen(E_Ritz, c)
            if self.use_range and self.low is not None and self.high is not None:
                eval_selected, evec_selected = select_values_and_vectors(E_Ritz, c, self.low, self.high)
            else:
                eval_selected, evec_selected = select_lowest_value_and_vectors(E_Ritz, c)

            if len(eval_selected) == 0:
                self.RitzValues.append(E_Ritz[i].tolist())
                self.RitzVectors.append(V @ c[:, i:i+1])
            else:
                self.RitzValues.append(eval_selected.tolist())
                self.RitzVectors.append(V @ evec_selected)
            
            self.outer_matvec.append(matvec_count)

            if i > 0 and abs(E_Ritz[0] - E_save) < self.tol:
                break

            E_save = E_Ritz[0]

            r = W @ c[:, 0:1] - (E_Ritz[0] * V) @ c[:, 0:1]
            vecvec_count += (2 * i + 3)
            # If the inverse of the diagonal part of the Hamiltonian is known, then we can apply it directly to the residual vector
            # which cause matvec to increase by 1
            # If not, we need to solve a linear system to apply the inverse of the diagonal part to the residual vector
            # which will cause matvec to increase by n_iter (stored in self.inner_matvec)
            if self.use_inv_diag:
                r = self.inv_diag * r
                matvec_count += 1
            else:
                r, n_iter = compute_inv_applied_to_state(E_Ritz[0], self.diag, r, self.n_qubits, tol=self.inner_tol)
                r=r.reshape(-1,1)
                self.inner_matvec.append(n_iter)
            r = r - V @ (V.T @ r)
            vecvec_count += (2 * i + 3)
            r = r / np.linalg.norm(r)
            vecvec_count += 1
            
            V = np.hstack((V, r))
            w = self.A @ r
            matvec_count += 1
            W = np.hstack((W, w))
        self.V = V


class CustomLanczos(CustomLinearSolver):
    def __init__(self, A, v, tol=1e-8, max_iter=100):
        self.method = "Lanczos"
        self.A = get_sparse_operator(A) if isinstance(A, QubitOperator) else A
        self.v = v
        self.tol = tol
        self.max_iter = max_iter
        self.alphas = []
        self.betas = []
        self.RitzValues = []
        self.RitzVectors = []
        self.overlaps = []
        self.V = None
    def run(self):
        v = self.v if self.v.ndim == 2 else self.v.reshape(-1, 1) # Ensure v is treated as a column vector
        v1 = v / np.linalg.norm(v)
        v_jm1 = v1
        w1_prime = self.A @ v1
        alpha = np.vdot(w1_prime, v1)
        self.alphas.append(alpha)
        w1 = w1_prime - alpha * v1
        # Store the vector v1
        self.V = copy.copy(v1)
        w_jm1 = w1
        v_jm1 = v1
        for j in range(1, self.max_iter+1):
            beta_j = np.linalg.norm(w_jm1)
            if beta_j > self.tol:
                v_j = w_jm1 / beta_j
                w_jprime = self.A @ v_j
                alpha_j = np.vdot(w_jprime, v_j)
                w_j = w_jprime - alpha_j * v_j - beta_j * v_jm1
                self.alphas.append(alpha_j)
                self.betas.append(beta_j)
                self.V = np.hstack((self.V, v_j))

                # Update the vectors
                v_jm1 = v_j
                w_jm1 = w_j
            else:
                print(f"beta_j = {beta_j} < tol = {self.tol}, terminating at iteration {j}.")
                break
    def solve_Ritz(self, return_all_itrs=False):
        if self.V is None:
            self.run()
        # Build the tridiagonal matrix from the alphas and betas
        H = np.diag(self.alphas) + np.diag(self.betas, -1) + np.diag(self.betas, 1)
        # if return_all_itrs, solve the eigenvalue problem for all sub-matrices of the tridiagonal matrix
        if return_all_itrs:
            self.RitzValues.append(self.alphas[0])
            self.RitzVectors.append(self.V[:, :1])
            for i in range(1, H.shape[0]):
                H_sub = H[:i+1, :i+1]
                w, v = np.linalg.eigh(H_sub)
                self.RitzValues.append(w)
                self.RitzVectors.append(self.V[:, :i+1] @ v)
        else:
            w, v = np.linalg.eigh(H)
            self.RitzValues.append(w)
            self.RitzVectors.append(self.V @ v)

class CustomShiftedParallelLanczos(CustomLinearSolver):
    def __init__(self, A, v, eigsh_results, n_qubits, tol=1e-8, max_iter=100, eval_index=0, n_contour_pts=8, contour_endpoints=None):
        self.method = "ShiftedParallelLanczos"
        self.A = get_sparse_operator(A) if isinstance(A, QubitOperator) else A
        self.v = v
        if contour_endpoints is None:
            start_idx, end_idx, self.exact_overlap, overlap_hf, lb, ub, self.contour, self.exact_eval = getContourDataFromEigsh(v, eigsh_results, eval_index, n_contour_pts) # idx=0 for ground state
        else:
            self.contour = getContourDataFromEndPoints(contour_endpoints[0], contour_endpoints[1], n_contour_pts)
            self.exact_overlap = eigsh_results["overlaps"][eval_index]   
        self.n_qubits = n_qubits
        self.tol = tol
        self.max_iter = max_iter
        self.eval_index = eval_index
        self.n_contour_pts = n_contour_pts
        self.overlaps = []
    def run(self):
        matvec_count = 0
        vecvec_count = 0

        v = self.v if self.v.ndim == 2 else self.v.reshape(-1, 1)
        v1 = v / np.linalg.norm(v)
        vecvec_count += 1
        s1 = self.A @ v1
        matvec_count += 1
        alpha_1 = np.vdot(s1, v1)
        vecvec_count += 1

        c = {shift: np.vdot(v, v) for shift in self.contour["Points"]}
        vecvec_count += len(self.contour["Points"])
        pi = {shift: 1 / (shift - alpha_1) for shift in self.contour["Points"]}
        L = {shift: c[shift] / (shift - alpha_1) for shift in self.contour["Points"]}
        self.expectation_values = [[L[shift] for shift in self.contour["Points"]]]
        self.iter_number = [1]
        self.matvec = [matvec_count]
        self.vecvec = [vecvec_count]

        L_old = L.copy()

        vk = v1
        sk = s1
        alpha_k = alpha_1

        for k in range(2, self.max_iter + 2):
            tk = sk - alpha_k * vk
            vecvec_count += 1
            beta_k = np.linalg.norm(tk)
            vecvec_count += 1

            vk_next = tk / beta_k
            sk_next = self.A @ vk_next - beta_k * vk
            matvec_count += 1
            vecvec_count += 1
            alpha_k_next = np.real_if_close(np.vdot(sk_next, vk_next))
            vecvec_count += 1

            converged = False
            for shift in self.contour["Points"]:
                tki = (beta_k ** 2 * pi[shift])
                pi_next = 1 / (shift - alpha_k_next - tki)
                ci_next = c[shift] * tki * pi[shift]
                L_old[shift] = L[shift]
                L[shift] += ci_next * pi_next
                vecvec_count += 1

                if abs(L[shift] - L_old[shift]) < self.tol:
                    converged = True

                pi[shift] = pi_next
                c[shift] = ci_next
            
            self.expectation_values.append([L[shift] for shift in self.contour["Points"]])
            self.iter_number.append(k)
            self.matvec.append(matvec_count)
            self.vecvec.append(vecvec_count)
            if converged:
                break

            # Update vk and sk for the next iteration
            vk = vk_next
            sk = sk_next
            alpha_k = alpha_k_next

        for computed_QF in self.expectation_values:
            self.overlaps.append(sum_gauss_points(self.contour, computed_QF))
        self.absolute_overlap_error = (np.array(self.overlaps) - self.exact_overlap).tolist()
        self.relative_overlap_error = (np.abs(self.absolute_overlap_error)/self.exact_overlap).tolist()

# Write a function to implement the conjugate gradient method to solve a linear system Ax=b, keeping track of the number of matrix-vector and vector-vector multiplications after each iteration as above.
def conjugate_gradient(A, b, x0=None, tol=1e-8, max_iter=100):
    # Generated by CoPilot, not tested yet
    if not isinstance(A, LinearOperator):
        raise ValueError("A must be a scipy.sparse.linalg.LinearOperator")

    matvec_count = 0
    vecvec_count = 0
    if x0 is None:
        x0 = np.zeros(A.shape[0])
    r0 = b - A @ x0
    matvec_count += 1
    p0 = r0
    vecvec_count += 1
    xk = x0
    rk = r0
    pk = p0

    # Initialize lists to store the number of matrix-vector and vector-vector multiplications after each iteration
    matvec_list = [matvec_count]
    vecvec_list = [vecvec_count]
    method_list = ["Conjugate Gradient"]

    for k in range(1, max_iter + 1):
        alpha_k = np.vdot(rk, rk) / np.vdot(pk, A @ pk)
        vecvec_count += 2
        xk_next = xk + alpha_k * pk
        vecvec_count += 1
        rk_next = rk - alpha_k * A @ pk
        matvec_count += 1
        beta_k = np.vdot(rk_next, rk_next) / np.vdot(rk, rk)
        vecvec_count += 2
        pk_next = rk_next + beta_k * pk
        vecvec_count += 1

        # Update values for the next iteration
        xk = xk_next
        rk = rk_next
        pk = pk_next

        # Append the number of matrix-vector and vector-vector multiplications to the lists
        matvec_list.append(matvec_count)
        vecvec_list.append(vecvec_count)
        method_list.append("Conjugate Gradient")
        # Check for convergence
        if np.linalg.norm(rk) < tol:
            break

    return {"method": method_list, "iterations": k, "matvec_list": matvec_list, "vecvec_list": vecvec_list}

def calculateInnerProducts(H_q, phi, zi, initial_state, preconditioner, tol=1e-5, maxiter=100):
    # Imported lazily: krypy 2.2.0 fails to import on recent SciPy (removed scipy.sparse.sputils.isintlike),
    # which would otherwise break every module that depends on alg_LinearOp.
    from krypy.linsys import LinearSystem, Gmres
    print("Solving for z = ", zi)
    linear_system = LinearSystem(get_sparse_operator(QubitOperator('', zi)-H_q), phi, Ml=preconditioner)
    solver = Gmres(linear_system, x0=initial_state, explicit_residual=True, store_all_xk=True, tol=tol, maxiter=maxiter)
    xks = solver.xk_all
    outList = [np.inner(phi, xk.flatten()) for xk in xks]
    print("First 5 integrands: ", outList[:5])
    return outList

class CustomGMRES(CustomLinearSolver):
    """
    GMRES method, currently using GMRES module from krypy.
    Future: Implement a custom GMRES method to keep track of the number of 
    matrix-vector and vector-vector multiplications using preconditioners.
    """
    def __init__(self, H:QubitOperator, phi, eigsh_results, n_contour_pts, tol=1e-5, max_iter=100, preconds_labels = ["None", "Diag"], eval_index=0, contour_endpoints=None):
        self.method = "GMRES"
        self.H = H
        self.phi = phi
        self.eigsh_results = eigsh_results
        # self.preconds = preconds
        self.preconds_labels = preconds_labels
        self.eval_index = eval_index
        self.n_contour_pts = n_contour_pts
        self.integration_center = (contour_endpoints[1]-contour_endpoints[0])/2 if contour_endpoints is not None else eigsh_results["eigenvalues"][eval_index]
        self.tol = tol
        self.max_iter = max_iter
        if contour_endpoints is None:
            start_idx, end_idx, self.exact_overlap, overlap_hf, lb, ub, self.contour, self.exact_eval = getContourDataFromEigsh(phi, eigsh_results, eval_index, n_contour_pts) # idx=0 for ground state
        else:
            self.contour = getContourDataFromEndPoints(contour_endpoints[0], contour_endpoints[1], n_contour_pts)
            self.exact_overlap = self.eigsh_results["overlaps"][eval_index]
        self.temp_results = {}
        self.overlaps = {}
        self.absolute_overlap_error = {}
        self.relative_overlap_error = {}

    def run(self):
        zi_values = self.contour["Points"]
        for i, zi in enumerate(zi_values):
            print(f"Calculating inner products for z = {zi}")
            initial_state = 1/(zi-self.integration_center)*self.phi # Change exact_eval to center of integration path in the future
            for j, preconditioner in enumerate(self.preconds_labels):
                # Use a tuple (i, j) as the key where i labels the zi value and j labels the preconditioner
                if preconditioner == "None":
                    self.temp_results[(i, j)] = calculateInnerProducts(self.H, self.phi, zi, initial_state, preconditioner=None, tol=self.tol, maxiter=self.max_iter)
                elif preconditioner == "Diag":
                    Ml_inv = QubitOperator('', zi) - get_diag_part_QubitOperator(self.H)
                    Ml = inv(get_sparse_operator(Ml_inv))
                    self.temp_results[(i, j)] = calculateInnerProducts(self.H, self.phi, zi, initial_state, preconditioner=Ml, tol=self.tol, maxiter=self.max_iter)
        # Process the results
        for j in range(len(self.preconds_labels)):
            min_length = min([len(self.temp_results[(i, j)]) for i in range(len(zi_values))])
            self.overlaps[j] = [sum_gauss_points(self.contour, [self.temp_results[(i, j)][k] for i in range(len(zi_values))]) for k in range(min_length)]
            self.absolute_overlap_error[j] = (np.array(self.overlaps[j]) - self.exact_overlap).tolist()
            self.relative_overlap_error[j] = (np.abs(self.absolute_overlap_error[j])/self.exact_overlap).tolist()
            self.matvec = [i for i in range(min_length)]

def resolvent_moments(H, v, contour_endpoints, n_contour_pts, max_moment, return_all=True):
    """
    Computes the moments of the resolvent operator, defined by the endpoints.
    """
    contour_dict = getContourDataFromEndPoints(contour_endpoints[0], contour_endpoints[1], n_contour_pts)
    
    n_evals = max_moment//2 + 1
    vecs = {0: v}
    moments = np.array([np.real_if_close(np.vdot(v, v))])
    for i in range(n_evals):
        for z in contour_dict['Points']:
            my_solver = linear_solver(H, v, x_guess=None)
            output_state, n_gmres_iter = my_solver.gmres_solve(tol=1e-6, maxiter=100)
        vecs[i+1] = H @ vecs[i]
    for i in range(1, max_moment+1):
        if i % 2 == 0:
            moments = np.append(moments,np.real_if_close(np.vdot(vecs[i//2], vecs[i//2])))
        else:
            moments = np.append(moments, np.real_if_close(np.vdot(vecs[i//2], vecs[i//2+1])))
    return moments if return_all else moments[-1]


def fix_point(zi, H, b, x0=None, tol=1e-6, max_iter=100):
    """ Solves (zI-H)x=b using the fixed-point iteration method.
    Does not seem to work; divergent behavior observed (20240828: need to scale spectrum of H?)
    """
    # if not isinstance(H, LinearOperator):
    #     raise ValueError("A must be a scipy.sparse.linalg.LinearOperator")
    if x0 is None:
        x0 = [np.zeros(H.shape[0]) for z in zi]
    elif len(x0) != len(zi):
        raise ValueError("x0 must have the same length as zi")
    
    if x0[0].ndim == 1:
        print("x0 must be a list of 2D arrays, attempting conversion")
        x0_2D = [[] for _ in range(len(x0))]
        for i in range(len(x0)):
            x0_2D[i] = x0[i].reshape(-1, 1) # Convert 1D arrays to 2D column vectors
        x0 = x0_2D
    
    xk = x0
    iter_number = [0]
    matvec_list = [0]
    vecvec_list = [0]
    method_list = ["Fixed Point"]
    expectation_values = [[np.vdot(b, x0[i]) for i, _ in enumerate(zi)]]
    matvec_count = 0
    vecvec_count = 0
    xk_next = [np.zeros(H.shape[0]) for z in zi]
    for k in range(max_iter):
        exp_values_per_iter = []
        for i, z in enumerate(zi):
            xk_next[i] = 1/z * (H @ xk[i] + b)
            matvec_count += 1
            vecvec_count += 1
            exp_values_per_iter.append(np.vdot(b, xk_next[i]))
            xk[i] = xk_next[i]
        expectation_values.append(exp_values_per_iter)
        matvec_list.append(matvec_count)
        vecvec_list.append(vecvec_count)
        method_list.append("Fixed Point")
        iter_number.append(k+1)
    return {"method": method_list, "expectation_values": expectation_values, "iterations": iter_number, "matvec_list": matvec_list, "vecvec_list": vecvec_list}

def find_diag_QubitOperator_inverse(A: QubitOperator):
    # Checks that A only contains Z and identity operators
    # if not all(map(lambda x: x[1] == 'Z' or x[1] == 'I', A.terms.keys())):
    #     raise ValueError("A must only contain Z and I operators")
    # Find the inverse of the diagonal part of A
    A_operator = get_sparse_operator(A)
    Ainv_operator = inv(A_operator)
    return Ainv_operator

def Dyson_iteration(phi:np.array, H:QubitOperator, z, max_iter, tol):
    from scipy.sparse.linalg import eigsh
    # Separate H into H_0 (diagonal) and V (off-diagonal)
    H_0_diag = get_diag_part_QubitOperator(H)
    V = H - H_0_diag

    # Compute the inverse of (z*I - H_0) as G_0 by inverting diagonal entries
    G_0_inv_vec = 1. / get_linear_qubit_operator_diagonal(QubitOperator('', z) - H_0_diag)
    G_0_inv = np.diag(G_0_inv_vec)

    # Convert phi to a column vector if it is not already
    if phi.ndim == 1:
        phi = phi[:, np.newaxis]

    # Initial computation of phi_tilde
    phi_tilde = G_0_inv @ phi
    phi_step = phi_tilde
    # Initialize variables for iteration
    norms = []
    # print eigenvalues of G_0 @ V
    print("Largest eigenvalues of G_0 @ V: ", eigsh(G_0_inv @ V, k=10, which='LM')[0])
    for _ in range(max_iter):
        # Compute phi_step
        phi_step = G_0_inv @ (V @ phi_step)
        

        # Update phi_tilde
        phi_tilde += phi_step

        # Calculate norm and check convergence
        current_norm = np.linalg.norm(phi_step)
        norms.append(current_norm)
        if current_norm < tol:
            break

    # Convert phi_tilde back to a 1D array if it was originally 1D
    # if phi_tilde.shape[1] == 1:
        # print("Converting phi_tilde to 1D array...")
        # phi_tilde = phi_tilde.flatten()
    return phi_tilde, norms

class Dyson(CustomLinearSolver):
    """
    Solves (zI-H)x=b using the Dyson iteration method.
    Variables:
    tol: minimum norm of phi_step to continue iteration
    exact_overlap: exact overlap of phi and phi_tilde
    """
    def __init__(self, H:QubitOperator, phi:np.array, contour:dict, max_iter:int, tol:float, exact_overlap:float):
        self.method = "Dyson"
        self.H = H
        self.phi = phi
        self.contour = contour
        self.max_iter = max_iter
        self.tol = tol
        self.overlaps = {}
        self.norms = {z: [] for z in self.contour["Points"]}
        self.temp_results = {z: [] for z in self.contour["Points"]}
        self.exact_overlap = exact_overlap
    def run(self):
        H_0_diag = get_diag_part_QubitOperator(self.H)
        V = get_sparse_operator(self.H - H_0_diag)
        for i, z in enumerate(self.contour["Points"]):
            # G_0_vec = 1. / get_linear_qubit_operator_diagonal(QubitOperator('', z) - H_0_diag) # 20240910: cannot cast float64 to np.complex128 
            # G_0 = np.diag(G_0_vec)
            G_0 = inv(get_sparse_operator(QubitOperator('', z) - H_0_diag))
            phi_tilde = G_0 @ self.phi
            phi_step = phi_tilde
            norms = []
            print("Largest eigenvalues of G_0 @ V: ", eigsh(G_0 @ V, k=10, which='LM')[0])
            for _ in range(self.max_iter):  
                phi_step = G_0 @ (V @ phi_step)
                phi_tilde += phi_step
                current_norm = np.linalg.norm(phi_step)
                norms.append(current_norm)
                if current_norm < self.tol:
                    break
                self.temp_results[z].append(np.vdot(self.phi, phi_tilde))
                self.norms[z].append(norms) 
        # Process the results
        min_length = min(len(self.temp_results[z]) for z in self.temp_results.keys())
        self.overlaps = [sum_gauss_points(self.contour, [self.temp_results[z][k] for z in self.temp_results.keys()]) for k in range(min_length)]
        self.absolute_overlap_error = (np.array(self.overlaps) - self.exact_overlap).tolist()
        self.relative_overlap_error = (np.abs(self.absolute_overlap_error)/self.exact_overlap).tolist()
        self.matvec = [2*i for i in range(min_length)]

if __name__ == '__main__':
    from overlapanalyzer.polynomial_majorisation_LB import K_matrix
    A = K_matrix(-3, [-2,-1,0], [0.1,0.2,0.3])
    my_lanczos = CustomLanczos(A, np.array([[1],[0],[0],[0]]))
    my_lanczos.solve_Ritz(return_all_itrs=False)
    print(my_lanczos.RitzValues)