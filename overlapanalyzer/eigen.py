import math
import json
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, identity
from scipy.sparse.linalg import eigsh
from openfermion import (
    load_operator,
    jw_configuration_state,
    get_sparse_operator,
    get_number_preserving_sparse_operator,
    jw_number_restrict_operator,
    jw_number_restrict_state,
    jw_sz_restrict_operator
)
from overlapanalyzer.read_ham import *
from overlapanalyzer.utils import exp_val_higher_moment, save_dict, index_to_occupied
import csv 

def save_filtered_results_to_csv(exact_energies_no_degen, overlaps, filename, threshold):
    """
    Save filtered results to a CSV file.

    Parameters:
    - exact_energies_no_degen (np.ndarray): Array of exact energies without degeneracy.
    - overlaps (np.ndarray): Array of overlap values.
    - filename (str): The name of the output CSV file.
    - threshold (float): Minimum overlap value to include in the output. Default is 1e-10.

    The first column of the output file contains `exact_energies_no_degen`, and the
    second column contains `overlaps`.
    """
    # Filter based on the threshold
    mask = overlaps > threshold
    filtered_energies = exact_energies_no_degen[mask]
    filtered_overlaps = overlaps[mask]

    # Save to CSV
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Exact Energy", "Overlap"])
        # Write the filtered data
        writer.writerows(zip(filtered_energies, filtered_overlaps))

def truncate_by_ovlp_threshold(vector, overlaps, threshold):
    truncated_vector = [v for v, o in zip(vector, overlaps) if o >= threshold]
    truncated_overlaps = [o for o in overlaps if o >= threshold]
    return truncated_vector, truncated_overlaps

def calc_eigen(fileDir, mol_name, geometry, spin, num_eigs=10, use_eigsh=False, hf_energy=None, **kwargs):
    # for spin in  ('s0', 't1'):
    if 'create_CAS' in kwargs.keys() and kwargs['create_CAS']:
        n_elec, n_spin_orb = load_mol_info(os.path.join(fileDir,'CAS_info.csv'))
        mol_name = mol_name+'CAS'
    else:
        n_elec, n_spin_orb = load_mol_info(os.path.join(fileDir,'info.csv'))

    # original_hams = find_files(os.path.join(fileDir,'ham_fer'), f"{spin}.data") # list of filenames ending with ".data"


    # for i, filename in enumerate(original_hams):
    filepath = os.path.join(fileDir,'ham_fer')
    filename = f'{mol_name}_{geometry}_{n_elec}_{n_spin_orb}_{spin}.data'
    print("Calculating eigenstates for: ", filename)
    H_loaded = load_operator(data_directory=filepath, file_name=filename, plain_text=True)
    print("Hamiltonian loaded.")
    # r = geometry[i]
    # H_sparse = get_number_preserving_sparse_operator(H_loaded, n_spin_orb, n_elec)
    H_sparse = get_sparse_operator(H_loaded)
    # H_sparse = jw_sz_restrict_operator(H_sparse, sz, n_electrons=n_elec, n_qubits=n_spin_orb)
    H_sparse = jw_number_restrict_operator(H_sparse, n_elec, n_qubits=n_spin_orb)
    print("Hamiltonian converted to sparse.")
    if use_eigsh:
        print(f"Calculating the first {num_eigs} eigenvalues using eigsh...")
        emax, _ = eigsh(H_sparse, k=1, which='LA')
        w, v = eigsh(H_sparse, k=num_eigs, which='SA')
        w_sorted, v_sorted = sort_eigen(w, v)
        e0 = w_sorted[0]
    else:
        print("Calculating all eigenvalues using eigh, this may take a while...")
        H_dense = H_sparse.todense()
        w, v = np.linalg.eigh(H_dense)
        v = np.array(v)
        w_sorted, v_sorted = sort_eigen(w, v)
        e0 = w_sorted[0]
        emax = w_sorted[-1]
    
    symmetries = get_exp_val_symmetries(v_sorted, n_spin_orb, n_elec=n_elec)
    # evals_filtered = filter_evals_by_symmetry(w_sorted, symmetries, nelec=n_elec)

    degens = count_degeneracies(w_sorted)
    w_no_degen = non_degenerate_values(w_sorted, degens)
    # Orthonormalize the eigenvectors within each degenerate subspace
    v_ON = orthonormalize(v_sorted, degens)
    print("Eigenstates computed.")
    if 'target_state' in kwargs.keys():
        state = kwargs.get('target_state')
    else: 
        if spin == 's0':
            # occ_list = hf_occupation_list(n_elec, 0)
            hf_state_full = jw_configuration_state([i for i in range(n_elec)], n_spin_orb)
        elif spin == 't1':
            hf_state_full = jw_configuration_state([i for i in range(n_elec-1)] + [n_elec], n_spin_orb)
        elif spin == 's1':
            hf_state_full = 1/np.sqrt(2)*(jw_configuration_state([i for i in range(n_elec-1)] + [n_elec+1], n_spin_orb) + jw_configuration_state([i for i in range(n_elec-2)] + [n_elec-1, n_elec], n_spin_orb))
        hf_state_full = np.expand_dims(hf_state_full, axis=1)
        state = jw_number_restrict_state(hf_state_full, n_elec, n_qubits=n_spin_orb)

    moments = exp_val_higher_moment(H_sparse, state, 40, return_all=True)
    moments_shifted = exp_val_higher_moment(H_sparse - moments[1]*identity(H_sparse.shape[0]), state, 40, return_all=True)
    (rescale_lower, rescale_upper) = kwargs.get('rescale_values') if kwargs.get('rescale_values') is not None else (emax, e0)
    moments_rescaled = exp_val_higher_moment((2*H_sparse - (rescale_upper+rescale_lower)*identity(H_sparse.shape[0]))/(rescale_upper-rescale_lower), state, 40, return_all=True)
    overlaps = overlap_with_vectors(state, v_ON, degens)
    threshold=kwargs.get('threshold', 0.0)
    leading_indices_and_gaps = compute_gaps(w_no_degen, overlaps, threshold=threshold)
    print("Exact HF overlap computed.")
    truncated_evals, truncated_overlaps = truncate_by_ovlp_threshold(w_no_degen, overlaps, threshold)
    multiplicity_list_large_ovlps = [degens[i] for i in range(len(degens)) if overlaps[i] > 1e-9]
    print("Multiplicities of significant overlaps: ", multiplicity_list_large_ovlps)
    print("Symmetry expectation values of of initial state: ", get_exp_val_symmetries(state, n_spin_orb, n_elec=n_elec))
    # print("Largest overlap and position: ", (overlaps_with_lowest[np.argmax(overlaps_with_lowest)], np.argmax(overlaps_with_lowest)))
    ensure_directory_exists(os.path.join(fileDir, 'eigen'))
    # save_operator(myiQCC.ham,file_name=filename, data_directory=dir+'/ham_dressed', allow_overwrite=True, plain_text=True)
    # quick_save({"Energy": myiQCC.energies_state_specific, "num_iterations": num_iterations, "num_generators": num_generators}, name=filename[:-5] + ".pkl", path=dir+'/ham_dressed')
    saving_dict = prepare_dict_to_save(hf_energy = hf_energy,
                                       exact_energies = w_sorted.tolist(),
                                    exact_energies_no_degen = w_no_degen.tolist(),
                                    exact_energies_truncated = truncated_evals,
                                    exact_energy_max = emax[0] if use_eigsh else emax,
                                    degen = degens.tolist(),
                                    eigen_states = v_ON,
                                    overlaps = overlaps,
                                    overlaps_truncated = truncated_overlaps,
                                    leading_indices_and_gaps = leading_indices_and_gaps,
                                    overlap_threshold = threshold,
                                    moments = moments.tolist(),
                                    moments_shifted = moments_shifted.tolist(),
                                    moments_rescaled = moments_rescaled.tolist(),
                                    rescale_values = (rescale_lower, rescale_upper),
                                    symmetries = symmetries,
                                    molecule=filename[:-5])
    tail_txt = '_CustomScale' if kwargs.get('rescale_values') is not None else ''
    save_dict(saving_dict, os.path.join(fileDir, 'eigen'), filename[:-5] + "_eigen" + tail_txt, save_pkl=kwargs.get('save_pkl', False))
    save_filtered_results_to_csv(w_no_degen, np.array(overlaps), os.path.join(fileDir, 'eigen', filename[:-5] + "_filtered.csv"), threshold)
    # with open(os.path.join(fileDir, 'eigen', filename[:-5] + "_eigen.pkl"), 'wb') as f:
    #     pickle.dump(saving_dict, f)
    # saving_dict.pop('eigen_states')
    # with open(os.path.join(fileDir, 'eigen', filename[:-5] + "_eigen.json"), 'w') as f:
    #     json.dump(saving_dict, f)
    
def get_exp_val_symmetries(v, n_spin_orb, n_elec=None, calc_sz=True, calc_n=True, calc_s2=True):
    """
    Return expectation values of Sz, N, and S^2 operators for the given eigenvectors v (v assumed to be obtained from eigsh).
    """
    from openfermion import sz_operator, number_operator, s_squared_operator, get_sparse_operator, jw_number_restrict_operator
    sparse_sz = get_sparse_operator(sz_operator(int(n_spin_orb//2)))
    sparse_n = get_sparse_operator(number_operator(int(n_spin_orb)))
    sparse_s2 = get_sparse_operator(s_squared_operator(int(n_spin_orb//2)))
    if n_elec is not None:
        sparse_sz = jw_number_restrict_operator(sparse_sz, n_elec, n_qubits=n_spin_orb)
        sparse_n = jw_number_restrict_operator(sparse_n, n_elec, n_qubits=n_spin_orb)
        sparse_s2 = jw_number_restrict_operator(sparse_s2, n_elec, n_qubits=n_spin_orb)
    exp_sz = [np.real(v[:,i].T @ sparse_sz @ v[:,i]) for i in range(len(v.T))] if calc_sz else None
    exp_n = [np.real(v[:,i].T @ sparse_n @ v[:,i]) for i in range(len(v.T))] if calc_n else None
    exp_s2 = [np.real(v[:,i].T @ sparse_s2 @ v[:,i]) for i in range(len(v.T))] if calc_s2 else None
    return (exp_sz, exp_n, exp_s2)

def filter_evals_by_symmetry(eigenvals, symmetry, sz=None, nelec=None, s2=None, tol=1e-7):
    filter1 = np.array(symmetry[0])  # Filtering array 1
    filter2 = np.array(symmetry[1])  # Filtering array 2
    filter3 = np.array(symmetry[2])  # Filtering array 3
    condition1 = np.abs(filter1 - sz) < tol if sz is not None else True
    condition2 = np.abs(filter2 - nelec) < tol if nelec is not None else True
    condition3 = np.abs(filter3 - s2) < tol if s2 is not None else True
    filtered_evals = eigenvals[condition1 & condition2 & condition3]
    return filtered_evals
    
def hf_occupation_list(n_electrons, n_unpaired):
    """
    Generate a list of occupation numbers for the Hartree-Fock state.

    :param n_electrons: the total number of electrons
    :param n_unpaired: the number of unpaired electrons, support aliasing: n_unpaired = 's0' or 't1'
    :return: the occupation list

    20240830: may remove if not useful
    """
    if n_unpaired == 's0':
        n_unpaired = 0
    elif n_unpaired == 't1':
        n_unpaired = 2
    # Ensure that both n_electrons and n_unpaired are even numbers
    if n_electrons % 2 != 0 or n_unpaired % 2 != 0:
        raise ValueError("Both n_electrons and n_unpaired must be even numbers.")
    n_paired = n_electrons - n_unpaired
    return list(range(0, n_paired)) + list(range(n_paired, n_paired + 2*(n_unpaired - 1)+1, 2))


def sort_eigen(eigenvalues, eigenvectors):
    # Get the indices that would sort the energy array
    sorted_indices = np.argsort(eigenvalues)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    return sorted_eigenvalues, sorted_eigenvectors

def calculate_lowest_eigen(matrix, n, is_hermitian=False, **kwargs):
    if is_hermitian:
        # If matrix is a csc or csr matrix, use eigsh
        if isinstance(matrix, (csc_matrix, csr_matrix)):
            print("Matrix is sparse Hermitian, using eigsh...")
            eigenvalues, eigenvectors = eigsh(matrix, k=n, v0=kwargs.get("v0"), which='SA')
        else:
            print("Matrix is Hermitian, using eigh...")
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    else:
        # Otherwise, use eig
        print("Matrix is not Hermitian, using eig...")
        eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Sort eigenvalues and corresponding eigenvectors
    sorted_eigenvalues, sorted_eigenvectors = sort_eigen(eigenvalues, eigenvectors)

    # Take the n lowest eigenvalues and their corresponding eigenvectors
    lowest_eigenvalues = sorted_eigenvalues[:n]
    lowest_eigenvectors = sorted_eigenvectors[:, :n]

    return lowest_eigenvalues, lowest_eigenvectors

def calculate_highest_eigen(matrix, n, is_hermitian=False, **kwargs):
    if is_hermitian:
        # If matrix is a csc or csr matrix, use eigsh
        if isinstance(matrix, (csc_matrix, csr_matrix)):
            print("Matrix is sparse Hermitian, using eigsh...")
            eigenvalues, eigenvectors = eigsh(matrix, k=n, v0=kwargs.get("v0"), which='LA')
        else:
            print("Matrix is Hermitian, using eigh...")
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    else:
        # Otherwise, use eig
        print("Matrix is not Hermitian, using eig...")
        eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Sort eigenvalues and corresponding eigenvectors
    sorted_eigenvalues, sorted_eigenvectors = sort_eigen(eigenvalues, eigenvectors)

    # Take the n highest eigenvalues and their corresponding eigenvectors
    highest_eigenvalues = sorted_eigenvalues[-n:]
    highest_eigenvectors = sorted_eigenvectors[:, -n:]

    return highest_eigenvalues, highest_eigenvectors

def calculate_lowest_and_highest_eigen(matrix, is_hermitian=False, **kwargs):
    """
    Calculate the smallest, mid, and largest eigenvalues.
    2024-08-30: potential redundancy, check eigsh.calc_eigen!!!
    """
    if is_hermitian:
        # If matrix is a csc or csr matrix, use eigsh
        if isinstance(matrix, (csc_matrix, csr_matrix)):
            print("Matrix is sparse Hermitian, using eigsh...")
            eigenvalues_lowest, _ = eigsh(matrix, k=2, v0=kwargs.get("v0"), which='SA')
            eigenvalues_highest, _ = eigsh(matrix, k=2, v0=kwargs.get("v0"), which='LA')
            eigenvalues = np.array(eigenvalues_lowest.tolist() + eigenvalues_highest.tolist())
        else:
            print("Matrix is Hermitian, using eigh...")
            eigenvalues = np.linalg.eigh(matrix)[0]
    else:
        # Otherwise, use eig
        print("Matrix is not Hermitian, using eig...")
        eigenvalues = np.linalg.eig(matrix)[0]

    sorted_eigenvalues = np.sort(eigenvalues)
    (E0, E1) = sorted_eigenvalues[0:2]
    highest_eigenvalue = sorted_eigenvalues[-1]

    return E0, E1, highest_eigenvalue

def count_degeneracies(eigenvals, tolerance=1e-8):
    """
    Count the degeneracies of the eigenvalues.
    Only works for eigenvals sorted.
    """
    # Raise error if the eigenvalues are not sorted from lowest to highest
    if not np.all(np.diff(eigenvals) >= 0):
        raise ValueError("Eigenvalues must be sorted from lowest to highest.")
    # Initialize an empty list to store the counts
    degeneracies = []

    # Iterate over the array
    i = 0
    while i < len(eigenvals):
        count = 1
        # Count the number of similar elements (within the tolerance)
        while i + 1 < len(eigenvals) and np.abs(eigenvals[i] - eigenvals[i + 1]) < tolerance:
            i += 1
            count += 1
        # Append the count to the degeneracies list
        degeneracies.append(count)
        i += 1

    return np.array(degeneracies)

def evals_no_degen(eigenvals, degen):
    """
    Return a list of eigenvalues without degeneracies.
    """
    start = 0
    evals = []
    for i, d in enumerate(degen):
        evals.append(eigenvals[start])
        start += d
    return evals

def overlaps_no_degen(overlaps, degen):
    """
    Return a list of overlaps without degeneracies.
    For a degenerate subspace, the overlap is equal to the sum of overlaps within that subspace.
    For now, no need to use since the calculated overlaps are already with a subspace.
    """
    start = 0
    overlaps_no_degen = []
    for i, d in enumerate(degen):
        overlaps_no_degen.append(np.sum(overlaps[start:start + d]))
        start += d
    return overlaps_no_degen
def non_degenerate_values(L, D):
    """
    Given an array L and a degeneracy list D, return a non-degenerate version of L.
    """
    # Ensure L and D are numpy arrays for easy indexing
    L = np.array(L)
    D = np.array(D)
    
    # Use the cumulative sum of D to get the index for each block of degenerate values
    indices = np.cumsum(D) - D
    # Return the unique (non-degenerate) values from L
    return L[indices]

def remove_dupl_by_multiplicity(L, D, multiplicity):
    """
    Given an array L and a degeneracy list D, return a list of values with multiplicity k.
    Example: if L = [a, b, b, b, c, c, d] and D = [1,3,2,1], then values_w_multiplicity_k(L, D, 1) = [a, d],
    values_w_multiplicity_k(L, D, 2) = [c], and values_w_multiplicity_k(L, D, 3) = [b].
    """
    result = []
    index = 0
    for d in D:
        if d == multiplicity:
            result.append(L[index])
        index += d  # Move index by the degeneracy count
    return result

def select_right_multiplicity(L, D, multiplicity):
    """
    Given an array L and a degeneracy list D of the same length, return a subset of L which contains only the values with multiplicity k.
    Example: if L = [a, b, c, d] and D = [1,3,2,1], then select_right_multiplicity(L, D, 1) = [a, d],
    select_right_multiplicity(L, D, 2) = [c], and select_right_multiplicity(L, D, 3) = [b].
    """
    result = []
    for i, elem in enumerate(L):
        if D[i] == multiplicity:
            result.append(elem)
    return result

def non_degenerate_vectors(vectors, degen):
    """
    Given a 2D array of column vectors and a degeneracy list, return a list of non-degenerate vectors.
    """
    degen = np.array(degen)
    indices = np.cumsum(degen) - degen
    return vectors[:,indices]

def compute_gaps(exact_energies_no_degen, overlaps, n=15, threshold=0.0):
    """
    Computes the gaps between eigenvalues for the indices of the largest n overlap values.

    Parameters:
    exact_energies_no_degen (numpy.ndarray): Array of exact energies, assumed to be sorted.
    overlaps (numpy.ndarray): Array of overlaps corresponding to the eigenvalues.
    n (int): Number of largest overlaps to consider.
    threshold (float): Minimum overlap value to consider when calculating gaps.

    Returns:
    list of tuples: Each tuple contains (index, overlap_value, (left_gap, right_gap)).
    """
    # Ensure inputs are numpy arrays
    exact_energies_no_degen = np.array(exact_energies_no_degen)
    overlaps = np.array(overlaps)
    
    # Get indices of the largest n overlap values
    largest_indices = np.argsort(overlaps)[-n:][::-1]

    # Calculate gaps
    results = []
    for idx in largest_indices:
        index = int(idx)  # Convert numpy int64 to Python int
        overlap_value = overlaps[index]

        # Filter indices based on the threshold
        valid_indices = np.where(overlaps > threshold)[0]

        # Calculate left_gap
        left_gap = None
        if index > 0:
            left_valid_indices = valid_indices[valid_indices < index]
            if len(left_valid_indices) > 0:
                left_index = left_valid_indices[-1]
                left_gap = abs(exact_energies_no_degen[index] - exact_energies_no_degen[left_index])

        # Calculate right_gap
        right_gap = None
        if index < len(overlaps) - 1:
            right_valid_indices = valid_indices[valid_indices > index]
            if len(right_valid_indices) > 0:
                right_index = right_valid_indices[0]
                right_gap = abs(exact_energies_no_degen[index] - exact_energies_no_degen[right_index])

        # Append results
        results.append((index, overlap_value, (left_gap, right_gap)))
    
    return results
# def find_subspace_index(degen, index):
#     """
#     Find the subspace index that the given index belongs to, based on the list of degeneracies.

#     :param degen: numpy array of degeneracies
#     :param index: the index to find the subspace for
#     :return: the index labelling the subspace that the given index belongs to
#     """
#     sum_degen = 0
#     for subspace_index, d in enumerate(degen):
#         sum_degen += d
#         if index <= sum_degen-1:
#             return subspace_index
#     return None  # If index is out of the range of the degeneracies

# def vectors_in_subspace(v, degen, original_index):
#     """
#     Return all vectors in v that belong to the same subspace as the vector labelled by original_index.

#     :param v: numpy array of vectors (assumed from eigh/eigsh)
#     :param degen: numpy array of degeneracies
#     :param index: the index to find the subspace for
#     :return: numpy array of vectors in the identified subspace
#     """
#     subspace_idx = find_subspace_index(degen, original_index)
#     if subspace_idx is None:
#         return np.array([])  # Return empty array if index is out of range

#     start = sum(degen[:subspace_idx])  # Start index of the subspace in v
#     end = start + degen[subspace_idx]  # End index (exclusive) of the subspace in v
#     return v[start:end]

def find_subspace_indices(degen, idx):
    start = 0
    for i in range(idx):
        start += degen[i]
    return (start, start+degen[idx])

def vecs_in_subspace(vectors, degen, idx):
    """
    Returns a part of the vectors matrix that corresponds to the subspace with the given index.
    Usage: get_subspace(vectors, degen, 0) returns the vectors corresponding to the first subspace defined by degen, etc.
    """
    start, end = find_subspace_indices(degen, idx)
    return vectors[:, start:end]

def overlap_with_ON_vecs(v, vectors):
    """
    Compute total overlap of vector v with a set of orthonormal vectors.
    :param v: the vector to compute overlap with
    :param vectors: the vectors to compute overlap with, given as a numpy array; each column represents a vector
    """
    overlaps = 0
    for i in range(vectors.shape[1]):
        overlaps += np.abs(np.vdot(v, vectors[:, i])) ** 2
    return overlaps

def overlap_with_vectors(v, vectors, degen):
    """
    Compute a list of overlaps of vector v with a set of vectors, where degen specifies how the vectors are grouped.
    """
    overlaps = []
    start = 0
    for d in degen:
        overlaps.append(overlap_with_ON_vecs(v, vectors[:, start:start+d]))
        start += d
    return overlaps


# def overlap_with_ON_vecs(v, vectors):
#     """
#     Compute total overlap of vector(s) v with a set of orthonormal vectors.
#     :param v: a single vector or a 2D array where each column represents an independent vector
#     :param vectors: the vectors to compute overlap with, given as a numpy array; each column represents a vector
#     :return: a single overlap value (if v is 1D) or a 1D array of overlaps (if v is 2D)
#     """
#     if v.ndim == 1:
#         v = v[:, np.newaxis]
#     overlaps = np.sum(np.abs(np.dot(vectors.T.conj(), v)) ** 2, axis=0)
#     return overlaps

# def overlap_with_vectors(v, vectors, degen):
#     """
#     Compute overlaps of vector(s) v with a set of vectors grouped by degeneracy.
#     :param v: a single vector or a 2D array where each column represents an independent vector
#     :param vectors: the vectors to compute overlap with, given as a numpy array; each column represents a vector
#     :param degen: an array indicating how the vectors are grouped (degeneracies)
#     :return: a 2D array where each row corresponds to the overlaps of a column in v with the vector groups
#     """
#     if v.ndim == 1:
#         v = v[:, np.newaxis]
    
#     overlaps = np.zeros((v.shape[1], len(degen)))
#     start = 0
#     for i, d in enumerate(degen):
#         overlaps[:, i] = overlap_with_ON_vecs(v, vectors[:, start:start+d])
#         start += d
#     return overlaps

def group_eigenvalues_with_overlaps(overlaps, exact_evals_non_degen, eigen_gap):
    """
    Group eigenvalues into non-overlapping groups based on eigen_gap and calculate total overlaps for each group.
    
    :param overlaps: List or array of overlaps corresponding to exact_evals_non_degen.
    :param exact_evals_non_degen: Sorted array of eigenvalues with degeneracy removed.
    :param eigen_gap: Minimum gap required to separate groups of eigenvalues.
    :return: Tuple of grouped overlaps and bounds: 
             ((total_overlap_group_1, (lower_bound_group_1, upper_bound_group_1)),
              (total_overlap_group_2, (lower_bound_group_2, upper_bound_group_2)), ...)
    """
    grouped_results = []
    current_group_overlap = 0
    current_group_start = exact_evals_non_degen[0]
    
    for i in range(len(exact_evals_non_degen)):
        if i > 0 and (exact_evals_non_degen[i] - exact_evals_non_degen[i - 1] >= eigen_gap):
            # Close current group and start a new one
            current_group_end = exact_evals_non_degen[i - 1]
            grouped_results.append((current_group_overlap, (current_group_start, current_group_end)))
            current_group_overlap = 0
            current_group_start = exact_evals_non_degen[i]
        
        # Add overlap of current eigenvalue to the current group
        current_group_overlap += overlaps[i]
    
    # Add the final group
    current_group_end = exact_evals_non_degen[-1]
    grouped_results.append((current_group_overlap, (current_group_start, current_group_end)))
    
    return tuple(grouped_results)

def orthonormalize(v, degen):
    """
    Orthonormalize the eigenvectors within each degenerate subspace.
    """
    start = 0
    v_ON = np.array([])
    for d in degen:
        # Orthonormalize the subspace using QR decomposition
        Q, _ = np.linalg.qr(v[:, start:start+d])
        v_ON = np.concatenate((v_ON, Q), axis=1) if v_ON.size else Q
        start += d
    return v_ON

def projection(v, basis):
    """Compute the projection of v onto the subspace spanned by the basis vectors."""
    proj = np.zeros_like(v, dtype=np.complex_)
    for w in basis:
        proj += np.vdot(w, v) / np.vdot(w, w) * w
    return proj

def residue_vector(v, basis):
    """Compute the residue vector of v with respect to the subspace spanned by basis vectors."""
    proj = projection(v, basis)
    return v - proj

