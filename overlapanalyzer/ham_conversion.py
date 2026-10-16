import os
from .read_ham import  find_files, extract_numbers_from_filenames, load_mol_info, ensure_directory_exists
from openfermion import load_operator, save_operator, jordan_wigner, is_hermitian #type: ignore
from qcc_240202.utils import f2q_map
# molecule = 'h2o/4e4o'

def fer_to_q(fileDir):
    original_hams = find_files(os.path.join(fileDir, 'ham_fer'),".data")
    for i, filename in enumerate(original_hams):
        
        H_fer = load_operator(data_directory=os.path.join(fileDir, 'ham_fer'), file_name=filename, plain_text=True)
        print(f'Loaded {filename} FermionOperator is Hermitian: {is_hermitian(H_fer)}')
        ensure_directory_exists(os.path.join(fileDir, 'ham_qubit'))
        H_qubit = jordan_wigner(H_fer)
        H_qubit.compress()
        print(f'Converted {filename} to QubitOperator is Hermitian: {is_hermitian(H_qubit)}')
        tol_initial = 1e-8
        while not is_hermitian(H_qubit):
            H_qubit.compress(abs_tol=tol_initial)
            tol_initial += 1e-8
            print(f'QubitOperator is Hermitian after compression with tol {tol_initial}: {is_hermitian(H_qubit)}')
        save_operator(H_qubit,file_name=filename, data_directory=os.path.join(fileDir, 'ham_qubit'), allow_overwrite=True, plain_text=True)
        # Save another one using iQCC utils
        # H_qubit_f2q, new_reference = f2q_map([H_fer], refs, 'jw', group_spins=False, z2_taper=False)
        # save_operator(H_qubit_f2q[0],file_name=filename[:-5]+'_f2q_map', data_directory=dir+'/ham_qubit', allow_overwrite=True, plain_text=True)
        # print(H_qubit_f2q[0] == H_qubit)
        # print(new_reference)

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    fileDir = os.path.join(current_dir, 'hamiltonian_gen_test/h4_linear')
    fer_to_q(fileDir)
