import os
import re
import json
from pyscf import gto, scf, mcscf, ao2mo
from qcc_240202.utilspyscf import construct_fermionic_operator, construct_fermionic_operator_from_mcscf
from qcc_240202.utils import f2q_map
from openfermion import (
    save_operator
)
from overlapanalyzer.read_ham import ensure_directory_exists, read_xyz
from overlapanalyzer.utils import get_molecular_data
import csv 

class CustomMole(gto.Mole):
    # Use this class for constructing molecular Hamiltonians etc.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    

def gen_ham_from_MD(mol_name, R10, basis='sto-3g'):
    """
    Generates Hamiltonian for a molecule from pre-stored molecular data.
    Args:
        mol_name (str): Name of the molecule.
        R10 (int): Distance between atoms in angstrom **TIMES 10**.
        basis (str): Basis set to use.
    """
    print(f'Generating Hamiltonian for: {mol_name}')
    # R10 = 10
    # for R in range(5, 35):
    geo = get_molecular_data(mol_name,R10/10,xyz_format=False)
    # geo = [ ['H', [0, 0, i*R/10]] for i in range(num_hydrogen) ]

    mol = gto.Mole()
    mol.atom = geo
    mol.verbose = 0
    mol.basis = basis
    mol.build()
    #-------------------------------------------------------------------------------
    #Obtain the FermionOperator for the PySCF molecule in a `n_elec`, `n_orbs` active space.
    #The fermionic Hamiltonian will be resolved in the HF orbitals.
    h_ferm = construct_fermionic_operator(mol)
    # Ensure directory exists
    directory_name = 'hamiltonian_gen/'+mol_name
    ensure_directory_exists(directory_name+'/ham_fer')
    save_operator(h_ferm,file_name=mol_name+f'_{R10/10}', data_directory=directory_name+'/ham_fer', allow_overwrite=True, plain_text=True)

def gen_ham_h_chain(num_hydrogen, R10, basis='sto-3g'):
    """
    Generates Hamiltonian for a hydrogen chain.
    Args:
        num_hydrogen (int): Number of hydrogen atoms.
        R10 (int): Distance between atoms in angstrom **TIMES 10**.
        basis (str): Basis set to use.
    """
    mol_name = f'h{num_hydrogen}_linear'
    R10 = 10
    geo = [ ['H', [0, 0, i*R10/10]] for i in range(num_hydrogen) ]

    mol = gto.Mole()
    mol.atom = geo
    mol.verbose = 0
    mol.basis = basis
    mol.build()
    #-------------------------------------------------------------------------------
    #Obtain the FermionOperator for the PySCF molecule in a `n_elec`, `n_orbs` active space.
    #The fermionic Hamiltonian will be resolved in the HF orbitals.
    h_ferm = construct_fermionic_operator(mol)
    # Ensure directory exists
    directory_name = 'hamiltonian_gen/'+mol_name
    ensure_directory_exists(directory_name+'/ham_fer')
    save_operator(h_ferm,file_name=mol_name+f'_{R10/10}', data_directory=directory_name+'/ham_fer', allow_overwrite=True, plain_text=True)

def gen_hamiltonian(fileDir, mol_name, mode, small_atom_basis='sto-3g', **kwargs):
    if mode == "from_MD":
        print("Loading molecular structure from molecular data.")
        geo = get_molecular_data(mol_name, kwargs.get('R10')/10, xyz_format=False)
        mol = gto.Mole()
        mol.atom = geo
        mol.basis = small_atom_basis # Same basis for all atoms
    elif mode == "h_chain":
        num_hydrogen = int(re.search(r'\d+', mol_name).group())
        geo = [ ['H', [0, 0, i*kwargs.get('R10')/10]] for i in range(num_hydrogen) ]
        mol = gto.Mole()
        mol.atom = geo
        mol.verbose = 0
        mol.basis = small_atom_basis

    elif mode == "OLED":
        # if mol_name.endswith(".xyz"):
        xyz = read_xyz(os.path.join(fileDir, 'OLEDs', 'xyzs', mol_name+'.xyz'))
        mol = gto.Mole()
        mol.max_memory = 32000 # 32GB, set to lower if run on laptop
        mol.atom = xyz
        mol.basis = {'N': small_atom_basis, 'H': small_atom_basis, 'C': small_atom_basis, 'F': small_atom_basis, 'O': small_atom_basis, 'Ir': 'sbkjc'}
        mol.pseudo = {'Ir': 'sbkjc'}
        mol.unit = 'Angstrom' #in Angstrom, use 'B' for Bohrs
        mol.symmetry = True
    else:
        raise ValueError("Invalid mode: must be 'from_MD', 'h_chain', or 'OLED'")
    
    if kwargs.get('spin') == 't1':
        mol.spin = 2
    mol.build()
    # num_orbs = mol.nao
    # Check if there is an existing .json file with the same filename under gen_hams, if yes, load it and run construct_fermionic_operator using it as the mo coefficients
    # if os.path.exists(os.path.join(current_dir, 'gen_hams', file_name[:-4]+'.json')):
    #     with open(os.path.join(current_dir, 'gen_hams', file_name[:-4]+'.json'), 'r') as f:
    #         mo = json.load(f)
    #     h_ferm = construct_fermionic_operator(mol, n_elec=8, n_orbs=8, mo=mo) # Limit number of qubits to 16 for now, can set to higher in future
    # else:


    # Save mf.mo_coeff as json
    # with open(os.path.join(current_dir, 'gen_hams', file_name[:-4]+'.json'), 'w') as f:
    #     json.dump(mf.mo_coeff.tolist(), f)
    
    # h_ferm = construct_fermionic_operator(mol, n_elec=n_elec, n_orbs=n_orbs, mo=mf.mo_coeff)
    if kwargs.get('create_CAS')==True:
        CAS_str = 'CAS'
        if kwargs.get('spin') in ('s0', 's1'):
            mf = scf.RHF(mol).run()
        elif kwargs.get('spin') == 't1':
            mf = scf.ROHF(mol).run()
        else:
            raise ValueError("Invalid spin: must be 's0' or 't1'")
        hf_energy = mf.e_tot
        n_elec = kwargs.get('n_elec')
        n_spin_orbs = kwargs.get('n_spin_orbs') 
        n_spatial_orbs = n_spin_orbs // 2 # Here the input n_orbs means the number of spin orbitals, so need to /2
        mycas = mcscf.CASCI(mf,n_spatial_orbs, n_elec)
        # Save a info.csv file with n_elec and n_orbs
        csv_filename = 'CAS_info.csv'

        h1e_cas, ecore = mycas.get_h1eff()
        h2e_cas = mycas.get_h2eff()
        h2e_cas = ao2mo.addons.restore('1', h2e_cas, n_spatial_orbs)
        h2e_cas = h2e_cas.transpose(0, 2, 3, 1)
        h_ferm = construct_fermionic_operator_from_mcscf(h1e_cas, h2e_cas, ecore)
        if kwargs.get('run_CASCI')==True:
            mycas.kernel()
    else:
        CAS_str=''
        h_ferm = construct_fermionic_operator(mol)
        mf = scf.RHF(mol).run()
        hf_energy = mf.e_tot # These two lines are redundant, change construct_fermionic_operator to a method under CustomMole class for the future
        n_elec = mol.nelectron
        n_spin_orbs = 2*mol.nao
        csv_filename = 'info.csv'
    
    with open(os.path.join(fileDir, csv_filename), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['N_electron', 'N_spin_orb'])
        writer.writerow([n_elec, n_spin_orbs])
    # if spin == 's0':
    #     hf_state = jw_configuration_state(hf_occupation_list(n_elec, 0), 2*n_orbs)
    # elif spin == 't1':
    #     hf_state = jw_configuration_state(hf_occupation_list(n_elec, 2), 2*n_orbs)
    # h_ferm_op = get_sparse_operator(h_ferm)
    # print("Checking expectation of h_ferm WRT HF state: ", vdot(hf_state, h_ferm_op.dot(hf_state)))

    ensure_directory_exists(os.path.join(fileDir, 'ham_fer'))
    if mode == "from_MD" or mode == "h_chain":
        geometry = f"_{kwargs.get('R10')/10}"
    elif mode == "OLED":
        geometry = ""
    # Save ecore, h1e_cas, h2e_cas
    # with open(os.path.join(fileDir, 'ham_fer', mol_name+geometry+f"_mo_ints_{kwargs.get('n_elec')}_{kwargs.get('n_orbs')}_{kwargs.get('spin')}.json"), 'w') as f:
    #     json.dump({'SCF energy': mf.e_tot,'basis': mol.basis, 'ecore': ecore, 'h1e_cas': h1e_cas.tolist(), 'h2e_cas': h2e_cas.tolist()}, f)
    filename_to_save = mol_name+CAS_str+geometry+f"_{n_elec}_{n_spin_orbs}_{kwargs.get('spin')}.data"
    save_operator(h_ferm,file_name=filename_to_save, data_directory=os.path.join(fileDir, 'ham_fer'), allow_overwrite=True, plain_text=True)
    return hf_energy

if __name__ == "__main__":
    target_dir = '/Users/junan/Documents/Research/Overlap/ham_dir'
    hf_energy = gen_hamiltonian(target_dir, mol_name='h2o', mode='from_MD', R10=21, create_CAS=True, spin='s0', n_elec=8, n_spin_orbs=12)
    print("HF energy: ", hf_energy)