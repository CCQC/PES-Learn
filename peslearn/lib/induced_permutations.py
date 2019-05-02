# This code aims to take an arbitrary molecular system AnBmCp... with any number of like atoms and:
# 1. Determine the atom permutation operations (cycles) of the permutation groups Sn, Sm, Sp ... 
# 2. Find the induced permutations of the atom permutation operations of Sn, Sm, Sp ...  on the set of interatomic distances
# 3. Export Singular code to derive the fundamental invariants
# Result: a generalized algorithm for obtaining a permutationally invariant basis for geometrical parameters so that the PES is permutation invariant

import numpy as np
import itertools as it
import math
import copy

def generate_permutations(k):
    """
    Generates a list of lists of all possible orderings of k indices
    """
    f_k = math.factorial(k)
    A = []
    for perm in (it.permutations(range(k))):
        A.append(list(perm)) 
    return A


def find_cycles(perm):
    """
    Finds the cycle(s) required to get the permutation. For example,
    the permutation [3,1,2] is obtained by permuting [1,2,3] with the cycle [1,2,3]
    read as "1 goes to 2, 2 goes to 3, 3 goes to 1".
    Sometimes cycles are products of more than one subcycle, e.g. (12)(34)(5678)
    """
    pi = {i: perm[i] for i in range(len(perm))}
    cycles = []

    while pi:
        elem0 = next(iter(pi)) # arbitrary starting element
        this_elem = pi[elem0]
        next_item = pi[this_elem]

        cycle = []
        while True:
            cycle.append(this_elem)
            del pi[this_elem]
            this_elem = next_item
            if next_item in pi:
                next_item = pi[next_item]
            else:
                break
        cycles.append(cycle[::-1])

    # only save cycles of size 2 and larger
    cycles[:] = [cyc for cyc in cycles if len(cyc) > 1]
    return cycles


def generate_bond_indices(natoms):
    """
    natoms: int
        The number of atoms
    Finds the array of bond indices of the upper triangle of an interatomic distance matrix, in column wise order
    ( or equivalently, lower triangle of interatomic distance matrix in row wise order):
    [[0,1], [0,2], [1,2], [0,3], [1,3], [2,3], ...,[0, natom], ...,[natom-1, natom]]
    """ 
    # initialize j as the number of atoms
    j = natoms - 1
    # now loop backward until you generate all bond indices 
    bond_indices = []
    while j > 0:
        i = j - 1
        while i >= 0:
            new = [i, j]
            bond_indices.insert(0, new)
            i -= 1
        j -= 1 
    return bond_indices

def molecular_cycles(atomtype_vector):
    """
    Finds the complete set of cycles that may act on a molecular system.
    Given an atomtype vector, containing the number of each atom:
         1.  generate the permutations of each atom
         2.  generate the cycles of each atom
         3.  adjust the indices to be nonoverlapping, so that each atom has a unique set of indices.
    For example, For an A2BC system, the indices may be assigned as follows: A 0,1; B 2; C 3; 
    while the methods generate_permutations and find_cycles index from 0 for every atom, so we adjust the indices of every atom appropriately
    """
    permutations_by_atom = [] 
    for atom in atomtype_vector:
        # add the set of permutations for each atom type to permutations_by_atom
        permutations_by_atom.append(generate_permutations(atom)) # an array of permutations is added for atom type X
    cycles_by_atom = [] 
    # each atom has a set of permutations, saved in permutations_by_atom 
    for i, perms in enumerate(permutations_by_atom):
        cycles = []
        # find the cycles of each permutation and append to cycles, then append cycles to cycles_by_atom
        for perm in perms:
            cyc = find_cycles(perm)
            if cyc:  # dont add empty cycles (identity permutation)
                cycles.append(cyc)
        cycles_by_atom.append(cycles)
    # now update the indices of the second atom through the last atom since they are currently indexed from zero
    # to do this we need to know the number of previous atoms, num_prev_atoms
    atomidx = 0
    num_prev_atoms = 0
    for atom in cycles_by_atom[1:]:
        num_prev_atoms += atomtype_vector[atomidx]
        for cycle in atom:
            for subcycle in cycle: # some cycles are composed of two or more subcycles (12)(34) etc.
                for i, idx in enumerate(subcycle): 
                    subcycle[i] = idx + num_prev_atoms
        atomidx += 1
    return cycles_by_atom


def permute_bond(bond, cycle):
    """
    Permutes a bond inidice if the bond indice is affected by the permutation cycle.
    There is certainly a better way to code this. Yikes.
    """
    count0 = 0
    count1 = 0
    # if the bond indice matches the cycle indice, set the bond indice equal to the next indice in the cycle
    # we count so we dont change a bond indice more than once.
    # If the cycle indice is at the end of the list, the bond indice should become the first element of the list since thats how cycles work.
    # theres probably a better way to have a list go back to the beginning
    for i, idx in enumerate(cycle):
        if (bond[0] == idx) and (count0 == 0):
            try:
                bond[0] = cycle[i+1]
            except:
                bond[0] = cycle[0]
            count0 += 1

        if (bond[1] == idx) and (count1 == 0):
            try:
                bond[1] = cycle[i+1]
            except:
                bond[1] = cycle[0]
            count1 += 1
    # sort if the permutation messed up the order. if you convert 1,2 to 2,1, for example    
    bond.sort()
    return bond 
   
def permute_bond_indices(atomtype_vector):
    """
    Permutes the set of bond indices of a molecule according to the complete set of valid molecular permutation cycles
    atomtype_vector: array-like
        A vector of the number of each atoms, the length is the total number of atoms.
        An A3B8C system would be [3, 8, 1]
    Returns many sets permuted bond indices, the number of which equal to the number of cycles
    """
    natoms = sum(atomtype_vector) 
    bond_indices = generate_bond_indices(natoms)    
    cycles_by_atom = molecular_cycles(atomtype_vector)
         
    bond_indice_permutations = [] # interatomic distance matrix permutations
    for atom in cycles_by_atom:
        for cycle in atom:
            tmp_bond_indices = copy.deepcopy(bond_indices) # need a deep copy, list of lists
            for subcycle in cycle:
                for i, bond in enumerate(tmp_bond_indices):
                    tmp_bond_indices[i] = permute_bond(bond, subcycle)
            bond_indice_permutations.append(tmp_bond_indices) 

    return bond_indice_permutations 

def induced_permutations(atomtype_vector, bond_indice_permutations):
    """
    Given the original bond indices list [[0,1],[0,2],[1,2]...] and a permutation of this bond indices list,
    find the permutation vector that maps the original to the permuted list. 
    Do this for all permutations of the bond indices list. 
    Result: The complete set induced interatomic distance matrix permutatations caused by the molecular permutation cycles 
    """
    natoms = sum(atomtype_vector) 
    bond_indices = generate_bond_indices(natoms)    
   
    induced_perms = [] 
    for bip in bond_indice_permutations:
        perm = []
        for bond1 in bond_indices:
            for i, bond2 in enumerate(bip):
                if bond1 == bond2:
                    perm.append(i)
        cycle = find_cycles(perm) 
        induced_perms.append(cycle)
    return induced_perms
                

def write_singular_input(natoms, induced_perms):
    # Singular doesnt tolerate 0 indexing, so we add 1 to every element
    for cycle in induced_perms:
        for subcycle in cycle:
            for i in range(len(subcycle)):
                subcycle[i] += 1

    # create interatom distance variables
    A = []
    nbonds = int((natoms**2 - natoms) / 2)
    for i in range(1,nbonds+1):
        A.append("x"+str(i))

    operators = ''
    count = 0
    for cycle in induced_perms:
        if count == 0:
            operators += 'list'
        else: 
            operators += ',list'
        count += 1

        if len(cycle) > 1:
            operators += '('
            for subcycle in cycle:
                operators += 'list'
                operators += str(tuple(subcycle))
                # add comma except at end
                if subcycle != cycle[-1]:
                    operators += ','
            operators += ')'
        else:
            if len(cycle) == 1:
                operators += '(list' + str(tuple(cycle[0])) + ')'

    line1 = "LIB \"finvar.lib\";\n"  
    line2 = "ring R=0,({}),dp;\n".format(",".join(map(str,A)))
    line3 = "def GEN=list({});\n".format(operators)
    line4 = "matrix G = invariant_algebra_perm(GEN,0);\n" 
    line5 = "G;"
    return (line1 + line2 + line3 + line4 + line5)


def atom_combinations(N):
    """
    Generates the combinations of atom numbers for a molecular system with total number of atoms equal to N
    """
    atomindices = []
    for i in range(1,N+1):
        atomindices.append(i)
    combos = [] 
    for i in range(1, N+1):
        for combo in it.combinations_with_replacement(atomindices, i):
            if sum(combo) == N: 
                combos.append(list(combo))
    return combos

# use this to test. vector should be in same order as axis of interatomic distances. i.e., if columns are indexed by H H H C C O, vector should be [3,2,1]
atomtype_vector = [3]
bond_indice_permutations = permute_bond_indices(atomtype_vector)
IP  = induced_permutations(atomtype_vector, bond_indice_permutations)
singular = write_singular_input(sum(atomtype_vector), IP)
print("Here is your Singular input file. Install Singular, copy paste text into a file, and run with 'Singular (inputfilename)'\n\n")
print(singular)
