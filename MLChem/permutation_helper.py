import numpy as np
import itertools as it
import copy


def generate_permutations(k):
    """
    Generates a list of lists of all possible orderings of k indices

    Parameters
    ----------
    k : int
        The number of elements in a permutation group. 

    Returns
    ---------
    permutations : list
        A list of all possible permutations of the set {1, 2, ..., k}
    """
    permutations = []
    for perm in (it.permutations(range(k))):
        permutations.append(list(perm))
    return permutations


def find_cycles(perm):
    """
    Finds every possible cycle which results in the given permutation

    Parameters
    ----------
    perm : list
        Some permutation vector

    Returns
    ----------
    cycles : list
        A list of cycles (permutation operations)

    Example
    -------
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
    Finds the array of bond indices of an interatomic distance matrix, in row wise order:
    [[0,1], [0,2], [1,2], [0,3], [1,3], [2,3], ..., [0, natoms], [1, natoms], ...,[natoms-1, natoms]]
    
    Parameters
    ----------
    natoms: int
        The number of atoms

    Returns
    ----------
    bond_indices : list
        A list of lists, where each sublist is the subscripts of an interatomic distance
        from an interatomic distance matrix representation of a molecular system.
        e.g. r_12, r_01, r_05 
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
    
    Parameters  
    ---------
    atomtype_vector : list
        A list of the number of each atom in a molecular system, e.g., for an A2BC system, 
        atomtype_vector would be [2,1,1].
    
    Returns
    --------
    cycles_by_atom : list
        The cycle permutation operators which act on each atom. (?)
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
    Permutes a bond indice if the bond indice is affected by the permutation cycle.

    Parameters
    ----------
    bond : list
        A list of length 2, a subscript of an interatom distance
    cycle : list
        A cycle-notation permutation operation.

    Returns
    -------
    bond : the permuted bond
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
    Permutes the set of bond indices of a molecule according to the complete set of 
    valid molecular permutation cycles.

    Parameters
    ----------
    atomtype_vector : list
        A list of the number of each atom in a molecular system, e.g., for an A3B8C system, 
        atomtype_vector would be [3,8,1].

    Returns
    --------
    bond_indice_permutations: list
        A list of all possible bond indice permutations of the interatomic distances 
        The length is equal to the number of atomic permutations
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
    Given the original bond indices list [[0,1],[0,2],[1,2]...] and a permutation of this bond indices 
    list (which is found by permute_bond_indices), find the permutation vector that maps the original 
    to the permuted list. Do this for all permutations of the bond indices list. 
    The result is complete set induced interatomic distance matrix permutatations caused 
    by the molecular permutation cycles.

    Parameters
    ---------
    atomtype_vector : list
        A list of the number of each atom in a molecular system, e.g., for an A3B8C system, 
        atomtype_vector would be [3,8,1].
    bond_indice_permutations: list
        A list of all possible bond indice permutations of the interatomic distances 
        The length is equal to the number of atomic permutations. (?)

    Returns
    ---------
    induced_perms : list
        The induced interatom distance permutations caused by the atomic permutation operations. 
        In row-wise order of the interatomic distance matrix: [r01, r02, r12, r03, r13, r12...]
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
       # cycle = find_cycles(perm) 
        induced_perms.append(perm)
    return induced_perms


