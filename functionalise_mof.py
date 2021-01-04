import math

import ase as ase
from ase import Atoms, io
import numpy as np
from numpy.linalg import norm

def atoms_of_type(types, element):
    """ returns all atom indices in types that match the symbol element """
    return [i for i, t in enumerate(types) if t == element]

def uc_neighbor_offsets(uc_vectors):
    multipliers = np.array(np.meshgrid([-1, 0, 1],[-1, 0, 1],[-1, 0, 1])).T.reshape(-1, 3)
    # offsets = np.array([[0, 0, 0]])
    # return np.array([(structure.cell * m).sum(axis=1) for m in multipliers]).flatten().reshape(27, 3)
    return [tuple((uc_vectors * m).sum(axis=1)) for m in multipliers]

def remove_duplicates(match_indices):
    match1 = set([tuple(sorted(matches)) for matches in match_indices])
    return [list(m) for m in match1]

def calc_sum_of_squares(positions):
    ss = np.zeros([len(positions), len(positions)])
    for j in range(len(positions)):
        for i in range(j, len(positions)):
            pdiff = positions[j] - positions[i]
            ss[i,j] = np.inner(pdiff, pdiff)
    return ss


def find_pattern_in_structure(structure, pattern):
    """find pattern in structure, where both are ASE atoms objects

    Returns:
        a list of indice lists for each set of matched atoms found
    """

    uc_offsets = uc_neighbor_offsets(structure.cell)

    s_positions = structure.positions
    s_types = list(structure.symbols)

    p_positions = pattern.positions
    p_types = list(pattern.symbols)

    p_ss = calc_sum_of_squares(p_positions)

    for i, pattern_atom_1 in enumerate(pattern):
        # Search instances of first atom in a search pattern
        if i == 0:
            match_indices = [[(idx,  (0., 0., 0.))] for idx in atoms_of_type(s_types, p_types[0])]
            print("round %d: " % i, match_indices)
            continue

        last_match_indices = match_indices
        match_indices = []
        for match in last_match_indices:
            for atom_idx in atoms_of_type(s_types, pattern_atom_1.symbol):
                for uc_offset in uc_offsets:
                    found_match = True
                    for j in range(i):
                        match_idx = match[j][0]
                        match_offset = match[j][1]

                        # we don't need an actual distance here, using the sum of squares instead
                        # saves us the square root calculation and allows us to use an inner product
                        # which is very fast in numpy
                        s_diff = s_positions[match_idx] + match_offset - s_positions[atom_idx] - uc_offset
                        s_ss = np.inner(s_diff, s_diff)

                        if not math.isclose(p_ss[i,j], s_ss, rel_tol=5e-2):
                            found_match = False
                            break

                    # anything that matches the distance to all prior pattern atoms is a good match so far
                    if found_match:
                        match_indices.append(match + [(atom_idx, uc_offset)])

        match_indices = remove_duplicates(match_indices)
        print("round %d: (%d) " % (i, len(match_indices)), match_indices)

    # get ASE atoms objects for each set of indices
    match_atoms = [structure.__getitem__([m[0] for m in match]) for match in match_indices]

    return match_indices, match_atoms


def replace_pattern_in_structure(structure, search_pattern, replace_pattern):
    pass
