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

def calc_norms(positions):
    p_norms = np.zeros([len(positions), len(positions)])
    for j in range(len(positions)):
        for i in range(j, len(positions)):
            p_norms[i,j] = norm(positions[j] - positions[i])
    return p_norms


def find_pattern_in_structure(structure, pattern):
    """find pattern in structure, where both are ASE atoms objects

    Returns:
        a list of indice lists for each set of matched atoms found
    """

    uc_offsets = uc_neighbor_offsets(structure.cell)
    print(uc_offsets)

    s_positions = structure.positions
    s_types = list(structure.symbols)

    p_positions = pattern.positions
    p_types = list(pattern.symbols)

    p_norms = calc_norms(p_positions)

    for i, pattern_atom_1 in enumerate(pattern):
        # Search instances of first atom in a search pattern
        if i == 0:
            match_indices = [[(idx,  (0., 0., 0.))] for idx in atoms_of_type(s_types, p_types[0])]
            print("round %d: " % i, match_indices)
            continue

        last_match_indices = match_indices
        match_indices = []
        # print(last_match_indices  )
        for match in last_match_indices:
            print("----------------------------------")
            print(match)
            for atom_idx in atoms_of_type(s_types, pattern_atom_1.symbol):
                print("--")
                for uc_offset in uc_offsets:
                    found_match = True
                    for j in range(i):
                        pdist = p_norms[i,j]
                        match_idx = match[j][0]
                        match_offset = match[j][1]
                        # print("NORM ARGS: ", structure[match[j]].position + uc_offset, structure_atom.position)
                        sdist = norm(s_positions[match_idx] + match_offset - s_positions[atom_idx] - uc_offset)

                        if not math.isclose(pdist, sdist, rel_tol=5e-2):
                            # print("[%d] %6.4f != %6.4f" % (pattern_atom_0.index, pdist, sdist))
                            found_match = False
                            break
                        # print("[%d] %6.4f ~= %6.4f" % (pattern_atom_0.index, pdist, sdist))
                        # print("[%d] %6.4f ~= %6.4f" % (pattern_atom_0.index, pdist, sdist))
                    # anything that matches the distance to all prior pattern atoms is a good match so far
                    if found_match:
                        print("found a match!", match, atom_idx, uc_offset)
                        match_indices.append(match + [(atom_idx, uc_offset)])

                        #TODO: need to save image index for MATCH!!

        # remove duplicates
        print("round %d: (%d) " % (i, len(match_indices)), match_indices)
        match_indices = remove_duplicates(match_indices)
        print("round %d deduped: (%d) " % (i, len(match_indices)), match_indices)

    # get ASE atoms objects for each set of indices
    match_atoms = [structure.__getitem__([m[0] for m in match]) for match in match_indices]

    return match_indices, match_atoms
