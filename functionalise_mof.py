import math

import ase as ase
from ase import Atoms, io
import numpy as np
from numpy.linalg import norm

def atoms_of_type(structure, element):
    return [a for a in structure if element == a.symbol]

def uc_neighbor_offsets(structure):
    multipliers = np.array(np.meshgrid([-1, 0, 1],[-1, 0, 1],[-1, 0, 1])).T.reshape(-1, 3)
    # offsets = np.array([[0, 0, 0]])
    # return np.array([(structure.cell * m).sum(axis=1) for m in multipliers]).flatten().reshape(27, 3)
    return [tuple((structure.cell * m).sum(axis=1)) for m in multipliers]


def find_pattern_in_structure(structure, pattern):
    """find pattern in structure, where both are ASE atoms objects

    Returns:
        a list of indice lists for each set of matched atoms found
    """

    uc_offsets = uc_neighbor_offsets(structure)
    print(uc_offsets)

    for i, pattern_atom_1 in enumerate(pattern):
        # Search instances of first atom in a search pattern
        if i == 0:
            match_indices = [[(a.index,  (0., 0., 0.))] for a in atoms_of_type(structure, pattern[0].symbol)]
            print("round %d: " % i, match_indices)
            continue

        last_match_indices = match_indices
        match_indices = []
        # print(last_match_indices  )
        for match in last_match_indices:
            print("----------------------------------")
            print(match)
            for structure_atom in atoms_of_type(structure, pattern_atom_1.symbol):
                print("--")
                for uc_offset in uc_offsets:
                    found_match = True
                    for j, pattern_atom_0 in enumerate(pattern[0:i]):
                        pdist = pattern.get_distance(i, j)
                        match_atom = structure[match[j][0]]
                        match_offset = match[j][1]
                        # print("NORM ARGS: ", structure[match[j]].position + uc_offset, structure_atom.position)
                        sdist = norm(match_atom.position + match_offset - structure_atom.position - uc_offset)

                        if not math.isclose(pdist, sdist, rel_tol=5e-2):
                            # print("[%d] %6.4f != %6.4f" % (pattern_atom_0.index, pdist, sdist))
                            found_match = False
                            break
                        # print("[%d] %6.4f ~= %6.4f" % (pattern_atom_0.index, pdist, sdist))
                        # print("[%d] %6.4f ~= %6.4f" % (pattern_atom_0.index, pdist, sdist))
                    # anything that matches the distance to all prior pattern atoms is a good match so far
                    if found_match:
                        print("found a match!", match, structure_atom.index, uc_offset)
                        match_indices.append(match + [(structure_atom.index, uc_offset)])

                        #TODO: need to save image index for MATCH!!

        print("round %d: (%d) " % (i, len(match_indices)), match_indices)
        # remove duplicates
        match_indices = set([tuple(sorted(matches)) for matches  in match_indices])
        match_indices = [list(m) for m in match_indices]
        print("round %d deduped: (%d) " % (i, len(match_indices)), match_indices)

    # get ASE atoms objects for each set of indices
    match_atoms = [structure.__getitem__([m[0] for m in match]) for match in match_indices]

    return match_indices, match_atoms
