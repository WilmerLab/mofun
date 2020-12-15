import math

import ase as ase
from ase import Atoms, io

def find_pattern_in_structure(structure, pattern):
    """find pattern in structure, where both are ASE atoms objects

    Returns:
        a list of indices of pattern found
    """


    # now search for every other pattern atoms
    for i, pattern_atom_1 in enumerate(pattern):
        # Search instances of first atom in a search pattern
        if i == 0:
            match_indices = []
            for structure_atom in structure:
                if pattern[0].symbol == structure_atom.symbol:
                    match_indices.append([structure_atom.index])

            print("round %d: " % i, match_indices)
            continue

        last_match_indices = match_indices
        match_indices = []
        for match in last_match_indices:
            for structure_atom in structure:
                if pattern_atom_1.symbol == structure_atom.symbol:
                    found_match = True
                    for j, pattern_atom_0 in enumerate(pattern[0:i]):
                        pdist = pattern.get_distance(i, j)
                        sdist = structure.get_distance(match[j], structure_atom.index)
                        if not math.isclose(pdist, sdist, rel_tol = 1e-3):
                            found_match = False
                            break
                    if found_match:
                        print("found a match!", match, structure_atom.index)
                        match_indices.append(match + [structure_atom.index])
        print("round %d: " % i, match_indices)

    # anything still a match is a good match

    # remove duplicates

    # get ASE atoms objects for each set of indices
    match_atoms = [structure.__getitem__(indices) for indices in match_indices]

    return match_indices, match_atoms
