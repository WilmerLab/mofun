import math

import ase as ase
from ase import Atoms, io



def atoms_of_type(structure, element):
    return [a for a in structure if element == a.symbol]

def find_pattern_in_structure(structure, pattern):
    """find pattern in structure, where both are ASE atoms objects

    Returns:
        a list of indices of pattern found
    """

    for i, pattern_atom_1 in enumerate(pattern):
        # Search instances of first atom in a search pattern
        if i == 0:
            match_indices = [[a.index] for a in atoms_of_type(structure, pattern[0].symbol)]
            print("round %d: " % i, match_indices)
            continue

        last_match_indices = match_indices
        match_indices = []
        for match in last_match_indices:
            for structure_atom in atoms_of_type(structure, pattern_atom_1.symbol):
                found_match = True
                for j, pattern_atom_0 in enumerate(pattern[0:i]):
                    pdist = pattern.get_distance(i, j)
                    sdist = structure.get_distance(match[j], structure_atom.index)
                    print("[%d] %6.4f ~= %6.4f" % (pattern_atom_0.index, pdist, sdist))
                    if not math.isclose(pdist, sdist, rel_tol=5e-2):
                        found_match = False
                        break

                # anything that matches the distance to all prior pattern atoms is a good match so far
                if found_match:
                    print("found a match!", match, structure_atom.index)
                    match_indices.append(match + [structure_atom.index])
        print("round %d: " % i, match_indices)

    # remove duplicates
    match_indices = set([tuple(sorted(indices)) for indices in match_indices])
    print("after deduplication: ", match_indices)

    # get ASE atoms objects for each set of indices
    match_atoms = [structure.__getitem__(indices) for indices in match_indices]

    return match_indices, match_atoms
