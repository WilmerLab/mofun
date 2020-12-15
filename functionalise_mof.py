import ase as ase
from ase import Atoms, io

def find_pattern_in_structure(structure, pattern):
    """find pattern in structure, where both are ASE atoms objects

    Returns:
        a list of indices of pattern found
    """

    # Search instances of first atom in a search pattern
    match_indices = []
    for i, pattern_atom in enumerate(pattern):
        for structure_atom in structure:
            if pattern_atom.symbol == structure_atom.symbol:
                if i == 0: # first atom in pattern, no distance check
                    match_indices.append([structure_atom.index])
                else:
                    raise Exception("needs to calculate distances here")

    # anything still a match is a good match

    # remove duplicates

    # get ASE atoms objects for each set of indices
    match_atoms = [structure.__getitem__(indices) for indices in match_indices]

    return match_indices, match_atoms
