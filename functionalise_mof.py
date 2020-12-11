import ase as ase
from ase import Atoms, io


def find_pattern_in_structure(structure, pattern):
    """find pattern in structure, where both are ASE atoms objects

    Returns:
        a list of indices of pattern found
    """

    # Search instances of first atom in a search pattern
    match_indices = []
    match_atoms = []
    for pattern_atom in pattern:
        if pattern_atom.symbol == "C":
            for structure_atom in structure:
                if pattern_atom.symbol == structure_atom.symbol:
                    match_indices.append(structure_atom.index)
                    # match_atoms.append(...)
    # for match indices:
    # for C case: [(3), (6), (9)...]
    # for CH2 case: [(3,4,5), (6,7,8), (9,10,11)...]
    return match_indices, match_atoms
