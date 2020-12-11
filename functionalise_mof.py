import ase as ase
from ase import Atoms, io


def find_pattern_in_structure(structure, pattern):
    """find pattern in structure, where both are ASE atoms objects

    Returns:
        a list of indices of pattern found
    """

    # Search instances of first atom in a search pattern
    list = []
    for pattern_atom in pattern:
        if pattern_atom.symbol == "C":
            for structure_atom in structure:
                if pattern_atom.symbol == structure_atom.symbol:
                    list.append(structure_atom.index)
    return list
