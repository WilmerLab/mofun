import ase as ase
from ase import Atoms, io

atoms_structure = ase.io.read("structure")
atoms_search = ase.io.read("search.pattern")

# Search instances of first atom in a search pattern
list = []
for pattern_atom in atoms_search:
    if pattern_atom.symbol == "C":
        for structure_atom in atoms_structure:
            if pattern_atom.symbol == structure_atom.symbol:
                list.append(structure_atom.index)
