import ase as ase
from ase import Atoms, io
from ase.io import write, read
from ase.visualize import view

def read_structure(filename):

    atoms = read(filename)

    # Some of them could be helpful later ...
    positions = atoms.get_positions()
    chemical_symbols = atoms.get_chemical_symbols()
    masses = atoms.get_masses()
    momenta = atoms.get_momenta()
    scaled_positions = atoms.get_scaled_positions()
    cell = atoms.get_cell()
    cell_lengths_and_angles = atoms.get_cell_lengths_and_angles()
    center_of_mass = atoms.get_center_of_mass()
    global_number_of_atoms = atoms.get_global_number_of_atoms()
    volume = atoms.get_volume()

    return atoms

def read_search_pattern(filename):

    atoms = read(filename)

    # Some of them could be helpful later ...
    positions = atoms.get_positions()
    chemical_symbols = atoms.get_chemical_symbols()
    masses = atoms.get_masses()
    momenta = atoms.get_momenta()
    scaled_positions = atoms.get_scaled_positions()
    cell = atoms.get_cell()
    cell_lengths_and_angles = atoms.get_cell_lengths_and_angles()
    center_of_mass = atoms.get_center_of_mass()
    global_number_of_atoms = atoms.get_global_number_of_atoms()

    return atoms

atoms_structure = read_structure("structure")
atoms_search = read_search_pattern("search.pattern")

# Search instances of first atom in a search pattern
list = []
for pattern in atoms_search:
    if pattern.symbol == "C":
        for find in atoms_structure:
            if pattern.symbol == find.symbol:
                list.append(find.index)
