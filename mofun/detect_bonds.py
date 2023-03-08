import numpy as np
from scipy.spatial import distance

from mofun import uc_neighbor_offsets

def max_bond_length(el1, el2):
    """Return the maximum length of a bond between two elements"""
    if el1 in NON_METALS or el2 in NON_METALS:
        return COVALENT_RADII[el1] + COVALENT_RADII[el2] + 0.45
    else:
        return COVALENT_RADII[el1] + COVALENT_RADII[el2]

def detect_bonds(structure):
    elements = structure.elements
    if structure.cell is not None:
        # look at all 27-1 neighbors
        uc_offsets = uc_neighbor_offsets(structure.cell)
    else:
        # look at only central cell since no boundaries
        uc_offsets = np.array([[0., 0., 0.]])

    bonds = []
    for idx1, atom1 in enumerate(structure.positions):
        # calculate all possible positions of atom1 in main and neighboring unit cells
        atom1_positions = atom1 + uc_offsets

        for i, atom2 in enumerate(structure.positions[idx1+1:]):
            idx2 = i + idx1 + 1
            # calculate distances between all images of atom1 and atom2
            ss = distance.cdist(atom1_positions, [atom2], "euclidean")
            if np.any(ss < max_bond_length(elements[idx1], elements[idx2])):
                bonds.append([idx1, idx2])

    return np.array(bonds)

NON_METALS = ['H', 'D', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Se', 'Br', 'I', 'Si']

#  Covalent radii from "Covalent radii revisited" DOI:10.1039/B801115J
COVALENT_RADII = {
    'H': 0.31,
    'D': 0.31,
    'He': 0.28,
    'Li': 1.28,
    'Be': 0.96,
    'B': 0.84,
    'C': 0.76,
    'N': 0.71,
    'O': 0.66,
    'F': 0.57,
    'Ne': 0.58,
    'Na': 1.66,
    'Mg': 1.41,
    'Al': 1.21,
    'Si': 1.11,
    'P': 1.07,
    'S': 1.05,
    'Cl': 1.02,
    'Ar': 1.06,
    'K': 2.03,
    'Ca': 1.76,
    'Sc': 1.7,
    'Ti': 1.6,
    'V': 1.53,
    'Cr': 1.39,
    'Mn': 1.30,
    'Fe': 1.32,
    'Co': 1.26,
    'Ni': 1.24,
    'Cu': 1.32,
    'Zn': 1.22,
    'Ga': 1.22,
    'Ge': 1.2,
    'As': 1.19,
    'Se': 1.2,
    'Br': 1.2,
    'Kr': 1.16,
    'Rb': 2.2,
    'Sr': 1.95,
    'Y': 1.9,
    'Zr': 1.75,
    'Nb': 1.64,
    'Mo': 1.54,
    'Tc': 1.47,
    'Ru': 1.46,
    'Rh': 1.42,
    'Pd': 1.39,
    'Ag': 1.45,
    'Cd': 1.44,
    'In': 1.42,
    'Sn': 1.39,
    'Sb': 1.39,
    'Te': 1.38,
    'I': 1.39,
    'Xe': 1.4,
    'Cs': 2.44,
    'Ba': 2.15,
    'La': 2.07,
    'Ce': 2.04,
    'Pr': 2.03,
    'Nd': 2.01,
    'Pm': 1.99,
    'Sm': 1.98,
    'Eu': 1.98,
    'Gd': 1.96,
    'Tb': 1.94,
    'Dy': 1.92,
    'Ho': 1.92,
    'Er': 1.89,
    'Tm': 1.9,
    'Yb': 1.87,
    'Lu': 1.87,
    'Hf': 1.75,
    'Ta': 1.7,
    'W': 1.62,
    'Re': 1.51,
    'Os': 1.44,
    'Ir': 1.41,
    'Pt': 1.36,
    'Au': 1.36,
    'Hg': 1.32,
    'Tl': 1.45,
    'Pb': 1.46,
    'Bi': 1.48,
    'Po': 1.4,
    'At': 1.5,
    'Rn': 1.5,
    'Fr': 2.6,
    'Ra': 2.21,
    'Ac': 2.15,
    'Th': 2.06,
    'Pa': 2,
    'U': 1.96,
    'Np': 1.9,
    'Pu': 1.87,
    'Am': 1.8,
    'Cm': 1.69
}
