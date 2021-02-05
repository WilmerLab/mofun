import copy

import ase
from ase.formula import Formula
from CifFile import ReadCif as read_cif
import numpy as np

class Atoms:

    def __init__(self, atom_types=[], positions=[], bonds=[], cell=[]):
        self.positions = np.array(positions, dtype=float)
        if isinstance(atom_types, str):
            self.atom_types = list(Formula(atom_types))
        else:
            self.atom_types = np.array(atom_types)
        self.cell = np.array(cell)
        self.bonds = np.array(bonds)
        if len(self.positions) != len(self.atom_types):
            raise Exception("len of positions and atom types must match")

    @classmethod
    def from_cif(cls, path):
        cf = read_cif(path)
        block = cf[cf.get_roots()[0][0]]

        cart_coord_tags = ["_atom_site_Cartn_x", "_atom_site_Cartn_y", "_atom_site_Cartn_z"]
        fract_coord_tags = ["_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"]
        use_fract_coords = False
        if np.array([block.has_key(tag) for tag in cart_coord_tags]).all():
            coords = [block[lbl] for lbl in cart_coord_tags]
        elif np.array([block.has_key(tag) for tag in fract_coord_tags]).all():
            use_fract_coords = True
            coords = [block[lbl] for lbl in fract_coord_tags]
        else:
            raise("no fractional or cartesian coords in CIF file")

        x = [float(c) for c in coords[0]]
        y = [float(c) for c in coords[1]]
        z = [float(c) for c in coords[2]]
        positions = np.array([x,y,z]).T

        atom_types = block['_atom_site_type_symbol']

        cell=None
        cell_tags = ['_cell_length_a', '_cell_length_b', '_cell_length_c', '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma']
        if np.array([block.has_key(tag) for tag in cell_tags]).all():
            a, b, c, alpha, beta, gamma = [float(block[tag]) for tag in cell_tags]
            if alpha != 90. or beta != 90 or gamma != 90.:
                raise Exception("No support for non orthorhombic UCs at the moment!")

            cell=np.identity(3) * (a, b, c)
            if use_fract_coords:
                positions *= (a,b,c)

        return cls(atom_types, positions, cell=cell)

    @classmethod
    def from_ase_atoms(cls, atoms):
        return cls(atoms.symbols, atoms.positions, atoms.cell)

    @property
    def symbols(self):
        return self.atom_types

    def copy(self):
        return copy.deepcopy(self)

    def translate(self, delta):
        if self.positions is not None and len(self) > 0:
            self.positions += delta

    def __len__(self):
        return len(self.positions)

    def extend(self, other):
        self.positions = np.append(self.positions, other.positions, axis=0)
        self.atom_types = np.append(self.atom_types, other.atom_types, axis=0)
        # self.bonds = np.append(self.bonds, other.bonds, axis=0)

    def __delitem__(self, i):
        self.positions = np.delete(self.positions, i, axis=0)
        self.atom_types = np.delete(self.atom_types, i, axis=0)
        # self.bonds = np.delete(self.bonds, i, axis=0)

    def pop(self, pos=-1):
        del(self, pos)

    def __getitem__(self, i):
        return Atoms(positions=np.take(self.positions, i, axis=0),
                     atom_types=np.take(self.atom_types, i, axis=0),
                     cell=self.cell)

    def to_ase(self):
        return ase.Atoms(self.atom_types, positions=self.positions, cell=   self.cell)
