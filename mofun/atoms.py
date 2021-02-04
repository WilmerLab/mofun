import copy

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
    @classmethod
    def fromcif(cls, path):
        cf = read_cif(path)

        coords = [cf.get_all(lbl) for lbl in ("_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z")]
        x = [float(c) for c in coords[0]]
        y = [float(c) for c in coords[1]]
        z = [float(c) for c in coords[2]]

        atom_types = cf.get_all("_atom_site_type_symbol")

        return cls(atom_types, np.array([x,y,z]).T[0])

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
        self.bonds = np.append(self.bonds, other.bonds, axis=0)

    def __delitem__(self, i):
        self.positions = np.delete(self.positions, i, axis=0)
        self.atom_types = np.delete(self.atom_types, i, axis=0)
        # self.bonds = np.delete(self.bonds, i, axis=0)

    def __getitem__(self, i):
        return Atoms(positions=np.take(self.positions, i, axis=0),
                     atom_types=np.take(self.atom_types, i, axis=0),
                     cell=self.cell)
