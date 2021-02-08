import copy

import ase
from ase.formula import Formula
from CifFile import ReadCif as read_cif
import numpy as np

class Atoms:

    def __init__(self, atom_types=[], positions=[], bond_types=[], bonds=[], cell=[]):
        self.positions = np.array(positions, dtype=float)
        if isinstance(atom_types, str):
            self.atom_types = list(Formula(atom_types))
        else:
            self.atom_types = np.array(atom_types)
        self.cell = np.array(cell)
        self.bonds = np.array(bonds)
        self.bond_types = np.array(bond_types)

        if len(self.positions) != len(self.atom_types):
            raise Exception("len of positions and atom types must match")
        if len(self.bonds) != len(self.bond_types):
            raise Exception("len of bonds and bond types must match")

    @classmethod
    def from_lammps_data(cls, f):
        def type_tuple(types, tup):
            return tuple([t(tup[i]) for i, t in enumerate(types)])

        atoms = []
        bonds = []
        angles = []
        dihedrals = []

        sections_handled = ["Atoms", "Bonds", "Angles", "Dihedrals"]
        current_section = None
        start_section = False

        for unprocessed_line in f:
            line = unprocessed_line.strip()
            if line in sections_handled:
                current_section = line
                start_section = True
                continue
            elif line == "":
                if not start_section:
                    # end of section or blank
                    current_section= None
                start_section = False
                continue
            tup = line.split()
            if current_section == "Atoms":
                atoms.append(type_tuple([int]*2 + [float]*3, tup))
            elif current_section == "Bonds":
                bonds.append(type_tuple([int]*4, tup))
            elif current_section == "Angles":
                angles.append(type_tuple([int]*5, tup))
            elif current_section == "Dihedrals":
                dihedrals.append(type_tuple([int]*6, tup))

        atoms = np.array(atoms)
        bonds = np.array(bonds)

        # note: bond indices in lammps-data file are 1-indexed and we are 0-indexed which is why
        # the bond pairs get a -1
        return cls(atom_types=atoms[:, 1] - 1, positions=atoms[:, 2:5],
                   bond_types=bonds[:, 1] - 1, bonds=bonds[:, 2:4] - 1)

    def to_lammps_data(self, f, file_comment=""):
        f.write("%s (written by mofun)\n\n" % file_comment)

        f.write('{} atoms\n'.format(len(self.atom_types)))
        f.write('{} bonds\n'.format(len(self.bond_types)))
        f.write('0 angles\n')
        f.write('0 dihedrals\n')
        f.write('0 impropers\n')
        f.write("\n")
        if (num_atom_types := len(set(self.atom_types))) > 0:
            f.write('{} atom types\n'.format(num_atom_types))
        if (num_bond_types := len(set(self.bond_types))) > 0:
            f.write('{} bond types\n'.format(num_bond_types))
        f.write("\n")

        f.write("Atoms\n\n")
        for i, (x, y, z) in enumerate(self.positions):
            f.write(" %d %d %10.6f %10.6f %10.6f\n" % (i + 1, self.atom_types[i] + 1, x, y, z))
        f.write("\n")

        f.write("Bonds\n\n")
        for i, (a1, a2) in enumerate(self.bonds):
            f.write(" %d %d %d %d\n" % (i + 1, self.bond_types[i] + 1, a1 + 1, a2 + 1))

    @classmethod
    def from_cif(cls, path):
        def has_all_tags(block, tags):
            return np.array([block.has_key(tag) for tag in tags]).all()

        cf = read_cif(path)
        block = cf[cf.get_roots()[0][0]]

        cart_coord_tags = ["_atom_site_Cartn_x", "_atom_site_Cartn_y", "_atom_site_Cartn_z", "_atom_site_label"]
        fract_coord_tags = ["_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z", "_atom_site_label"]
        use_fract_coords = False
        if has_all_tags(block, cart_coord_tags):
            coords = [block[lbl] for lbl in cart_coord_tags]
        elif has_all_tags(block, fract_coord_tags):
            use_fract_coords = True
            coords = [block[lbl] for lbl in fract_coord_tags]
        else:
            raise("no fractional or cartesian coords in CIF file")

        x = [float(c) for c in coords[0]]
        y = [float(c) for c in coords[1]]
        z = [float(c) for c in coords[2]]
        atom_name = coords[3]
        positions = np.array([x,y,z]).T

        atom_types = block['_atom_site_type_symbol']

        bonds = []
        bond_tags = ["_geom_bond_atom_site_label_1", "_geom_bond_atom_site_label_2"]
        if has_all_tags(block, bond_tags):
            cifbonds = zip(*[block[lbl] for lbl in bond_tags])
            bonds = [(atom_name.index(a), atom_name.index(b)) for (a,b) in cifbonds]

        cell=None
        cell_tags = ['_cell_length_a', '_cell_length_b', '_cell_length_c', '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma']
        if has_all_tags(block, cell_tags):
            a, b, c, alpha, beta, gamma = [float(block[tag]) for tag in cell_tags]
            if alpha != 90. or beta != 90 or gamma != 90.:
                raise Exception("No support for non orthorhombic UCs at the moment!")

            cell=np.identity(3) * (a, b, c)
            if use_fract_coords:
                positions *= (a,b,c)

        return cls(atom_types, positions, bonds, cell)

    @classmethod
    def from_ase_atoms(cls, atoms):
        return cls(atoms.symbols, atoms.positions, cell=atoms.cell)

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
        self.bonds = np.append(self.bonds, other.bonds + len(self.positions), axis=0)
        self.positions = np.append(self.positions, other.positions, axis=0)
        self.atom_types = np.append(self.atom_types, other.atom_types, axis=0)

    def __delitem__(self, indices):
        self.positions = np.delete(self.positions, indices, axis=0)
        self.atom_types = np.delete(self.atom_types, indices, axis=0)

        if len(self.bonds) > 0:
            # delete bonds
            bond_idx_to_delete = []
            for i, (a,b) in enumerate(self.bonds):
                if a in indices or b in indices:
                    bond_idx_to_delete.append(i)
            self.bonds = np.delete(self.bonds, bond_idx_to_delete, axis=0)

            # reindex bonds
            sorted_indices = sorted(indices, reverse=True)
            for i in sorted_indices:
                self.bonds = np.subtract(self.bonds, 1, where=self.bonds>i)


    def pop(self, pos=-1):
        del(self, pos)

    def __getitem__(self, i):
        return Atoms(positions=np.take(self.positions, i, axis=0),
                     atom_types=np.take(self.atom_types, i, axis=0),
                     cell=self.cell)

    def to_ase(self):
        return ase.Atoms(self.atom_types, positions=self.positions, cell=   self.cell)
