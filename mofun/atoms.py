import copy

import ase
from ase.formula import Formula
from CifFile import ReadCif as read_cif
import numpy as np

class Atoms:

    def __init__(self, atom_types=[], positions=[], bond_types=[], bonds=[],
                    angle_types=[], angles=[], dihedrals=[], dihedral_types=[], cell=[]):

        self.positions = np.array(positions, dtype=float)
        if isinstance(atom_types, str):
            self.atom_types = list(Formula(atom_types))
        else:
            self.atom_types = np.array(atom_types)
        self.cell = np.array(cell)
        self.bonds = np.array(bonds)
        self.bond_types = np.array(bond_types)
        self.angles = np.array(angles)
        self.angle_types = np.array(angle_types)
        self.dihedrals = np.array(dihedrals)
        self.dihedral_types = np.array(dihedral_types)

        if len(self.positions) != len(self.atom_types):
            raise Exception("len of positions and atom types must match")
        if len(self.bonds) != len(self.bond_types):
            raise Exception("len of bonds and bond types must match")
        if len(self.angles) != len(self.angle_types):
            raise Exception("len of angles and angle types must match")
        if len(self.dihedrals) != len(self.dihedral_types):
            raise Exception("len of dihedrals and dihedral_types must match")

    @classmethod
    def from_lammps_data(cls, f):
        def get_types_tups(arr):
            types = tups = []
            if len(arr) > 0:
                types = arr[:, 1] - 1
                tups = arr[:, 2:] - 1
            return types, tups

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
                atoms.append(tup)
            elif current_section == "Bonds":
                bonds.append(tup)
            elif current_section == "Angles":
                angles.append(tup)
            elif current_section == "Dihedrals":
                dihedrals.append(tup)

        atoms = np.array(atoms, dtype=float)
        bonds = np.array(bonds, dtype=int)
        angles = np.array(angles, dtype=int)
        dihedrals = np.array(dihedrals, dtype=int)

        # note: bond indices in lammps-data file are 1-indexed and we are 0-indexed which is why
        # the bond pairs get a -1
        atom_types = np.array(atoms[:, 1] - 1, dtype=int)
        atom_coords = atoms[:, 2:5]

        bond_types, bond_tups = get_types_tups(bonds)
        angle_types, angle_tups = get_types_tups(angles)
        dihedral_types, dihedral_tups = get_types_tups(dihedrals)

        return cls(atom_types=atom_types, positions=atom_coords,
                   bond_types=bond_types, bonds=bond_tups,
                   angle_types=angle_types, angles=angle_tups,
                   dihedral_types=dihedral_types, dihedrals=dihedrals)

    def to_lammps_data(self, f, file_comment=""):
        f.write("%s (written by mofun)\n\n" % file_comment)

        f.write('%d atoms\n' % len(self.atom_types))
        f.write('%d bonds\n' % len(self.bond_types))
        f.write('%d angles\n' % len(self.angle_types))
        f.write('%d dihedrals\n' % len(self.dihedral_types))
        f.write('0 impropers\n')
        f.write("\n")

        if (num_atom_types := len(set(self.atom_types))) > 0:
            f.write('%d atom types\n' % num_atom_types)
        if (num_bond_types := len(set(self.bond_types))) > 0:
            f.write('%d bond types\n' % num_bond_types)
        if (num_angle_types := len(set(self.angle_types))) > 0:
            f.write('%d angle types\n' % num_angle_types)
        if (num_dihedral_types := len(set(self.dihedral_types))) > 0:
            f.write('%d dihedral types\n' % num_dihedral_types)
        f.write("\n")

        f.write("Atoms\n\n")
        for i, (x, y, z) in enumerate(self.positions):
            f.write(" %d %d %10.6f %10.6f %10.6f\n" % (i + 1, self.atom_types[i] + 1, x, y, z))

        if len(self.bonds) > 0:
            f.write("\nBonds\n\n")
            for i, tup in enumerate(self.bonds):
                f.write(" %d %d %d %d\n" % (i + 1, self.bond_types[i] + 1, *(np.array(tup) + 1)))

        if len(self.angles) > 0:
            f.write("\nAngles\n\n")
            for i, tup in enumerate(self.angles):
                f.write(" %d %d %d %d %d\n" % (i + 1, self.angle_types[i] + 1, *(np.array(tup) + 1)))

        if len(self.dihedrals) > 0:
            f.write("\nDihedrals\n\n")
            for i, tup in enumerate(self.dihedrals):
                f.write(" %d %d %d %d %d %d\n" % (i + 1, self.dihedral_types[i] + 1, *(np.array(tup) + 1)))

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
            print("WARNING: cif read doesn't handle bond types at present; bonding info is discarded. Use LAMMPS data file format if you need bonds")
            bonds = []

        cell=None
        cell_tags = ['_cell_length_a', '_cell_length_b', '_cell_length_c', '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma']
        if has_all_tags(block, cell_tags):
            a, b, c, alpha, beta, gamma = [float(block[tag]) for tag in cell_tags]
            if alpha != 90. or beta != 90 or gamma != 90.:
                raise Exception("No support for non orthorhombic UCs at the moment!")

            cell=np.identity(3) * (a, b, c)
            if use_fract_coords:
                positions *= (a,b,c)

        return cls(atom_types, positions, cell=cell)

    @classmethod
    def from_ase_atoms(cls, atoms):
        return cls(atoms.symbols, atoms.positions, cell=atoms.cell)

    def map_atom_types(self, type_map):
        """ converts lammps style atom types (indices) to string atom types based on a provided map"""
        return [type_map[i] for i in self.atom_types]

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
        if len(self.bonds) == 0:
            self.bonds = other.bonds.copy() + len(self.positions)
            self.bond_types = other.bond_types.copy() + len(self.positions)
        else:
            self.bonds = np.append(self.bonds, other.bonds + len(self.positions), axis=0)
            self.bond_types = np.append(self.bond_types, other.bond_types + len(self.bond_types), axis=0)

        if len(self.angles) == 0:
            self.angles = other.angles.copy() + len(self.positions)
            self.angle_types = other.angle_types.copy() + len(self.angle_types)
        else:
            self.angles = np.append(self.angles, other.angles + len(self.positions), axis=0)
            self.angle_types = np.append(self.angle_types, other.angle_types + len(self.angle_types), axis=0)

        if len(self.dihedrals) == 0:
            self.dihedrals = other.dihedrals.copy() + len(self.positions)
            self.dihedral_types = other.dihedral_types.copy() + len(self.dihedral_types)
        else:
            self.dihedrals = np.append(self.dihedrals, other.dihedrals + len(self.positions), axis=0)
            self.dihedral_types = np.append(self.dihedral_types, other.dihedral_types + len(self.dihedral_types), axis=0)

        self.positions = np.append(self.positions, other.positions, axis=0)
        self.atom_types = np.append(self.atom_types, other.atom_types, axis=0)

    def _delete_and_reindex_atom_index_array(self, arr, sorted_deleted_indices):
        updated_arr = arr.copy()
        arr_idx_to_delete = []
        for i, atom_idx_tuple in enumerate(arr):
            if np.any([a in sorted_deleted_indices for a in atom_idx_tuple]):
                arr_idx_to_delete.append(i)
        updated_arr = np.delete(updated_arr, arr_idx_to_delete, axis=0)

        # reindex
        for i in sorted_deleted_indices:
            updated_arr = np.subtract(updated_arr, 1, where=updated_arr>i)

        return updated_arr


    def __delitem__(self, indices):
        self.positions = np.delete(self.positions, indices, axis=0)
        self.atom_types = np.delete(self.atom_types, indices, axis=0)

        sorted_indices = sorted(indices, reverse=True)
        if len(self.bonds) > 0:
            self.bonds = self._delete_and_reindex_atom_index_array(self.bonds, sorted_indices)
        if len(self.angles) > 0:
            self.angles = self._delete_and_reindex_atom_index_array(self.angles, sorted_indices)
        if len(self.dihedrals) > 0:
            self.dihedrals = self._delete_and_reindex_atom_index_array(self.dihedrals, sorted_indices)



    def pop(self, pos=-1):
        del(self, pos)

    def __getitem__(self, i):
        return Atoms(positions=np.take(self.positions, i, axis=0),
                     atom_types=np.take(self.atom_types, i, axis=0),
                     cell=self.cell)

    def to_ase(self):
        return ase.Atoms(self.atom_types, positions=self.positions, cell=   self.cell)
