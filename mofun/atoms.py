import copy

import ase
from ase.formula import Formula
from CifFile import ReadCif as read_cif
import numpy as np
from scipy.linalg import norm

from mofun.helpers import guess_elements_from_masses, ELEMENT_MASSES

class Atoms:

    def __init__(self, atom_types=[], positions=[], bond_types=[], bonds=[],
                    angle_types=[], angles=[], dihedrals=[], dihedral_types=[],
                    atom_type_masses=[], cell=[], elements=[], atom_type_elements=[]):

        self.atom_type_masses = np.array(atom_type_masses, ndmin=1)
        self.positions = np.array(positions, dtype=float, ndmin=1)

        if len(atom_type_elements) > 0:
            # this is a __getitem__ subset
            self.atom_types = np.array(atom_types)
            self.atom_type_elements = atom_type_elements
        elif len(atom_types) > 0:
            # from atom type ids and masses, such as from LAMMPS:
            self.atom_types = np.array(atom_types, ndmin=1)
            self.atom_type_elements = guess_elements_from_masses(self.atom_type_masses)
        elif len(elements) > 0:
            # from element array, such as from ASE atoms
            # i.e. Propane ['C', 'H', 'H', 'H', 'C', 'H', 'H', 'C', 'H', 'H', 'H']:
            if isinstance(elements, str):
                # from element string, i.e. Propane "CHHHCHHCHHH" (shorthand):
                elements = list(Formula(elements))

            self.atom_type_elements = list(set(elements))
            self.atom_types = np.array([self.atom_type_elements.index(s) for s in elements])
            self.atom_type_masses = [ELEMENT_MASSES[s] for s in self.atom_type_elements]
        else:
            # no atom_types, atom_type_elements or elements passed
            self.atom_types = np.array([], ndmin=1)

        self.cell = np.array(cell)
        self.bonds = np.array(bonds)
        self.bond_types = np.array(bond_types)
        self.angles = np.array(angles)
        self.angle_types = np.array(angle_types)
        self.dihedrals = np.array(dihedrals)
        self.dihedral_types = np.array(dihedral_types)

        if len(self.positions) != len(self.atom_types):
            raise Exception("len of positions (%d) and atom types (%d) must match" % (len(self.positions), len(self.atom_types)))
        if len(self.bonds) != len(self.bond_types):
            raise Exception("len of bonds and bond types must match")
        if len(self.angles) != len(self.angle_types):
            raise Exception("len of angles and angle types must match")
        if len(self.dihedrals) != len(self.dihedral_types):
            raise Exception("len of dihedrals and dihedral_types must match")

    @classmethod
    def from_lammps_data(cls, f, atom_format="full"):
        def get_types_tups(arr):
            types = tups = []
            if len(arr) > 0:
                types = arr[:, 1] - 1
                tups = arr[:, 2:] - 1
            return types, tups

        masses = []
        atoms = []
        bonds = []
        angles = []
        dihedrals = []

        sections_handled = ["Atoms", "Bonds", "Angles", "Dihedrals", "Masses"]
        current_section = None
        start_section = False

        for unprocessed_line in f:
            # handle comments
            line = unprocessed_line.split('#')[0].strip()
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
            if current_section == "Masses":
                masses.append(tup[1])
            elif current_section == "Atoms":
                atoms.append(tup)
            elif current_section == "Bonds":
                bonds.append(tup)
            elif current_section == "Angles":
                angles.append(tup)
            elif current_section == "Dihedrals":
                dihedrals.append(tup)

        atom_type_masses = np.array(masses, dtype=float)
        atoms = np.array(atoms, dtype=float)
        bonds = np.array(bonds, dtype=int)
        angles = np.array(angles, dtype=int)
        dihedrals = np.array(dihedrals, dtype=int)

        # note: bond indices in lammps-data file are 1-indexed and we are 0-indexed which is why
        # the bond pairs get a -1
        if atom_format == "atomic":
            atom_types = np.array(atoms[:, 1] - 1, dtype=int)
            atom_tups = atoms[:, 2:5]
        elif atom_format == "full":
            atom_types = np.array(atoms[:, 2] - 1, dtype=int)
            atom_charges = np.array(atoms[:, 3], dtype=float)
            atom_tups = atoms[:, 3:6]

        bond_types, bond_tups = get_types_tups(bonds)
        angle_types, angle_tups = get_types_tups(angles)
        dihedral_types, dihedral_tups = get_types_tups(dihedrals)

        return cls(atom_types=atom_types, positions=atom_tups,
                   bond_types=bond_types, bonds=bond_tups,
                   angle_types=angle_types, angles=angle_tups,
                   dihedral_types=dihedral_types, dihedrals=dihedral_tups,
                   atom_type_masses=atom_type_masses)

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

        f.write("Masses\n\n")
        for i, m in enumerate(self.atom_type_masses):
            f.write(" %d %5.4f\n" % (i + 1, m))

        f.write("\nAtoms\n\n")
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
        return cls(elements=atoms.symbols, positions=atoms.positions, cell=atoms.cell)

    @property
    def elements(self):
        return [self.atom_type_elements[i] for i in self.atom_types]

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

    @property
    def num_atom_types(self):
        return max(self.atom_types) + 1

    def extend_atom_types(self, other):
        self.atom_type_elements = np.append(self.atom_type_elements, other.atom_type_elements)
        self.atom_type_masses = np.append(self.atom_type_masses, other.atom_type_masses)

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

    def _delete_and_reindex_atom_index_array(self, arr, sorted_deleted_indices, secondary_arr=None):
        updated_arr = arr.copy()
        arr_idx_to_delete = []
        for i, atom_idx_tuple in enumerate(arr):
            if np.any([a in sorted_deleted_indices for a in atom_idx_tuple]):
                arr_idx_to_delete.append(i)
        updated_arr = np.delete(updated_arr, arr_idx_to_delete, axis=0)

        # reindex
        for i in sorted_deleted_indices:
            updated_arr = np.subtract(updated_arr, 1, where=updated_arr>i)

        if secondary_arr is not None:
            # remove same indices from secondary array; this is used for types arrays
            secondary_arr = np.delete(secondary_arr, arr_idx_to_delete, axis=0)
            return (updated_arr, secondary_arr)
        else:
            return updated_arr


    def __delitem__(self, indices):
        self.positions = np.delete(self.positions, indices, axis=0)
        self.atom_types = np.delete(self.atom_types, indices, axis=0)

        sorted_indices = sorted(indices, reverse=True)
        if len(self.bonds) > 0:
            self.bonds, self.bond_types = self._delete_and_reindex_atom_index_array(self.bonds, sorted_indices, self.bond_types)
        if len(self.angles) > 0:
            self.angles, self.angle_types = self._delete_and_reindex_atom_index_array(self.angles, sorted_indices, self.angle_types)
        if len(self.dihedrals) > 0:
            self.dihedrals, self.dihedral_types = self._delete_and_reindex_atom_index_array(self.dihedrals, sorted_indices, self.dihedral_types)



    def pop(self, pos=-1):
        del(self, pos)

    def __getitem__(self, i):
        idx = np.array(i, ndmin=1)
        return Atoms(positions=np.take(self.positions, idx, axis=0),
                     atom_types=np.take(self.atom_types, idx, axis=0),
                     atom_type_masses=self.atom_type_masses,
                     atom_type_elements=self.atom_type_elements,
                     cell=self.cell)

    def to_ase(self):
        return ase.Atoms(self.atom_types, positions=self.positions, cell=self.cell)

def find_unchanged_atom_pairs(orig_structure, final_structure, max_delta=1e-5):
    """returns array of tuple pairs, where each pair contains the indices in the original and the final
    structure that match.

    does not work across pbcs"""
    match_pairs = []
    for i, p1 in enumerate(orig_structure.positions):
        for j, p2 in enumerate(final_structure.positions):
            if norm(np.array(p2) - p1) < max_delta and orig_structure.elements[i] == final_structure.elements[j]:
                match_pairs.append((i,j))
                break
    return match_pairs
