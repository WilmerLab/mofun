import copy
import itertools
import xml.etree.ElementTree as ET

import ase
from ase.formula import Formula
from CifFile import ReadCif as read_cif
import networkx as nx
import numpy as np
from scipy.linalg import norm

from mofun.helpers import guess_elements_from_masses, ATOMIC_MASSES

class Atoms:
    """
indexed one per atom:
    atom_types=[], types for each atom
    positions=[]: coordinates (x,y,z) for each atom
    charges=[]: charges for each atom
    atom_group=[]: which "group" atom is part of. For LAMMPS, gets mapped to a molecule id.

indexed one per atom type:
    atom_type_masses=[]: mass for each atom type
    atom_type_elements=[]: periodic table element for each atom type
    atom_type_labels=[]: (optional) type labels to be output in line comments for lammps output.

indexed one per bond:
    bonds=[]: atom index tuple for each bond, i.e. (1,2) would be a bond connecting atoms 1 and 2.
    bond_types=[]: type of the bond

indexed one per angle:
    angles=[]: atom index tuple for each angle, i.e. (1,2,3 )
    angle_types=[]: type of the angle

indexed one per dihedral:
    dihedrals=[]: atom index tuple for each dihedral, i.e. (1,2,3,4)
    dihedral_types=[]: type of the dihedral

cell=[]: unit cell matrix (same definition as in ASE)

*_params:

    pair_params=[]
    bond_type_params=[]
    angle_type_params=[]
    dihedral_type_params=[]

    The *_params variables are lists of strings, where the item index corresponds to the atom type, and
    the string is the full lampps coeffs def string. we do not interpret any of the LAMMPS coefficient
    specifics, we just store it in its original form, i.e. this Angle Coeffs section

    ```
    Angle Coeffs

     1 cosine/periodic  72.500283  -1  1   # C_R O_1 H_
     2 cosine/periodic  277.164705  -1  3   # C_R C_R O_1
     ```
    would be interpreted like this:

    ```
    angle_type_params= ["cosine/periodic  72.500283  -1  1   # C_R O_1 H_",
                        "cosine/periodic  277.164705  -1  3   # C_R C_R O_1"]
    ```"""

    def __init__(self, atom_types=[], positions=[], charges=[], bond_types=[], bonds=[],
                    angle_types=[], angles=[], dihedrals=[], dihedral_types=[],
                    atom_type_masses=[], cell=[], elements=[], atom_type_elements=[],
                    pair_params=[], bond_type_params=[], angle_type_params=[],
                    dihedral_type_params=[], atom_type_labels=[], atom_groups=[]):

        self.atom_type_masses = np.array(atom_type_masses, ndmin=1)
        self.positions = np.array(positions, dtype=float, ndmin=1)
        if len(charges) == 0:
            self.charges = np.zeros(len(self.positions), dtype=float)
        else:
            self.charges = np.array(charges, dtype=float)

        if len(atom_groups) == 0:
            self.atom_groups = np.zeros(len(self.positions), dtype=int)
        else:
            self.atom_groups = np.array(atom_groups, dtype=int)

        self.atom_type_labels=atom_type_labels
        if len(atom_type_elements) > 0:
            # this is a __getitem__ subset
            self.atom_types = np.array(atom_types)
            self.atom_type_elements = atom_type_elements
        elif len(atom_types) > 0:
            # from atom type ids and masses, such as from LAMMPS:
            self.atom_types = np.array(atom_types, ndmin=1)
            self.atom_type_elements = guess_elements_from_masses(self.atom_type_masses)
        elif len(elements) > 0:
            # from element array, such as from ASE atoms or read CML
            # i.e. Propane ['C', 'H', 'H', 'H', 'C', 'H', 'H', 'C', 'H', 'H', 'H']:
            if isinstance(elements, str):
                # from element string, i.e. Propane "CHHHCHHCHHH" (shorthand):
                elements = list(Formula(elements))

            # preserve order of types
            self.atom_type_elements = list(dict.fromkeys(elements).keys())
            self.atom_types = np.array([self.atom_type_elements.index(s) for s in elements])
            self.atom_type_masses = [ATOMIC_MASSES[s] for s in self.atom_type_elements]
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

        self.pair_params = np.array(pair_params)
        self.bond_type_params = np.array(bond_type_params)
        self.angle_type_params = np.array(angle_type_params)
        self.dihedral_type_params = np.array(dihedral_type_params)

        self.assert_arrays_are_consistent_sizes()

    def assert_arrays_are_consistent_sizes(self):
        if len(self.positions) != len(self.atom_types):
            raise Exception("len of positions (%d) and atom types (%d) must match" % (len(self.positions), len(self.atom_types)))
        if len(self.positions) != len(self.charges):
            raise Exception("len of positions (%d) and charges (%d) must match" % (len(self.positions), len(self.charges)))

        if len(self.bonds) != len(self.bond_types):
            raise Exception("len of bonds and bond types must match")
        if len(self.angles) != len(self.angle_types):
            raise Exception("len of angles and angle types must match")
        if len(self.dihedrals) != len(self.dihedral_types):
            raise Exception("len of dihedrals and dihedral_types must match")

    @classmethod
    def from_lammps_data(cls, f, atom_format="full", atom_type_labels=[]):
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

        pair_coeffs = []
        bond_coeffs = []
        angle_coeffs = []
        dihedral_coeffs = []

        sections_handled = ["Pair Coeffs", "Bond Coeffs", "Angle Coeffs", "Dihedral Coeffs",
                            "Atoms", "Bonds", "Angles", "Dihedrals", "Masses"]
        current_section = None
        start_section = False

        for unprocessed_line in f:
            # handle comments
            comment = ""
            if "#" in unprocessed_line:
                line, comment = unprocessed_line.split('#')
                comment ="   #" + comment.rstrip()
            else:
                line = unprocessed_line.split('#')[0]
            line = line.strip()
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
            elif current_section == "Pair Coeffs":
                pair_coeffs.append("%s%s" % (" ".join(tup[1:]), comment))
            elif current_section == "Bond Coeffs":
                bond_coeffs.append("%s%s" % (" ".join(tup[1:]), comment))
            elif current_section == "Angle Coeffs":
                angle_coeffs.append("%s%s" % ("  ".join(tup[1:]), comment))
            elif current_section == "Dihedral Coeffs":
                dihedral_coeffs.append("%s%s" % (" ".join(tup[1:]), comment))
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
            charges = np.zeros(len(atom_types))
            atom_tups = atoms[:, 2:5]
        elif atom_format == "full":
            atom_types = np.array(atoms[:, 2] - 1, dtype=int)
            charges = np.array(atoms[:, 3], dtype=float)
            atom_tups = atoms[:, 4:7]

        bond_types, bond_tups = get_types_tups(bonds)
        angle_types, angle_tups = get_types_tups(angles)
        dihedral_types, dihedral_tups = get_types_tups(dihedrals)

        return cls(atom_types=atom_types, positions=atom_tups, charges=charges,
                   bond_types=bond_types, bonds=bond_tups,
                   angle_types=angle_types, angles=angle_tups,
                   dihedral_types=dihedral_types, dihedrals=dihedral_tups,
                   atom_type_masses=atom_type_masses,
                   pair_params=pair_coeffs, bond_type_params=bond_coeffs,
                   angle_type_params=angle_coeffs, dihedral_type_params=dihedral_coeffs,
                   atom_type_labels=atom_type_labels)

    def to_lammps_data(self, f, atom_format="full", file_comment=""):
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

        f.write("\nMasses\n\n")
        for i, m in enumerate(self.atom_type_masses):
            f.write(" %d %5.4f   # %s\n" % (i + 1, m, self.label_atoms(i)))

        if len(self.pair_params) > 0:
            f.write('\nPair Coeffs\n\n')
            for i, params in enumerate(self.pair_params):
                f.write(' %d %s\n' % (i + 1, params))

        if len(self.bond_type_params) > 0:
            f.write('\nBond Coeffs\n\n')
            for i, params in enumerate(self.bond_type_params):
                f.write(' %d %s\n' % (i + 1, params))

        if len(self.angle_type_params) > 0:
            f.write('\nAngle Coeffs\n\n')
            for i, params in enumerate(self.angle_type_params):
                f.write(' %d %s\n' % (i + 1, params))

        if len(self.dihedral_type_params) > 0:
            f.write('\nDihedral Coeffs\n\n')
            for i, params in enumerate(self.dihedral_type_params):
                f.write(' %d %s\n' % (i + 1, params))

        f.write("\nAtoms\n\n")
        if atom_format == "atomic":
            for i, (x, y, z) in enumerate(self.positions):
                f.write(" %d %d %10.6f %10.6f %10.6f   # %s\n" % (i + 1, self.atom_types[i] + 1, x, y, z, self.label_atoms(self.atom_types[i])))
        elif atom_format == "full":
            for i, (x, y, z) in enumerate(self.positions):
                f.write(" %d %d %d %10.6f %10.6f %10.6f %10.6f   # %s\n" %
                    (i + 1, self.atom_groups[i] + 1, self.atom_types[i] + 1, self.charges[i], x, y, z, self.label_atoms(self.atom_types[i])))

        if len(self.bonds) > 0:
            f.write("\nBonds\n\n")
            for i, tup in enumerate(self.bonds):
                f.write(" %d %d %d %d   # %s\n" % (i + 1, self.bond_types[i] + 1, *(np.array(tup) + 1), self.label_atoms(tup, atom_indices=True)))

        if len(self.angles) > 0:
            f.write("\nAngles\n\n")
            for i, tup in enumerate(self.angles):
                f.write(" %d %d %d %d %d   # %s\n" % (i + 1, self.angle_types[i] + 1, *(np.array(tup) + 1), self.label_atoms(tup, atom_indices=True)))

        if len(self.dihedrals) > 0:
            f.write("\nDihedrals\n\n")
            for i, tup in enumerate(self.dihedrals):
                f.write(" %d %d %d %d %d %d   # %s\n" % (i + 1, self.dihedral_types[i] + 1, *(np.array(tup) + 1), self.label_atoms(tup, atom_indices=True)))

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
        positions = np.array([x,y,z], dtype=float).T

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

        return cls(elements=atom_types, positions=positions, cell=cell)

    @classmethod
    def from_cml(cls, path):
        tree = ET.parse(path)
        root = tree.getroot()

        atom_dicts = [a.attrib for a in root.findall('.//atom')]
        atom_tuples = [(a['id'], a['elementType'],
                       float(a['x3']), float(a['y3']), float(a['z3'])) for a in atom_dicts]
        ids, elements, x, y, z = zip(*atom_tuples)
        id_to_idx = {id:i for i, id in enumerate(ids)}
        positions = np.array([x,y,z]).T

        bond_dicts = [a.attrib for a in root.findall('.//bond')]
        bond_tuples = [(a['atomRefs2'].split(), float(a['order'])) for a in bond_dicts]
        bonds_by_ids, bond_orders = zip(*bond_tuples)
        bonds = [(id_to_idx[b1], id_to_idx[b2]) for (b1,b2) in bonds_by_ids]
        bond_types = [0 for b in bonds]
        return cls(elements=elements, positions=positions, bonds=bonds, bond_types=bond_types)

    @classmethod
    def from_ase_atoms(cls, atoms):
        return cls(elements=atoms.symbols, positions=atoms.positions, cell=atoms.cell)


    def label_atoms(self, atoms, atom_indices=False):
        if not hasattr(atoms, '__iter__'):
            atoms = [atoms]

        if atom_indices:
            atoms = [self.atom_types[i] for i in atoms]

        return " ".join([self.atom_type_labels[x] for x in atoms])

    def retype_atoms_from_uff_types(self, new_types):
        """ takes a list of new_types that are strings, converts to integer types, and populates
        atom_type_labels"""

        ptable_order = lambda x: list(ATOMIC_MASSES.keys()).index(x.split("_")[0])
        unique_types = list(set(new_types))

        # sort by string ordering, so types like 'C_1', 'C_2', 'C_3', 'C_R' will show up in order
        unique_types.sort()
        # sort by periodic element # order
        unique_types.sort(key=ptable_order)

        self.atom_type_labels = unique_types
        self.atom_type_elements = [s.split("_")[0] for s in unique_types]
        self.atom_type_masses = [ATOMIC_MASSES[s] for s in self.atom_type_elements]

        self.atom_types = [unique_types.index(s) for s in new_types]

    @property
    def elements(self):
        return [self.atom_type_elements[i] for i in self.atom_types]

    @property
    def symbols(self):
        return self.elements

    def copy(self):
        return copy.deepcopy(self)

    def translate(self, delta):
        if self.positions is not None and len(self) > 0:
            self.positions += delta

    def __len__(self):
        return len(self.positions)

    @property
    def num_atom_types(self):
        return 0 if len(self.atom_types) == 0 else max(self.atom_types) + 1

    @property
    def num_bond_types(self):
        return 0 if len(self.bond_types) == 0 else max(self.bond_types) + 1

    @property
    def num_angle_types(self):
        return 0 if len(self.angle_types) == 0 else max(self.angle_types) + 1

    @property
    def num_dihedral_types(self):
        return 0 if len(self.dihedral_types) == 0 else max(self.dihedral_types) + 1

    def extend_types(self, other):
        offsets = (self.num_atom_types, self.num_bond_types,
                   self.num_angle_types, self.num_dihedral_types)

        self.atom_type_elements = np.append(self.atom_type_elements, other.atom_type_elements)
        self.atom_type_masses = np.append(self.atom_type_masses, other.atom_type_masses)
        self.atom_type_labels = np.append(self.atom_type_labels, other.atom_type_labels)
        self.pair_params = np.append(self.pair_params, other.pair_params)

        self.bond_type_params = np.append(self.bond_type_params, other.bond_type_params)
        self.angle_type_params = np.append(self.angle_type_params, other.angle_type_params)
        self.dihedral_type_params = np.append(self.dihedral_type_params, other.dihedral_type_params)

        return offsets


    def extend(self, other, offsets=None):
        """ adds other Atoms object's arrays to its own.

        The Default behavior is for all the types and params from other structure to be appended to
        this structure.

        Alternatively, an offsets tuple may be passed with the results of calling extend_types(). No
        new types will be added, but the newly added atoms, bonds, etc will refer to types by their
        value in the other Atoms object plus the offset. Use this when you are adding the same set
        of atoms multiple times, or if your other atoms already share the same type ids as this
        object. For the later case, the tuple (0,0,0,0) may be passed in.
        """
        atom_idx_offset = len(self.positions)
        if offsets is None:
            print("auto offset: extending types")
            offsets = self.extend_types(other)

        self.positions = np.append(self.positions, other.positions, axis=0)
        self.atom_types = np.append(self.atom_types, other.atom_types + offsets[0])
        self.charges = np.append(self.charges, other.charges)
        self.atom_groups = np.append(self.atom_groups, other.atom_groups)

        self.bonds = np.append(self.bonds, other.bonds + atom_idx_offset).reshape((-1,2))
        self.bond_types = np.append(self.bond_types, other.bond_types + offsets[1])

        self.angles = np.append(self.angles, other.angles + atom_idx_offset).reshape((-1,3))
        self.angle_types = np.append(self.angle_types, other.angle_types + offsets[2])

        self.dihedrals = np.append(self.dihedrals, other.dihedrals + atom_idx_offset).reshape((-1,4))
        self.dihedral_types = np.append(self.dihedral_types, other.dihedral_types + offsets[3])



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
        self.charges = np.delete(self.charges, indices, axis=0)
        self.atom_groups = np.delete(self.atom_groups, indices, axis=0)

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
                     charges=np.take(self.charges, idx, axis=0),
                     atom_type_masses=self.atom_type_masses,
                     atom_type_elements=self.atom_type_elements,
                     atom_groups=self.atom_groups,
                     cell=self.cell)

    def to_ase(self):
        kwargs = dict(positions=self.positions)
        if self.cell is not None and len(self.cell) > 0:
            kwargs['cell'] = self.cell
        return ase.Atoms(self.elements, **kwargs)

    def calc_angles(self):
        g = nx.Graph()
        g.add_edges_from([tuple(x) for x in self.bonds])

        angles = []
        for n in g.nodes:
            angles += [(a, n, b) for (a,b) in itertools.combinations(g.neighbors(n), 2)]
        self.angles = np.array(angles)

    def calc_dihedrals(self):
        g = nx.Graph()
        g.add_edges_from([tuple(x) for x in self.bonds])

        dihedrals = []
        for a, b in g.edges:
            a_neighbors = list(g.adj[a])
            a_neighbors.remove(b)
            b_neighbors = list(g.adj[b])
            b_neighbors.remove(a)

            dihedrals += [(a1, a, b, b1) for a1 in a_neighbors for b1 in b_neighbors]
        self.dihedrals = np.array(dihedrals)


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
