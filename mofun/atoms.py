import copy
import io
import os
import pathlib
import sys
import xml.etree.ElementTree as ET

import ase
from ase.formula import Formula
from CifFile import ReadCif as read_cif

import numpy as np
from scipy.linalg import norm
from scipy.spatial.distance import cdist

from mofun.helpers import guess_elements_from_masses, ATOMIC_MASSES, use_or_open

class Atoms:
    """
    An Atoms object is a container for all the information required to keep track of a structure and
    the structure's force field.

    See the documentation on `__init__` or `load` to see how to create an Atoms object.

    An Atoms object is made up primarily of numpy arrays and a couple normal lists.

    Some arrays are per-atom (one entry per atom): `atom_types`, `positions`, `charges`, and
    `groups`.

    Some arrays are per-atom-type (one entry per atom type): `atom_type_masses`, `atom_type_elements`, and
    `atom_type_labels`.

    Some arrays are per-bond (one entry per bond): `bonds`, and `bond_types`.

    Some arrays are per-angle (one entry per angle): `angles`, and `angle_types`.

    Some arrays are per-dihedral (one entry per dihedral): `dihedrals`, and `dihedral_types`.

    Some arrays are per-improper (one entry per improper): `impropers`, and `improper_types`.

    Atoms also stores information on the coefficients needed to define each force field term:
    `pair_coeffs`, `bond_type_coeffs`, `angle_type_coeffs`, `dihedral_type_coeffs`, and
    `improper_type_coeffs`. The \\*\\_coeffs variables are lists of strings, where the item index
    corresponds to the \\*\\_type, and the string is the full LAMMPS coeffs definition string. we do
    not interpret any of the LAMMPS coefficient specifics, we just store it in its original form,
    i.e. this Angle Coeffs section:

    ```
    Angle Coeffs

    1 cosine/periodic  72.500283  -1  1   # C_R O_1 H_
    2 cosine/periodic  277.164705  -1  3   # C_R C_R O_1
    ```

    would be interpreted like this:

    ```
    angle_type_coeffs= ["cosine/periodic  72.500283  -1  1   # C_R O_1 H_",
                        "cosine/periodic  277.164705  -1  3   # C_R C_R O_1"]
    ```"""

    def __init__(self, atom_types=[], positions=[], charges=[], groups=[],
                    elements=[], atom_type_masses=[], atom_type_elements=[], atom_type_labels=[],
                    bonds=[], bond_types=[], angles=[], angle_types=[],
                    dihedrals=[], dihedral_types=[], impropers=[], improper_types=[],
                    pair_coeffs=[], bond_type_coeffs=[], angle_type_coeffs=[],
                    dihedral_type_coeffs=[], improper_type_coeffs=[], cell=None):
        """Create an Atoms object.

        An Atoms object can be created without any atoms using `Atoms()`. For creating more
        interesting Atoms objects, there are a few rules to keep in mind. The parameters
        `atom_types`, `positions`, `charges`, and `groups`  are all lists that should have a
        size equal to the number of atoms in the system. `positions` is mandatory; `charges`, and
        `groups` are optional (both default to 0 for each atom) and there are two ways to
        specify atom types: 1) specify the atom_types and atom_type_elements explicitly, which is
        how the `load_lmpdat` method loads Atoms objects from a LAMMPS data file, or 2) specify
        per-atom elements and and have MOFUN auto-number the atom types for you, which is more
        convenient when specifying small molecules in code or when loading from other file formats
        such as CML or CIF which may store element information but not type information.

        When explicitly setting the types, you must pass `atom_types` and `atom_type_elements`.
        `atom_types` is a list of int type ids >= 0, one for each atom in the system.
        `atom_type_elements` is a list of element names (e.g. "C", "N", "Zr") per _atom type_. For
        example, if your system is propane, your atom_types list could be [0, 1, 1, 1] and your
        atom_type_elements list would then be ["C", "H"].

        To specify per-atom elements, you must pass `elements` with either a list of elements, such
        as `Atoms(elements=["C", "C"], ...)` or with a string `Atoms(elements="CC", ...)`. If you
        use the `elements` parameter, then type ids are automatically generated.

        Passing `atom_type_masses` is optional, and masses will be inferred from the elements if
        missing. If you are using atom types with masses that do not correspond to periodic table
        elements, then you will need to specify the masses explicitly.

        Passing force field term information for `bonds`, `angles`, `dihedrals`, and `impropers` is optional, as well
        as passing force field coefficients for LAMMPS with `pair_coeffs`, `bond_type_coeffs`,
        `angle_type_coeffs`, `dihedral_type_coeffs`, and `improper_type_coeffs`.

        Examples:

        ```
         a = Atoms() # create an empty Atoms object with no atoms
         a = Atoms(atom_types=[0], positions=[[0,0,0]]) # create one atom of type 0
         a = Atoms(elements=["C"], positions=[[0,0,0]]) # create one Carbon
         a = Atoms(elements=["C", "C"], positions=[[0,0,0], [1,0,0]]) # create two Carbons
         a = Atoms(elements="CC", positions=[[0,0,0], [1,0,0]]) # create two Carbons using shorthand element notation
        ```

        Args:
            atom_types (List[int]): list of integer type ids, one per atom.
            positions (List[Tuple[float, float, float]]): list of tuple atom x,y,z coordinates, one per atom.
            charges (List[float]): list of charges, one per atom. Defaults to 0 for each atom if not passed.
            groups (List[int]): list of integer groups, one per atom. For LAMMPS this gets mapped to a "molecule id". Defaults to 0 for each atom if not passed.
            elements (List[str], str): either a list of elements, (e.g. ["C", "H", "H", "H"]) or an element string (e.g. "CHHH")
            atom_type_masses (List[float]): list of atom type masses, one per atom type. If masses are not passed, they will be inferred from the `atom_type_elements`.
            atom_type_elements (List[str]): list of atom type elements, one per atom type.
            atom_type_labels (List[str]):  list of atom type labels, one per atom type. Used in LAMMPS data file line comments as type labels.
            bonds (List[Tuple[int, int]]): list of bond tuples where each tuple defines a pair of atoms that are bonded. Each value in the tuple is an index of an atom in the atom_* lists.
            bond_types (List[int]): list of bond type id ints for each bond defined in `bonds`.
            angles (List[Tuple[int, int, int]]): list of angle tuples where each tuple defines a triplet of atoms making up the angle. Each value in the tuple is an index of an atom in the atom_* lists.
            angle_types (List[int]): list of angle type id ints for each angle defined in `angles`.
            dihedrals (List[Tuple[int, int, int, int]]): list of dihedral tuples where each dihedral defines a quartet of atoms making up the dihedral. Each value in the tuple is an index of an atom in the atom_* lists.
            dihedral_types (List[int]): list of dihedral type id ints for each dihedral defined in `dihedrals`.
            impropers (List[Tuple[int, int, int, int]]): list of improper tuples where each improper defines a quartet of atoms making up the improper. Each value in the tuple is an index of an atom in the atom_* lists.
            improper_types (List[int]): list of improper type id ints for each improper defined in `impropers`.
            pair_coeffs (List[str]): pair coefficients definition string (anything supported by LAMMPS in a data file but without the type id). One per atom type.
            bond_type_coeffs (List[str]): bond coefficients definition string (anything supported by LAMMPS in a data file but without the type id). One per bond type.
            angle_type_coeffs (List[str]): angle coefficients definition string (anything supported by LAMMPS in a data file but without the type id). One per angle type.
            dihedral_type_coeffs (List[str]): dihedral coefficients definition string (anything supported by LAMMPS in a data file but without the type id). One per dihedral type.
            improper_type_coeffs (List[str]): improper coefficients definition string (anything supported by LAMMPS in a data file but without the type id). One per improper type.
            cell (Array(3x3)): 3x3 array of unit cell vectors.

        Returns:
            Atoms: the atoms object.
        """

        self.atom_type_masses = np.array(atom_type_masses, ndmin=1)
        self.positions = np.array(positions, dtype=float, ndmin=1)

        if cell is not None:
            self.cell = np.array(cell)
        else:
            self.cell = None

        self.bonds = np.array(bonds, dtype=int)
        self.bond_types = np.array(bond_types, dtype=int)
        self.angles = np.array(angles, dtype=int)
        self.angle_types = np.array(angle_types, dtype=int)
        self.dihedrals = np.array(dihedrals, dtype=int)
        self.dihedral_types = np.array(dihedral_types, dtype=int)
        self.impropers = np.array(impropers, dtype=int)
        self.improper_types = np.array(improper_types, dtype=int)

        self.pair_coeffs = np.array(pair_coeffs)
        self.bond_type_coeffs = np.array(bond_type_coeffs)
        self.angle_type_coeffs = np.array(angle_type_coeffs)
        self.dihedral_type_coeffs = np.array(dihedral_type_coeffs)
        self.improper_type_coeffs = np.array(improper_type_coeffs)

        if len(charges) > 0:
            self.charges = np.array(charges, dtype=float)
        else:
            self.charges = np.zeros(len(self.positions), dtype=float)

        if len(groups) > 0:
            self.groups = np.array(groups, dtype=int)
        else:
            self.groups = np.zeros(len(self.positions), dtype=int)

        # load atom_types and atom_type_elements
        if len(atom_types) > 0:
            # default case or a __getitem__ subset
            self.atom_types = np.array(atom_types)
            self.atom_type_elements = atom_type_elements
        elif len(elements) > 0:
            # from element array, such as from ASE atoms or read CML
            # i.e. Propane ['C', 'H', 'H', 'H', 'C', 'H', 'H', 'C', 'H', 'H', 'H']:
            # or from element string, i.e. Propane "CHHHCHHCHHH" (shorthand):
            if isinstance(elements, str):
                elements = list(Formula(elements))

            # preserve order of types
            self.atom_type_elements = list(dict.fromkeys(elements).keys())
            self.atom_types = np.array([self.atom_type_elements.index(s) for s in elements])
        else:
            # no atom_type_elements or elements passed
            # this should be the `Atoms()` case; if not, it will fail the asserts below
            self.atom_types = np.array([], ndmin=1)
            self.atom_type_elements = []

        # automatically determine masses from elements if masses are not passed
        if len(self.atom_type_masses) == 0 and len(self.atom_type_elements) > 0:
            self.atom_type_masses = [ATOMIC_MASSES[s] for s in self.atom_type_elements]

        if len(atom_type_labels) > 0:
            self.atom_type_labels = atom_type_labels
        else:
            print("WARNING: using the atom elements as the atom_type_labels since labels were not supplied.", file=sys.stderr)
            # use default atom types equal to the element; this may not be unique!
            self.atom_type_labels = self.atom_type_elements

        self.assert_arrays_are_consistent_sizes()

    def assert_arrays_are_consistent_sizes(self):
        # we always should have these lists, though charges and groups may be zero defaults
        if len(self.positions) != len(self.atom_types):
            raise Exception("len of positions (%d) and atom types (%d) must match" % (len(self.positions), len(self.atom_types)))
        if len(self.positions) != len(self.charges):
            raise Exception("len of positions (%d) and charges (%d) must match" % (len(self.positions), len(self.charges)))
        if len(self.positions) != len(self.groups):
            raise Exception("len of positions (%d) and groups (%d) must match" % (len(self.positions), len(self.groups)))

        # these list pairs should always match
        if len(self.bonds) != len(self.bond_types):
            raise Exception("len of bonds and bond types must match")
        if len(self.angles) != len(self.angle_types):
            raise Exception("len of angles and angle types must match")
        if len(self.dihedrals) != len(self.dihedral_types):
            raise Exception("len of dihedrals and dihedral_types must match")
        if len(self.impropers) != len(self.improper_types):
            raise Exception("len of impropers and improper_types must match")

        # these arrays must have at least as many atom_type_* entries as the num_atom_types. The
        # reason they do not have to match _exactly_ is because a subset of an Atoms object gets
        # _all_ the atom_type_* arrays and does not reindex the atom_types array, so the
        # num_atom_types may be fewer than the number of atom types in the atom_type_* arrays.
        if len(self.atom_type_labels) < self.num_atom_types:
            raise Exception("len of atom_type_labels (%d) must be >= num_atom_types (%d)" % (len(self.atom_type_labels), self.num_atom_types))
        if len(self.atom_type_elements) < self.num_atom_types:
            raise Exception("len of atom_type_elements (%d) must be >= num_atom_types (%d)" % (len(self.atom_type_elements), self.num_atom_types))
        if len(self.atom_type_masses) < self.num_atom_types:
            raise Exception("len of atom_type_masses (%d) must be >= num_atom_types (%d)" % (len(self.atom_type_masses), self.num_atom_types))

    @classmethod
    def load(cls, f, filetype=None, **kwargs):
        """Creates an Atoms object from either a path or a file object and filetype.

        Can load any of the supported types:

        - lammps data file: "lmpdat"
        - cif
        - cml

        Args:
            f (Str or Path or File): either a path to a file or an open File to load from
            filetype (Str): filetype ('lmpdat', 'cif', or 'cml') of passed f File object, or
                explicit filetype to override default filetype implied from file extension.
            kwargs: keyword args passed on to individual load functions.
        """

        fd = None
        path = None

        if isinstance(f, io.TextIOBase):
            fd = f
            if filetype is None:
                raise Exception("If a File object is passed, a filetype must be passed with it")
        else:
            # other cases are treated as either Pathlib path or strings
            path = f
            if filetype is None:
                _, filetype = os.path.splitext(path)
                filetype = filetype[1:]

        if filetype == "lmpdat":
            with use_or_open(fd, path) as fh:
                atoms = cls.load_lmpdat(fh, **kwargs)
                return atoms
        elif filetype == "cml":
            return cls.load_cml(fd or path, **kwargs)
        elif filetype == "cif":
            with use_or_open(fd, path) as fh:
                return cls.load_cif(fh, **kwargs)
        else:
            raise Exception("Unsupported filetype")

    def save(self, f, filetype=None, **kwargs):
        """Saves an Atoms object to either a path or a file object and filetype.

        Can save any of the supported types:

        - lammps data file: "lmpdat"
        - mol

        Args:
            f (Str or Path or File): either a path to a file or an open File to save to
            filetype (Str): filetype ('lmpdat', 'cif', or 'cml') of passed f File object, or
                explicit filetype to override default filetype implied from file extension.
            kwargs: keyword args passed on to individual save functions.
        """

        fd = None
        path = None

        if isinstance(f, io.TextIOBase):
            fd = f
            if filetype is None:
                raise Exception("If a File object is passed, a filetype must be passed with it")
        else:
            # other cases are treated as either Pathlib path or strings
            path = f
            if filetype is None:
                _, filetype = os.path.splitext(path)
                filetype = filetype[1:]

        if filetype == "lmpdat":
            with use_or_open(fd, path, mode='w') as fh:
                atoms = self.save_lmpdat(fh, **kwargs)
                return atoms
        elif filetype == "mol":
            with use_or_open(fd, path, mode='w') as fh:
                return self.save_mol(fh, **kwargs)
        else:
            raise Exception("Unsupported filetype")

    @classmethod
    def load_lmpdat(cls, f, atom_format="full", guess_atol=0.1):
        """Load Atoms object from lammps data file (.lmpdat) format.

        LAMMPS data files store only atom ids and masses, but do not store two other things we need:
        elements and atom type labels. These are the rules for inferring atom type labels and elements.

        In priority order, for elements, we:

        1. guess the elements using the masses by seeing if there is a periodic table element within
        0.1 g/mol of the mass. If any atom types doe not match to an existing periodic table
        element, this method fails.
        2. use the atom ids as the elements (and print a warning).

        In priority order, for atom type labels, we:

        1. use the comments after each line in the Masses section as the atom type. If any line is
        missing a comment, this method fails.
        2. use the elements, if we have them.
        3. use the atom ids (and print a warning).

        Args:
            f (File): File-like object to read from.
            atom_format (str): atom format of lammps data file. Currently supported atom formats are
                'full' and 'atomic'.
            guess_atol (float): absolute tolerance a read mass can differ from a periodic table mass
                and still be considered that element. Default: 0.1.

        Returns:
            Atoms: loaded Atoms object
        """
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
        impropers = []

        pair_coeffs = []
        bond_coeffs = []
        angle_coeffs = []
        dihedral_coeffs = []
        improper_coeffs = []
        atom_type_labels = []
        atom_type_elements = []

        cellx = celly = cellz = 0.0

        sections_handled = ["Pair Coeffs", "Bond Coeffs", "Angle Coeffs", "Dihedral Coeffs",
                            "Improper Coeffs", "Atoms", "Bonds", "Angles", "Dihedrals", "Impropers",
                            "Masses"]
        current_section = None
        start_section = False

        for unprocessed_line in f:
            # handle comments
            comment_string = ""
            comment = None
            if "#" in unprocessed_line:
                line, comment = unprocessed_line.split('#')
                comment = comment.strip()
                comment_string ="   # " + comment
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
                atom_type_labels.append(comment)
            elif current_section == "Pair Coeffs":
                pair_coeffs.append("%s%s" % (" ".join(tup[1:]), comment_string))
            elif current_section == "Bond Coeffs":
                bond_coeffs.append("%s%s" % (" ".join(tup[1:]), comment_string))
            elif current_section == "Angle Coeffs":
                angle_coeffs.append("%s%s" % ("  ".join(tup[1:]), comment_string))
            elif current_section == "Dihedral Coeffs":
                dihedral_coeffs.append("%s%s" % (" ".join(tup[1:]), comment_string))
            elif current_section == "Improper Coeffs":
                improper_coeffs.append("%s%s" % (" ".join(tup[1:]), comment_string))
            elif current_section == "Atoms":
                atoms.append(tup)
            elif current_section == "Bonds":
                bonds.append(tup)
            elif current_section == "Angles":
                angles.append(tup)
            elif current_section == "Dihedrals":
                dihedrals.append(tup)
            elif current_section == "Impropers":
                impropers.append(tup)
            elif current_section is None:
                if "xlo xhi" in line:
                    cellx = float(tup[1]) - float(tup[0])
                elif "ylo yhi" in line:
                    celly = float(tup[1]) - float(tup[0])
                elif "zlo zhi" in line:
                    cellz = float(tup[1]) - float(tup[0])

        cell = None
        if cellx > 0. and celly > 0. and cellz > 0.:
            cell = np.identity(3) * (cellx, celly, cellz)
        atom_type_masses = np.array(masses, dtype=float)
        atoms = np.array(atoms, dtype=float)
        bonds = np.array(bonds, dtype=int)
        angles = np.array(angles, dtype=int)
        dihedrals = np.array(dihedrals, dtype=int)
        impropers = np.array(impropers, dtype=int)

        # note: bond indices in lmpdat file are 1-indexed and we are 0-indexed which is why
        # the bond pairs get a -1
        if atom_format == "atomic":
            atom_types = np.array(atoms[:, 1] - 1, dtype=int)
            groups = np.zeros(len(atom_types))
            charges = np.zeros(len(atom_types))
            atom_tups = atoms[:, 2:5]
        elif atom_format == "full":
            groups = np.array(atoms[:, 1] - 1, dtype=int)
            atom_types = np.array(atoms[:, 2] - 1, dtype=int)
            charges = np.array(atoms[:, 3], dtype=float)
            atom_tups = atoms[:, 4:7]

        # guess the atom elements; if this fails, use the atoms ids as the elements
        try:
            atom_type_elements = guess_elements_from_masses(atom_type_masses, max_delta=guess_atol)
        except Exception:
            print("WARNING: using type ids for elements since some masses do not correspond to periodic table elements within the set tolerance.", file=sys.stderr)
            atom_type_elements = [str(i + 1) for i in range(len(masses))]

        # infer the atom type labels
        if (atom_type_labels.count(None) == 0):
            # then loading atom types from the labels worked
            pass
        else:
            print("WARNING: using elements for atom type labels since there is not an atom type label comment for every atom type in the Masses section.", file=sys.stderr)
            atom_type_labels = atom_type_elements.copy()


        bond_types, bond_tups = get_types_tups(bonds)
        angle_types, angle_tups = get_types_tups(angles)
        dihedral_types, dihedral_tups = get_types_tups(dihedrals)
        improper_types, improper_tups = get_types_tups(impropers)

        return cls(atom_types=atom_types, positions=atom_tups, charges=charges,
                   atom_type_masses=atom_type_masses, atom_type_elements=atom_type_elements,
                   bond_types=bond_types, bonds=bond_tups,
                   angle_types=angle_types, angles=angle_tups,
                   dihedral_types=dihedral_types, dihedrals=dihedral_tups,
                   improper_types=improper_types, impropers=improper_tups,
                   pair_coeffs=pair_coeffs, bond_type_coeffs=bond_coeffs,
                   angle_type_coeffs=angle_coeffs, dihedral_type_coeffs=dihedral_coeffs,
                   improper_type_coeffs=improper_coeffs, atom_type_labels=atom_type_labels,
                   groups=groups, cell=cell)

    def save_lmpdat(self, f, atom_format="full", file_comment=""):
        """Saves a lmpdat file

        Args:
            f (File): an open file to write to
            atom_format (str): LAMMPS atom format. Supports only 'atomic' and 'full'.
            file_comment (str): written in first line of output file.
        """

        f.write("%s (written by mofun)\n\n" % file_comment)

        f.write('%d atoms\n' % len(self.atom_types))
        f.write('%d bonds\n' % len(self.bond_types))
        f.write('%d angles\n' % len(self.angle_types))
        f.write('%d dihedrals\n' % len(self.dihedral_types))
        f.write('%d impropers\n' % len(self.improper_types))
        f.write("\n")

        if self.num_atom_types > 0:
            f.write('%d atom types\n' % self.num_atom_types)
        if self.num_bond_types > 0:
            f.write('%d bond types\n' % self.num_bond_types)
        if self.num_angle_types > 0:
            f.write('%d angle types\n' % self.num_angle_types)
        if self.num_dihedral_types > 0:
            f.write('%d dihedral types\n' % self.num_dihedral_types)
        if self.num_improper_types > 0:
            f.write('%d improper types\n' % self.num_improper_types)

        # TODO: support triclinic
        if self.cell is not None and self.cell.shape == (3,3):
            xlohi, ylohi, zlohi = zip([0,0,0], np.diag(self.cell))
            f.write(" %10.6f %10.6f xlo xhi\n" % xlohi)
            f.write(" %10.6f %10.6f ylo yhi\n" % ylohi)
            f.write(" %10.6f %10.6f zlo zhi\n" % zlohi)

        f.write("\nMasses\n\n")
        for i, m in enumerate(self.atom_type_masses):
            f.write(" %d %10.6f   # %s\n" % (i + 1, m, self.label_atoms(i)))

        if len(self.pair_coeffs) > 0:
            f.write('\nPair Coeffs\n\n')
            for i, coeffs in enumerate(self.pair_coeffs):
                f.write(' %d %s\n' % (i + 1, coeffs))

        if len(self.bond_type_coeffs) > 0:
            f.write('\nBond Coeffs\n\n')
            for i, coeffs in enumerate(self.bond_type_coeffs):
                f.write(' %d %s\n' % (i + 1, coeffs))

        if len(self.angle_type_coeffs) > 0:
            f.write('\nAngle Coeffs\n\n')
            for i, coeffs in enumerate(self.angle_type_coeffs):
                f.write(' %d %s\n' % (i + 1, coeffs))

        if len(self.dihedral_type_coeffs) > 0:
            f.write('\nDihedral Coeffs\n\n')
            for i, coeffs in enumerate(self.dihedral_type_coeffs):
                f.write(' %d %s\n' % (i + 1, coeffs))

        if len(self.improper_type_coeffs) > 0:
            f.write('\nImproper Coeffs\n\n')
            for i, coeffs in enumerate(self.improper_type_coeffs):
                f.write(' %d %s\n' % (i + 1, coeffs))

        f.write("\nAtoms\n\n")
        if atom_format == "atomic":
            for i, (x, y, z) in enumerate(self.positions):
                f.write(" %d %d %10.6f %10.6f %10.6f   # %s\n" % (i + 1, self.atom_types[i] + 1, x, y, z, self.label_atoms(self.atom_types[i])))
        elif atom_format == "full":
            for i, (x, y, z) in enumerate(self.positions):
                f.write(" %d %d %d %10.6f %10.6f %10.6f %10.6f   # %s\n" %
                    (i + 1, self.groups[i] + 1, self.atom_types[i] + 1, self.charges[i], x, y, z, self.label_atoms(self.atom_types[i])))

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

        if len(self.impropers) > 0:
            f.write("\nImpropers\n\n")
            for i, tup in enumerate(self.impropers):
                f.write(" %d %d %d %d %d %d   # %s\n" % (i + 1, self.improper_types[i] + 1, *(np.array(tup) + 1), self.label_atoms(tup, atom_indices=True)))


    def save_mol(self, f, file_comment=""):
        """Writes .mol file for structural information."""

        f.write(" Molecule_name: %s\n" % file_comment)
        f.write("\n")
        f.write("  Coord_Info: Listed Cartesian None\n")
        f.write("        %d\n" % len(self))

        for i, (x, y, z) in enumerate(self.positions):
            f.write("%6d %10.4f %10.4f %10.4f  %5s %10.8f  0  0\n" % (i + 1, x, y, z,
                self.elements[i], self.charges[i]))

        if self.cell is not None:
            f.write("\n\n\n")
            f.write("  Fundcell_Info: Listed\n")
            f.write("        %10.4f       %10.4f       %10.4f\n" % tuple(np.diag(self.cell)))
            f.write("           90.0000          90.0000          90.0000\n")
            f.write("           0.00000          0.00000          0.00000\n")
            f.write("        %10.4f       %10.4f       %10.4f\n" % tuple(np.diag(self.cell)))


    @classmethod
    def load_cif(cls, f):
        """Loads a CIF file, including bonding information.

        Args:
            f (File): File-like object to read from.

        Returns:
            Atoms: loaded Atoms object
        """

        def has_all_tags(block, tags):
            return np.array([block.has_key(tag) for tag in tags]).all()

        # PyCifRw supports file descriptors and path strings, but doesn't not support PathLib paths.
        if isinstance(f, pathlib.PurePath):
            f = str(f)

        cf = read_cif(f)
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

        charges = []
        if block.has_key('_atom_site_charge'):
            charges = block['_atom_site_charge']

        bonds = []
        bond_tags = ["_geom_bond_atom_site_label_1", "_geom_bond_atom_site_label_2"]
        if has_all_tags(block, bond_tags):
            cifbonds = zip(*[block[lbl] for lbl in bond_tags])
            bonds = [(atom_name.index(a), atom_name.index(b)) for (a,b) in cifbonds]
            print("WARNING: cif read doesn't handle bond types at present; bonding info is discarded. Use LAMMPS data file format if you need bonds", file=sys.stderr)
            bonds = []

        cell=None
        cell_tags = ['_cell_length_a', '_cell_length_b', '_cell_length_c', '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma']
        if has_all_tags(block, cell_tags):
            a, b, c, alpha, beta, gamma = [float(block[tag]) for tag in cell_tags]
            # TODO: Fix for triclinic
            if alpha != 90. or beta != 90 or gamma != 90.:
                raise Exception("No support for non orthorhombic UCs at the moment!")

            cell=np.identity(3) * (a, b, c)
            if use_fract_coords:
                positions *= (a,b,c)

        return cls(elements=atom_types, positions=positions, cell=cell, charges=charges)

    @classmethod
    def load_cml(cls, f, verbose=False):
        """Loads a CML file, including bonding information.

        Args:
            f (Path or File): Path or File-like object to read from.

        Returns:
            Atoms: loaded Atoms object
        """

        tree = ET.parse(f)
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
        if verbose:
            print("Found %d atoms: %s" % (len(elements), elements))
            print("Found %d atom positions: %s" % (len(positions), positions))
            print("Found %d bonds: %s" % (len(bonds), bonds))
            print("Found %d bond_types: %s" % (len(bond_types), bond_types))

        return cls(elements=elements, positions=positions, bonds=bonds, bond_types=bond_types)

    @classmethod
    def from_ase_atoms(cls, atoms):
        """Create an Atoms object from an ASE Atoms object.

        Only supports importing the atom positions, elements, and the unit cell.

        Args:
            atoms (ASE Atoms object): atoms to load from

        Returns:
            Atoms: atoms loaded from an ASE Atoms object.
        """
        return cls(elements=atoms.symbols, positions=atoms.positions, cell=atoms.cell)


    def label_atoms(self, atoms, atom_indices=False):
        if not hasattr(atoms, '__iter__'):
            atoms = [atoms]

        if atom_indices:
            atoms = [self.atom_types[i] for i in atoms]

        return " ".join([self.atom_type_labels[x] for x in atoms])

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
        if len(self.atom_types) == 0:
            return 0
        return len(self.atom_type_elements)

    @property
    def num_bond_types(self):
        if len(self.bond_types) == 0:
            return 0
        return len(self.bond_type_coeffs) or max(self.bond_types) + 1

    @property
    def num_angle_types(self):
        if len(self.angle_types) == 0:
            return 0
        return len(self.angle_type_coeffs) or max(self.angle_types) + 1

    @property
    def num_dihedral_types(self):
        if len(self.dihedral_types) == 0:
            return 0
        return len(self.dihedral_type_coeffs) or max(self.dihedral_types) + 1

    @property
    def num_improper_types(self):
        if len(self.improper_types) == 0:
            return 0
        return len(self.improper_type_coeffs) or max(self.improper_types) + 1

    def extend_types(self, other):
        offsets = (self.num_atom_types, self.num_bond_types,
                   self.num_angle_types, self.num_dihedral_types, self.num_improper_types)

        self.atom_type_elements = np.append(self.atom_type_elements, other.atom_type_elements)
        self.atom_type_masses = np.append(self.atom_type_masses, other.atom_type_masses)
        self.atom_type_labels = np.append(self.atom_type_labels, other.atom_type_labels)
        self.pair_coeffs = np.append(self.pair_coeffs, other.pair_coeffs)

        self.bond_type_coeffs = np.append(self.bond_type_coeffs, other.bond_type_coeffs)
        self.angle_type_coeffs = np.append(self.angle_type_coeffs, other.angle_type_coeffs)
        self.dihedral_type_coeffs = np.append(self.dihedral_type_coeffs, other.dihedral_type_coeffs)
        self.improper_type_coeffs = np.append(self.improper_type_coeffs, other.improper_type_coeffs)

        return offsets


    def extend(self, other, offsets=None, structure_index_map={}, verbose=False):
        """Adds other Atoms object's arrays to its own.

        The default behavior is for all the types and params from other structure to be appended to
        this structure.

        Alternatively, an offsets tuple may be passed with the results of calling extend_types(). No
        new types will be added, but the newly added atoms, bonds, etc will refer to types by their
        value in the other Atoms object plus the offset. Use this when you are adding the same set
        of atoms multiple times, or if your other atoms already share the same type ids as this
        object. For the later case, the tuple (0,0,0,0) may be passed in.

        Args:
            other (Atoms): atoms to add to self
            offsets: an offsets tuple with the results of calling extend_types().
            structure_index_map: dictionary where key is an index in other and value is an index in
                self, where entries only exist if the position and element of the entries are
                identical and can be considered to be the same atom.
            verbose (bool): print debugging info.
        """

        atom_idx_offset = len(self.positions)
        if offsets is None:
            if verbose:
                print("auto offset: extending types")
            offsets = self.extend_types(other)

        # update atom types for atoms that are already part of self Atoms object
        for other_index, self_index in structure_index_map.items():
            self.atom_types[self_index] = other.atom_types[other_index] + offsets[0]

        # add atoms that are not part of self Atoms object
        atoms_to_add = [i for i in range(len(other)) if i not in structure_index_map.keys()]
        self.positions = np.append(self.positions, other.positions[atoms_to_add], axis=0)
        self.atom_types = np.append(self.atom_types, other.atom_types[atoms_to_add] + offsets[0])
        self.charges = np.append(self.charges, other.charges[atoms_to_add])
        self.groups = np.append(self.groups, other.groups[atoms_to_add])

        # update structure index map
        structure_index_map2 = {a:i + atom_idx_offset for i,a in enumerate(atoms_to_add)}
        structure_index_map2.update(structure_index_map)
        convert2structureindex = np.vectorize(structure_index_map2.get)

        def find_existing_topo(topo, new_topo):
            """ find existing topo tuples between the same atoms as a new topo set.

            Used to allow an override of an existing force field term by finding old terms between
            the same atoms to delete"""
            if len(topo) == 0:
                return []
            return list(np.nonzero(cdist(topo, new_topo, 'cityblock') == 0)[0])

        if len(other.bonds) > 0:
            new_bonds = convert2structureindex(other.bonds)
            existing_bond_indices = find_existing_topo(self.bonds, new_bonds)
            self.bonds = np.append(self.bonds, new_bonds).reshape((-1,2))
            self.bond_types = np.append(self.bond_types, other.bond_types + offsets[1])
            self.bonds = np.delete(self.bonds, existing_bond_indices, axis=0)
            self.bond_types = np.delete(self.bond_types, existing_bond_indices)

        if len(other.angles) > 0:
            new_angles = convert2structureindex(other.angles)
            existing_angle_indices = find_existing_topo(self.angles, new_angles)
            self.angles = np.append(self.angles, new_angles).reshape((-1,3))
            self.angle_types = np.append(self.angle_types, other.angle_types + offsets[2])
            self.angles = np.delete(self.angles, existing_angle_indices, axis=0)
            self.angle_types = np.delete(self.angle_types, existing_angle_indices)

        if len(other.dihedrals) > 0:
            new_dihedrals = convert2structureindex(other.dihedrals)
            existing_dihedral_indices = find_existing_topo(self.dihedrals, new_dihedrals)
            self.dihedrals = np.append(self.dihedrals, new_dihedrals).reshape((-1,4))
            self.dihedral_types = np.append(self.dihedral_types, other.dihedral_types + offsets[3])
            self.dihedrals = np.delete(self.dihedrals, existing_dihedral_indices, axis=0)
            self.dihedral_types = np.delete(self.dihedral_types, existing_dihedral_indices)

        if len(other.impropers) > 0:
            new_impropers = convert2structureindex(other.impropers)
            existing_improper_indices = find_existing_topo(self.impropers, new_impropers)
            self.impropers = np.append(self.impropers, new_impropers).reshape((-1,4))
            self.improper_types = np.append(self.improper_types, other.improper_types + offsets[4])
            self.impropers = np.delete(self.impropers, existing_improper_indices, axis=0)
            self.improper_types = np.delete(self.improper_types, existing_improper_indices)

        self.assert_arrays_are_consistent_sizes()

    def replicate(self, repldims=(1,1,1)):
        """Replicate atoms object across xyz dimensions

        Warnings:
        * does not magically handle any bonds that may cross periodic boundary conditions!

        Args:
            repldims (Tuple(int, int, int)): number of times to replicate in each dimension

        Returns:
            Atoms: replicated atoms.
        """
        if self.cell is None:
            raise Exception("Can't replicate if no unit cell has been defined")

        repl_atoms = self.copy()
        ucmults = np.array(np.meshgrid(*[range(r) for r in repldims])).T.reshape(-1, 3)
        ucmults = ucmults[np.any(ucmults != 0, axis=1)] # remove [0,0,0] since in copy
        for ucmult in ucmults:
            transatoms = self.copy()
            transatoms.translate(np.matmul(transatoms.cell.T, ucmult))
            repl_atoms.extend(transatoms, offsets=(0,0,0,0))

        repl_atoms.cell = self.cell * repldims
        return repl_atoms

    def _delete_and_reindex_atom_index_array(self, arr, sorted_deleted_indices, secondary_arr=None):
        updated_arr = arr.copy()
        arr_idx_to_delete = []
        for i, atom_idx_tuple in enumerate(arr):
            if np.any([a in sorted_deleted_indices for a in atom_idx_tuple]):
                arr_idx_to_delete.append(i)
        updated_arr = np.delete(updated_arr, arr_idx_to_delete, axis=0)

        # reindex
        for i in sorted_deleted_indices:
            np.subtract(updated_arr, 1, out=updated_arr, where=updated_arr>i)

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
        self.groups = np.delete(self.groups, indices, axis=0)

        sorted_indices = sorted(indices, reverse=True)
        if len(self.bonds) > 0:
            self.bonds, self.bond_types = self._delete_and_reindex_atom_index_array(self.bonds, sorted_indices, self.bond_types)
        if len(self.angles) > 0:
            self.angles, self.angle_types = self._delete_and_reindex_atom_index_array(self.angles, sorted_indices, self.angle_types)
        if len(self.dihedrals) > 0:
            self.dihedrals, self.dihedral_types = self._delete_and_reindex_atom_index_array(self.dihedrals, sorted_indices, self.dihedral_types)
        if len(self.impropers) > 0:
            self.impropers, self.improper_types = self._delete_and_reindex_atom_index_array(self.impropers, sorted_indices, self.improper_types)

        self.assert_arrays_are_consistent_sizes()

    def pop(self, pos=-1):
        del(self, pos)

    def __getitem__(self, i):
        idx = np.array(i, ndmin=1)
        return Atoms(positions=np.take(self.positions, idx, axis=0),
                     atom_types=np.take(self.atom_types, idx, axis=0),
                     charges=np.take(self.charges, idx, axis=0),
                     atom_type_masses=self.atom_type_masses,
                     atom_type_elements=self.atom_type_elements,
                     groups=np.take(self.groups, idx, axis=0),
                     cell=self.cell)

    def to_ase(self):
        """Convert to ASE atoms object.

        Only supports export of the positions and elements.
        """
        kwargs = dict(positions=self.positions)
        if self.cell is not None:
            kwargs['cell'] = self.cell
            kwargs['pbc'] = True
        return ase.Atoms(self.elements, **kwargs)

def find_unchanged_atom_pairs(orig_structure, final_structure, max_delta=1e-5):
    """Returns array of tuple pairs, where each pair contains the indices in the original and the final
    structure that match.

    Does not work across PBCs."""
    match_pairs = []
    for i, p1 in enumerate(orig_structure.positions):
        for j, p2 in enumerate(final_structure.positions):
            if norm(np.array(p2) - p1) < max_delta and orig_structure.elements[i] == final_structure.elements[j]:
                match_pairs.append((i,j))
                break
    return match_pairs
