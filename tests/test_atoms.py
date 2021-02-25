
import io
from io import StringIO

from tests.fixtures import *
from mofun import Atoms
from mofun.atoms import find_unchanged_atom_pairs

def test_atoms_del__deletes_bonds_attached_to_atoms(linear_cnnc):
    del(linear_cnnc[[1]])
    assert list(linear_cnnc.elements) == ["C", "N", "C"]
    assert (linear_cnnc.bonds == [[1,2]]).all()

def test_atoms_del__deletes_types_with_all_topologies(linear_cnnc):
    linear_cnnc.bond_types = [0, 1, 2]
    linear_cnnc.angle_types = [0, 1]
    linear_cnnc.dihedral_types = [0]
    del(linear_cnnc[[0]])
    assert (linear_cnnc.bond_types == (1, 2)).all()
    assert linear_cnnc.angle_types == (1)
    assert len(linear_cnnc.dihedral_types) == 0

def test_atoms_del__deletes_angles_attached_to_atoms(linear_cnnc):
    del(linear_cnnc[[0]])
    assert (linear_cnnc.angles == [(0,1,2)]).all()

def test_atoms_del__deletes_dihedrals_attached_to_atoms(linear_cnnc):
    del(linear_cnnc[[1]])
    assert len(linear_cnnc.dihedrals) == 0

def test_atoms_extend__on_nonbonded_structure_reindexes_new_bonds_to_proper_atoms(linear_cnnc):
    linear_cnnc_no_bonds = Atoms(elements='CNNC', positions=[(0., 0., 0), (1.0, 0., 0.), (2.0, 0., 0.), (3.0, 0., 0.)])
    linear_cnnc_no_bonds.extend(linear_cnnc)
    assert np.array_equal(linear_cnnc_no_bonds.bonds, [(4,5), (5,6), (6,7)])

def test_atoms_extend__reindexes_new_bonds_to_proper_atoms(linear_cnnc):
    double_cnnc = linear_cnnc.copy()
    double_cnnc.extend(linear_cnnc)
    assert (double_cnnc.bonds == [(0,1), (1,2), (2,3), (4,5), (5,6), (6,7)]).all()

def test_atoms_extend__reindexes_new_angles_to_proper_atoms(linear_cnnc):
    double_cnnc = linear_cnnc.copy()
    double_cnnc.extend(linear_cnnc)
    assert (double_cnnc.angles == [(0,1,2), (1,2,3), (4,5,6), (5,6,7)]).all()

def test_atoms_extend__reindexes_new_dihedrals_to_proper_atoms(linear_cnnc):
    double_cnnc = linear_cnnc.copy()
    double_cnnc.extend(linear_cnnc)
    assert (double_cnnc.dihedrals == [(0,1,2,3), (4,5,6,7)]).all()

def test_atoms_to_lammps_data__output_file_identical_to_one_read():
    with importlib.resources.open_text(tests, "uio66-linker.lammps-data") as f:
        sin = StringIO(f.read())
        uio66_linker_ld = Atoms.from_lammps_data(sin, atom_format="atomic")

    sout = io.StringIO("")
    uio66_linker_ld.to_lammps_data(sout, file_comment="uio66-linker.lammps-data")
    sout.seek(0)
    sin.seek(0)
    assert sout.read() == sin.read()

def test_find_unchanged_atom_pairs__same_structure_is_unchanged(linear_cnnc):
    assert find_unchanged_atom_pairs(linear_cnnc, linear_cnnc) == [(0,0), (1,1), (2,2), (3,3)]

def test_find_unchanged_atom_pairs__different_atom_type_is_changed(linear_cnnc):
    cncc = linear_cnnc.copy()
    cncc.atom_types[2] = cncc.atom_type_elements.index("C")
    assert find_unchanged_atom_pairs(linear_cnnc, cncc) == [(0,0), (1,1), (3,3)]

def test_find_unchanged_atom_pairs__different_position_is_changed(linear_cnnc):
    offset_linear_cnns = linear_cnnc.copy()
    offset_linear_cnns.positions[2] += 0.5
    assert find_unchanged_atom_pairs(linear_cnnc, offset_linear_cnns) == [(0,0), (1,1), (3,3)]

def test_atoms_elements__finds_cnnc_for_masses_12_14():
    atoms = Atoms(atom_type_masses=[12.0, 14.0], atom_types=[0, 1, 1, 0], positions=[[0,0,0]] * 4)
    linear_cnnc.elements = ["C", "N", "N", "C"]


def test_atoms_getitem__has_all_atom_types():
    atoms = Atoms(atom_type_masses=[12.0, 14.0], atom_types=[1, 0, 0, 1], positions=[[0,0,0]] * 4)
    assert atoms[0].elements[0] == "N"
    assert atoms[1].elements[0] == "C"
    assert atoms[(0,1)].elements == ["N", "C"]
    assert atoms[(2,3)].elements == ["C", "N"]

def test_atoms_from_cml__loads_elements_bonds():
    with importlib.resources.path(tests, "uio66.cml") as path:
        atoms = Atoms.from_cml(path)

    assert atoms.elements == ["C", "O", "O", "C", "C", "H", "H", "C", "C", "C", "C", "H", "H", "O", "C", "O"]
    assert (atoms.bonds == [[0, 1], [10, 11], [0, 2], [0, 3], [3, 10], [9, 10], [3, 4], [9, 12],
                           [8, 9], [4, 5], [4, 7], [7, 8], [8, 14], [6, 7], [13, 14], [14, 15]]).all()

def test_atoms_from_cif__loads_elements():
    with importlib.resources.open_text(tests, "uio66-linker.cif") as fd:
        atoms = Atoms.from_cif(fd)

    assert atoms.elements == ["C", "O", "O", "C", "C", "H", "H", "C", "C", "C", "C", "H", "H", "O", "C", "O"]

def test_atoms_calc_angles__ethane_has_12_angles():
    ethane = Atoms(elements='HHHCCHHH', positions=[(1., 1., 1)] * 8,
                bonds=[(0,3), (1,3), (2,3), (3,4), (4,5), (4,6), (4,7)], bond_types=[0] * 7)

    ethane.calc_angles()
    unique_angles = set([tuple(x) for x in ethane.angles])
    assert len(ethane.angles) == 12
    assert unique_angles == {(0,3,1), (0,3,2), (1,3,2), (0,3,4), (1,3,4), (2,3,4),
                             (3,4,5), (3,4,6), (3,4,7), (5,4,6), (5,4,7), (6,4,7)}

def test_atoms_calc_dihedrals__ethane_has_9_dihedrals():
    ethane = Atoms(elements='HHHCCHHH', positions=[(1., 1., 1)] * 8,
                bonds=[(0,3), (1,3), (2,3), (3,4), (4,5), (4,6), (4,7)], bond_types=[0] * 7)

    ethane.calc_dihedrals()
    unique_dihedrals = set([tuple(x) for x in ethane.dihedrals])
    assert len(ethane.dihedrals) == 9
    assert unique_dihedrals == {(0,3,4,5), (1,3,4,5), (2,3,4,5),
                                (0,3,4,6), (1,3,4,6), (2,3,4,6),
                                (0,3,4,7), (1,3,4,7), (2,3,4,7)}
