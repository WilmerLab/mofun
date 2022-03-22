
import io
from io import StringIO

from numpy.testing import assert_equal as np_assert_equal

from tests.fixtures import *
from mofun import Atoms
from mofun.atoms import find_unchanged_atom_pairs
from mofun.helpers import assert_positions_are_unchanged, assert_structure_positions_are_unchanged


def test_atoms__type_labels_are_inferred_from_elements():
    atoms = Atoms(elements="CNZrPt", positions=random_positions(4))
    assert np.array_equal(atoms.atom_type_labels, ["C", "N", "Zr", "Pt"])

def test_atoms__masses_are_inferred_from_elements():
    atoms = Atoms(elements="CNZrPt", positions=random_positions(4))
    assert np.allclose(atoms.atom_type_masses, [12, 14, 91.2, 195.1], atol=0.1)

def test__delete_and_reindex_atom_index_array():
    a = np.array([[1,2], [3,4], [5,6], [1,6]])
    updated_atoms, _ = Atoms()._delete_and_reindex_atom_index_array(a, [3])
    np_assert_equal(updated_atoms, np.array([[1,2],[4,5],[1,5]]))

def test_atoms_del__deletes_bonds_attached_to_atoms(linear_cnnc):
    del(linear_cnnc[[1]])
    assert list(linear_cnnc.elements) == ["C", "N", "C"]
    assert (linear_cnnc.bonds == [[1,2]]).all()

def test_atoms_del__deletes_types_with_all_topologies(linear_cnnc):
    linear_cnnc.impropers = np.array([(0,1,2,3)])
    linear_cnnc.bond_types = np.array([0, 1, 2])
    linear_cnnc.angle_types = np.array([0, 1])
    linear_cnnc.dihedral_types = np.array([0])
    linear_cnnc.improper_types = np.array([0])
    linear_cnnc.assert_arrays_are_consistent_sizes()
    del(linear_cnnc[[0]])
    assert (linear_cnnc.bond_types == (1, 2)).all()
    assert linear_cnnc.angle_types == (1)
    assert len(linear_cnnc.dihedral_types) == 0
    assert len(linear_cnnc.improper_types) == 0

def test_atoms_del__deletes_angles_attached_to_atoms(linear_cnnc):
    del(linear_cnnc[[0]])
    assert (linear_cnnc.angles == [(0,1,2)]).all()

def test_atoms_del__deletes_dihedrals_attached_to_atoms(linear_cnnc):
    del(linear_cnnc[[1]])
    assert len(linear_cnnc.dihedrals) == 0

def test_atoms_del__deletes_impropers_attached_to_atoms():
    atoms = Atoms(elements="HCHH", positions=random_positions(4), impropers=[(0,1,2,3)], improper_types=[0])
    del(atoms[[1]])
    assert len(atoms.impropers) == 0

def test_atoms_extend__on_nonbonded_structure_reindexes_new_bonds_to_proper_atoms(linear_cnnc):
    linear_cnnc_no_bonds = Atoms(elements='CNNC', positions=[(0., 0., 0), (1.0, 0., 0.), (2.0, 0., 0.), (3.0, 0., 0.)])
    linear_cnnc_no_bonds.extend(linear_cnnc)
    assert np.array_equal(linear_cnnc_no_bonds.bonds, [(4,5), (5,6), (6,7)])

def test_atoms_extend__new_types_come_after_old_types1(linear_cnnc):
    a = Atoms(elements="C", positions=[[0, 0, 0]])
    b = Atoms(elements="H", positions=[[1, 1, 1]])
    a.extend(b)
    assert np.array_equal(a.elements, ["C", "H"])

def test_atoms_extend__new_types_come_after_old_types(linear_cnnc):
    linear_cnnc.impropers = np.array([(0,1,2,3)])

    linear_cnnc.atom_types = np.array([0,1,1,0])
    linear_cnnc.bond_types = np.array([0,1,0])
    linear_cnnc.angle_types = np.array([0,1])
    linear_cnnc.dihedral_types = np.array([0])
    linear_cnnc.improper_types = np.array([0])
    double_cnnc = linear_cnnc.copy()
    double_cnnc.extend(linear_cnnc)
    assert np.array_equal(double_cnnc.atom_types, [0, 1, 1, 0, 2, 3, 3, 2])
    assert np.array_equal(double_cnnc.bond_types, [0, 1, 0, 2, 3, 2])
    assert np.array_equal(double_cnnc.angle_types, [0, 1, 2, 3])
    assert np.array_equal(double_cnnc.dihedral_types, [0, 1])
    assert np.array_equal(double_cnnc.improper_types, [0, 1])
    assert np.array_equal(double_cnnc.elements, ["C", "N", "N", "C"] * 2)

def test_atoms_extend__with_structure_map_reindexes_new_bonds_to_proper_atoms(linear_cnnc):
    fn_pattern = Atoms(elements='NCH', atom_type_labels=["Nx", "Cx", "Hx"],
                positions=[(2.0, 0., 0.), (3.0, 0., 0.), (4.0, 0., 0.)],
                bonds=[(0,1), (1,2)], bond_types=[0] * 2,
                angles=[(0,1,2)], angle_types=[0])
    fn_linear_cnnc = linear_cnnc.copy()
    fn_linear_cnnc.extend(fn_pattern, structure_index_map={0:2, 1:3})

    assert len(fn_linear_cnnc) == 5
    assert np.array_equal(fn_linear_cnnc.bonds, [[0,1], [1,2], [2,3], [3,4]])
    assert np.array_equal(fn_linear_cnnc.angles, [[0,1,2], [1,2,3], [2,3,4]])
    assert np.array_equal(fn_linear_cnnc.atom_types, [0, 1, 2, 3, 4])
    assert np.array_equal(fn_linear_cnnc.atom_type_labels, ["C", "N", "Nx", "Cx", "Hx"])

def test_atoms_extend__reindexes_new_bonds_to_proper_atoms(linear_cnnc):
    double_cnnc = linear_cnnc.copy()
    double_cnnc.extend(linear_cnnc)
    assert np.array_equal(double_cnnc.bonds, [(0,1), (1,2), (2,3), (4,5), (5,6), (6,7)])

def test_atoms_extend__reindexes_new_angles_to_proper_atoms(linear_cnnc):
    double_cnnc = linear_cnnc.copy()
    double_cnnc.extend(linear_cnnc)
    assert (double_cnnc.angles == [(0,1,2), (1,2,3), (4,5,6), (5,6,7)]).all()

def test_atoms_extend__reindexes_new_dihedrals_to_proper_atoms(linear_cnnc):
    double_cnnc = linear_cnnc.copy()
    double_cnnc.extend(linear_cnnc)
    assert (double_cnnc.dihedrals == [(0,1,2,3), (4,5,6,7)]).all()

def test_atoms_extend__reindexes_new_impropers_to_proper_atoms(linear_cnnc):
    linear_cnnc.impropers = np.array([(0,1,2,3)])
    linear_cnnc.improper_types = np.array([0])
    double_cnnc = linear_cnnc.copy()
    double_cnnc.extend(linear_cnnc)
    assert (double_cnnc.impropers == [(0,1,2,3), (4,5,6,7)]).all()

def test_atoms_extend__uses_bonds_regardless_of_atom_order():
    atoms = Atoms(atom_type_elements=["C", "N"], atom_types=[1, 0, 0, 1],
        positions=[[0,0,1], [0,0,2], [0,0,3], [0,0,4]],
        bonds=[[0,1],[1,2],[2,3]], bond_types=[0,0,0]
    )
    atoms2 = atoms.copy()
    atoms2.bonds = np.array([[1,0], [2,1], [3,2]])

    atoms.extend(atoms2, structure_index_map={0:0, 1:1, 2:2, 3:3})

    assert len(atoms.bonds) == 3

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

def test_atoms_getitem__has_all_atom_types_and_charges():
    atoms = Atoms(atom_type_elements=["C", "N"], atom_types=[1, 0, 0, 1], positions=[[0,0,0]] * 4, charges=[1,2,3,4])
    assert atoms[0].elements[0] == "N"
    assert atoms[1].elements[0] == "C"
    assert atoms[0].charges[0] == 1
    assert atoms[1].charges[0] == 2
    assert atoms[(0,1)].elements == ["N", "C"]
    assert atoms[(2,3)].elements == ["C", "N"]
    assert (atoms[(0,1)].charges ==[1, 2]).all()

def test_atoms_replicate__111_is_unchanged(octane):
    reploctane = octane.replicate((1,1,1))
    assert np.array_equal(octane.positions, reploctane.positions)
    assert np.array_equal(octane.atom_types, reploctane.atom_types)
    assert np.array_equal(octane.charges, reploctane.charges)
    assert np.array_equal(octane.groups, reploctane.groups)

def test_atoms_replicate__211_has_replicate_in_x_dim(octane):
    reploctane = octane.replicate((2,1,1))
    assert (octane.positions == reploctane.positions[0:26, :]).all()
    assert (octane.positions[:, 0] + 60 == reploctane.positions[26:52, 0]).all()
    assert (octane.positions[:, 1:] == reploctane.positions[26:52, 1:]).all()

    assert np.array_equal(np.tile(octane.atom_types,2), reploctane.atom_types)
    assert np.array_equal(np.tile(octane.charges,2), reploctane.charges)
    assert np.array_equal(np.tile(octane.groups,2), reploctane.groups)

def test_atoms_replicate__213_has_replicates_in_xz_dims(octane):
    reploctane = octane.replicate((2,1,3))
    na = 26 # atoms in octane

    # check [1, 1, 1]
    assert (octane.positions == reploctane.positions[0:26, :]).all()

    # check [2, 1, 1]
    i = 1
    assert (octane.positions[:, 0] + 60 == reploctane.positions[na*i:na*(i+1), 0]).all()
    assert (octane.positions[:, 1] +  0 == reploctane.positions[na*i:na*(i+1), 1]).all()
    assert (octane.positions[:, 2] +  0 == reploctane.positions[na*i:na*(i+1), 2]).all()

    # check [1, 1, 2]
    i = 2
    assert (octane.positions[:, 0] +  0 == reploctane.positions[na*i:na*(i+1), 0]).all()
    assert (octane.positions[:, 1] +  0 == reploctane.positions[na*i:na*(i+1), 1]).all()
    assert (octane.positions[:, 2] + 60 == reploctane.positions[na*i:na*(i+1), 2]).all()

    # check [2, 1, 2]
    i = 3
    assert (octane.positions[:, 0] + 60 == reploctane.positions[na*i:na*(i+1), 0]).all()
    assert (octane.positions[:, 1] +  0 == reploctane.positions[na*i:na*(i+1), 1]).all()
    assert (octane.positions[:, 2] + 60 == reploctane.positions[na*i:na*(i+1), 2]).all()

    # check [1, 1, 3]
    i = 4
    assert (octane.positions[:, 0] +   0 == reploctane.positions[na*i:na*(i+1), 0]).all()
    assert (octane.positions[:, 1] +   0 == reploctane.positions[na*i:na*(i+1), 1]).all()
    assert (octane.positions[:, 2] + 120 == reploctane.positions[na*i:na*(i+1), 2]).all()

    # check [2, 1, 3]
    i = 5
    assert (octane.positions[:, 0] +  60 == reploctane.positions[na*i:na*(i+1), 0]).all()
    assert (octane.positions[:, 1] +   0 == reploctane.positions[na*i:na*(i+1), 1]).all()
    assert (octane.positions[:, 2] + 120 == reploctane.positions[na*i:na*(i+1), 2]).all()

    assert np.array_equal(np.tile(octane.atom_types, 6), reploctane.atom_types)
    assert np.array_equal(np.tile(octane.charges, 6), reploctane.charges)
    assert np.array_equal(np.tile(octane.groups, 6), reploctane.groups)

def test_atoms_replicate__triclinic_222_has_replicates_in_three_dims():
    a = Atoms(elements="H", positions=[(1, 1, 1)], cell=np.array([[10, 0, 0], [10, 10, 0], [0, 0, 10]]))
    expected = Atoms(elements="HHHHHHHH", cell=np.array([[20, 0, 0], [20, 20, 0], [0, 0, 20]]),
                     positions=[[1,1,1], [11,1,1], [11,11,1], [21,11,1],
                                [1,1,11], [11,1,11], [11,11,11], [21,11,11]])

    ra = a.replicate((2, 2, 2))
    assert len(ra) == 8
    assert_structure_positions_are_unchanged(ra, expected)
    assert np.array_equal(ra.cell, expected.cell)
