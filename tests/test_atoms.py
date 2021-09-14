
import io
from io import StringIO

from tests.fixtures import *
from mofun import Atoms
from mofun.atoms import find_unchanged_atom_pairs


def test_atoms__type_labels_are_inferred_from_elements():
    atoms = Atoms(elements="CNZrPt", positions=random_positions(4))
    assert np.array_equal(atoms.atom_type_labels, ["C", "N", "Zr", "Pt"])

def test_atoms__masses_are_inferred_from_elements():
    atoms = Atoms(elements="CNZrPt", positions=random_positions(4))
    assert np.allclose(atoms.atom_type_masses, [12, 14, 91.2, 195.1], atol=0.1)

def test__delete_and_reindex_atom_index_array():
    a = np.array([[1,2], [3,4], [5,6], [1,6]])
    updated_atoms = Atoms()._delete_and_reindex_atom_index_array(a, [3])
    assert(updated_atoms == np.array([[1,2],[4,5],[1,5]])).all()

def test_atoms_del__deletes_bonds_attached_to_atoms(linear_cnnc):
    del(linear_cnnc[[1]])
    assert list(linear_cnnc.elements) == ["C", "N", "C"]
    assert (linear_cnnc.bonds == [[1,2]]).all()

def test_atoms_del__deletes_types_with_all_topologies(linear_cnnc):
    linear_cnnc.impropers = [(0,1,2,3)]

    linear_cnnc.bond_types = [0, 1, 2]
    linear_cnnc.angle_types = [0, 1]
    linear_cnnc.dihedral_types = [0]
    linear_cnnc.improper_types = [0]
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
    linear_cnnc.impropers = [(0,1,2,3)]

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
    linear_cnnc.impropers = [(0,1,2,3)]
    linear_cnnc.improper_types = [0]
    double_cnnc = linear_cnnc.copy()
    double_cnnc.extend(linear_cnnc)
    assert (double_cnnc.impropers == [(0,1,2,3), (4,5,6,7)]).all()

def test_atoms_save_lmpdat__from_load_cif_is_successful():
    with importlib.resources.path(tests, "uio66.cif") as path:
        uio66 = Atoms.load_cif(path)
        uio66.atom_type_labels = uio66.atom_type_elements

    sout = io.StringIO("")
    uio66.save_lmpdat(sout)

def test_atoms_load_lmpdat__uio66_has_arrays_of_right_size():
    with importlib.resources.open_text(tests, "uio66-F.lmpdat") as f:
        atoms = Atoms.load_lmpdat(f, atom_format="full", use_comment_for_type_labels=True)

    assert len(atoms.atom_type_masses) == 4
    assert len(atoms.pair_coeffs) == 4
    assert len(atoms.bond_type_coeffs) == 2
    assert len(atoms.angle_type_coeffs) == 2
    assert atoms.positions.shape == (16,3)
    assert len(atoms.groups) == 16
    assert len(atoms.charges) == 16
    assert len(atoms.atom_types) == 16
    assert len(atoms.bonds) == 4
    assert len(atoms.angles) == 8


def test_atoms_save_lmpdat__with_no_atom_type_labels_outputs_file_with_type_id_labels():
    with importlib.resources.open_text(tests, "uio66-linker-arb-terms-no-labels.lmpdat") as f:
        atoms = Atoms.load_lmpdat(f, atom_format="full", use_ids_for_type_labels_and_elements=True)

    sout = io.StringIO("")
    atoms.save_lmpdat(sout, file_comment="uio66-linker-arb-terms-no-labels.lmpdat")

    # output file code, in case we need to update the lmpdat file because of new format changes
    # with open("uio66-linker.lmpdat", "w") as f:
    #     sout.seek(0)
    #     f.write(sout.read())

    with importlib.resources.open_text(tests, "uio66-linker-arb-terms-atom-type-id-labels.lmpdat") as f:
        sout.seek(0)
        assert sout.read() == f.read()

def test_atoms_load_lmpdat__no_atom_type_labels():
    with importlib.resources.open_text(tests, "uio66-linker-arb-terms-no-labels.lmpdat") as f:
        atoms = Atoms.load_lmpdat(f, atom_format="full", use_ids_for_type_labels_and_elements=True)


def test_atoms_load_lmpdat__use_comment_for_type_labels_with_no_atom_type_labels_raises_exception():
    with importlib.resources.open_text(tests, "uio66-linker-arb-terms-no-labels.lmpdat") as f:
        with pytest.raises(Exception):
            atoms = Atoms.load_lmpdat(f, atom_format="full", use_comment_for_type_labels=True)

def test_atoms_save_lmpdat__output_file_identical_to_one_read():
    with importlib.resources.open_text(tests, "uio66-linker-arb-terms.lmpdat") as f:
        sin = StringIO(f.read())
        uio66_linker_ld = Atoms.load_lmpdat(sin, atom_format="full", use_comment_for_type_labels=True)

    sout = io.StringIO("")
    uio66_linker_ld.save_lmpdat(sout, file_comment="uio66-linker-arb-terms.lmpdat")

    # output file code, in case we need to update the lmpdat file because of new format changes
    # with open("uio66-linker-arb-terms.lmpdat", "w") as f:
    #     sout.seek(0)
    #     f.write(sout.read())

    sout.seek(0)
    sin.seek(0)
    assert sout.read() == sin.read()

def test_atoms_load__loads_lmpdat_from_file_or_path():
    with importlib.resources.path(tests, "uio66-linker.lmpdat") as path:
        atoms = Atoms.load(path, atom_format="atomic")
        assert len(atoms) == 16

    with importlib.resources.open_text(tests, "uio66-linker.lmpdat") as fd:
        atoms = Atoms.load(fd, filetype="lmpdat", atom_format="atomic")
        assert len(atoms) == 16

def test_atoms_load__loads_cml():
    with importlib.resources.path(tests, "uio66-linker.cml") as path:
        atoms = Atoms.load(path)
        assert len(atoms) == 16

def test_atoms_load__loads_cif_from_file_or_path():
    with importlib.resources.path(tests, "uio66-linker.cif") as path:
        atoms = Atoms.load(path)
        assert len(atoms) == 16

    with importlib.resources.open_text(tests, "uio66-linker.cif") as fd:
        atoms = Atoms.load(fd, filetype="cif")
        assert len(atoms) == 16

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

def test_atoms_load_cml__loads_elements_bonds():
    with importlib.resources.path(tests, "uio66-linker.cml") as path:
        atoms = Atoms.load_cml(path)

    assert atoms.elements == ["C", "O", "O", "C", "C", "H", "H", "C", "C", "C", "C", "H", "H", "O", "C", "O"]
    assert (atoms.bonds == [[0, 1], [10, 11], [0, 2], [0, 3], [3, 10], [9, 10], [3, 4], [9, 12],
                           [8, 9], [4, 5], [4, 7], [7, 8], [8, 14], [6, 7], [13, 14], [14, 15]]).all()

def test_atoms_load_cif__loads_elements():
    with importlib.resources.open_text(tests, "uio66-linker.cif") as fd:
        atoms = Atoms.load_cif(fd)

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
