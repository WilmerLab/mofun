import io
from io import StringIO
from pathlib import Path

import ase.io

from tests.fixtures import *
# import tests.test_atoms_load_save
from mofun import Atoms
from mofun.helpers import assert_structure_positions_are_unchanged


###### load/save LMPDAT format
def check_uio66_arb_terms_arrays(atoms):
    assert len(atoms.atom_type_masses) == 4
    assert len(atoms.pair_coeffs) == 4
    assert len(atoms.bond_type_coeffs) == 1
    assert len(atoms.angle_type_coeffs) == 2
    assert len(atoms.dihedral_type_coeffs) == 3
    assert len(atoms.improper_type_coeffs) == 1
    assert atoms.positions.shape == (16,3)
    assert len(atoms.groups) == 16
    assert len(atoms.charges) == 16
    assert len(atoms.atom_types) == 16
    assert len(atoms.bonds) == 4
    assert len(atoms.angles) == 8
    assert len(atoms.dihedrals) == 14
    assert len(atoms.impropers) == 2

def test_atoms_load_lmpdat__has_arrays_of_right_size():
    with Path("tests/uio66/uio66-linker-arb-terms.lmpdat").open() as f:
        atoms = Atoms.load_lmpdat(f, atom_format="full")

    check_uio66_arb_terms_arrays(atoms)

def test_atoms_load_lmpdat__atomic_masses_are_guessed_correctly():
    with Path("tests/test_atoms_load_save/atomic-masses.lmpdat").open() as f:
        atoms = Atoms.load_lmpdat(f, atom_format="full")
    assert np.array_equal(atoms.atom_type_elements, ["C", "N"])

def test_atoms_load_lmpdat__if_masses_are_nonatomic_then_elements_are_type_ids():
    with Path("tests/test_atoms_load_save/nonatomic-masses.lmpdat").open() as f:
        atoms = Atoms.load_lmpdat(f, atom_format="full")
    assert np.array_equal(atoms.atom_type_masses, [12., 1000.])
    assert np.array_equal(atoms.atom_type_elements, ["1", "2"])

def test_atoms_load_lmpdat__atom_type_labels_are_loaded_from_comments():
    with Path("tests/test_atoms_load_save/atomic-masses.lmpdat").open() as f:
        atoms = Atoms.load_lmpdat(f, atom_format="full")
    assert np.array_equal(atoms.atom_type_masses, [12., 14.])
    assert np.array_equal(atoms.atom_type_labels, ["C1", "N1"])

def test_atoms_load_lmpdat__atom_type_labels_are_loaded_from_elements_if_no_comments():
    with Path("tests/test_atoms_load_save/atomic-masses-no-comments.lmpdat").open() as f:
        atoms = Atoms.load_lmpdat(f, atom_format="full")
    assert np.array_equal(atoms.atom_type_masses, [12., 14.])
    assert np.array_equal(atoms.atom_type_labels, ["C", "N"])

def test_atoms_load_lmpdat__if_masses_are_nonatomic_with_no_comments_then_elements_and_labels_are_type_ids():
    with Path("tests/test_atoms_load_save/nonatomic-masses-no-comments.lmpdat").open() as f:
        atoms = Atoms.load_lmpdat(f, atom_format="full")
    assert np.array_equal(atoms.atom_type_masses, [12., 1000.])
    assert np.array_equal(atoms.atom_type_elements, ["1", "2"])
    assert np.array_equal(atoms.atom_type_labels, ["1", "2"])

def test_atoms_save_lmpdat__outputs_file_identical_to_input_file():
    with Path("tests/uio66/uio66-linker-arb-terms.lmpdat").open() as f:
        sin = StringIO(f.read())
        uio66_linker_ld = Atoms.load_lmpdat(sin, atom_format="full")

    sout = io.StringIO("")
    uio66_linker_ld.save_lmpdat(sout, file_comment="uio66-linker-arb-terms.lmpdat")

    # output file code, in case we need to update the lmpdat file because of new format changes
    # with open("uio66-linker-arb-terms.lmpdat", "w") as f:
    #     sout.seek(0)
    #     f.write(sout.read())

    sout.seek(0)
    sin.seek(0)
    assert sout.read() == sin.read()

def test_atoms_save_lmpdat__triclinic_file_outputs_file_identical_to_input_file():
    with Path("tests/uio66/uio66-triclinic.lmpdat").open() as f:
        sin = StringIO(f.read())
        uio66_linker_ld = Atoms.load_lmpdat(sin, atom_format="full")

    sout = io.StringIO("")
    uio66_linker_ld.save_lmpdat(sout, file_comment="uio66-triclinic.lmpdat")
    sout.seek(0)
    sin.seek(0)
    assert sout.read() == sin.read()

###### load CML format
def test_atoms_load_cml__w_path_loads_elements_bonds():
    atoms = Atoms.load_cml("tests/uio66/uio66-linker.cml")

    assert atoms.elements == ["C", "O", "O", "C", "C", "H", "H", "C", "C", "C", "C", "H", "H", "O", "C", "O"]
    assert (atoms.bonds == [[0, 1], [10, 11], [0, 2], [0, 3], [3, 10], [9, 10], [3, 4], [9, 12],
                           [8, 9], [4, 5], [4, 7], [7, 8], [8, 14], [6, 7], [13, 14], [14, 15]]).all()

def test_atoms_load_cml__w_file_loads_elements_bonds():
    with open("tests/uio66/uio66-linker.cml", 'r') as f:
        atoms = Atoms.load_cml(f)

    assert atoms.elements == ["C", "O", "O", "C", "C", "H", "H", "C", "C", "C", "C", "H", "H", "O", "C", "O"]
    assert (atoms.bonds == [[0, 1], [10, 11], [0, 2], [0, 3], [3, 10], [9, 10], [3, 4], [9, 12],
                           [8, 9], [4, 5], [4, 7], [7, 8], [8, 14], [6, 7], [13, 14], [14, 15]]).all()

###### load / save CIF format
def test_atoms_load_p1_cif__loads_elements():
    with Path("tests/uio66/uio66-linker.cif").open() as fd:
        atoms = Atoms.load_p1_cif(fd)

    assert atoms.elements == ["C", "O", "O", "C", "C", "H", "H", "C", "C", "C", "C", "H", "H", "O", "C", "O"]

def test_atoms_load_p1_cif__raises_exception_if_not_p1():
    with pytest.raises(Exception):
        with Path("tests/othermofs/SIFSIX-3-Cu-P4.cif").open() as fd:
            atoms = Atoms.load_p1_cif(fd)

def test_atoms_load_cif__same_cell_and_pos_as_ase_io():
    uio66cif = Atoms.load_p1_cif("tests/uio66/uio66-triclinic.cif")
    uio66asecif = ase.io.read("tests/uio66/uio66-triclinic.cif")
    assert np.allclose(uio66cif.cell, uio66asecif.cell)
    assert np.allclose(uio66cif.positions, uio66asecif.positions)

def test_atoms_load_p1_cif__same_cell_and_pos_as_load_lmpdat():
    uio66cif = Atoms.load_p1_cif("tests/uio66/uio66-triclinic.cif")
    uio66lmp = Atoms.load("tests/uio66/uio66-triclinic.lmpdat", atom_format="full")
    assert np.allclose(uio66cif.cell, uio66lmp.cell)
    assert_structure_positions_are_unchanged(uio66cif, uio66lmp)

def test_atoms_load_p1_cif__outputs_file_identical_to_input_file():
    with Path("tests/uio66/uio66-linker-arb-terms.cif").open() as f:
        sin = StringIO(f.read())
        cif = Atoms.load_p1_cif(sin)

    sout = io.StringIO("")
    cif.save_p1_cif(sout)

    # output file code, in case we need to update the lmpdat file because of new format changes
    with open("test-01.cif", "w") as f:
        sout.seek(0)
        f.write(sout.read())

    sout.seek(0)
    sin.seek(0)
    assert sout.read() == sin.read()


###### load method
def test_atoms_load__loads_lmpdat_from_file_or_path():
    with Path("tests/uio66/uio66-linker.lmpdat") as path:
        atoms = Atoms.load(path, atom_format="atomic")
        assert len(atoms) == 16

    with Path("tests/uio66/uio66-linker.lmpdat").open() as fd:
        atoms = Atoms.load(fd, filetype="lmpdat", atom_format="atomic")
        assert len(atoms) == 16

def test_atoms_load__loads_cml_from_file_or_path():
    with Path("tests/uio66/uio66-linker.cml") as path:
        atoms = Atoms.load(path)
        assert len(atoms) == 16

    with Path("tests/uio66/uio66-linker.cml").open() as fd:
        atoms = Atoms.load(fd, filetype="cml")
        assert len(atoms) == 16

def test_atoms_load__loads_cif_from_file_or_path():
    with Path("tests/uio66/uio66-linker.cif") as path:
        atoms = Atoms.load(path)
        assert len(atoms) == 16

    with Path("tests/uio66/uio66-linker.cif").open() as fd:
        atoms = Atoms.load(fd, filetype="cif")
        assert len(atoms) == 16
