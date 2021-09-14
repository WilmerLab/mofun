import io
from io import StringIO
from pathlib import Path

from tests.fixtures import *
# import tests.test_atoms_load_save
from mofun import Atoms


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
    with importlib.resources.open_text(tests, "uio66-linker-arb-terms.lmpdat") as f:
        atoms = Atoms.load_lmpdat(f, atom_format="full", use_comment_for_type_labels=True)

    check_uio66_arb_terms_arrays(atoms)

def test_atoms_load_lmpdat__no_atom_type_labels_uses_type_ids():
    with importlib.resources.open_text(tests, "uio66-linker-arb-terms-no-labels.lmpdat") as f:
        atoms = Atoms.load_lmpdat(f, atom_format="full", use_ids_for_type_labels_and_elements=True)

    check_uio66_arb_terms_arrays(atoms)
    assert np.array_equal(atoms.atom_type_labels, ["1", "2", "3", "4"])
    assert np.array_equal(atoms.atom_type_elements, ["1", "2", "3", "4"])

def test_atoms_load_lmpdat__when_no_atom_type_labels_use_comment_for_type_labels_raises_exception():
    with Path("tests/test_atoms_load_save/nonatomic-masses.lmpdat").open() as f:
        with pytest.raises(Exception):
            atoms = Atoms.load_lmpdat(f, atom_format="full", use_comment_for_type_labels=True)

def test_atoms_load_lmpdat__nonatomic_masses_raise_exception():
    with Path("tests/test_atoms_load_save/nonatomic-masses.lmpdat").open() as f:
        with pytest.raises(Exception):
            atoms = Atoms.load_lmpdat(f, atom_format="full")

def test_atoms_load_lmpdat__nonatomic_masses_with_use_ids_for_type_labels_and_elements_is_OK():
    with Path("tests/test_atoms_load_save/nonatomic-masses.lmpdat").open() as f:
        atoms = Atoms.load_lmpdat(f, atom_format="full", use_ids_for_type_labels_and_elements=True)

    assert np.array_equal(atoms.atom_type_masses, [12., 1000.])


def test_atoms_save_lmpdat__outputs_file_identical_to_input_file():
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

###### load CML format
def test_atoms_load_cml__loads_elements_bonds():
    with importlib.resources.path(tests, "uio66-linker.cml") as path:
        atoms = Atoms.load_cml(path)

    assert atoms.elements == ["C", "O", "O", "C", "C", "H", "H", "C", "C", "C", "C", "H", "H", "O", "C", "O"]
    assert (atoms.bonds == [[0, 1], [10, 11], [0, 2], [0, 3], [3, 10], [9, 10], [3, 4], [9, 12],
                           [8, 9], [4, 5], [4, 7], [7, 8], [8, 14], [6, 7], [13, 14], [14, 15]]).all()

###### load CIF format
def test_atoms_load_cif__loads_elements():
    with importlib.resources.open_text(tests, "uio66-linker.cif") as fd:
        atoms = Atoms.load_cif(fd)

    assert atoms.elements == ["C", "O", "O", "C", "C", "H", "H", "C", "C", "C", "C", "H", "H", "O", "C", "O"]


###### load method
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
