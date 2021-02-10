
import io
from io import StringIO

from tests.fixtures import *
from mofun import Atoms

def test_atoms_del__deletes_bonds_attached_to_atoms(linear_cnnc):
    del(linear_cnnc[[1]])
    assert list(linear_cnnc.atom_types) == ["C", "N", "C"]
    assert (linear_cnnc.bonds == [[1,2]]).all()

def test_atoms_del__deletes_angles_attached_to_atoms(linear_cnnc):
    del(linear_cnnc[[0]])
    assert (linear_cnnc.angles == [(0,1,2)]).all()

def test_atoms_del__deletes_dihedrals_attached_to_atoms(linear_cnnc):
    del(linear_cnnc[[1]])
    assert len(linear_cnnc.dihedrals) == 0

def test_atoms_extend__on_nonbonded_structure_reindexes_new_bonds_to_proper_atoms(linear_cnnc):
    linear_cnnc_no_bonds = Atoms('CNNC', positions=[(0., 0., 0), (1.0, 0., 0.), (2.0, 0., 0.), (3.0, 0., 0.)])
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
        uio66_linker_ld = Atoms.from_lammps_data(sin)

    sout = io.StringIO("")
    uio66_linker_ld.to_lammps_data(sout, file_comment="uio66-linker.lammps-data")
    sout.seek(0)
    sin.seek(0)
    assert sout.read() == sin.read()
