

from tests.fixtures import *
from mofun import Atoms

def test_atoms_del__deletes_bonds_attached_to_atoms(linear_cnnc):
    del(linear_cnnc[[1]])
    assert list(linear_cnnc.atom_types) == ["C", "N", "C"]
    assert (linear_cnnc.bonds == np.array([[1,2]])).all()
