

from tests.fixtures import *
from mofun import Atoms

def test_atoms_del__deletes_bonds_attached_to_atoms(linear_cnnc):
    del(linear_cnnc[[1]])
    assert list(linear_cnnc.atom_types) == ["C", "N", "C"]
    assert (linear_cnnc.bonds == np.array([[1,2]])).all()


def test_atoms_extend__reindexes_new_bonds_to_proper_atoms(linear_cnnc):
    double_cnnc = linear_cnnc.copy()
    double_cnnc.extend(linear_cnnc)
    assert list(double_cnnc.atom_types) == ["C", "N", "N", "C"] * 2
    assert (double_cnnc.bonds == np.concatenate([linear_cnnc.bonds, linear_cnnc.bonds + 4])).all()
