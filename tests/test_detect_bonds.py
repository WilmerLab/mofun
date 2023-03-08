import numpy as np
from numpy.testing import assert_equal as np_assert_equal

from mofun import Atoms
from mofun.detect_bonds import detect_bonds, COVALENT_RADII

def test_detect_bonds__bonds_carbons_lt_1_97_A_apart():
    structure = Atoms(elements="CCCCC", positions=[[0,0,0], [0.3,0,0], [2,0,0], [3.98,0,0], [5,0,0]])
    bonds = detect_bonds(structure)
    np_assert_equal(bonds, [[0,1],[1,2],[3,4]])

def test_detect_bonds__metal_li_li_has_no_buffer():
    r = COVALENT_RADII["Fe"]
    structure = Atoms(elements="FeFeFe", positions=[[0,0,0], [2*r-0.001,0,0], [4*r,0,0]])
    bonds = detect_bonds(structure)
    np_assert_equal(bonds, [[0,1]])

def test_detect_bonds__bonds_across_periodic_boundaries():
    structure = Atoms(elements="CCC", cell=10*np.identity(3),
                      positions=[[0.2, 0.2, 0.2], [9.8, 0.2, 0.2], [9.8, 9.8, 9.8]])
    bonds = detect_bonds(structure)
    np_assert_equal(bonds, [[0,1], [0,2], [1,2]])

