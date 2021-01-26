import importlib
from importlib import resources

import ase
from ase import io
import numpy as np
import pytest

import tests

def assert_positions_should_be_unchanged(orig_structure, final_structure, decimal_points=5):
    p = orig_structure.positions.round(decimal_points)
    p_ordered = p[np.lexsort((p[:,0], p[:,1], p[:,2]))]
    new_p = final_structure.positions.round(decimal_points)
    new_p_ordered = new_p[np.lexsort((new_p[:,0], new_p[:,1], new_p[:,2]))]
    for i, p1 in enumerate(p_ordered):
        assert (p1 == new_p_ordered[i]).all()


@pytest.fixture
def octane():
    # CH3 CH2 CH2 CH2 CH2 CH2 CH2 CH3 #
    with importlib.resources.path(tests, "octane.xyz") as path:
        structure = ase.io.read(path)
        structure.positions += 30
        structure.set_cell(60 * np.identity(3))
        yield structure

@pytest.fixture
def hkust1_cif():
    with importlib.resources.path(tests, "HKUST-1_withbonds.cif") as path:
        yield ase.io.read(path)

@pytest.fixture
def hkust1_3x3x3_xyz():
    with importlib.resources.path(tests, "HKUST-1_3x3x3.xyz") as path:
        structure = ase.io.read(path)
        structure.set_cell(79.0290 * np.identity(3))
        yield structure

@pytest.fixture
def hkust1_3x3x3_cif():
    with importlib.resources.path(tests, "HKUST-1_3x3x3.cif") as path:
        yield ase.io.read(path)

@pytest.fixture
def benzene():
    with importlib.resources.path(tests, "benzene.xyz") as path:
        yield ase.io.read(path)

@pytest.fixture
def benzene_at_origin():
    with importlib.resources.path(tests, "benzene_at_origin.xyz") as path:
        yield ase.io.read(path)

@pytest.fixture
def benzene_rotated():
    with importlib.resources.path(tests, "benzene_rotated.xyz") as path:
        yield ase.io.read(path)
