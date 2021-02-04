import importlib
from importlib import resources
from math import sqrt

import ase
from ase import io, Atoms
import numpy as np
from numpy.linalg import norm
import pytest
from pytest import approx

import tests

sqrt2_2 = sqrt(2) / 2
sqrt3_2 = sqrt(3) / 2

def assert_positions_are_unchanged(orig_structure, final_structure, max_delta=1e-5, verbose=False):
    p = orig_structure.positions
    p_ordered = p[np.lexsort((p[:,0], p[:,1], p[:,2]))]
    new_p = final_structure.positions
    new_p_ordered = list(new_p[np.lexsort((new_p[:,0], new_p[:,1], new_p[:,2]))])

    p_corresponding = []
    distances = np.full(len(p), max(9.99, 9.99 * max_delta))
    for i, p1 in enumerate(p_ordered):
        found_match = False
        for j, p2 in enumerate(new_p_ordered):
            if p2[2] - p1[2] > 1:
                break
            elif (np21 := norm(np.array(p2) - p1)) < 0.1:
                found_match = True
                p_corresponding.append(new_p_ordered.pop(j))
                distances[i] = np21
                break
        if not found_match:
            p_corresponding.append([])

    distances = np.array(distances)
    if verbose:
        for i, p1 in enumerate(p_ordered):
            annotation = ""
            if distances[i] > max_delta:
                annotation = " * "
            print(i, p1, p_corresponding[i], distances[i], annotation)
        print("UNMATCHED coords: ")
        for p1 in new_p_ordered:
            print(p1)
    assert (distances < max_delta).all()

def assert_benzene(coords):
    # incomplete sample
    p = coords
    assert norm(p[0] - p[1]) == approx(2.42, 5e-2)
    assert norm(p[0] - p[3]) == approx(1.40, 5e-2)
    assert norm(p[0] - p[4]) == approx(2.79, 5e-2)
    assert norm(p[5] - p[8]) == approx(0.93, 5e-2)

@pytest.fixture
def linear_cnnc():
    yield Atoms('CNNC', positions=[(0., 0., 0), (1.0, 0., 0.), (2.0, 0., 0.), (3.0, 0., 0.)], cell=[15]*3)

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
