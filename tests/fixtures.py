
from math import sqrt
from pathlib import Path

import ase.io
import numpy as np
from numpy.linalg import norm
import pytest
from pytest import approx

import tests
from mofun import Atoms
from mofun.helpers import typekey

sqrt2_2 = sqrt(2) / 2
sqrt3_2 = sqrt(3) / 2

def random_positions(num):
    return np.random.rand(num, 3) * 100

def assert_structure_positions_are_unchanged(orig_structure, final_structure, max_delta=1e-5, verbose=False):
    return assert_positions_are_unchanged(orig_structure.positions, final_structure.positions, max_delta, verbose)

def assert_positions_are_unchanged(p, new_p, max_delta=1e-5, verbose=False):
    p_ordered = p[np.lexsort((p[:,0], p[:,1], p[:,2]))]
    new_p_ordered = list(new_p[np.lexsort((new_p[:,0], new_p[:,1], new_p[:,2]))])

    p_corresponding = []
    distances = np.full(len(p), max(9.99, 9.99 * max_delta))
    for i, p1 in enumerate(p_ordered):
        found_match = False
        for j, p2 in enumerate(new_p_ordered):
            if p2[2] - p1[2] > 1:
                break
            elif (np21 := norm(np.array(p2) - p1)) < max_delta:
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

def assert_topo(topo, expected_topo, types=None, expected_types=None, coeffs=None, expected_coeffs=None):

    # check right atoms are part of the topo
    sorted_topo = sorted([typekey(t) for t in topo])
    sorted_expected_topo = sorted([typekey(t) for t in expected_topo])
    assert np.array_equal(sorted_topo, sorted_expected_topo)

    # check types are mapped (assume coeffs are ordered the same!)
    if types is not None and expected_types is not None:
        sorted_topo_w_types = sorted([(*typekey(t), types[i]) for i, t in enumerate(topo)])
        sorted_expected_topo_w_types = sorted([(*typekey(t), expected_types[i]) for i, t in enumerate(expected_topo)])
        assert np.array_equal(sorted_topo_w_types, sorted_expected_topo_w_types)

    # check coeffs for each type are equal
    if coeffs is not None and expected_coeffs is not None:
        assert np.array_equal(coeffs, expected_coeffs)

def assert_benzene(coords):
    # incomplete sample
    p = coords
    assert norm(p[0] - p[1]) == approx(2.42, 5e-2)
    assert norm(p[0] - p[3]) == approx(1.40, 5e-2)
    assert norm(p[0] - p[4]) == approx(2.79, 5e-2)
    assert norm(p[5] - p[8]) == approx(0.93, 5e-2)

@pytest.fixture
def linear_cnnc():
    yield Atoms(elements='CNNC', positions=[(0., 0., 0), (1.0, 0., 0.), (2.0, 0., 0.), (3.0, 0., 0.)],
                bonds=[(0,1), (1,2), (2,3)], bond_types=[0] * 3,
                angles=[(0,1,2), (1,2,3)], angle_types=[0,0],
                dihedrals=[(0,1,2,3)], dihedral_types=[0], cell=15*np.identity(3))

@pytest.fixture
def octane():
    # CH3 CH2 CH2 CH2 CH2 CH2 CH2 CH3 #
    with Path("tests/molecules/octane.xyz") as path:
        structure = Atoms.from_ase_atoms(ase.io.read(path))
        structure.cell = 60 * np.identity(3)
        structure.translate((30., 30., 30.))
        yield structure

@pytest.fixture
def hkust1_cif():
    with Path("tests/hkust-1/hkust-1-with-bonds.cif") as path:
        yield Atoms.load_cif(path)

@pytest.fixture
def hkust1_3x3x3_xyz():
    with Path("tests/hkust-1/hkust-1-3x3x3.xyz") as path:
        structure = Atoms.from_ase_atoms(ase.io.read(path))
        structure.cell = 79.0290 * np.identity(3)
        yield structure

@pytest.fixture
def hkust1_3x3x3_cif():
    with Path("tests/hkust-1/hkust-1-3x3x3.cif") as path:
        yield Atoms.load_cif(path)

@pytest.fixture
def benzene():
    with Path("tests/molecules/benzene.xyz") as path:
        yield Atoms.from_ase_atoms(ase.io.read(path))

@pytest.fixture
def uio66_linker_no_bonds():
    with Path("tests/uio66/uio66-linker-no-bonds.lmpdat").open() as fd:
        yield Atoms.load_lmpdat(fd, atom_format="atomic")

@pytest.fixture
def uio66_linker_some_bonds():
    # this was a modified UIO-66-F linker with bonds defined for the C-F bond. The F's have been
    # replaced by H's.
    with Path("tests/uio66/uio66-linker.lmpdat").open() as fd:
        yield Atoms.load_lmpdat(fd, atom_format="atomic")

@pytest.fixture
def uio66_linker_cml():
    with Path("tests/uio66/uio66-linker.cml") as path:
        yield Atoms.load_cml(path)
