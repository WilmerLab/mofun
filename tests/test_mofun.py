from collections import Counter

from math import sqrt

import ase
from ase.visualize import view
import numpy as np
import pytest
from pytest import approx
from scipy.spatial.transform import Rotation as R

from mofun import find_pattern_in_structure, replace_pattern_in_structure, Atoms

from tests.fixtures import *


def test_find_pattern_in_structure__octane_has_8_carbons(octane):
    pattern = Atoms('C', positions=[(0, 0, 0)])
    match_indices = find_pattern_in_structure(octane, pattern)
    assert len(match_indices) == 8
    for indices in match_indices:
        assert octane[indices].atom_types == ["C"]

def test_find_pattern_in_structure__octane_has_2_CH3(octane):
    pattern = Atoms('CHHH', positions=[(0, 0, 0), (-0.538, -0.635,  0.672), (-0.397,  0.993,  0.052), (-0.099, -0.371, -0.998)])
    match_indices = find_pattern_in_structure(octane, pattern)
    assert len(match_indices) == 2
    for indices in match_indices:
        pattern_found = octane[indices]
        assert (pattern_found.atom_types == ["C", "H", "H", "H"]).all()
        cpos = pattern_found.positions[0]
        assert ((pattern_found.positions[1] - cpos) ** 2).sum() == approx(1.18704299, 5e-2)
        assert ((pattern_found.positions[2] - cpos) ** 2).sum() == approx(1.18704299, 5e-2)
        assert ((pattern_found.positions[3] - cpos) ** 2).sum() == approx(1.18704299, 5e-2)

def test_find_pattern_in_structure__match_indices_returned_in_order_of_pattern():
    structure = ase.Atoms('HOH', positions=[(4., 0, 0), (5., 0., 0), (6., 0., 0.),], cell=15*np.identity(3))
    search_pattern = Atoms('HO', positions=[(-1., 0, 0), (0., 0., 0.)])
    match_indices = find_pattern_in_structure(structure, search_pattern)
    assert set(match_indices) == {(0, 1), (2, 1)}

def test_find_pattern_in_structure__octane_has_12_CH2(octane):
    # there are technically 12 matches, since each CH3 makes 3 variations of CH2
    pattern = Atoms('CHH', positions=[(0, 0, 0),(-0.1  , -0.379, -1.017), (-0.547, -0.647,  0.685)])
    match_indices = find_pattern_in_structure(octane, pattern)

    assert len(match_indices) == 12
    for indices in match_indices:
        pattern_found = octane[indices]
        assert (pattern_found.atom_types == ["C", "H", "H"]).all()
        cpos = pattern_found.positions[0]
        assert ((pattern_found.positions[1] - cpos) ** 2).sum() == approx(1.18704299, 5e-2)
        assert ((pattern_found.positions[2] - cpos) ** 2).sum() == approx(1.18704299, 5e-2)

def test_find_pattern_in_structure__cnnc_over_x_pbc_has_positions_across_x_pbc(linear_cnnc):
    linear_cnnc.positions = (linear_cnnc.positions + (-0.5, 0.0, 0.0)) % 15
    linear_cnnc.pop(-1) #don't match final NC
    search_pattern = Atoms('CN', positions=[(0.0, 0., 0), (1.0, 0., 0.)])
    match_indices, match_positions = find_pattern_in_structure(linear_cnnc, search_pattern, return_positions=True)
    assert (linear_cnnc[match_indices[0]].positions == [(14.5, 0., 0.), (0.5, 0., 0.)]).all()
    assert (match_positions[0] == np.array([(14.5, 0., 0.), (15.5, 0., 0.)])).all()

def test_find_pattern_in_structure__cnnc_over_xy_pbc_has_positions_across_xy_pbc(linear_cnnc):
    v2_2 = sqrt(2.0) / 2
    linear_cnnc = Atoms('CNNC', positions=[(0., 0., 0), (v2_2, v2_2, 0.), (2.*v2_2, 2.*v2_2, 0.), (3.*v2_2, 3.*v2_2, 0.)], cell=15*np.identity(3))
    linear_cnnc.positions = (linear_cnnc.positions + (-0.5, -0.5, 0.0)) % 15
    linear_cnnc.pop() #don't match final NC
    print(linear_cnnc.positions)
    search_pattern = Atoms('CN', positions=[(0.0, 0., 0), (1.0, 0., 0.)])
    match_indices, match_positions = find_pattern_in_structure(linear_cnnc, search_pattern, return_positions=True)
    assert np.isclose(linear_cnnc[match_indices[0]].positions, np.array([(14.5, 14.5, 0.), (sqrt2_2 - 0.5, sqrt2_2 - 0.5, 0.)])).all()
    assert (match_positions[0] == np.array([(14.5, 14.5, 0.), (14.5 + sqrt2_2, 14.5 + sqrt2_2, 0.)])).all()

def test_find_pattern_in_structure__octane_over_pbc_has_2_CH3(octane):
    # CH3 CH2 CH2 CH2 CH2 CH2 CH2 CH3 #
    # move atoms across corner boundary
    octane.positions += -1.8
    # move coordinates into main 15 Å unit cell
    octane.positions %= 15
    octane.cell = (15 * np.identity(3))

    pattern = Atoms('CHHH', positions=[(0, 0, 0), (-0.538, -0.635,  0.672), (-0.397,  0.993,  0.052), (-0.099, -0.371, -0.998)])
    match_indices = find_pattern_in_structure(octane, pattern)
    assert len(match_indices) == 2
    for indices in match_indices:
        assert (octane[indices].atom_types == ["C", "H", "H", "H"]).all()

@pytest.mark.slow
def test_find_pattern_in_structure__hkust1_unit_cell_has_32_benzene_rings(hkust1_cif, benzene):
    match_indices = find_pattern_in_structure(hkust1_cif, benzene)

    assert len(match_indices) == 32
    for indices in match_indices:
        pattern_found = hkust1_cif[indices]
        assert list(pattern_found.atom_types) == ['C','C','C','C','C','C','H','H','H']
        assert_benzene(pattern_found.positions)

@pytest.mark.slow
def test_find_pattern_in_structure__hkust1_unit_cell_offset_has_32_benzene_rings(hkust1_cif, benzene):
    hkust1_cif.translate((-4,-4,-4))
    hkust1_cif.positions = hkust1_cif.positions % np.diag(hkust1_cif.cell)
    match_indices, coords = find_pattern_in_structure(hkust1_cif, benzene, return_positions=True)
    for i, indices in enumerate(match_indices):
        assert list(hkust1_cif[indices].atom_types) == ['C','C','C','C','C','C','H','H','H']
        assert_benzene(coords[i])
    assert len(match_indices) == 32

@pytest.mark.slow
def test_find_pattern_in_structure__hkust1_unit_cell_has_48_Cu_metal_nodes(hkust1_cif):
    pattern = Atoms('Cu', positions=[(0, 0, 0)])
    match_indices = find_pattern_in_structure(hkust1_cif, pattern)

    assert len(match_indices) == 48
    for indices in match_indices:
        pattern_found = hkust1_cif[indices]
        assert list(pattern_found.atom_types) == ['Cu']


@pytest.mark.veryslow
def test_find_pattern_in_structure__hkust1_cif_3x3x3_supercell_has_864_benzene_rings(hkust1_3x3x3_cif, benzene):
    match_indices = find_pattern_in_structure(hkust1_3x3x3_cif, benzene)

    assert len(match_indices) == 864
    for indices in match_indices:
        pattern_found = hkust1_3x3x3_cif[indices]
        assert list(pattern_found.atom_types) == ['C','C','C','C','C','C','H','H','H']
        assert_benzene(pattern_found.positions)

@pytest.mark.veryslow
def test_find_pattern_in_structure__hkust1_xyz_3x3x3_supercell_has_864_benzene_rings(hkust1_3x3x3_xyz, benzene):
    match_indices = find_pattern_in_structure(hkust1_3x3x3_xyz, benzene)

    assert len(match_indices) == 864
    for indices in match_indices:
        pattern_found = hkust1_3x3x3_xyz[indices]
        assert list(pattern_found.atom_types) == ['C','C','C','C','C','C','H','H','H']
        assert_benzene(pattern_found.positions)

@pytest.mark.veryslow
def test_find_pattern_in_structure__hkust1_cif_3x3x3_supercell_has_1296_Cu_metal_nodes(hkust1_3x3x3_cif):
    pattern = Atoms('Cu', positions=[(0, 0, 0)])
    match_indices = find_pattern_in_structure(hkust1_3x3x3_cif, pattern)

    assert len(match_indices) == 1296
    for indices in match_indices:
        pattern_found = hkust1_3x3x3_cif[indices]
        assert list(pattern_found.atom_types) == ['Cu']

@pytest.mark.veryslow
def test_find_pattern_in_structure__hkust1_xyz_3x3x3_supercell_has_1296_Cu_metal_nodes(hkust1_3x3x3_xyz):
    pattern = Atoms('Cu', positions=[(0, 0, 0)])
    match_indices = find_pattern_in_structure(hkust1_3x3x3_xyz, pattern)

    assert len(match_indices) == 1296
    for indices in match_indices:
        pattern_found = hkust1_3x3x3_xyz[indices]
        assert list(pattern_found.atom_types) == ['Cu']

def test_replace_pattern_in_structure__replace_hydrogens_in_octane_with_nothing(octane):
    # CH3 CH2 CH2 CH2 CH2 CH2 CH2 CH3 #
    search_pattern = Atoms('H', positions=[(0, 0, 0)])
    replace_pattern = Atoms()

    final_structure = replace_pattern_in_structure(octane, search_pattern, replace_pattern)
    assert list(final_structure.atom_types) == ["C"] * 8

def test_replace_pattern_in_structure__replace_hydrogens_in_octane_with_hydrogens(octane):
    search_pattern = Atoms('H', positions=[(0, 0, 0)])
    replace_pattern = search_pattern

    final_structure = replace_pattern_in_structure(octane, search_pattern, replace_pattern)
    assert Counter(final_structure.atom_types) == {"H": 18, "C": 8}
    assert_positions_are_unchanged(octane, final_structure)

def test_replace_pattern_in_structure__replace_hydrogens_in_octane_with_fluorines(octane):
    search_pattern = Atoms('H', positions=[(0, 0, 0)])
    replace_pattern = Atoms('F', positions=[(0, 0, 0)])
    final_structure = replace_pattern_in_structure(octane, search_pattern, replace_pattern)
    assert Counter(final_structure.atom_types) == {"F": 18, "C": 8}
    assert_positions_are_unchanged(octane, final_structure)

def test_replace_pattern_in_structure__replace_CH3_in_octane_with_fluorines(octane):
    search_pattern = Atoms('CHHH', positions=[(0, 0, 0), (-0.538, -0.635,  0.672), (-0.397,  0.993,  0.052), (-0.099, -0.371, -0.998)])
    replace_pattern = Atoms('FFFF', positions=search_pattern.positions)
    final_structure = replace_pattern_in_structure(octane, search_pattern, replace_pattern)
    assert Counter(final_structure.atom_types) == {"F": 8, "C": 6, "H": 12}
    # note the positions are not EXACTLY the same because the original structure has slightly
    # different coordinates for the two CH3 groups!
    assert_positions_are_unchanged(octane, final_structure, max_delta=0.1)

def test_replace_pattern_in_structure__replace_CH3_in_octane_with_CH3(octane):
    search_pattern = Atoms('CHHH', positions=[(0, 0, 0), (-0.538, -0.635,  0.672), (-0.397,  0.993,  0.052), (-0.099, -0.371, -0.998)])
    replace_pattern = Atoms('CHHH', positions=search_pattern.positions)
    final_structure = replace_pattern_in_structure(octane, search_pattern, replace_pattern)
    assert Counter(final_structure.atom_types) == {"C": 8, "H": 18}
    # note the positions are not EXACTLY the same because the original structure has slightly
    # different coordinates for the two CH3 groups!
    assert_positions_are_unchanged(octane, final_structure, max_delta=0.1)

def test_replace_pattern_in_structure__two_points_on_x_axis_positions_are_unchanged():
    structure = Atoms('CNNC', positions=[(0., 0., 0), (1.0, 0., 0.), (2.0, 0., 0.), (3.0, 0., 0.)], cell=100*np.identity(3))
    search_pattern = Atoms('NN', positions=[(0.0, 0., 0), (1.0, 0., 0.)])
    replace_pattern = Atoms('FF', positions=[(0., 0., 0), (1.0, 0., 0.)])

    final_structure = replace_pattern_in_structure(structure, search_pattern, replace_pattern)
    assert Counter(final_structure.atom_types) == {"C":2, "F": 2}
    assert_positions_are_unchanged(structure, final_structure)

def test_replace_pattern_in_structure__pattern_and_reverse_pattern_on_x_axis_positions_are_unchanged():
    structure = Atoms('HHCCCHH', positions=[(-1., 1, 0), (-1., -1, 0), (0., 0., 0), (1., 0., 0.), (3., 0., 0.), (4., 1, 0), (4., -1, 0)], cell=7*np.identity(3))
    structure.translate([2, 2, 0.1])
    search_pattern = Atoms('HHC', positions=[(0., 1, 0), (0., -1, 0), (1., 0., 0.)])
    replace_pattern = Atoms('HHC', positions=[(0., 1, 0), (0., -1, 0), (1., 0., 0.)])

    final_structure = replace_pattern_in_structure(structure, search_pattern, replace_pattern)

    assert Counter(final_structure.atom_types) == {"C":3, "H": 4}
    assert_positions_are_unchanged(structure, final_structure)

def test_replace_pattern_in_structure__replacement_pattern_across_pbc_gets_coordinates_within_unit_cell():
    structure = Atoms('CNNX', positions=[(9.1, 0., 0), (0.1, 0., 0), (1.1, 0., 0.), (2.1, 0., 0.)], cell=10*np.identity(3))
    search_pattern = Atoms('CN', positions=[(0., 0., 0), (1.0, 0., 0.)])
    replace_pattern = search_pattern
    final_structure = replace_pattern_in_structure(structure, search_pattern, replace_pattern)
    assert Counter(final_structure.atom_types) == Counter(structure.atom_types)
    assert_positions_are_unchanged(structure, final_structure)

def test_replace_pattern_in_structure__100_randomly_rotated_patterns_replaced_with_itself_does_not_change_positions():
    search_pattern = Atoms('CCH', positions=[(0., 0., 0.), (4., 0., 0.),(0., 1., 0.)])
    replace_pattern = Atoms('FFHe', positions=search_pattern.positions)
    structure = search_pattern.copy()
    structure.cell = 15 * np.identity(3)
    for _ in range(100):
        r = R.random(1)
        print(r.as_quat())
        structure.positions = r.apply(structure.positions)
        dp = np.random.random(3) * 15
        print(dp)
        structure.translate(dp)
        structure.positions = structure.positions % 15
        final_structure = replace_pattern_in_structure(structure, search_pattern, replace_pattern, axis1a_idx=0, axis1b_idx=1)
        assert_positions_are_unchanged(structure, final_structure)

def test_replace_pattern_in_structure__special_rotated_pattern_replaced_with_itself_does_not_change_positions():
    search_pattern = Atoms('CCH', positions=[(0., 0., 0.), (4., 0., 0.),(0., 1., 0.)])
    replace_pattern = Atoms('FFHe', positions=search_pattern.positions)
    structure = search_pattern.copy()
    structure.cell = [15] * np.identity(3)
    r = R.from_quat([-0.4480244,  -0.50992783,  0.03212454, -0.7336319 ])
    structure.positions = r.apply(structure.positions)
    structure.positions = structure.positions % 15
    final_structure = replace_pattern_in_structure(structure, search_pattern, replace_pattern, axis1a_idx=0, axis1b_idx=1)
    assert_positions_are_unchanged(structure, final_structure)

def test_replace_pattern_in_structure__special2_rotated_pattern_replaced_with_itself_does_not_change_positions():
    search_pattern = Atoms('CCH', positions=[(0., 0., 0.), (4., 0., 0.),(0., 1., 0.)])
    replace_pattern = Atoms('FFHe', positions=search_pattern.positions)
    structure = search_pattern.copy()
    structure.cell = 15 * np.identity(3)
    r = R.from_quat([ 0.02814096,  0.99766676,  0.03984918, -0.04776152])
    structure.positions = r.apply(structure.positions)
    structure.positions = structure.positions % 15
    final_structure = replace_pattern_in_structure(structure, search_pattern, replace_pattern, axis1a_idx=0, axis1b_idx=1)
    assert_positions_are_unchanged(structure, final_structure)

def test_replace_pattern_in_structure__two_points_at_angle_are_unchanged():
    structure = Atoms('CNNC', positions=[(0., 0., 0), (1.0, 0., 0.),
                                        (1+ sqrt(2)/2, sqrt(2)/2, 0.),
                                        (2 + sqrt(2)/2, sqrt(2)/2, 0.)], cell=100*np.identity(3))
    search_pattern = Atoms('NN', positions=[(0.0, 0., 0), (1.0, 0., 0.)])
    replace_pattern = Atoms('FF', positions=[(0., 0., 0), (1.0, 0., 0.)])

    final_structure = replace_pattern_in_structure(structure, search_pattern, replace_pattern)
    assert Counter(final_structure.atom_types) == {"C":2, "F": 2}
    assert_positions_are_unchanged(structure, final_structure)

@pytest.mark.slow
def test_replace_pattern_in_structure__in_hkust1_replacing_benzene_with_benzene_does_not_change_positions(hkust1_cif, benzene):
    final_structure = replace_pattern_in_structure(hkust1_cif, benzene, benzene, axis1a_idx=0, axis1b_idx=7)
    assert Counter(final_structure.atom_types) == Counter(hkust1_cif.atom_types)
    assert_positions_are_unchanged(hkust1_cif, final_structure, max_delta=0.1)

@pytest.mark.slow
def test_replace_pattern_in_structure__in_hkust1_offset_replacing_benzene_with_benzene_does_not_change_positions(hkust1_cif, benzene):
    hkust1_cif.translate((-4,-4,-4))
    hkust1_cif.positions = hkust1_cif.positions % np.diag(hkust1_cif.cell)
    final_structure = replace_pattern_in_structure(hkust1_cif, benzene, benzene, axis1a_idx=0, axis1b_idx=7)
    assert Counter(final_structure.atom_types) == Counter(hkust1_cif.atom_types)
    assert_positions_are_unchanged(hkust1_cif, final_structure, max_delta=0.1)

@pytest.mark.slow
def test_replace_pattern_in_structure__in_uio66_replacing_linker_with_linker_does_not_change_positions():
    with importlib.resources.path(tests, "uio66.cif") as path:
        structure = Atoms.from_cif(str(path))
    with importlib.resources.path(tests, "uio66-linker.cif") as path:
        search_pattern = Atoms.from_cif(str(path))
    with importlib.resources.path(tests, "uio66-linker-fluorinated.cif") as path:
        replace_pattern =  Atoms.from_cif(str(path))

    final_structure = replace_pattern_in_structure(structure, search_pattern, replace_pattern)
    assert Counter(final_structure.atom_types) == {'C': 192, 'O': 120, 'F': 96, 'Zr': 24}
    # assert_positions_are_unchanged(structure, final_structure, max_delta=0.25, verbose=True)
