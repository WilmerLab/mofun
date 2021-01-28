from collections import Counter

from math import sqrt

from ase import Atoms
import numpy as np
import pytest
from pytest import approx
from scipy.spatial.transform import Rotation as R

from mofun import find_pattern_in_structure, replace_pattern_in_structure

from tests.fixtures import *


def test_find_pattern_in_structure__octane_has_8_carbons(octane):
    pattern = Atoms('C', positions=[(0, 0, 0)])
    match_indices = find_pattern_in_structure(octane, pattern)
    assert len(match_indices) == 8
    for indices in match_indices:
        assert octane[indices].get_chemical_symbols() == ["C"]

def test_find_pattern_in_structure__octane_has_2_CH3(octane):
    pattern = Atoms('CHHH', positions=[(0, 0, 0), (-0.538, -0.635,  0.672), (-0.397,  0.993,  0.052), (-0.099, -0.371, -0.998)])
    match_indices = find_pattern_in_structure(octane, pattern)
    assert len(match_indices) == 2
    for indices in match_indices:
        pattern_found = octane[indices]
        assert pattern_found.get_chemical_symbols() == ["C", "H", "H", "H"]
        cpos = pattern_found[0].position
        assert ((pattern_found[1].position - cpos) ** 2).sum() == approx(1.18704299, 5e-2)
        assert ((pattern_found[2].position - cpos) ** 2).sum() == approx(1.18704299, 5e-2)
        assert ((pattern_found[3].position - cpos) ** 2).sum() == approx(1.18704299, 5e-2)

def test_find_pattern_in_structure__match_indices_returned_in_order_of_pattern():
    structure = Atoms('HOH', positions=[(4., 0, 0), (5., 0., 0), (6., 0., 0.),], cell=[15]*3)
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
        assert pattern_found.get_chemical_symbols() == ["C", "H", "H"]
        cpos = pattern_found[0].position
        assert ((pattern_found[1].position - cpos) ** 2).sum() == approx(1.18704299, 5e-2)
        assert ((pattern_found[2].position - cpos) ** 2).sum() == approx(1.18704299, 5e-2)

def test_find_pattern_in_structure__cnnc_over_x_pbc_has_positions_across_x_pbc(linear_cnnc):
    linear_cnnc.positions = (linear_cnnc.positions + (-0.5, 0.0, 0.0)) % 15
    linear_cnnc.pop() #don't match final NC
    search_pattern = Atoms('CN', positions=[(0.0, 0., 0), (1.0, 0., 0.)])
    match_indices, match_positions = find_pattern_in_structure(linear_cnnc, search_pattern, return_positions=True)
    assert (linear_cnnc[match_indices[0]].positions == [(14.5, 0., 0.), (0.5, 0., 0.)]).all()
    assert (match_positions[0] == np.array([(14.5, 0., 0.), (15.5, 0., 0.)])).all()

def test_find_pattern_in_structure__cnnc_over_xy_pbc_has_positions_across_xy_pbc(linear_cnnc):
    linear_cnnc.rotate(45, 'z')
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
    octane.set_cell(15 * np.identity(3))

    pattern = Atoms('CHHH', positions=[(0, 0, 0), (-0.538, -0.635,  0.672), (-0.397,  0.993,  0.052), (-0.099, -0.371, -0.998)])
    match_indices = find_pattern_in_structure(octane, pattern)
    assert len(match_indices) == 2
    for indices in match_indices:
        assert octane[indices].get_chemical_symbols() == ["C", "H", "H", "H"]

@pytest.mark.slow
def test_find_pattern_in_structure__hkust1_unit_cell_has_32_benzene_rings(hkust1_cif, benzene):
    match_indices = find_pattern_in_structure(hkust1_cif, benzene)

    assert len(match_indices) == 32
    for indices in match_indices:
        pattern_found = hkust1_cif[indices]
        assert pattern_found.get_chemical_symbols() == ['C','C','C','C','C','C','H','H','H']
        assert ((pattern_found[0].position - pattern_found[1].position) ** 2).sum() == approx(5.8620934418, 5e-2)
        assert ((pattern_found[0].position - pattern_found[3].position) ** 2).sum() == approx(1.9523164046, 5e-2)
        assert ((pattern_found[0].position - pattern_found[4].position) ** 2).sum() == approx(7.8072193204, 5e-2)
        assert ((pattern_found[5].position - pattern_found[8].position) ** 2).sum() == approx(0.8683351588, 5e-2)

def test_find_pattern_in_structure__hkust1_unit_cell_has_48_Cu_metal_nodes(hkust1_cif):
    pattern = Atoms('Cu', positions=[(0, 0, 0)])
    match_indices = find_pattern_in_structure(hkust1_cif, pattern)

    assert len(match_indices) == 48
    for indices in match_indices:
        pattern_found = hkust1_cif[indices]
        assert pattern_found.get_chemical_symbols() == ['Cu']

@pytest.mark.slow
def test_find_pattern_in_structure__hkust1_cif_3x3x3_supercell_has_864_benzene_rings(hkust1_3x3x3_cif, benzene):
    match_indices = find_pattern_in_structure(hkust1_3x3x3_cif, benzene)

    assert len(match_indices) == 864
    for indices in match_indices:
        pattern_found = hkust1_3x3x3_cif[indices]
        assert pattern_found.get_chemical_symbols() == ['C','C','C','C','C','C','H','H','H']
        assert ((pattern_found[0].position - pattern_found[1].position) ** 2).sum() == approx(5.8620934418, 5e-2)
        assert ((pattern_found[0].position - pattern_found[3].position) ** 2).sum() == approx(1.9523164046, 5e-2)
        assert ((pattern_found[0].position - pattern_found[4].position) ** 2).sum() == approx(7.8072193204, 5e-2)
        assert ((pattern_found[5].position - pattern_found[8].position) ** 2).sum() == approx(0.8683351588, 5e-2)

@pytest.mark.slow
def test_find_pattern_in_structure__hkust1_xyz_3x3x3_supercell_has_864_benzene_rings(hkust1_3x3x3_xyz, benzene):
    match_indices = find_pattern_in_structure(hkust1_3x3x3_xyz, benzene)

    assert len(match_indices) == 864
    for indices in match_indices:
        pattern_found = hkust1_3x3x3_xyz[indices]
        assert pattern_found.get_chemical_symbols() == ['C','C','C','C','C','C','H','H','H']
        assert ((pattern_found[0].position - pattern_found[1].position) ** 2).sum() == approx(5.8620934418, 5e-2)
        assert ((pattern_found[0].position - pattern_found[3].position) ** 2).sum() == approx(1.9523164046, 5e-2)
        assert ((pattern_found[0].position - pattern_found[4].position) ** 2).sum() == approx(7.8072193204, 5e-2)
        assert ((pattern_found[5].position - pattern_found[8].position) ** 2).sum() == approx(0.8683351588, 5e-2)

@pytest.mark.slow
def test_find_pattern_in_structure__hkust1_cif_3x3x3_supercell_has_1296_Cu_metal_nodes(hkust1_3x3x3_cif):
    pattern = Atoms('Cu', positions=[(0, 0, 0)])
    match_indices = find_pattern_in_structure(hkust1_3x3x3_cif, pattern)

    assert len(match_indices) == 1296
    for indices in match_indices:
        pattern_found = hkust1_3x3x3_cif[indices]
        assert pattern_found.get_chemical_symbols() == ['Cu']

@pytest.mark.slow
def test_find_pattern_in_structure__hkust1_xyz_3x3x3_supercell_has_1296_Cu_metal_nodes(hkust1_3x3x3_xyz):
    pattern = Atoms('Cu', positions=[(0, 0, 0)])
    match_indices = find_pattern_in_structure(hkust1_3x3x3_xyz, pattern)

    assert len(match_indices) == 1296
    for indices in match_indices:
        pattern_found = hkust1_3x3x3_xyz[indices]
        assert pattern_found.get_chemical_symbols() == ['Cu']

def test_replace_pattern_in_structure__replace_hydrogens_in_octane_with_nothing(octane):
    # CH3 CH2 CH2 CH2 CH2 CH2 CH2 CH3 #
    search_pattern = Atoms('H', positions=[(0, 0, 0)])
    replace_pattern = Atoms()

    final_structure = replace_pattern_in_structure(octane, search_pattern, replace_pattern)
    assert final_structure.get_chemical_symbols() == ["C"] * 8

def test_replace_pattern_in_structure__replace_hydrogens_in_octane_with_hydrogens(octane):
    search_pattern = Atoms('H', positions=[(0, 0, 0)])
    replace_pattern = search_pattern

    final_structure = replace_pattern_in_structure(octane, search_pattern, replace_pattern)
    assert Counter(final_structure.symbols) == {"H": 18, "C": 8}
    assert_positions_should_be_unchanged(octane, final_structure)

def test_replace_pattern_in_structure__replace_hydrogens_in_octane_with_fluorines(octane):
    search_pattern = Atoms('H', positions=[(0, 0, 0)])
    replace_pattern = Atoms('F', positions=[(0, 0, 0)])
    final_structure = replace_pattern_in_structure(octane, search_pattern, replace_pattern)
    assert Counter(final_structure.symbols) == {"F": 18, "C": 8}
    assert_positions_should_be_unchanged(octane, final_structure)

def test_replace_pattern_in_structure__replace_CH3_in_octane_with_fluorines(octane):
    search_pattern = Atoms('CHHH', positions=[(0, 0, 0), (-0.538, -0.635,  0.672), (-0.397,  0.993,  0.052), (-0.099, -0.371, -0.998)])
    replace_pattern = Atoms('FFFF', positions=search_pattern.positions)
    final_structure = replace_pattern_in_structure(octane, search_pattern, replace_pattern)
    assert Counter(final_structure.symbols) == {"F": 8, "C": 6, "H": 12}
    # note the positions are not EXACTLY the same because the original structure has slightly
    # different coordinates for the two CH3 groups!
    assert_positions_should_be_unchanged(octane, final_structure, 0)

def test_replace_pattern_in_structure__replace_CH3_in_octane_with_CH3(octane):
    search_pattern = Atoms('CHHH', positions=[(0, 0, 0), (-0.538, -0.635,  0.672), (-0.397,  0.993,  0.052), (-0.099, -0.371, -0.998)])
    replace_pattern = Atoms('CHHH', positions=search_pattern.positions)
    final_structure = replace_pattern_in_structure(octane, search_pattern, replace_pattern)
    assert Counter(final_structure.symbols) == {"C": 8, "H": 18}
    # note the positions are not EXACTLY the same because the original structure has slightly
    # different coordinates for the two CH3 groups!
    assert_positions_should_be_unchanged(octane, final_structure, 0)

def test_replace_pattern_in_structure__two_points_on_x_axis_positions_are_unchanged():
    structure = Atoms('CNNC', positions=[(0., 0., 0), (1.0, 0., 0.), (2.0, 0., 0.), (3.0, 0., 0.)], cell=[100]*3)
    search_pattern = Atoms('NN', positions=[(0.0, 0., 0), (1.0, 0., 0.)])
    replace_pattern = Atoms('FF', positions=[(0., 0., 0), (1.0, 0., 0.)])

    final_structure = replace_pattern_in_structure(structure, search_pattern, replace_pattern)
    assert Counter(final_structure.symbols) == {"C":2, "F": 2}
    assert_positions_should_be_unchanged(structure, final_structure, decimal_points=5)

def test_replace_pattern_in_structure__pattern_and_reverse_pattern_on_x_axis_positions_are_unchanged():
    structure = Atoms('HHCCCHH', positions=[(-1., 1, 0), (-1., -1, 0), (0., 0., 0), (1., 0., 0.), (3., 0., 0.), (4., 1, 0), (4., -1, 0)], cell=[7]*3)
    structure.translate([2, 2, 0.1])
    print("structure.positions: ", structure.positions)
    search_pattern = Atoms('HHC', positions=[(0., 1, 0), (0., -1, 0), (1., 0., 0.)])
    replace_pattern = Atoms('HHC', positions=[(0., 1, 0), (0., -1, 0), (1., 0., 0.)])

    final_structure = replace_pattern_in_structure(structure, search_pattern, replace_pattern)

    assert Counter(final_structure.symbols) == {"C":3, "H": 4}
    assert_positions_should_be_unchanged(structure, final_structure, decimal_points=5)

def test_replace_pattern_in_structure__replacement_pattern_across_pbc_gets_coordinates_within_unit_cell():
    structure = Atoms('CNNX', positions=[(9.1, 0., 0), (0.1, 0., 0), (1.1, 0., 0.), (2.1, 0., 0.)], cell=[10]*3)
    search_pattern = Atoms('CN', positions=[(0., 0., 0), (1.0, 0., 0.)])
    replace_pattern = search_pattern
    final_structure = replace_pattern_in_structure(structure, search_pattern, replace_pattern)
    assert Counter(final_structure.symbols) == Counter(structure.symbols)
    assert_positions_should_be_unchanged(structure, final_structure, decimal_points=5)

def test_replace_pattern_in_structure__two_points_at_angle_are_unchanged():
    structure = Atoms('CNNC', positions=[(0., 0., 0), (1.0, 0., 0.),
                                        (1+ sqrt(2)/2, sqrt(2)/2, 0.),
                                        (2 + sqrt(2)/2, sqrt(2)/2, 0.)], cell=[100]*3)
    search_pattern = Atoms('NN', positions=[(0.0, 0., 0), (1.0, 0., 0.)])
    replace_pattern = Atoms('FF', positions=[(0., 0., 0), (1.0, 0., 0.)])

    final_structure = replace_pattern_in_structure(structure, search_pattern, replace_pattern)
    assert Counter(final_structure.symbols) == {"C":2, "F": 2}
    assert_positions_should_be_unchanged(structure, final_structure, decimal_points=5)

# def test_replace_pattern_orient__in_hkust1_replacing_benzene_with_benzene_does_not_change_positions(hkust1_cif, benzene):
#     match_indices = find_pattern_in_structure(hkust1_cif, benzene)
#     replace_pattern = [benzene.copy() for _ in match_indices]
#
#     translate_molecule_origin(benzene)
#     for i in range(len(replace_pattern)):
#         translate_molecule_origin(replace_pattern[i])
#
#     for i in range(len(replace_pattern)):
#         replace_pattern_orient(benzene, replace_pattern[i])
#
#     for i in range(len(replace_pattern)):
#         replace_pattern_orient(match_atoms[i], replace_pattern[i])
#
#     for i in range(len(replace_pattern)):
#         translate_replace_pattern(replace_pattern[i], match_atoms[i])
#
#     for i in range(len(replace_pattern)):
#         diff = match_atoms[i].positions - replace_pattern[i].positions
#         for j in range(len(diff)):
#             for k in range(3):
#                 assert (diff[j][k])**2 == approx(0, abs=1e-3)
#
#
#         assert len(replace_pattern[i]) == 9
#         assert replace_pattern[i].get_chemical_symbols() == ["C", "C", "C", "C", "C", "C", "H", "H", "H"]
