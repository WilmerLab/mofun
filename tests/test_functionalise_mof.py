import importlib
from importlib import resources

import ase
from ase import Atoms
import numpy as np
import pytest
from pytest import approx

from functionalise_mof import find_pattern_in_structure, replace_pattern_in_structure
import tests


def test_find_pattern_in_structure__octane_has_8_carbons():
    # CH3 CH2 CH2 CH2 CH2 CH2 CH2 CH3 #
    with importlib.resources.path(tests, "octane.xyz") as octane_path:
        structure = ase.io.read(octane_path)
    pattern = Atoms('C', positions=[(0, 0, 0)])
    match_indices, match_atoms = find_pattern_in_structure(structure, pattern)
    assert len(match_atoms) == 8
    for pattern_found in match_atoms:
        assert pattern_found.get_chemical_symbols() == ["C"]

def test_find_pattern_in_structure__octane_has_2_CH3():
    # CH3 CH2 CH2 CH2 CH2 CH2 CH2 CH3 #
    with importlib.resources.path(tests, "octane.xyz") as octane_path:
        structure = ase.io.read(octane_path)
    pattern = Atoms('CHHH', positions=[(0, 0, 0), (-0.538, -0.635,  0.672), (-0.397,  0.993,  0.052), (-0.099, -0.371, -0.998)])
    match_indices, match_atoms = find_pattern_in_structure(structure, pattern)
    assert len(match_atoms) == 2
    for pattern_found in match_atoms:
        assert pattern_found.get_chemical_symbols() == ["C", "H", "H", "H"]
        cpos = pattern_found[0].position
        assert ((pattern_found[1].position - cpos) ** 2).sum() == approx(1.18704299, 5e-2)
        assert ((pattern_found[2].position - cpos) ** 2).sum() == approx(1.18704299, 5e-2)
        assert ((pattern_found[3].position - cpos) ** 2).sum() == approx(1.18704299, 5e-2)

def test_find_pattern_in_structure__octane_has_12_CH2():
    # there are technically 12 matches, since each CH3 makes 3 variations of CH2
    # CH3 CH2 CH2 CH2 CH2 CH2 CH2 CH3 #
    with importlib.resources.path(tests, "octane.xyz") as octane_path:
        structure = ase.io.read(octane_path)
    pattern = Atoms('CHH', positions=[(0, 0, 0),(-0.1  , -0.379, -1.017), (-0.547, -0.647,  0.685)])
    match_indices, match_atoms = find_pattern_in_structure(structure, pattern)

    assert len(match_atoms) == 12
    for pattern_found in match_atoms:
        assert pattern_found.get_chemical_symbols() == ["C", "H", "H"]
        cpos = pattern_found[0].position
        assert ((pattern_found[1].position - cpos) ** 2).sum() == approx(1.18704299, 5e-2)
        assert ((pattern_found[2].position - cpos) ** 2).sum() == approx(1.18704299, 5e-2)

def test_find_pattern_in_structure__octane_over_pbc_has_2_CH3():
    # CH3 CH2 CH2 CH2 CH2 CH2 CH2 CH3 #
    with importlib.resources.path(tests, "octane.xyz") as octane_path:
        structure = ase.io.read(octane_path)#[0:4]
        positions = structure.get_positions()

        # move positions to get part of CH3 across two boundary conditions
        positions += -1.8

        # move coordinates into main 15 Å unit cell
        positions %= 15
        structure.set_positions(positions)
        structure.set_cell(15 * np.identity(3))

    pattern = Atoms('CHHH', positions=[(0, 0, 0), (-0.538, -0.635,  0.672), (-0.397,  0.993,  0.052), (-0.099, -0.371, -0.998)])
    match_indices, match_atoms = find_pattern_in_structure(structure, pattern)
    assert len(match_atoms) == 2
    for pattern_found in match_atoms:
        assert pattern_found.get_chemical_symbols() == ["C", "H", "H", "H"]

def test_find_pattern_in_structure__hkust1_unit_cell_has_32_benzene_rings():
    with importlib.resources.path(tests, "HKUST-1_withbonds.cif") as hkust1_path:
        structure = ase.io.read(hkust1_path)
    with importlib.resources.path(tests, "HKUST-1_benzene.xyz") as linker_path:
        pattern = ase.io.read(linker_path)
    match_indices, match_atoms = find_pattern_in_structure(structure, pattern)

    assert len(match_atoms) == 32
    for pattern_found in match_atoms:
        assert pattern_found.get_chemical_symbols() == ['C','C','C','C','C','C','H','H','H']
        assert ((pattern_found[0].position - pattern_found[1].position) ** 2).sum() == approx(5.8620934418, 5e-2)
        assert ((pattern_found[0].position - pattern_found[3].position) ** 2).sum() == approx(1.9523164046, 5e-2)
        assert ((pattern_found[0].position - pattern_found[4].position) ** 2).sum() == approx(7.8072193204, 5e-2)
        assert ((pattern_found[5].position - pattern_found[8].position) ** 2).sum() == approx(0.8683351588, 5e-2)

def test_find_pattern_in_structure__hkust1_unit_cell_has_48_Cu_metal_nodes():
    with importlib.resources.path(tests, "HKUST-1_withbonds.cif") as hkust1_path:
        structure = ase.io.read(hkust1_path)
    pattern = Atoms('Cu', positions=[(0, 0, 0)])
    match_indices, match_atoms = find_pattern_in_structure(structure, pattern)

    assert len(match_atoms) == 48
    for pattern_found in match_atoms:
        assert pattern_found.get_chemical_symbols() == ['Cu']

def test_find_pattern_in_structure__hkust1_3x3x3_supercell_has_864_benzene_rings():
    with importlib.resources.path(tests, "HKUST-1_3x3x3.cif") as hkust1_path:
        structure = ase.io.read(hkust1_path)
    with importlib.resources.path(tests, "HKUST-1_benzene.xyz") as linker_path:
        pattern = ase.io.read(linker_path)
    match_indices, match_atoms = find_pattern_in_structure(structure, pattern)

    assert len(match_atoms) == 864
    for pattern_found in match_atoms:
        assert pattern_found.get_chemical_symbols() == ['C','C','C','C','C','C','H','H','H']
        assert ((pattern_found[0].position - pattern_found[1].position) ** 2).sum() == approx(5.8620934418, 5e-2)
        assert ((pattern_found[0].position - pattern_found[3].position) ** 2).sum() == approx(1.9523164046, 5e-2)
        assert ((pattern_found[0].position - pattern_found[4].position) ** 2).sum() == approx(7.8072193204, 5e-2)
        assert ((pattern_found[5].position - pattern_found[8].position) ** 2).sum() == approx(0.8683351588, 5e-2)

def test_find_pattern_in_structure__hkust1_3x3x3_supercell_has_1296_Cu_metal_nodes():
    with importlib.resources.path(tests, "HKUST-1_3x3x3.cif") as hkust1_path:
        structure = ase.io.read(hkust1_path)
    pattern = Atoms('Cu', positions=[(0, 0, 0)])
    match_indices, match_atoms = find_pattern_in_structure(structure, pattern)

    assert len(match_atoms) == 1296
    for pattern_found in match_atoms:
        assert pattern_found.get_chemical_symbols() == ['Cu']

def test_replace_pattern_in_structure__replace_hydrogens_in_octane_with_nothing():
    # CH3 CH2 CH2 CH2 CH2 CH2 CH2 CH3 #
    with importlib.resources.path(tests, "octane.xyz") as octane_path:
        structure = ase.io.read(octane_path)
    search_pattern = Atoms('H', positions=[(0, 0, 0)])
    replace_pattern = search_pattern

    replaced_structure = replace_pattern_in_structure(structure, search_pattern, replace_pattern)
    assert len(replaced_structure) == 8
    assert replaced_structure.get_chemical_symbols() == ["C"] * 8

def test_replace_pattern_in_structure__replace_hydrogens_in_octane_with_hydrogens():
    # CH3 CH2 CH2 CH2 CH2 CH2 CH2 CH3 #
    with importlib.resources.path(tests, "octane.xyz") as octane_path:
        structure = ase.io.read(octane_path)
    search_pattern = Atoms('H', positions=[(0, 0, 0)])
    replace_pattern = search_pattern

    replaced_structure = replace_pattern_in_structure(structure, search_pattern, replace_pattern)
    assert len(replaced_structure) == 26
    assert replaced_structure.get_chemical_symbols() == ["C", "H", "H", "H", "C", "H", "H",
        "C", "H", "H", "C", "H", "H", "C", "H", "H", "C", "H", "H", "C", "H", "H", "C", "H", "H", "H"]

    # TODO: assert positions are the same as when we started
