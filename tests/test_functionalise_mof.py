import importlib
from importlib import resources

import ase
from ase import Atoms

from functionalise_mof import find_pattern_in_structure
import tests

def test_find_pattern_in_structure__find_all_carbons_in_octane():
    # CH3 CH2 CH2 CH2 CH2 CH2 CH2 CH3 #
    with importlib.resources.path(tests, "octane.xyz") as octane_path:
        structure = ase.io.read(octane_path)
    pattern = Atoms('C', positions=[(0, 0, 0)])
    search_results = find_pattern_in_structure(structure, pattern)
    assert len(search_results) == 8


#
# def find_pattern_in_structure():
#     pass

# def test_find_pattern_in_structure__find_all_ch2_in_octane():
#     # CH3 CH2 CH2 CH2 CH2 CH2 CH2 CH3 #
#     structure = ... [octane]
#     pattern = ... [ch2]
#     search_results = find_pattern_in_structure(structure, pattern)
#     assert len(search_results) == 6
#
# def test_find_pattern_in_structure__find_all_ch2_in_octane_over_pbc():
#     # CH3 CH2 CH2 CH2 C|PBC| H2 CH2 CH2 CH3 #
#     structure = ... [octane]
#     pattern = ... [ch2]
#     search_results = find_pattern_in_structure(structure, pattern)
#     assert len(search_results) == 6
