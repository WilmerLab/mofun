
from mofun.helpers import remove_duplicates, position_index_farthest_from_axis
from tests.fixtures import *

def test_remove_duplicates__leaves_order_untouched():
    assert remove_duplicates([(3, 2, 1)]) == [(3, 2, 1)]
    assert remove_duplicates([(3, 2, 1), (1, 2, 3)]) == [(3, 2, 1)]

def test_remove_duplicates__with_constant_key_is_one_item():
    assert remove_duplicates([(3, 2, 1), (3, 6, 5), (3, 6, 5)], key=lambda m: "asdf") == [(3, 2, 1)]

def test_remove_duplicates__with_no_key_uses_sorted_tuples():
    assert remove_duplicates([(3, 2, 1), (3, 6, 5), (3, 5, 6)]) == [(3, 2, 1), (3, 6, 5)]

def test_position_index_farthest_from_axis__for_octane_is_index_6_or_20(octane):
    # index 20 is one of the H on the benzene ring
    assert position_index_farthest_from_axis(octane.positions[-1] - octane.positions[0], octane) in [6,20]
