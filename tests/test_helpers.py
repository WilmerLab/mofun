
from mofun.helpers import remove_duplicates, position_index_farthest_from_axis
from tests.fixtures import *

def test_remove_duplicates__should_leave_order_untouched():
    assert remove_duplicates([(3, 2, 1)]) == [(3, 2, 1)]
    assert remove_duplicates([(3, 2, 1), (1, 2, 3)]) == [(3, 2, 1)]

def test_position_index_farthest_from_axis__for_octane_is_index_6_or_20(octane):
    # index 20 is one of the H on the benzene ring
    assert position_index_farthest_from_axis(octane.positions[-1] - octane.positions[0], octane) in [6,20]
