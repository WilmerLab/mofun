
import networkx as nx

from mofun.rough_uff import assign_uff_atom_types


def test_assign_uff_atom_types__carbon_in_methane_is_tetrahedral():
    g = nx.Graph()
    g.add_edges_from([(0,1), (0,2), (0,3), (0,4)])
    uff_atom_types = assign_uff_atom_types(g, ["C", "H", "H", "H", "H"])
    assert uff_atom_types == ["C_3", "H_", "H_", "H_", "H_"]

def test_assign_uff_atom_types__carbon_in_co2_is_linear():
    g = nx.Graph()
    g.add_edges_from([(0,1), (0,2)])
    uff_atom_types = assign_uff_atom_types(g, ["C", "O", "O"])
    assert uff_atom_types == ["C_1", "O_1", "O_1"]

def test_assign_uff_atom_types__carbon_in_benzene_ring_is_aromatic():
    g = nx.Graph()
    g.add_edges_from([(0,1), (1,2), (2,3), (3,4), (4,5), (5,0)])
    uff_atom_types = assign_uff_atom_types(g, ["C"] * 6)
    assert uff_atom_types == ["C_R"] * 6
