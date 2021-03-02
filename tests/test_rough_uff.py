import os

import networkx as nx

from mofun import Atoms
from mofun.rough_uff import assign_uff_atom_types, add_aromatic_flag
from tests.fixtures import uio66_linker_cml


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

def test_assign_uff_atom_types__elements_with_only_one_uff_atom_type_wo_hybridization_get_that_type():
    g = nx.Graph()
    g.add_edges_from([(0,1), (1,2), (2,3), (3,4), (4,5), (5,0)])
    uff_atom_types = assign_uff_atom_types(g, ["F", "Rb", "Li", "Cl", "K", "Br"])
    assert uff_atom_types == ["F_", "Rb", "Li", "Cl", "K_", "Br"]

def test_add_aromatic_flag__uio66_linker_has_aromatic_benzene_ring(uio66_linker_cml):
    g = nx.Graph()
    g.add_edges_from(uio66_linker_cml.bonds)
    add_aromatic_flag(g)
    atom_benzene_ring = ["aromatic" in g.nodes[n] for n in sorted(g.nodes())]
    # Trues indicate where the benzene carbons are in the uio_linker_cml file
    assert atom_benzene_ring == [False, False, False, True, True, False, False, True, True, True, True, False, False, False, False, False]

def test_assign_uff_atom_types__uio66_linker(uio66_linker_cml):
    g = nx.Graph()
    g.add_edges_from(uio66_linker_cml.bonds)
    uff_atom_types = assign_uff_atom_types(g, uio66_linker_cml.elements)
    assert uff_atom_types == ["C_2", "O_1", "O_1", "C_R", "C_R", "H_", "H_", "C_R", "C_R", "C_R",
                              "C_R", "H_", "H_", "O_1", "C_2", "O_1"]
