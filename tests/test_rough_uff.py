import os

import networkx as nx

from mofun import Atoms
from mofun.rough_uff import assign_uff_atom_types, add_aromatic_flag, \
                            bond_params, angle_params, dihedral_params
from tests.fixtures import uio66_linker_cml

from pytest import approx

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

def test_assign_uff_atom_types__N_in_RNH2_is_tetrahedral():
    g = nx.Graph()
    g.add_edges_from([(0,1), (0,2), (0,3), (1,4)])
    uff_atom_types = assign_uff_atom_types(g, ["N", "C", "H", "H", "H"])
    assert uff_atom_types[0] == "N_3"

def test_assign_uff_atom_types__N_in_RNNN_is_trigonal():
    g = nx.Graph()
    g.add_edges_from([(0,1), (1,2), (0,3), (3,4)])
    uff_atom_types = assign_uff_atom_types(g, ["N", "N", "N", "C", "H"])
    assert uff_atom_types[0] == "N_2"

def test_assign_uff_atom_types__N_in_RNC4_ring_is_aromatic():
    g = nx.Graph()
    g.add_edges_from([(0,1), (1,2), (2,3), (3,4), (4,0), (0,5), (5,6)])
    uff_atom_types = assign_uff_atom_types(g, ["N", "C", "C", "C", "C", "C", "H"])
    assert uff_atom_types[0] == "N_R"


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

def test_bond_params__C_N_amide_is_1293():
    # normative values for bond length and force constant from Towhee
    force_k, bond_length = bond_params("C_R", "N_R", bond_order=1.41)
    assert bond_length == approx(1.3568, 1e-4)
    # /2 here because we are currently returning constants compatible with LAMMPS
    assert force_k == approx(1293.18/2, 5e-2)

def test_angle_params__C_N_C_amide_is_105_5():
    # normative values for force constant from Towhee
    angle_style, force_k, b, n = angle_params("C_3", "N_R", "C_R", bond_orders=[None, 1.41])
    print(angle_style, force_k, b, n)
    assert angle_style  == "cosine/periodic"
    assert force_k == approx(210.97397, 1e-5)

def test_dihedral_params__force_constant_should_match_table_2_kind_of():
    # Note that none of the calculated force constants match table 2 because the dihderal potential
    # is ill-defined. The normative values here come from our attempt at calculating what they
    # should be.

    # test SP3-SP3
    assert dihedral_params("H_", "C_3", "C_3", "H_")[1] * 2 == approx(2.119, abs=1e-3)
    assert dihedral_params("H_", "C_3", "Si3", "H_")[1] * 2 == approx(1.611, abs=1e-3)
    assert dihedral_params("H_", "C_3", "Ge3", "H_")[1] * 2 == approx(1.219, abs=1e-3)
    assert dihedral_params("H_", "C_3", "Sn3", "H_")[1] * 2 == approx(0.649, abs=1e-3)
    assert dihedral_params("H_", "C_3", "N_3", "H_")[1] * 2 == approx(0.977, abs=1e-3)
    assert dihedral_params("H_", "C_3", "P_3+3", "H_")[1] * 2 == approx(2.255, abs=1e-3)
    assert dihedral_params("H_", "C_3", "As3+3", "H_")[1] * 2 == approx(1.783, abs=1e-3)
    assert dihedral_params("H_", "C_3", "O_3", "H_")[1] * 2 == approx(0.195, abs=1e-3)
    assert dihedral_params("H_", "C_3", "S_3+2", "H_")[1] * 2 == approx(1.013, abs=1e-3)
    assert dihedral_params("H_", "C_3", "Se3+2", "H_")[1] * 2 == approx(0.843, abs=1e-3)

    # test SP3-SP3 exception for oxygen group elements
    assert dihedral_params("H_", "O_3", "O_3", "H_")[1] * 2 == approx(2.0, abs=1e-3)
    assert dihedral_params("H_", "S_3+2", "S_3+2", "H_")[1] * 2 == approx(6.8, abs=1e-3)

    # test SP2-SP2
    assert dihedral_params("C_", "O_2", "C_2", "C_")[1] * 2 == approx(10., abs=1e-3)
    assert dihedral_params("C_", "O_2", "S_2", "C_")[1] * 2 == approx(7.906, abs=1e-3)

    # test SP3-SP2
    # propene
    assert dihedral_params("H_", "C_3", "C_2", "C_2")[1] * 2 == approx(2., abs=1e-3)
    assert dihedral_params("C_2", "C_2", "C_3", "H_")[1] * 2 == approx(2., abs=1e-3)
