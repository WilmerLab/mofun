import os

import networkx as nx

from mofun import Atoms
from mofun.rough_uff import calc_uff_atom_types, add_aromatic_flag, guess_bond_order, \
                            bond_params, angle_params, dihedral_params, calc_angles, calc_dihedrals, retype_atoms_from_uff_types
from tests.fixtures import uio66_linker_cml

from pytest import approx

def test_guess_bond_order__two_tetrahedrals_is_1():
    assert guess_bond_order('C_3', 'C_3') == 1
    assert guess_bond_order('N_3', 'N_3') == 1
    assert guess_bond_order('O_3', 'O_3') == 1
    assert guess_bond_order('O_3', 'C_3') == 1
    assert guess_bond_order('N_3', 'C_3') == 1
    assert guess_bond_order('N_3', 'O_3') == 1

def test_guess_bond_order__H_and_group_7_elements_is_1():
    assert guess_bond_order('H_', 'C_3') == 1
    assert guess_bond_order('F_', 'C_3') == 1
    assert guess_bond_order('Cl', 'C_3') == 1
    assert guess_bond_order('Br', 'C_3') == 1
    assert guess_bond_order('I', 'C_3') == 1

def test_guess_bond_order__CNO_trigonal_pairs_is_2():
    assert guess_bond_order('C_2', 'C_2') == 2
    assert guess_bond_order('N_2', 'N_2') == 2
    assert guess_bond_order('O_2', 'O_2') == 2
    assert guess_bond_order('C_2', 'N_2') == 1
    assert guess_bond_order('N_2', 'O_2') == 1
    assert guess_bond_order('O_2', 'C_2') == 1

def test_guess_bond_order__CNO_aromatic_pairs_is_1_5():
    assert guess_bond_order('C_R', 'C_R') == 1.5
    assert guess_bond_order('N_R', 'N_R') == 1.5
    assert guess_bond_order('O_R', 'O_R') == 1.5
    assert guess_bond_order('C_R', 'N_2') == 1
    assert guess_bond_order('N_R', 'O_2') == 1
    assert guess_bond_order('O_R', 'C_2') == 1

def test_guess_bond_order__rules_for_azido_is_2():
    rules = [({'N_1'}, 2), ({'N_1', 'N_2'}, 2)]
    assert guess_bond_order('N_1', 'N_1') == 1
    assert guess_bond_order('N_1', 'N_1', rules) == 2
    assert guess_bond_order('N_1', 'N_2') == 1
    assert guess_bond_order('N_1', 'N_2', rules) == 2

def test_calc_uff_atom_types__carbon_in_methane_is_tetrahedral():
    uff_atom_types = calc_uff_atom_types([(0,1), (0,2), (0,3), (0,4)], ["C", "H", "H", "H", "H"])
    assert uff_atom_types == ["C_3", "H_", "H_", "H_", "H_"]

def test_calc_uff_atom_types__carbon_in_co2_is_linear():
    uff_atom_types = calc_uff_atom_types([(0,1), (0,2)], ["C", "O", "O"])
    assert uff_atom_types == ["C_1", "O_1", "O_1"]

def test_calc_uff_atom_types__carbon_in_benzene_ring_is_aromatic():
    uff_atom_types = calc_uff_atom_types([(0,1), (1,2), (2,3), (3,4), (4,5), (5,0)], ["C"] * 6)
    assert uff_atom_types == ["C_R"] * 6

def test_calc_uff_atom_types__N_in_RNH2_is_tetrahedral():
    uff_atom_types = calc_uff_atom_types([(0,1), (0,2), (0,3), (1,4)], ["N", "C", "H", "H", "H"])
    assert uff_atom_types[0] == "N_3"

def test_calc_uff_atom_types__N_in_RNNN_is_trigonal():
    uff_atom_types = calc_uff_atom_types([(0,1), (1,2), (0,3), (3,4)], ["N", "N", "N", "C", "H"])
    assert uff_atom_types[0] == "N_2"

def test_calc_uff_atom_types__N_in_RNC4_ring_is_aromatic():
    uff_atom_types = calc_uff_atom_types([(0,1), (1,2), (2,3), (3,4), (4,0), (0,5), (5,6)], ["N", "C", "C", "C", "C", "C", "H"])
    assert uff_atom_types[0] == "N_R"

def test_calc_uff_atom_types__NNN_override_rule_neighbors_assigns_N1():
    override_rules = {
        "N": [
            ("N_1", dict(neighbors=("N","N"))),
        ]
    }
    uff_atom_types = calc_uff_atom_types([(0,1), (1,2), (0,3), (3,4)], ["N", "N", "N", "C", "H"], override_rules=override_rules)
    assert uff_atom_types[0:3] == ["N_2", "N_1", "N_1"]

def test_calc_uff_atom_types__elements_with_only_one_uff_atom_type_wo_hybridization_get_that_type():
    uff_atom_types = calc_uff_atom_types([(0,1), (1,2), (2,3), (3,4), (4,5), (5,0)], ["F", "Rb", "Li", "Cl", "K", "Br"])
    assert uff_atom_types == ["F_", "Rb", "Li", "Cl", "K_", "Br"]

def test_add_aromatic_flag__uio66_linker_has_aromatic_benzene_ring(uio66_linker_cml):
    g = nx.Graph()
    g.add_edges_from(uio66_linker_cml.bonds)
    add_aromatic_flag(g)
    atom_benzene_ring = ["aromatic" in g.nodes[n] for n in sorted(g.nodes())]
    # Trues indicate where the benzene carbons are in the uio_linker_cml file
    assert atom_benzene_ring == [False, False, False, True, True, False, False, True, True, True, True, False, False, False, False, False]

def test_calc_uff_atom_types__uio66_linker(uio66_linker_cml):
    uff_atom_types = calc_uff_atom_types(uio66_linker_cml.bonds, uio66_linker_cml.elements)
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

def test_dihedral_params__X_1_is_none():
    assert dihedral_params('C_R', 'N_2', 'N_1', 'N_1', 1) is None

def test_calc_angles__ethane_has_12_angles():
    angles = calc_angles([(0,3), (1,3), (2,3), (3,4), (4,5), (4,6), (4,7)])
    unique_angles = set([tuple(x) for x in angles])
    assert len(angles) == 12
    assert unique_angles == {(0,3,1), (0,3,2), (1,3,2), (0,3,4), (1,3,4), (2,3,4),
                             (3,4,5), (3,4,6), (3,4,7), (5,4,6), (5,4,7), (6,4,7)}

def test__dihedrals__ethane_has_9_dihedrals():
    dihedrals = calc_dihedrals([(0,3), (1,3), (2,3), (3,4), (4,5), (4,6), (4,7)])
    unique_dihedrals = set([tuple(x) for x in dihedrals])
    assert len(dihedrals) == 9
    assert unique_dihedrals == {(0,3,4,5), (1,3,4,5), (2,3,4,5),
                                (0,3,4,6), (1,3,4,6), (2,3,4,6),
                                (0,3,4,7), (1,3,4,7), (2,3,4,7)}

def test__retype_atoms_from_uff_types__boron_gives_all_boron():
    cnnc = Atoms(elements='CNNC', positions=[(0., 0., 0), (1.0, 0., 0.), (2.0, 0., 0.), (3.0, 0., 0.)])
    retype_atoms_from_uff_types(cnnc, ["B"] * 4)
    assert cnnc.atom_types == [0] * 4
    assert cnnc.atom_type_labels == ["B"]
    assert cnnc.atom_type_elements == ["B"]
    assert cnnc.atom_type_masses == [10.811]

def test__retype_atoms_from_uff_types__BZrZrB_gives_BZrZrB():
    cnnc = Atoms(elements='CNNC', positions=[(0., 0., 0), (1.0, 0., 0.), (2.0, 0., 0.), (3.0, 0., 0.)])
    retype_atoms_from_uff_types(cnnc, ["B", "Zr", "Zr", "B"])
    assert cnnc.atom_types == [0, 1, 1, 0]
    assert cnnc.atom_type_labels == ["B", "Zr"]
    assert cnnc.atom_type_elements == ["B", "Zr"]
    assert cnnc.atom_type_masses == [10.811, 91.224]

def test__retype_atoms_from_uff_types__BZrZrB_with_full_atom_types_gives_BZrZrB():
    cnnc = Atoms(elements='CNNC', positions=[(0., 0., 0), (1.0, 0., 0.), (2.0, 0., 0.), (3.0, 0., 0.)])
    retype_atoms_from_uff_types(cnnc, ["B_3", "Zr8f4", "Zr8f4", "B_3"])
    assert cnnc.atom_types == [0, 1, 1, 0]
    assert cnnc.atom_type_labels == ["B_3", "Zr8f4"]
    assert cnnc.atom_type_elements == ["B", "Zr"]
    assert cnnc.atom_type_masses == [10.811, 91.224]

def test__retype_atoms_from_uff_types__C2C1ZrC3H_with_full_atom_types_orders_as_HCCCZr():
    cnnnc = Atoms(elements='CNNNC', positions=[(0., 0., 0), (1.0, 0., 0.), (2.0, 0., 0.), (3.0, 0., 0.), (4.0, 0., 0.)])
    retype_atoms_from_uff_types(cnnnc, ["C_2", "C_1", "Zr8f4", "C_3", "H_"])
    assert cnnnc.atom_types == [2, 1, 4, 3, 0]
    assert cnnnc.atom_type_labels == ["H_", "C_1", "C_2", "C_3", "Zr8f4"]
    assert cnnnc.atom_type_elements == ["H", "C", "C", "C", "Zr"]
    assert cnnnc.atom_type_masses == [1.00794, 12.0107, 12.0107, 12.0107, 91.224]
