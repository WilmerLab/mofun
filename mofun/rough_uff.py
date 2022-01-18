from collections import Counter
import itertools
from math import sqrt, log, cos, sin, pi
import sys

import networkx as nx
import numpy as np

from mofun.atomic_masses import ATOMIC_MASSES
from mofun.uff4mof import UFF4MOF, uff_key_starts_with, MAIN_GROUP_ELEMENTS
from mofun.helpers import typekey

def default_uff_rules():
    return {
        "H": [
            ("H_b", dict(n=2)),
            ("H_", {})
        ],
        "O": [
            ("O_1", dict(n=1))
        ],
        "C": [
            ("C_R", dict(aromatic=True))
        ],
        "N": [
            ("N_R", dict(aromatic=True))
        ]
    }

def add_aromatic_flag(g):
    cycles = nx.cycle_basis(g)
    for cycle in cycles:
        if 5 <= len(cycle) <= 7:
            for n in cycle:
                g.nodes[n]['aromatic'] = True

def retype_atoms_from_uff_types(atoms, new_types):
    """Retypes an atoms object with new UFF types.

    Takes a new list of types (new_types) of length equal to the number of atoms in the system and
    creates a new set of unique atom_types, where atom_type_labels are the UFF atom types, and atom_type_elements
    and atom_type_masses are the appropriate element and mass for the UFF atom type. New atom_types are
    assigned to the atoms object, based on the new atom type indices.

    Args:
        atoms (Atoms): atoms object to retype
        new_types (List[Str]): list of UFF atom types, one per atom in the system.
    """

    # sort by string ordering, so types like 'C_1', 'C_2', 'C_3', 'C_R' will show up in order
    unique_types = list(set(new_types))
    unique_types.sort()

    # sort by periodic element # order
    ptable_order = lambda x: list(ATOMIC_MASSES.keys()).index(x[0:2].replace('_', ''))
    unique_types.sort(key=ptable_order)

    atoms.atom_type_labels = unique_types
    atoms.atom_type_elements = [s[0:2].replace('_', '') for s in unique_types]
    atoms.atom_type_masses = [ATOMIC_MASSES[s] for s in atoms.atom_type_elements]

    atoms.atom_types = [unique_types.index(s) for s in new_types]

def calc_uff_atom_types(bonds, elements, override_rules=None):
    """
    """

    g = nx.Graph()
    g.add_edges_from(bonds)

    if override_rules is None:
        override_rules = default_uff_rules()

    uff_keys = UFF4MOF.keys()

    atom_types = []
    add_aromatic_flag(g)
    for n in sorted(g.nodes):
        # handle override rules
        el = elements[n]
        found_type = False

        if el == "C":
            h = g.degree(n) - 1
        elif el == "N":
            h = g.degree(n)
        elif el == "O":
            if g.degree(n) == 1:
                h = 2
            else:
                h = 3
        else:
            h = g.degree(n) - 1

        if el in override_rules:
            for ufftype, reqs in override_rules[el]:
                if "n" in reqs and g.degree(n) != reqs['n']:
                    continue
                if "h" in reqs and h != reqs['h']:
                    continue
                if "aromatic" in reqs and "aromatic" not in g.nodes[n]:
                    continue
                if "neighbors" in reqs and \
                        sorted([elements[i] for i in g.neighbors(n)]) != sorted(reqs['neighbors']):
                    continue
                found_type = True
                atom_types.append(ufftype)
                break
            if found_type:
                continue

        # handle default cases
        # 1: try element + hybridization
        uff_key = "%s%1d" % (el.ljust(2, "_"), h)
        possible_uff_keys = uff_key_starts_with(uff_key)
        if len(possible_uff_keys) == 1:
            atom_types.append(possible_uff_keys[0])
            continue
        elif len(possible_uff_keys) >= 2:
            if uff_key in UFF4MOF:
                atom_types.append(uff_key)
                print("WARNING: multiple possible UFF keys: %s. Choosing the simplest one: %s" % (possible_uff_keys, uff_key), file=sys.stderr)
                continue
            else:
                raise Exception("Error: too many possible UFF keys that starts with %s with no way of discerning which one: %s" % (uff_key, possible_uff_keys))

        # 2: try element w/o hybridization
        # Note that UFF4MOF adds some fancy types, i.e. Li3f2, and if the hybridization doesn't match
        # above, then this will still default to the original UFF Li term.
        if (el := uff_key[0:2]) in UFF4MOF:
            atom_types.append(el)
            continue

        raise Exception("no appropriate UFF key that starts with %s" % uff_key)

    return atom_types

def pair_coeffs(a1):
    lj_sigma = UFF4MOF[a1][2] * (2**(-1./6.))
    lj_epsilon = UFF4MOF[a1][3]
    return [lj_epsilon, lj_sigma]

def guess_bond_order(a1, a2, rules=[]):
    """
    Args:
        a1 (str): UFF atom type of atom 1 in bond 1-2, e.g. "C_R"
        a2 (str): UFF atom type of atom 2 in bond 1-2, e.g. "C_R"
        rules (optional): a list of tuple pairs where each pair contains a set of atom_types
            representing a bond, and a bo. E.g.: [({N_1}, 2), ({N_1, N_2}, 2)] where that means a
            bond between two N_1 atoms would have a bond order of 2, and a bond between an N_1 and a
            N_2 would also have a bond_order of 2.

    This method is 'hacky' at best and could be replaced by something more sophisticated.
    This is roughly the same as what is used in Pete Boyd's 'lammps-interface'.
    """
    bond_atom_types = {a1, a2}

    for rule_atom_types, bo in rules:
        if bond_atom_types == rule_atom_types:
            return bo

    if len({'H_', 'F_', 'Cl', 'Br', 'I_', 'C_3', 'N_3', 'O_3'} & bond_atom_types) > 0:
        return 1
    elif len(bond_atom_types) == 1 and bond_atom_types <= {'C_2', 'N_2', 'O_2'}:
        return 2
    elif len(bond_atom_types) == 1 and bond_atom_types <= {'C_R', 'N_R', 'O_R'}:
        return 1.5
    else:
        print('%s %s Bond order not explicitly assigned. Using default value of 1.' % (a1, a2), file=sys.stderr)
        return 1

def bond_params(a1, a2, bond_order=None, bond_order_rules=[]):
    """calculate the UFF bond parameters in a form suitable for calculations in LAMMPS.

    Args:
        a1 (str): UFF atom type of atom 1 in bond 1-2, e.g. "C_R"
        a2 (str): UFF atom type of atom 2 in bond 1-2, e.g. "C_R"
        bond_order (float): bond order for the bond

    Returns:
        (float, float): tuple of bond force constant and bond length.
    """

    if bond_order is None:
        bond_order = guess_bond_order(a1, a2, bond_order_rules)

    # 1. Calculate rij and Kij
    ri, zi, chii = [UFF4MOF[a1][k] for k in (0, 5, 8)]
    rj, zj, chij = [UFF4MOF[a2][k] for k in (0, 5, 8)]

    rBO = -0.1332 * (ri + rj) * log(bond_order)
    rEN = (ri * rj * (chii**0.5 - chij**0.5)**2) / (chii * ri + chij * rj)

    rij = ri + rj + rBO - rEN
    kij = 664.12 * zi * zj / (rij**3)

    return (kij / 2, rij)

def angle_params(a1, a2, a3, bond_orders=[None, None], bond_order_rules=[]):
    """calculate the UFF angle parameters in a form suitable for calculations in LAMMPS.

    Args:
        a1 (str): UFF atom type of atom 1 in angle 1-2-3, e.g. "C_R"
        a2 (str): UFF atom type of atom 2 in angle 1-2-3, e.g. "C_R"
        a3 (str): UFF atom type of atom 3 in angle 1-2-3, e.g. "C_R"
        a2_coord_num (int): coordination # of a2 atom.
        bond_orders ([float, float]): bond orders for the 1-2 and 2-3 bonds.

    Returns:
        (str, float, int, int): For linear cases, returns a tuple of ('cosine/periodic', C, b, n)
            where C, b, n are defined by https://lammps.sandia.gov/doc/angle_cosine_periodic.html.
        (str, float, float, float): For the nonlinear general case, returns a tuple of
            ('fourier', K, c0, c1, c2) where the args are defined by
            https://lammps.sandia.gov/doc/angle_fourier.html
    """

    a2_coord_is_4 = (a2[2] == "3") if len(a2) > 2 else False

    theta0deg = UFF4MOF[a2][1]
    theta0rad = theta0deg * 2 * pi / 360

    # Determine force constant
    rij = bond_params(a1, a2, bond_order=bond_orders[0], bond_order_rules=bond_order_rules)[1]
    rjk = bond_params(a2, a3, bond_order=bond_orders[1], bond_order_rules=bond_order_rules)[1]
    rik = sqrt(rij**2 + rjk**2 - 2 * rij * rjk * cos(theta0rad))

    zi = UFF4MOF[a1][5]
    zk = UFF4MOF[a3][5]
    kijk = 664.12 * (zi * zk / rik**5) * (3 * rij * rjk * (1 - cos(theta0rad)**2) -
            (rik**2 * cos(theta0rad)))

    if theta0deg in [180., 120., 90.]:
        # Linear cases use a simplified fourier expansion. This is written in the Rappe paper as:
        # E = Kijk/n^2[1 - cos(n*theta)]
        # The corresponding term in LAMMPS is a cosine/periodic term of the form:
        # E = C[1 - B(-1)^n cos(n*theta)]

        if theta0deg == 180.:
            n = 1
            # use 1 + cos x
            b = 1
        elif theta0deg == 120.:
            n = 3
            b = -1
        elif theta0deg == 90. and a2_coord_is_4:
            n = 2
            # use 1 + cos x
            b = -1
        elif theta0deg == 90.:
            n = 4
            b = 1

        return ('cosine/periodic', kijk, b, n)
    else:
        # General nonlinear cases use a three-term fourier expansion of the form:
        # E = Kijk * [C0 + C1 cos(theta) + c2 cos(2*theta)]
        # The corresponding term in LAMMPS is a fourier term.

        c2 = 1 / (4 * sin(theta0rad)**2)
        c1 = -4 * c2 * cos(theta0rad)
        c0 = c2 * (2 * cos(theta0rad)**2 + 1)
        return ('fourier', kijk, c0, c1, c2)

def dihedral_params(a1, a2, a3, a4, num_dihedrals_about_bond=1, bond_order=None, bond_order_rules=[]):
    """Use a small cosine Fourier expansion

    E_phi = 1/2*V_phi * [1 - cos(n*phi0)*cos(n*phi)]

    this is available in Lammps in the form of a harmonic potential
    E = K * [1 + d*cos(n*phi)]

    K = V_phi / 2
    d = -cos(n*phi0)
    n = n

    NB: the d term must be negated to recover the UFF potential.
    """

    # get the elements and hybridizations from each UFF atom type
    ut = [a1, a2, a3, a4]
    el = [s[0:2].strip('_') for s in ut]
    h = [s[2] if len(s) > 2 else 0 for s in ut]

    if bond_order is None:
        bond_order = guess_bond_order(a2, a3, bond_order_rules)

    oxygen_group = {'O', 'S', 'Se', 'Te', 'Po'}

    print("dihedral: %s-%s-%s-%s M=%d" % (*tuple(ut), num_dihedrals_about_bond))
    if {h[1], h[2]} <= {'3'}:
        print("sp3-sp3")
        # center atoms are sp3, use eq 16 from Rappe
        # n=3, phi0 = 180 or 60
        # cos(n*phi0) = cos(3*180), cos(3*60) => cos(540), cos(180) = -1
        # for oxy exception: cos(2*90) = cos(180) = -1
        # hence, d = 1

        n = 3
        v1 = UFF4MOF[ut[1]][6]
        v2 = UFF4MOF[ut[2]][6]

        # exception for when both atoms are from group 6
        if {el[1], el[2]} <= oxygen_group:
            print("exception: both oxys")
            n = 2
            v1 = 2. if el[1] == "O" else 6.8
            v2 = 2. if el[2] == "O" else 6.8

        v = sqrt(v1*v2) / num_dihedrals_about_bond
        return ("harmonic", v/2, 1, n)

    elif {h[1], h[2]} <= {'2', 'R'}:
        print("sp2-sp2")
        # center atoms are sp2 (or aromatic), use eq 17 from Rappe,
        # phi0=180, n=2, cos(2*180) = 1, hence d = -1
        v = 5.0 * sqrt(UFF4MOF[a2][7] * UFF4MOF[a3][7]) * (1. + 4.18 * log(bond_order)) / num_dihedrals_about_bond
        n = 2
        return ("harmonic", v/2, -1, n)

    elif {h[1], h[2]} <= {'2', 'R', '3'}:
        print("sp2-sp3")
        # mixed sp2 / sp3 / aromatic case
        if {h[0], h[1]} <= {'2'} or {h[2], h[3]} <= {'2'}:
            print("exception: sp2 to another sp2")
            # exception for when the sp2 is bonded to another sp2,
            # n = 3, phi0 = 180, cos(3*180) = -1, d=1

            n = 3
            v = 2. / num_dihedrals_about_bond
            return ("harmonic", v/2, 1, n)

        if (h[1] == '3' and el[1] in oxygen_group and el[2] not in oxygen_group) or \
             (h[2] == '3' and el[2] in oxygen_group and el[1] not in oxygen_group):
            print("exception: sp3 oxy, sp2/resonant other")
            # exception for sp3 from oxygen column and sp2 or resonant from another column
            # phi0=90, n=2, cos(2*90) = -1, d = 1
            # use eq 17 from rappe
            v = 5.0 * sqrt(UFF4MOF[ut[1]][7] * UFF4MOF[ut[2]][7]) * (1. + 4.18 * log(bond_order)) / num_dihedrals_about_bond
            n = 2
            return ("harmonic", v/2, 1, n)

        # default mixed sp2 / sp3 case
        # phi0 = 0, cos 6*0 = 1, d = -1
        n = 6
        v = 1. / num_dihedrals_about_bond
        return ("harmonic", v/2, -1, n)
    elif '1' in {h[1], h[2]}:
        # no dihedrals for "sp-hybridized centers X_1"
        return None
    elif not {el[1], el[2]} <= set(MAIN_GROUP_ELEMENTS):
        # no dihedrals for non main group elements
        return None
    else:
        raise Exception("we don't know how to handle this dihedral: %s-%s-%s-%s" % tuple(ut))

def calc_angles(bonds):
    """ Returns all possible angle tuples from a list of all system bond tuples."""
    g = nx.Graph()
    g.add_edges_from(bonds)

    angles = []
    for n in g.nodes:
        angles += [(a, n, b) for (a,b) in itertools.combinations(g.neighbors(n), 2)]
    return np.array(angles)

def calc_dihedrals(bonds):
    """ Returns all possible dihedral tuples from a list of all system bond tuples."""
    g = nx.Graph()
    g.add_edges_from(bonds)
    # g.add_edges_from([tuple(x) for x in bonds])

    dihedrals = []
    for a, b in g.edges:
        a_neighbors = list(g.adj[a])
        a_neighbors.remove(b)
        b_neighbors = list(g.adj[b])
        b_neighbors.remove(a)

        dihedrals += [(a1, a, b, b1) for a1 in a_neighbors for b1 in b_neighbors]
    return np.array(dihedrals)

def delete_if_all_in_set(arr, s):
    deletion_list = []
    for i, tup in enumerate(arr):
        if len(set(tup) - s) == 0:
            deletion_list.append(i)
    return np.delete(arr, deletion_list, axis=0)

def assign_pair_coeffs(atoms, assign_atom_type_labels_from_elements=False):
    if assign_atom_type_labels_from_elements:
        atoms.atom_type_labels = [uff_key_starts_with(el.ljust(2, "_"))[0] for el in atoms.atom_type_elements]

    atoms.pair_coeffs = ['%10.6f %10.6f # %s' % (*pair_coeffs(a1), a1) for a1 in atoms.atom_type_labels]

def assign_bond_types(atoms, uff_atom_types, bond_order_rules=[], exclude=[]):
    if len(exclude) >= 2:
        atoms.bonds = delete_if_all_in_set(atoms.bonds, exclude)

    # get an ordered list of bond types for every bond, e.g. [(1,2), (4,5)]
    bond_types = [typekey([uff_atom_types[a] for a in atup]) for atup in atoms.bonds]
    # reduce the list so every type is unique
    unique_bond_types = list(dict.fromkeys(bond_types).keys())
    # bond_types are the index of the type in the unique_bond_types list
    atoms.bond_types = [unique_bond_types.index(bt) for bt in bond_types]
    # calculate bond_params from bond types and bond_order_rules and assign
    params = [(*bond_params(a1, a2, bond_order_rules=bond_order_rules), a1, a2) for (a1, a2) in unique_bond_types]
    atoms.bond_type_coeffs = ['%10.6f %10.6f # %s %s' % p for p in params]

def angle2lammpsdat(params):
    if params[0] == "fourier":
        return '%s %10.6f %10.6f %10.6f %10.6f # %s' % params
    elif params[0] == "cosine/periodic":
        return '%s %10.6f %d %d # %s' % params
    else:
        raise Exception("Unhandled angle style '%s'" % params[0])

def assign_angle_types(atoms, uff_atom_types, bond_order_rules=[], exclude=[]):
    if len(exclude) >= 3:
        atoms.angles = delete_if_all_in_set(atoms.angles, exclude)

    angle_types = [typekey([uff_atom_types[a] for a in atup]) for atup in atoms.angles]
    unique_angle_types = list(dict.fromkeys(angle_types).keys())
    atoms.angle_types = [unique_angle_types.index(a) for a in angle_types]
    params = [(*angle_params(*a_ids, bond_order_rules=bond_order_rules), "%s %s %s" % a_ids) for a_ids in unique_angle_types]
    atoms.angle_type_coeffs = [angle2lammpsdat(a) for a in params]

def assign_dihedral_types(atoms, uff_atom_types, bond_order_rules=[], exclude=[]):
    """

    Args:
        exclude (List): removes dihedrals from list if the dihedral's atom
            indices are ALL contained within exclude.
    """
    num_dihedrals_per_bond = Counter([typekey([a2, a3]) for _, a2, a3, _ in atoms.dihedrals])
    if len(exclude) >= 4:
        atoms.dihedrals = delete_if_all_in_set(atoms.dihedrals, exclude)

    # dihedral type is a tuple of four atom indices, and the number of dihedrals about the center bond
    dihedral_types = [(*typekey([uff_atom_types[a] for a in atup]), num_dihedrals_per_bond[typekey([atup[1], atup[2]])]) for atup in atoms.dihedrals]
    unique_dihedral_types = list(dict.fromkeys(dihedral_types).keys())
    # params = [dihedral_params(*a_ids, bond_order_rules=bond_order_rules) for a_ids in unique_dihedral_types]
    params = [(dihedral_params(*a_ids, bond_order_rules=bond_order_rules), a_ids) for a_ids in unique_dihedral_types]

    # delete any dihedrals when the params come back None (i.e. for *_1)
    for i in reversed(range(len(params))):
        if params[i][0] is None:
            # delete from atoms.dihedrals and dihedral_types
            d_to_del = unique_dihedral_types[i]
            atoms.dihedrals = [d for j, d in enumerate(atoms.dihedrals) if dihedral_types[j] != d_to_del]
            dihedral_types = [d for d in dihedral_types if d != d_to_del]
            # delete from params and types
            del(unique_dihedral_types[i])
            del(params[i])

    # assign dihedral types
    atoms.dihedral_types = [unique_dihedral_types.index(a) for a in dihedral_types]
    atoms.dihedral_type_coeffs = ['%s %10.6f %d %d # %s %s %s %s M=%d' % (*p1, *p2) for p1, p2 in params]
