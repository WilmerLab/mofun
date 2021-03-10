from math import sqrt, log, cos, sin, pi

import networkx as nx

from mofun.uff4mof import UFF4MOF, uff_key_starts_with, MAIN_GROUP_ELEMENTS

def default_uff_rules():
    return {
        "H": [
            ("H_b", dict(h=2)),
            ("H_", {})
        ],
        "O": [
            ("O_1", dict(h=0))
        ],
        "C": [
            ("C_R", dict(aromatic=True))
        ]
    }

def add_aromatic_flag(g):
    cycles = nx.cycle_basis(g)
    for cycle in cycles:
        if 5 <= len(cycle) <= 7:
            for n in cycle:
                g.nodes[n]['aromatic'] = True

def assign_uff_atom_types(g, elements, override_rules=None):
    """ g is a networkx Graph object
    """
    if override_rules is None:
        override_rules = default_uff_rules()

    uff_keys = UFF4MOF.keys()

    atom_types = []
    add_aromatic_flag(g)
    for n in sorted(g.nodes):
        # handle override rules
        el = elements[n]
        found_type = False
        h = g.degree(n) - 1

        if el in override_rules:
            for ufftype, reqs in override_rules[el]:
                if "h" in reqs and h != reqs['h']:
                    continue
                if "aromatic" in reqs and "aromatic" not in g.nodes[n]:
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
            raise Exception("too many possible UFF keys that starts with %s with no way of discerning which one: " % (uff_key, possible_uff_keys))

        # 2: try element w/o hybridization
        # Note that UFF4MOF adds some fancy types, i.e. Li3f2, and if the hybridization doesn't match
        # above, then this will still default to the original UFF Li term.
        if (el := uff_key[0:2]) in UFF4MOF:
            atom_types.append(el)
            continue

        raise Exception("no appropriate UFF key that starts with %s" % uff_key)

    return atom_types

def pair_params(a1):
    lj_sigma = UFF4MOF[a1][2] * (2**(-1./6.))
    lj_epsilon = UFF4MOF[a1][3]
    return [lj_epsilon, lj_sigma]

def guess_bond_order(atom1, atom2):
    # This method is 'hacky' at best and could be replaced by something more sophisticated.
    # This is roughly the same as what is used in Pete Boyd's 'lammps-interface'.
    bond_atom_types = {atom1, atom2}
    if len({'H_', 'F_', 'Cl', 'Br', 'I_'}.intersection(bond_atom_types)) > 0:
        return 1
    elif len({'C_3', 'N_3', 'O_3'}.intersection(bond_atom_types)) > 0:
        return 1
    elif len(bond_atom_types) == 1 and bond_atom_types.issubset({'C_2', 'N_2', 'O_2'}):
        return 2
    elif len(bond_atom_types) == 1 and bond_atom_types.issubset({'C_R', 'N_R', 'O_R'}):
        return 1.5
    else:
        print('%s %s Bond order not properly assigned. Using default value of 1.' % (atom1, atom2))
        return 1

def bond_params(a1, a2, bond_order=None):
    """calculate the UFF bond parameters in a form suitable for calculations in LAMMPS.

    Args:
        a1 (str): UFF atom type of atom 1 in bond 1-2, e.g. "C_R"
        a2 (str): UFF atom type of atom 2 in bond 1-2, e.g. "C_R"
        bond_order (float): bond order for the bond

    Returns:
        (float, float): tuple of bond force constant and bond length.
    """

    if bond_order is None:
        bond_order = guess_bond_order(a1, a2)

    # 1. Calculate rij and Kij
    ri, zi, chii = [UFF4MOF[a1][k] for k in (0, 5, 8)]
    rj, zj, chij = [UFF4MOF[a2][k] for k in (0, 5, 8)]

    rBO = -0.1332 * (ri + rj) * log(bond_order)
    rEN = (ri * rj * (chii**0.5 - chij**0.5)**2) / (chii * ri + chij * rj)

    rij = ri + rj + rBO - rEN
    kij = 664.12 * zi * zj / (rij**3)

    return (kij, rij)

def angle_params(a1, a2, a3, bond_orders=[None, None]):
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

    a2_coord_is_4 = a2[2] == "3" if len(a2) > 2 else False

    theta0deg = UFF4MOF[a2][1]
    theta0rad = theta0deg * 2 * pi / 360

    # Determine force constant
    rij = bond_params(a1, a2, bond_order=bond_orders[0])[1]
    rjk = bond_params(a2, a3, bond_order=bond_orders[1])[1]
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

def dihedral_params(a1, a2, a3, a4, num_dihedrals_about_bond=1, bond_order=None):
    """Use a small cosine Fourier expansion

    E_phi = 1/2*V_phi * [1 - cos(n*phi0)*cos(n*phi)]


    this is available in Lammps in the form of a harmonic potential
    E = K * [1 + d*cos(n*phi)]

    NB: the d term must be negated to recover the UFF potential.
    """

    # get the elements and hybridizations from each UFF atom type
    ut = [a1, a2, a3, a4]
    el = [s[0:2].strip('_') for s in ut]
    h = [s[2] if len(s) > 2 else 0 for s in ut]

    if bond_order is None:
        bond_order = guess_bond_order(a2, a3)

    oxygen_group = {'O', 'S', 'Se', 'Te', 'Po'}
    #need bond order!

    print("dihedral: %s-%s-%s-%s M=%d" % (*tuple(ut), num_dihedrals_about_bond))
    if {h[1], h[2]} <= {'3'}:
        print("sp3-sp3")
        # center atoms are sp3, use eq 16 from Rappe
        # for both cases handled here, n*theta = 180, so cos(n*theta0)=-1, hence d=1
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
        # center atoms are sp2 (or aromatic), use eq 17 from Rappe, theta0=180, hence d=1
        v = 5.0 * sqrt(UFF4MOF[a2][7] * UFF4MOF[a3][7]) * (1. + 4.18 * log(bond_order)) / num_dihedrals_about_bond
        n = 2
        return ("harmonic", v/2, 1, n)

    elif {h[1], h[2]} <= {'2', 'R', '3'}:
        print("sp2-sp3")
        # mixed sp2 / sp3 / aromatic case
        if {h[0], h[1]} <= {'2'} or {h[2], h[3]} <= {'2'}:
            print("exception: sp2 to another sp2")
            # exception for when the sp2 is bonded to another sp2, d=1
            n = 3
            v = 2. / num_dihedrals_about_bon
            return ("harmonic", v/2, 1, n)

        # use eq 17 from rappe
        v = 5.0 * sqrt(UFF4MOF[ut[1]][7] * UFF4MOF[ut[2]][7]) * (1. + 4.18 * log(bond_order)) / num_dihedrals_about_bond

        if (h[1] == '3' and el[1] in oxygen_group and el[2] not in oxygen_group) or \
             (h[2] == '3' and el[2] in oxygen_group and el[1] not in oxygen_group):
            print("exception: sp3 oxy, sp2/resonant other")
            # exception for sp3 from oxygen column and sp2 or resonant from another column, d=1
            n = 2
            return ("harmonic", v/2, 1, n)
        else:
            # default mixed sp2 / sp3 case
            n = 6
            return ("harmonic", v/2, -1, n)
    elif '1' in {h[1], h[2]}:
        # no dihedrals for "sp-hybridized centers X_1"
        return None
    elif not {el[1], el[2]} <= set(MAIN_GROUP_ELEMENTS):
        # no dihedrals for non main group elements
        return None
    else:
        raise Exception("we don't know how to handle this dihedral: %s-%s-%s-%s" % tuple(ut))
