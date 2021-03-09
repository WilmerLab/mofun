import math

import networkx as nx

from mofun.uff4mof import UFF4MOF, uff_key_starts_with

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

    rBO = -0.1332 * (ri + rj) * math.log(bond_order)
    rEN = (ri * rj * (chii**0.5 - chij**0.5)**2) / (chii * ri + chij * rj)

    rij = ri + rj + rBO - rEN
    kij = 664.12 * zi * zj / (rij**3)

    return (kij, rij)

def angle_params(a1, a2, a3, a2_coord_num, bond_orders=[None, None]):
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

    theta0deg = UFF4MOF[a2][1]
    theta0rad = theta0deg * 2 * math.pi / 360

    # Determine force constant
    rij = bond_params(a1, a2, bond_order=bond_orders[0])[1]
    rjk = bond_params(a2, a3, bond_order=bond_orders[1])[1]
    rik = math.sqrt(rij**2 + rjk**2 - 2 * rij * rjk * math.cos(theta0rad))

    zi = UFF4MOF[a1][5]
    zk = UFF4MOF[a3][5]
    kijk = 664.12 * (zi * zk / rik**5) * (3 * rij * rjk * (1 - math.cos(theta0rad)**2) -
            (rik**2 * math.cos(theta0rad)))

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
        elif theta0deg == 90. and a2_coord_num == 4:
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

        c2 = 1 / (4 * math.sin(theta0rad)**2)
        c1 = -4 * c2 * math.cos(theta0rad)
        c0 = c2 * (2 * math.cos(theta0rad)**2 + 1)
        return ('fourier', kijk, c0, c1, c2)
