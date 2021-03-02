
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
