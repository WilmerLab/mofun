
from mofun.uff4mof import UFF4MOF, uff_key_starts_with

def default_uff_rules():
    return {
        "H": [
            ("H_b", dict(bo=2)),
            ("H_", {})
        ],
        "O": [
            ("O_1", dict(bo=0))
        ],
    }

def assign_uff_atom_types(g, elements, override_rules=None):
    """ g is a networkx Graph object
    """
    if override_rules is None:
        override_rules = default_uff_rules()

    uff_keys = UFF4MOF.keys()

    atom_types = []

    for n in g.nodes:
        # handle override rules
        el = elements[n]
        found_type = False
        bo = g.degree(n) - 1

        if el in override_rules:
            for ufftype, reqs in override_rules[el]:
                if "bo" in reqs and bo != reqs['bo']:
                    continue
                found_type = True
                atom_types.append(ufftype)
                break
            if found_type:
                continue

        # handle default cases
        uff_key = "%s%1d" % (el.ljust(2, "_"), bo)
        possible_uff_keys = uff_key_starts_with(uff_key)
        if len(possible_uff_keys) == 1:
            atom_types.append(possible_uff_keys[0])
        elif len(possible_uff_keys) == 0:
            raise Exception("no appropriate UFF key that starts with %s" % uff_key)
        elif len(possible_uff_keys) >= 2:
            raise Exception("too many possible UFF keys that starts with %s with no way of discerning which one: " % (uff_key, possible_uff_keys))

    return atom_types
