from contextlib import contextmanager
import random

import numpy as np
from scipy.linalg import norm
from scipy.spatial.transform import Rotation as R

from mofun.atomic_masses import ATOMIC_MASSES

def atoms_of_type(types, element):
    """ returns all atom indices in types that match the symbol element """
    return [i for i, t in enumerate(types) if t == element]

def atoms_by_type_dict(atom_types):
    atoms_by_type = {k:[] for k in set(atom_types)}
    for i, k in enumerate(atom_types):
        atoms_by_type[k].append(i)
    return atoms_by_type

def remove_duplicates(match_indices, key=lambda m: tuple(sorted(m)), pick_random=False):
    keyed_tuples = {}
    for m in match_indices:
        mkey = key(m)
        if mkey not in keyed_tuples:
            keyed_tuples[mkey] = [m]
        else:
            keyed_tuples[mkey].append(m)
    if pick_random:
        return [random.choice(matches) for _, matches in keyed_tuples.items()]
    else: # pick first
        return [matches[0] for _, matches in keyed_tuples.items()]

def position_index_farthest_from_axis(axis, atoms):
    q = quaternion_from_two_vectors(axis, [1., 0., 0.])
    ratoms = q.apply(atoms.positions)
    ss = (ratoms[:,1:3] ** 2).sum(axis=1)
    return np.nonzero(ss==ss.max())[0][0]

def quaternion_from_two_vectors(p1, p2):
    """ returns the quaternion necessary to rotate p1 to p2"""
    v1 = np.array(p1) / norm(p1)
    v2 = np.array(p2) / norm(p2)

    angle = np.arccos(max(-1.0, min(np.dot(v1, v2), 1)))
    axis = np.cross(v1, v2)

    if np.isclose(axis, [0., 0., 0.], 1e-3).all() and angle != 0.0:
        # the antiparallel case requires we arbitrarily find a orthogonal rotation axis, since the
        # cross product of a two parallel / antiparallel vectors is 0.
        axis = np.cross(v1, np.random.random(3))

    if norm(axis) > 1e-15:
        axis /= norm(axis)
    return R.from_quat([*(axis*np.sin(angle / 2)), np.cos(angle/2)])

def quaternion_from_two_vectors_around_axis(p1, p2, axis):
    """ returns the quaternion necessary to rotate p1 to p2"""
    v1 = np.array(p1) / norm(p1)
    v2 = np.array(p2) / norm(p2)
    axis = np.array(axis)

    angle = np.arccos(max(-1.0, min(np.dot(v1, v2), 1)))

    if norm(axis) > 1e-15:
        axis /= norm(axis)

    v1v2cross = np.cross(v1, v2)
    if angle != 0. and norm(v1v2cross) == 0.:
        print("POSSIBLY PROBLEMATIC ARGUMENT SET:")
        print("p1: ", p1)
        print("p2:", p2)
        print("axis: ", axis)
        print("v1: ", v1)
        print("v2:", v2)
        print("angle:", angle)
        print("v1v2cross: ", v1v2cross)

    if angle != 0. and np.isclose(axis, v1v2cross / norm(v1v2cross), 1e-3).all():
        angle *= -1
    return R.from_quat([*(axis*np.sin(angle / 2)), np.cos(angle/2)])

def guess_elements_from_masses(masses, max_delta=1e-2):
    def find_element(elmass):
        for sym, mass in ATOMIC_MASSES.items():
            if elmass - mass < max_delta:
                return sym
        raise Exception("no element matching mass %8.5f in elements list. Please add one?")

    return [find_element(m) for m in masses]

@contextmanager
def use_or_open(fh, path, mode='r'):
    if fh is None:
        with open(path, mode) as f:
            yield f
    else:
        yield fh
