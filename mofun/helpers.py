from contextlib import contextmanager
import functools
import math
import random
import warnings

import numpy as np
from scipy.linalg import norm
from scipy.spatial.transform import Rotation as R

from mofun.atomic_masses import ATOMIC_MASSES

def suppress_warnings(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return wrapper_decorator

def atoms_of_type(types, element):
    """ returns all atom indices in types that match the symbol element """
    return [i for i, t in enumerate(types) if t == element]

def atoms_by_type_dict(atom_types):
    atoms_by_type = {k:[] for k in set(atom_types)}
    for i, k in enumerate(atom_types):
        atoms_by_type[k].append(i)
    return atoms_by_type

def group_duplicates(match_indices, key=lambda m: tuple(sorted(m))):
    keyed_tuples = {}
    for m in match_indices:
        mkey = key(m)
        if mkey not in keyed_tuples:
            keyed_tuples[mkey] = [m]
        else:
            keyed_tuples[mkey].append(m)
    return keyed_tuples

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
    axis = np.array(axis)

    # convert p1 / p2 to be vectors orthogonal to axis
    p1 = p1 - (np.dot(p1, axis) / np.dot(axis, axis)) * axis
    p2 = p2 - (np.dot(p2, axis) / np.dot(axis, axis)) * axis

    v1 = np.array(p1) / norm(p1)
    v2 = np.array(p2) / norm(p2)

    angle = np.arccos(max(-1.0, min(np.dot(v1, v2), 1)))

    if norm(axis) > 1e-15:
        axis /= norm(axis)

    if angle not in [0., math.pi] and np.isclose(axis, np.cross(v1, v2) / norm(np.cross(v1, v2)), 1e-3).all():
        angle *= -1
    return R.from_quat([*(axis*np.sin(-angle / 2)), np.cos(-angle/2)])

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

def typekey(tup):
    rev = list(tup)
    rev.reverse()
    if tuple(rev) <= tuple(tup):
        return tuple(rev)
    return tuple(tup)

class PositionsNotEquivalent(Exception):
    pass

def assert_structure_positions_are_unchanged(orig_structure, final_structure, max_delta=1e-5, verbose=True):
    return assert_positions_are_unchanged(orig_structure.positions, final_structure.positions, max_delta, verbose)

def assert_positions_are_unchanged(p, new_p, max_delta=1e-5, verbose=True, raise_exception=False):
    if raise_exception:
        if not positions_are_unchanged(p, new_p, max_delta, verbose):
            raise PositionsNotEquivalent()
    else:
         assert positions_are_unchanged(p, new_p, max_delta, verbose)

def positions_are_unchanged(p, new_p, max_delta=1e-5, verbose=True):
    p_ordered = p[np.lexsort((p[:,0], p[:,1], p[:,2]))]
    new_p_ordered = new_p[np.lexsort((new_p[:,0], new_p[:,1], new_p[:,2]))]

    if verbose:
        print("** positions_are_unchanged? **")
        print("p = \n", p)
        print("new_p = \n", new_p)
        print("p     (sorted) = \n", p_ordered)
        print("new_p (sorted) = \n", new_p_ordered)
    p_matched = []
    p_corresponding = []
    distances = np.full(len(p), max(9.99, 9.99 * max_delta))
    for i, p1 in enumerate(p_ordered):
        found_match = False
        for j, p2 in enumerate(new_p_ordered):
            # print(p2, p1, norm(np.array(p2) - p1))
            if p2[2] - p1[2] > 1:
                break
            elif (np21 := norm(np.array(p2) - p1)) < max_delta:
                found_match = True
                p_corresponding.append(new_p_ordered[j, :])
                new_p_ordered = np.delete(new_p_ordered, j, axis=0)
                p_matched.append(i)
                distances[i] = np21
                break
        if not found_match:
            p_corresponding.append([])

    p_unmatched = np.delete(p_ordered, p_matched, 0)
    distances = np.array(distances)
    if verbose:
        for i, p1 in enumerate(p_ordered):
            annotation = ""
            if distances[i] > max_delta:
                annotation = " * "
            print(i, p1, p_corresponding[i], distances[i], annotation)
        print("UNMATCHED coords in old positions: ")
        for p1 in p_unmatched:
            print(p1)
        print("UNMATCHED coords in new positions: ")
        for p1 in new_p_ordered:
            print(p1)
        print("--")
    return (distances < max_delta).all()
