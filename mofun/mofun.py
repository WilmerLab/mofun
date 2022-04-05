import math
import random
import sys

import numpy as np
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as R

from mofun.atoms import find_unchanged_atom_pairs
from mofun.helpers import atoms_of_type, atoms_by_type_dict, position_index_farthest_from_axis, \
                          quaternion_from_two_vectors, quaternion_from_two_vectors_around_axis, \
                          remove_duplicates, suppress_warnings, group_duplicates

def uc_neighbor_offsets(uc_vectors):
    multipliers = np.array(np.meshgrid([-1, 0, 1],[-1, 0, 1],[-1, 0, 1])).T.reshape(-1, 1, 3)
    return np.array([np.matmul(uc_vectors.T, mult[0]) for mult in multipliers])

def _get_positions_from_all_adjacent_unit_cells(structure, distance):
    """Calculates the atom positions for all atoms in the given unit cell and every adjacent unit cell, as determined by
    the periodic boundaries.

    Limits this comprehensive list to only the atoms within a given distance from one of the borders of the unit cell.
    There is a general version, which supports triclinic unit cells, and an optimized version for orthorhombic unit
    cells.

    Returns the list of all calculated positions `all_pos`, and a set of three lists for atoms within the distance
    cutoff for the positions `near_pos`, the atom types `near_types`, and the nearby atoms index in the master
    positions list `near_indices`.

    Args:
        structure (Atoms): periodic structure to calculate positions
        distance: nearby atoms are defined as being either in the main unit cell, or within `distance` of the main unit cell's boundary.

    Returns
        List(): position of every nearby atom, aka `near_pos`
        List(): atom type for every nearby atom, aka `near_types`
        List(): index of nearby atom in master positions list for every nearby atom, aka `near_indices`
        numpy.Array(): positions of all atoms across in all UCs, aka `all_pos`
    """

    cell = structure.cell
    uc_offsets = uc_neighbor_offsets(cell)
    # move (0., 0., 0.) to be at the 0 index
    uc_offsets[np.where(np.all(uc_offsets == (0,0,0), axis=1))[0][0]] = uc_offsets[0]
    uc_offsets[0] = (0.0, 0.0, 0.0)

    all_positions = [structure.positions + uc_offset for uc_offset in uc_offsets]
    all_positions = np.array([x for y in all_positions for x in y])

    s_types = list(structure.elements) * len(uc_offsets)

    near_pos = []
    near_types = []
    near_indices = []

    if not structure.cell_is_orthorhombic():
        # search within triclinic space + buffer by looking at three planes that go through origin

        # normal vectors for planes: xy, xz, yz
        nvs = np.array([np.cross(cell[0], cell[1]), np.cross(cell[0], cell[2]), np.cross(cell[1], cell[2])])
        nvnorms = np.linalg.norm(nvs, axis=1)
        planedists = np.abs(np.array([np.dot(cell[2], nvs[0]) / nvnorms[0],
                                      np.dot(cell[1], nvs[1]) / nvnorms[1],
                                      np.dot(cell[0], nvs[2]) / nvnorms[2]]))
        # calculate distance to center point; from a plane boundary to inside the unit cell
        # (inwards) should have a negative distance. If not, we will multiply the distances below
        # by -1 to account for this.
        centerpos = cell.sum(axis=0) / 2
        centerdist = np.dot(nvs, centerpos)
        nmults = -centerdist / np.abs(centerdist)

        for i, pos in enumerate(all_positions):
            if ((-planedists[0] - distance <= nmults[0] * np.dot(nvs[0], pos) / nvnorms[0] <= distance) and
                (-planedists[1] - distance <= nmults[1] * np.dot(nvs[1], pos) / nvnorms[1] <= distance) and
                (-planedists[2] - distance <= nmults[2] * np.dot(nvs[2], pos) / nvnorms[2] <= distance)):

                near_indices.append(i)
                near_pos.append(pos)
                near_types.append(s_types[i])

    else: # orthorhombic
        cell = list(np.diag(cell))

        for i, pos in enumerate(all_positions):
            if (pos[0] >= -distance and pos[0] < distance + cell[0] and
                pos[1] >= -distance and pos[1] < distance + cell[1] and
                pos[2] >= -distance and pos[2] < distance + cell[2]):

                near_indices.append(i)
                near_pos.append(pos)
                near_types.append(s_types[i])

    return near_pos, near_types, near_indices, all_positions

@suppress_warnings
def find_pattern_in_structure(structure, pattern, axisp1_idx=None, axisp2_idx=None, opoint_idx=None,
        return_positions_and_quats=False, atol=5e-2, verbose=False):
    """Looks for instances of `pattern` in `structure`, where a match in the structure has the same number of atoms, the
    same elements and the same relative coordinates as in `pattern`.

    Returns a list of tuples, one tuple per match found in `structure` where each tuple has the size `len(pattern)` and
    contains the indices in the structure that matched the pattern. If `return_positions_and_quats=True` then two
    additional lists are returned containing positions for each matched index for each match, and the quaternions
    required to rotate the search pattern to the match pattern.

    Args:
        structure (Atoms): an Atoms object to search in.
        pattern (Atoms): an Atoms object to search for.
        axisp1_idx (float): index in search_pattern of first point defining the directional axis of the search_pattern. May help with performance for large problems.
        axisp2_idx (float): index in search_pattern of second point defining the directional axis of the search_pattern. May help with performance for large problems.
        opoint_idx (float): index in search_pattern of the orientation point that we will use to align the search pattern (the search pattern will be rotated so that the orientation point in the search pattern will have the same coordinates as the same point in the match pattern).
        return_positions_and_quats (bool): additionally returns the positions for each index and the quaternions that rotation the search pattern to the match pattern.
        atol (float): the absolute tolerance (how close an atom must be in the structure to the position in pattern to be considered a match).
        verbose (bool): print debugging info.
    Returns:
        List [tuple(len(pattern))],  {List [tuple(len(pattern))], List [scipy.spatial.transform.Rotation]} : returns a
            list of tuples of size `len(pattern)` containing the indices in the structure that matched the pattern, one
            tuple per each match, and optionally (if return_positions_and_quats is True) a corresponding list of tuples
            with positions instead of indices, and a list of quaternions (scipy.spatial.transform.Rotation), one per
            each match.
    """

    if verbose:
        print("calculating point distances...")
    p_ss = distance.cdist(pattern.positions, pattern.positions, "sqeuclidean")
    if axisp1_idx is None and axisp2_idx is None:
        # pick the pair of points that are farthest away from each other
        axisp1_idx, axisp2_idx = np.unravel_index(np.argmax(p_ss, axis=None), p_ss.shape)
    elif axisp1_idx is None or axisp2_idx is None:
        # if we are given one point, find the point that is farthest away from it
        axisp1_idx = axisp1_idx or axisp2_idx # axisp1_idx is whichever point is not none
        axisp2_idx = np.argmax(p_ss[axisp1_idx, :])

    pattern_length = p_ss.max() ** 0.5 + 2 * atol
    near_pos, near_types, near_indices, all_positions = _get_positions_from_all_adjacent_unit_cells(structure, pattern_length)
    atoms_by_type = atoms_by_type_dict(near_types)

    # created sorted coords array for creating search subsets
    p = np.array(sorted([(*r, i) for i, r in enumerate(near_pos)]))

    # Search instances of first atom in a search pattern
    # 0,0,0 uc atoms are always indexed first from 0 to # atoms in structure.
    starting_atoms = [idx for idx in atoms_of_type(near_types[0: len(structure)], pattern.elements[0])]
    if verbose:
        print("round %d (%d) [%s]: " % (0, len(starting_atoms), pattern.elements[0]), starting_atoms)

    def get_nearby_atoms(p, near_pos, pattern_length, a):
        p1 = p[(p[:, 0] <= near_pos[a][0] + pattern_length) & (p[:, 0] >= near_pos[a][0] - pattern_length)]
        p2 = p1[(p1[:, 1] <= near_pos[a][1] + pattern_length) & (p1[:, 1] >= near_pos[a][1] - pattern_length)]
        return (p2[(p2[:, 2] <= near_pos[a][2] + pattern_length) & (p2[:, 2] >= near_pos[a][2] - pattern_length)])

    pattern_elements = pattern.elements
    all_match_index_tuples = []
    for a_idx, a in enumerate(starting_atoms):
        match_index_tuples = [[a]]

        nearby = get_nearby_atoms(p, near_pos, pattern_length, a)
        nearby_atom_indices = nearby[:,3].astype(np.int32)
        nearby_positions = nearby[:,0:3]

        idx2ssidx = {atom_idx:i for i, atom_idx in enumerate(nearby_atom_indices)}
        s_ss = distance.cdist(nearby_positions, nearby_positions, "sqeuclidean")

        # start for loop at one since we've searched for starting atoms (index == 0) above
        for i in range(1, len(pattern)):
            if len(match_index_tuples) == 0:
                break
            last_match_index_tuples = match_index_tuples
            match_index_tuples = []
            for match in last_match_index_tuples:
                for ss_idx, atom_idx in enumerate(nearby_atom_indices):
                    if near_types[atom_idx] == pattern_elements[i]:
                        found_match = True
                        # check all distances to this new proposed atom
                        for j in range(0, i):
                            if not math.isclose(p_ss[i,j]**0.5, s_ss[idx2ssidx[match[j]], ss_idx]**0.5, abs_tol=atol):
                                found_match = False
                                break

                        # anything that matches the distance to all prior pattern atoms is a good match so far
                        if found_match:
                            match_index_tuples.append(match + [atom_idx])
            if verbose:
                print("round %d (%d) [%s]: " % (i, len(match_index_tuples), pattern.elements[i]), match_index_tuples)
        if verbose:
            print("starting atom %d: found %d matches: %s" % (a_idx, len(match_index_tuples), match_index_tuples))
        all_match_index_tuples += match_index_tuples

    ## remove duplicates by calculating quaternions necessary to align atoms, applying it to the search pattern and if
    # there is symmetry or chirality, eliminating possible matches where the atoms rotate into different places than we
    # found them in the structure.
    pattern = pattern.copy()
    pattern.translate(-pattern.positions[axisp1_idx])
    search_axis = pattern.positions[axisp2_idx]
    if len(pattern) > 2 and opoint_idx is None:
        # note that we find an orientation point if there are nore than two atoms in the search pattern, but it is still
        # possible that all the atoms like on the search axis. In that, case a point on the axis will be returned and
        # there is an unnecessary final rotation to "align" that point.
        opoint_idx = position_index_farthest_from_axis(search_axis, pattern)

    grouped_tuples = group_duplicates(all_match_index_tuples, key=lambda m: tuple(sorted([near_indices[i] % len(structure) for i in m])))
    grouped_tuples2 = []

    good_match_index_tuples = []
    good_match_quats = []
    for _, match_tuples in grouped_tuples.items():
        quats = []
        good_indices = []
        for i, match_tuple in enumerate(match_tuples):
            atom_positions = np.array([all_positions[near_indices[m]] for m in match_tuple])
            q = R.identity()
            if len(atom_positions) > 1:
                # the first quaternion aligns the search pattern axis points with the axis points
                # found in the structure
                match_axis = atom_positions[axisp2_idx] - atom_positions[axisp1_idx]
                q = quaternion_from_two_vectors(search_axis, match_axis)

                if len(atom_positions) > 2:
                    # the second quaternion is a rotation around the found axis in the structure and
                    # aligns the orientation axis point to its placement in the structure.
                    match_orientation_point = atom_positions[opoint_idx] - atom_positions[axisp1_idx]
                    rotated_orientation_point = q.apply(pattern.positions[opoint_idx])
                    q = quaternion_from_two_vectors_around_axis(rotated_orientation_point, match_orientation_point, match_axis) * q

            quats.append(q)
            chk_pattern = pattern.copy()
            chk_pattern.positions = q.apply(chk_pattern.positions)
            chk_pattern.translate(atom_positions[axisp1_idx])

            # note that we can use positions are unchanged here, which does not handle periodic boundaries, because all
            # our coordinates are unwrapped.
            if np.allclose(atom_positions, chk_pattern.positions, atol=atol):
                good_indices.append(i)

        if len(good_indices) > 1:
            # Likely it is because of symmetry if we found more than one good match. Randomly choose one.
            match_chosen = random.choice(good_indices)
            good_match_index_tuples.append(match_tuples[match_chosen])
            good_match_quats.append(quats[match_chosen])
        elif len(good_indices) == 1:
            # Found one good match; either we have no symmetry, or we have symmetry where only one numbering of the
            # atoms can be rotated into place
            good_match_index_tuples.append(match_tuples[good_indices[0]])
            good_match_quats.append(quats[good_indices[0]])
        else:
            print("""WARNING: Search pattern was matched, but there is no possible way to rotate the search patttern to meet the
                match pattern. This is likely due to finding a match of the opposite chirality.
                """, file=sys.stderr)

    match_index_tuples_in_uc = [tuple([near_indices[m] % len(structure) for m in match]) for match in good_match_index_tuples]
    if return_positions_and_quats:
        match_index_tuple_positions = np.array([[all_positions[near_indices[m]] for m in match] for match in good_match_index_tuples])
        return match_index_tuples_in_uc, match_index_tuple_positions, np.array(good_match_quats)
    else:
        return match_index_tuples_in_uc

class AtomsShouldNotBeDeletedTwice(Exception):
    pass

@suppress_warnings
def replace_pattern_in_structure(
    structure, search_pattern, replace_pattern, replace_fraction=1.0, atol=5e-2,
    axisp1_idx=None, axisp2_idx=None, opoint_idx=None,
    return_num_matches=False, replace_all=False, verbose=False,
    positions_check_max_delta=0.1,
    ignore_atoms_should_not_be_deleted_twice=False):
    """Replaces all instances of `pattern` in `structure` with the `replace_pattern`.

    Works across periodic boundary conditions.

    WARNING: the replace pattern _MUST_ be on the same coordinate system as the search_pattern. If
    there are atoms that remain the same between the search and replace patterns, they must have the
    exact same coordinates. If these were to be offset, or moved, then when the replacement pattern
    gets inserted into the structure, then the replacement will also be offset.

    Args:
        structure (Atoms): an Atoms object to search in.
        search_pattern (Atoms): an Atoms object to search for.
        replace_pattern (Atoms): an Atoms object to search for.
        replace_fraction (float): how many instances of the search_pattern found in the structure get replaced by the replace pattern.
        atol (float): absolute tolerance in Angstroms for atom posistions to be considered matching.
        axisp1_idx (float): index in search_pattern of first point defining the directional axis of the search_pattern.
        axisp2_idx (float): index in search_pattern of second point defining the directional axis of the search_pattern.
        opoint_idx (float): index in search_pattern of third point defining the orientational axis of the search_pattern.
        replace_all (bool): replaces all atoms even if positions and elements match exactly
        verbose (bool): print debugging info.
        ignore_atoms_should_not_be_deleted_twice (bool): don't raise an AtomsShouldNotBeDeletedTwice exception when
            two matches would delete the same atoms.
    Returns:
        Atoms: the structure after search_pattern is replaced by replace_pattern.
    """
    search_pattern = search_pattern.copy()
    replace_pattern = replace_pattern.copy()

    # translate both search and replace patterns so that first atom of search pattern is at the origin
    replace_pattern.translate(-search_pattern.positions[0])
    search_pattern.translate(-search_pattern.positions[0])

    match_indices, match_positions, quats = find_pattern_in_structure(structure, search_pattern, atol=atol,
        axisp1_idx=axisp1_idx, axisp2_idx=axisp2_idx, opoint_idx=opoint_idx, return_positions_and_quats=True,
        verbose=verbose)

    if replace_fraction < 1.0:
        replace_indices = random.sample(list(range(len(match_positions))), k=round(replace_fraction * len(match_positions)))
        match_indices = [match_indices[i] for i in replace_indices]
        match_positions = match_positions[replace_indices]
        quats = quats[replace_indices]

    if verbose: print("match_indices / positions: ", match_indices, match_positions)

    replace2search_pattern_map = {k:v for (k,v) in find_unchanged_atom_pairs(replace_pattern, search_pattern)}

    new_structure = structure.copy()
    to_delete = set()
    if len(replace_pattern) == 0:
        to_delete |= set([idx for match in match_indices for idx in match])
    else:
        offsets = new_structure.extend_types(replace_pattern)

        for m_i, atom_positions in enumerate(match_positions):
            q = quats[m_i]
            new_atoms = replace_pattern.copy()
            new_atoms.positions = q.apply(new_atoms.positions)
            new_atoms.translate(atom_positions[0])
            new_atoms.positions %= np.diag(new_structure.cell)

            if verbose:
                print("new atoms after translate:\n", new_atoms.positions)

            structure_index_map = {}
            if not replace_all:
                structure_index_map = {k: match_indices[m_i][v] for k,v in replace2search_pattern_map.items()}
                new_structure.extend(new_atoms, offsets=offsets, structure_index_map=structure_index_map)
            else:
                new_structure.extend(new_atoms, offsets=offsets)

            to_delete_linker = set(match_indices[m_i]) - set(structure_index_map.values())
            if (to_delete.isdisjoint(to_delete_linker) or ignore_atoms_should_not_be_deleted_twice):
                to_delete |= set(to_delete_linker)
            else:
                raise AtomsShouldNotBeDeletedTwice()

    del(new_structure[list(to_delete)])

    if return_num_matches:
        return new_structure, len(match_indices)
    else:
        return new_structure
