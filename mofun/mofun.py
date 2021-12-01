import math
import random

import numpy as np
from scipy.spatial import distance

from mofun.atoms import find_unchanged_atom_pairs
from mofun.helpers import atoms_of_type, atoms_by_type_dict, position_index_farthest_from_axis, \
                          quaternion_from_two_vectors, quaternion_from_two_vectors_around_axis, \
                          remove_duplicates, assert_positions_are_unchanged

def uc_neighbor_offsets(uc_vectors):
    multipliers = np.array(np.meshgrid([-1, 0, 1],[-1, 0, 1],[-1, 0, 1])).T.reshape(-1, 1, 3)
    return (uc_vectors * multipliers).sum(axis=1)

def get_types_ss_map_limited_near_uc(structure, length):
    """
    structure:
    length: the length of the longest dimension of the search pattern

    creates master lists of indices, types and positions, for all atoms in the structure and all
    atoms across the PBCs. Limits atoms across PBCs to those that are within a distance of the
    boundary that is less than the length of the search pattern (i.e. atoms further away from the
    boundary than this will never match the search pattern).
    """

    cell = structure.cell
    uc_offsets = uc_neighbor_offsets(cell)
    # move (0., 0., 0.) to be at the 0 index
    uc_offsets[np.where(np.all(uc_offsets == (0,0,0), axis=1))[0][0]] = uc_offsets[0]
    uc_offsets[0] = (0.0, 0.0, 0.0)

    s_positions = [structure.positions + uc_offset for uc_offset in uc_offsets]
    s_positions = np.array([x for y in s_positions for x in y])

    s_types = list(structure.elements) * len(uc_offsets)

    index_mapper = []
    s_pos_view = []
    s_types_view = []

    is_triclinic = not (np.diag(cell) * np.identity(3) == cell).all()
    if is_triclinic:
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

        for i, pos in enumerate(s_positions):
            if ((-planedists[0] - length <= nmults[0] * np.dot(nvs[0], pos) / nvnorms[0] <= length) and
                (-planedists[1] - length <= nmults[1] * np.dot(nvs[1], pos) / nvnorms[1] <= length) and
                (-planedists[2] - length <= nmults[2] * np.dot(nvs[2], pos) / nvnorms[2] <= length)):

                index_mapper.append(i)
                s_pos_view.append(pos)
                s_types_view.append(s_types[i])

    else: # orthorhombic
        cell = list(np.diag(cell))

        for i, pos in enumerate(s_positions):
            if (pos[0] >= -length and pos[0] < length + cell[0] and
                pos[1] >= -length and pos[1] < length + cell[1] and
                pos[2] >= -length and pos[2] < length + cell[2]):

                index_mapper.append(i)
                s_pos_view.append(pos)
                s_types_view.append(s_types[i])

    return s_types_view, index_mapper, s_pos_view, s_positions

def find_pattern_in_structure(structure, pattern, return_positions=False, rel_tol=5e-2, verbose=False):
    """Looks for instances of `pattern` in `structure`, where a match in the structure has the same number
    of atoms, the same elements and the same relative coordinates as in `pattern`.

    Returns a list of tuples, one tuple per match found in `structure` where each tuple has the size
    `len(pattern)` and contains the indices in the structure that matched the pattern. If
    `return_postions=True` then  an additional list is returned containing positions for each
    matched index for each match.

    Args:
        structure (Atoms): an Atoms object to search in.
        pattern (Atoms): an Atoms object to search for.
        return_positions (bool): additionally returns the positions for each index
        rel_tol (float): the relative tolerance (how close an atom must be in the structure to the position in pattern to be consdired a match).
        verbose (bool): print debugging info.
    Returns:
        List [tuple(len(pattern))]: returns a tuple of size `len(pattern)` containing the indices in structure that matched the pattern, one tuple per each match.
    """
    # the relative tolerance needs adjusted to squared relative tolerance
    rel_tol_sq = 1 - (1 - rel_tol)**2
    if verbose:
        print("calculating point distances...")
    p_ss = distance.cdist(pattern.positions, pattern.positions, "sqeuclidean")
    pattern_length = p_ss.max() ** 0.5
    s_types_view, index_mapper, s_pos_view, s_positions = get_types_ss_map_limited_near_uc(structure, pattern_length)
    atoms_by_type = atoms_by_type_dict(s_types_view)

    # created sorted coords array for creating search subsets
    p = np.array(sorted([(*r, i) for i, r in enumerate(s_pos_view)]))

    # Search instances of first atom in a search pattern
    # 0,0,0 uc atoms are always indexed first from 0 to # atoms in structure.
    starting_atoms = [idx for idx in atoms_of_type(s_types_view[0: len(structure)], pattern.elements[0])]
    if verbose:
        print("round %d (%d) [%s]: " % (0, len(starting_atoms), pattern.elements[0]), starting_atoms)

    def get_nearby_atoms(p, s_pos_view, pattern_length, a):
        p1 = p[(p[:, 0] <= s_pos_view[a][0] + pattern_length) & (p[:, 0] >= s_pos_view[a][0] - pattern_length)]
        p2 = p1[(p1[:, 1] <= s_pos_view[a][1] + pattern_length) & (p1[:, 1] >= s_pos_view[a][1] - pattern_length)]
        return (p2[(p2[:, 2] <= s_pos_view[a][2] + pattern_length) & (p2[:, 2] >= s_pos_view[a][2] - pattern_length)])

    pattern_elements = pattern.elements
    all_match_index_tuples = []
    for a_idx, a in enumerate(starting_atoms):
        match_index_tuples = [[a]]

        nearby = get_nearby_atoms(p, s_pos_view, pattern_length, a)
        nearby_atom_indices = nearby[:,3].astype(np.int32)
        nearby_positions = nearby[:,0:3]
        idx2ssidx = {atom_idx:i for i, atom_idx in enumerate(nearby_atom_indices)}
        s_ss = distance.cdist(nearby_positions, nearby_positions, "sqeuclidean")

        for i in range(1, len(pattern)):
            if len(match_index_tuples) == 0:
                break
            last_match_index_tuples = match_index_tuples
            match_index_tuples = []
            for match in last_match_index_tuples:
                for ss_idx, atom_idx in enumerate(nearby_atom_indices):
                    if s_types_view[atom_idx]==pattern_elements[i]:
                        found_match = True
                        for j in range(0, i):
                            if not math.isclose(p_ss[i,j], s_ss[idx2ssidx[match[j]], ss_idx], rel_tol=rel_tol_sq):
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

    all_match_index_tuples = remove_duplicates(all_match_index_tuples,
        key=lambda m: tuple(sorted([index_mapper[i] % len(structure) for i in m])))

    match_index_tuples_in_uc = [tuple([index_mapper[m] % len(structure) for m in match]) for match in all_match_index_tuples]
    if return_positions:
        match_index_tuple_positions = np.array([[s_positions[index_mapper[m]] for m in match] for match in all_match_index_tuples])
        return match_index_tuples_in_uc, match_index_tuple_positions
    else:
        return match_index_tuples_in_uc

def replace_pattern_in_structure(
    structure, search_pattern, replace_pattern, replace_fraction=1.0,
    axis1a_idx=0, axis1b_idx=-1, axis2_idx=None,
    return_num_matches=False, replace_all=False, verbose=False,
    ignore_positions_check=False, positions_check_max_delta=0.1):
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
        axis1a_idx (float): index in search_pattern of first point defining the directional axis of the search_pattern. Mostly useful for testing and debugging.
        axis1b_idx (float): index in search_pattern of second point defining the directional axis of the search_pattern. Mostly useful for testing and debugging.
        replace_all (bool): replaces all atoms even if positions and elements match exactly
        verbose (bool): print debugging info.
    Returns:
        Atoms: the structure after search_pattern is replaced by replace_pattern.
    """
    search_pattern = search_pattern.copy()
    replace_pattern = replace_pattern.copy()

    match_indices, match_positions = find_pattern_in_structure(structure, search_pattern, return_positions=True)

    if replace_fraction < 1.0:
        replace_indices = random.sample(list(range(len(match_positions))), k=round(replace_fraction * len(match_positions)))
        match_indices = [match_indices[i] for i in replace_indices]
        match_positions = match_positions[replace_indices]

    if verbose: print("match_indices / positions: ", match_indices, match_positions)

    # translate both search and replace patterns so that first atom of search pattern is at the origin
    replace_pattern.translate(-search_pattern.positions[axis1a_idx])
    search_pattern.translate(-search_pattern.positions[axis1a_idx])
    search_axis = search_pattern.positions[axis1b_idx]
    if verbose: print("search pattern axis: ", search_axis)

    replace2search_pattern_map = {k:v for (k,v) in find_unchanged_atom_pairs(replace_pattern, search_pattern)}

    if len(search_pattern) > 2:
        if axis2_idx is None:
            search_orientation_point_idx = position_index_farthest_from_axis(search_axis, search_pattern)
        else:
            search_orientation_point_idx = axis2_idx

        search_orientation_point = search_pattern.positions[search_orientation_point_idx]
        search_orientation_axis = search_orientation_point - (np.dot(search_orientation_point, search_axis) / np.dot(search_axis, search_axis)) * search_axis
        if verbose:
            print("search pattern orientation point index: ", search_orientation_point_idx)
            print("search pattern orientation axis: ", search_orientation_axis)

    new_structure = structure.copy()
    to_delete = set()
    if len(replace_pattern) == 0:
        to_delete |= set([idx for match in match_indices for idx in match])
    else:
        offsets = new_structure.extend_types(replace_pattern)
        for m_i, atom_positions in enumerate(match_positions):
            new_atoms = replace_pattern.copy()
            if not ignore_positions_check:
                chk_search_pattern = search_pattern.copy()
            if verbose:
                print("--------------")
                print(m_i)
                print("average position: ", np.average(atom_positions, axis=0))

            if len(atom_positions) > 1:
                match_axis = atom_positions[axis1b_idx] - atom_positions[axis1a_idx]
                if verbose: print("match axis: ", match_axis)

                # the first quaternion aligns the search pattern axis points with the axis points
                # found in the structure and is used to rotate the replacement pattern to match
                q1 = quaternion_from_two_vectors(search_axis, match_axis)
                new_atoms.positions = q1.apply(new_atoms.positions)
                if not ignore_positions_check:
                    chk_search_pattern.positions = q1.apply(chk_search_pattern.positions)
                if verbose:
                    print("q1: ", q1.as_quat())

                if len(atom_positions) > 2:
                    match_orientation_point = atom_positions[search_orientation_point_idx] - atom_positions[axis1a_idx]
                    match_orientation_axis = match_orientation_point - (np.dot(match_orientation_point, match_axis) / np.dot(match_axis, match_axis)) * match_axis
                    if verbose: print("match orientation axis: ", match_orientation_axis)
                    q1_o_axis = q1.apply(search_orientation_axis)

                    # the second quaternion is a rotation around the found axis in the structure and
                    # aligns the orientation axis point to its placement in the structure.
                    q2 = quaternion_from_two_vectors_around_axis(match_orientation_axis, q1_o_axis, match_axis)
                    if verbose:
                        print("orienting using match orientation point: ", match_orientation_point)
                        print("from match orientation axis: ", match_orientation_axis)
                        print("to (rotated) search pattern orientation axis: ", q1_o_axis)
                        print("q2: ", q2.as_quat())

                    new_atoms.positions = q2.apply(new_atoms.positions)
                    if not ignore_positions_check:
                        chk_search_pattern.positions = q2.apply(chk_search_pattern.positions)

            # move replacement atoms into correct position
            new_atoms.translate(atom_positions[axis1a_idx])

            if not ignore_positions_check:
                chk_search_pattern.translate(atom_positions[axis1a_idx])

            if not ignore_positions_check:
                assert_positions_are_unchanged(atom_positions, chk_search_pattern.positions,
                    max_delta=positions_check_max_delta, verbose=verbose, raise_exception=True)

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
            to_delete |= set(to_delete_linker)

    del(new_structure[list(to_delete)])

    if return_num_matches:
        return new_structure, len(match_indices)
    else:
        return new_structure
