import math
import random

import numpy as np
from scipy.spatial import distance

from mofun.atoms import find_unchanged_atom_pairs

from mofun.helpers import atoms_of_type, atoms_by_type_dict, position_index_farthest_from_axis, \
                          quaternion_from_two_vectors, quaternion_from_two_vectors_around_axis, \
                          remove_duplicates

def uc_neighbor_offsets(uc_vectors):
    multipliers = np.array(np.meshgrid([-1, 0, 1],[-1, 0, 1],[-1, 0, 1])).T.reshape(-1, 1, 3)
    return (uc_vectors * multipliers).sum(axis=1)

def get_types_ss_map_limited_near_uc(structure, length, cell):
    """
    structure:
    length: the length of the longest dimension of the search pattern
    cell:

    creates master lists of indices, types and positions, for all atoms in the structure and all
    atoms across the PBCs. Limits atoms across PBCs to those that are within a distance of the
    boundary that is less than the length of the search pattern (i.e. atoms further away from the
    boundary than this will never match the search pattern).
    """
    # if not (cell.angles() == [90., 90., 90.]).all():
    #     raise Exception("Currently optimizations do not support unit cell angles != 90")

    uc_offsets = uc_neighbor_offsets(structure.cell)
    # move (0., 0., 0.) to be at the 0 index
    uc_offsets[np.where(np.all(uc_offsets == (0,0,0), axis=1))[0][0]] = uc_offsets[0]
    uc_offsets[0] = (0.0, 0.0, 0.0)

    s_positions = [structure.positions + uc_offset for uc_offset in uc_offsets]
    s_positions = np.array([x for y in s_positions for x in y])

    s_types = list(structure.elements) * len(uc_offsets)
    cell = list(np.diag(cell))
    index_mapper = []
    s_pos_view = []
    s_types_view = []

    for i, pos in enumerate(s_positions):
        # only currently works for orthorhombic crystals
        if (pos[0] >= -length and pos[0] < length + cell[0] and
                pos[1] >= -length and pos[1] < length + cell[1] and
                pos[2] >= -length and pos[2] < length + cell[2]):
            index_mapper.append(i)
            s_pos_view.append(pos)
            s_types_view.append(s_types[i])

    s_ss = distance.cdist(s_pos_view, s_pos_view, "sqeuclidean")
    return s_types_view, s_ss, index_mapper, s_pos_view, s_positions

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
    s_types_view, s_ss, index_mapper, s_pos_view, s_positions = get_types_ss_map_limited_near_uc(structure, pattern_length, structure.cell)
    atoms_by_type = atoms_by_type_dict(s_types_view)

    # created sorted coords array for creating search subsets
    p = np.array(sorted([(*r, i) for i, r in enumerate(s_pos_view)]))

    # Search instances of first atom in a search pattern
    # 0,0,0 uc atoms are always indexed first from 0 to # atoms in structure.
    starting_atoms = [idx for idx in atoms_of_type(s_types_view[0: len(structure)], pattern.elements[0])]
    if verbose:
        print("round %d (%d) [%s]: " % (0, len(starting_atoms), pattern.elements[0]), starting_atoms)

    pattern_elements = pattern.elements
    all_match_index_tuples = []
    for a_idx, a in enumerate(starting_atoms):
        match_index_tuples = [[a]]

        nearby = p[(p[:, 0] <= s_pos_view[a][0] + pattern_length) & (p[:, 0] >= s_pos_view[a][0] - pattern_length) &
                   (p[:, 1] <= s_pos_view[a][1] + pattern_length) & (p[:, 1] >= s_pos_view[a][1] - pattern_length) &
                   (p[:, 2] <= s_pos_view[a][2] + pattern_length) & (p[:, 2] >= s_pos_view[a][2] - pattern_length)]

        nearby_atom_indices = nearby[:,3].astype(np.int16)

        for i in range(1, len(pattern)):
            last_match_index_tuples = match_index_tuples
            match_index_tuples = []
            for match in last_match_index_tuples:
                for atom_idx in nearby_atom_indices:
                    if s_types_view[atom_idx]==pattern_elements[i]:
                        found_match = True
                        for j in range(0, i):
                            if not math.isclose(p_ss[i,j], s_ss[match[j], atom_idx], rel_tol=rel_tol_sq):
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

def replace_pattern_in_structure(structure, search_pattern, replace_pattern, replace_fraction=1.0, axis1a_idx=0, axis1b_idx=-1, verbose=False):
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

    if verbose: print(match_indices, match_positions)

    # translate both search and replace patterns so that first atom of search pattern is at the origin
    replace_pattern.translate(-search_pattern.positions[axis1a_idx])
    search_pattern.translate(-search_pattern.positions[axis1a_idx])
    search_axis = search_pattern.positions[axis1b_idx]
    if verbose: print("search_axis: ", search_axis)

    replace2search_pattern_map = {k:v for (k,v) in find_unchanged_atom_pairs(replace_pattern, search_pattern)}

    if len(search_pattern) > 2:
        orientation_point_index = position_index_farthest_from_axis(search_axis, search_pattern)
        orientation_point = search_pattern.positions[orientation_point_index]
        orientation_axis = orientation_point - (np.dot(orientation_point, search_axis) / np.dot(search_axis, search_axis)) * search_axis
        if verbose: print("orientation_axis: ", orientation_axis)

    new_structure = structure.copy()
    to_delete = set()
    if len(replace_pattern) == 0:
        to_delete |= set([idx for match in match_indices for idx in match])
    else:
        offsets = new_structure.extend_types(replace_pattern)
        for m_i, atom_positions in enumerate(match_positions):
            new_atoms = replace_pattern.copy()
            if verbose:
                print(atom_positions)
                print("--------------")
                print("original atoms:\n", atom_positions)
                print("new atoms:\n", new_atoms.positions)

            if len(atom_positions) > 1:
                found_axis = atom_positions[axis1b_idx] - atom_positions[axis1a_idx]
                if verbose: print("found axis: ", found_axis)
                q1 = quaternion_from_two_vectors(search_axis, found_axis)
                if q1 is not None:
                    new_atoms.positions = q1.apply(new_atoms.positions)
                    if verbose:
                        print("q1: ", q1.as_quat())
                        print("new atoms after q1:\n", new_atoms.positions)
                        print("new atoms after q1 (translated):\n", new_atoms.positions + atom_positions[axis1a_idx])

                if len(atom_positions) > 2:
                    found_orientation_point = atom_positions[orientation_point_index] - atom_positions[axis1a_idx]
                    found_orientation_axis = found_orientation_point - (np.dot(found_orientation_point, found_axis) / np.dot(found_axis, found_axis)) * found_axis
                    if verbose: print("found orientation_axis: ", found_orientation_axis)
                    q1_o_axis = orientation_axis
                    if q1 is not None:
                        q1_o_axis = q1.apply(q1_o_axis)

                    q2 = quaternion_from_two_vectors_around_axis(found_orientation_axis, q1_o_axis, found_axis)
                    if verbose:
                        print("(transformed) orientation_axis: ", q1_o_axis)
                        print("orienting: ", found_orientation_point, q1_o_axis, found_orientation_axis, q2)
                    if q2 is not None:

                        new_atoms.positions = q2.apply(new_atoms.positions)
                        if verbose:
                            print("q2: ", q2.as_quat())
                            print("new aif verbose: toms after q2:\n", new_atoms.positions)

            # move replacement atoms into correct position
            new_atoms.translate(atom_positions[axis1a_idx])
            new_atoms.positions %= np.diag(new_structure.cell)

            if verbose: print("new atoms after translate:\n", new_atoms.positions)

            structure_index_map = {k: match_indices[m_i][v] for k,v in replace2search_pattern_map.items()}
            new_structure.extend(new_atoms, offsets=offsets, structure_index_map=structure_index_map)

            to_delete_linker = set(match_indices[m_i]) - set(structure_index_map.values())
            to_delete |= set(to_delete_linker)

    del(new_structure[list(to_delete)])

    return new_structure