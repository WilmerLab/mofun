import math

import numpy as np
from scipy.spatial import distance

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
    if not (cell.angles() == [90., 90., 90.]).all():
        raise Exception("Currently optimizations do not support unit cell angles != 90")

    uc_offsets = uc_neighbor_offsets(structure.cell)
    # move (0., 0., 0.) to be at the 0 index
    uc_offsets[np.where(np.all(uc_offsets == (0,0,0), axis=1))[0][0]] = uc_offsets[0]
    uc_offsets[0] = (0.0, 0.0, 0.0)

    s_positions = [structure.positions + uc_offset for uc_offset in uc_offsets]
    s_positions = np.array([x for y in s_positions for x in y])

    s_types = list(structure.symbols) * len(uc_offsets)
    cell = list(structure.cell.lengths())
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
    return s_types_view, s_ss, index_mapper, s_positions

def find_pattern_in_structure(structure, pattern, return_positions=False):
    """find pattern in structure, where both are ASE atoms objects

    Returns:
        a list of indice lists for each set of matched atoms found
    """
    p_ss = distance.cdist(pattern.positions, pattern.positions, "sqeuclidean")
    s_types_view, s_ss, index_mapper, s_positions = get_types_ss_map_limited_near_uc(structure, p_ss.max(), structure.cell)
    atoms_by_type = atoms_by_type_dict(s_types_view)

    for i, pattern_atom_1 in enumerate(pattern):
        # Search instances of first atom in a search pattern
        if i == 0:
            # 0,0,0 uc atoms are always indexed first from 0 to # atoms in structure.
            match_index_tuples = [[idx] for idx in atoms_of_type(s_types_view[0: len(structure)], pattern.symbols[0])]
            print("round %d (%d): " % (i, len(match_index_tuples)), match_index_tuples)
            continue

        last_match_index_tuples = match_index_tuples
        match_index_tuples = []
        for match in last_match_index_tuples:
            for atom_idx in atoms_by_type[pattern_atom_1.symbol]:
                found_match = True
                for j in range(i):
                    if not math.isclose(p_ss[i,j], s_ss[match[j], atom_idx], rel_tol=5e-2):
                        found_match = False
                        break

                # anything that matches the distance to all prior pattern atoms is a good match so far
                if found_match:
                    match_index_tuples.append(match + [atom_idx])

        print("round %d: (%d) " % (i, len(match_index_tuples)), match_index_tuples)

    match_index_tuples = remove_duplicates(match_index_tuples,
        key=lambda m: tuple(sorted([index_mapper[i] % len(structure) for i in m])))

    match_index_tuples_in_uc = [tuple([index_mapper[m] % len(structure) for m in match]) for match in match_index_tuples]
    if return_positions:
        match_index_tuple_positions = np.array([[s_positions[index_mapper[m]] for m in match] for match in match_index_tuples])
        return match_index_tuples_in_uc, match_index_tuple_positions
    else:
        return match_index_tuples_in_uc

def replace_pattern_in_structure(structure, search_pattern, replace_pattern, axis1a_idx=0, axis1b_idx=-1):
    search_pattern = search_pattern.copy()
    replace_pattern = replace_pattern.copy()

    match_indices, match_positions = find_pattern_in_structure(structure, search_pattern, return_positions=True)
    print(match_indices)

    # translate both search and replace patterns so that first atom of search pattern is at the origin
    replace_pattern.translate(-search_pattern.positions[axis1a_idx])
    search_pattern.translate(-search_pattern.positions[axis1a_idx])
    search_axis = search_pattern.positions[axis1b_idx]
    print("search_axis: ", search_axis)

    if len(search_pattern) > 2:
        orientation_point_index = position_index_farthest_from_axis(search_axis, search_pattern)
        orientation_point = search_pattern.positions[orientation_point_index]
        orientation_axis = orientation_point - (np.dot(orientation_point, search_axis) / np.dot(search_axis, search_axis)) * search_axis
        print("orientation_axis: ", orientation_axis)

    new_structure = structure.copy()
    if len(replace_pattern) > 0:

        for atom_positions in match_positions:
            print(atom_positions)
            print("--------------")
            print("original atoms:\n", atom_positions)
            new_atoms = replace_pattern.copy()
            print("new atoms:\n", new_atoms.positions)
            if len(atom_positions) > 1:
                found_axis = atom_positions[axis1b_idx] - atom_positions[axis1a_idx]
                print("found axis: ", found_axis)
                q1 = quaternion_from_two_vectors(search_axis, found_axis)
                if q1 is not None:
                    new_atoms.positions = q1.apply(new_atoms.positions)
                    print("q1: ", q1.as_quat())
                    print("new atoms after q1:\n", new_atoms.positions)
                    print("new atoms after q1 (translated):\n", new_atoms.positions + atom_positions[axis1a_idx])

                if len(atom_positions) > 2:
                    found_orientation_point = atom_positions[orientation_point_index] - atom_positions[axis1a_idx]
                    found_orientation_axis = found_orientation_point - (np.dot(found_orientation_point, found_axis) / np.dot(found_axis, found_axis)) * found_axis
                    print("found orientation_axis: ", found_orientation_axis)
                    q1_o_axis = orientation_axis
                    if q1:
                        q1_o_axis = q1.apply(q1_o_axis)

                    print("(transformed) orientation_axis: ", q1_o_axis)
                    q2 = quaternion_from_two_vectors_around_axis(found_orientation_axis, q1_o_axis, found_axis)
                    print("orienting: ", found_orientation_point, q1_o_axis, found_orientation_axis, q2)
                    if q2 is not None:
                        print("q2: ", q2.as_quat())
                        new_atoms.positions = q2.apply(new_atoms.positions)
                        print("new atoms after q2:\n", new_atoms.positions)

            # move replacement atoms into correct position
            new_atoms.translate(atom_positions[axis1a_idx])
            new_atoms.positions %= new_structure.cell.lengths()
            print("new atoms after translate:\n", new_atoms.positions)
            new_structure.extend(new_atoms)

    indices_to_delete = [idx for match in match_indices for idx in match]
    del(new_structure[indices_to_delete])

    return new_structure
