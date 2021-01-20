import math

import ase as ase
from ase import Atoms, io
import numpy as np
from numpy.linalg import norm
from scipy.spatial import distance

def atoms_of_type(types, element):
    """ returns all atom indices in types that match the symbol element """
    return [i for i, t in enumerate(types) if t == element]

def uc_neighbor_offsets(uc_vectors):
    multipliers = np.array(np.meshgrid([-1, 0, 1],[-1, 0, 1],[-1, 0, 1])).T.reshape(-1, 3)
    return {tuple((uc_vectors * m).sum(axis=1)) for m in multipliers}

def remove_duplicates(match_indices):
    match1 = set([tuple(sorted(matches)) for matches in match_indices])
    return [list(m) for m in match1]

def get_types_ss_map_limited_near_uc(structure, length, cell):
    if not (cell.angles() == [90., 90., 90.]).all():
        raise Exception("Currently optimizations do not support unit cell angles != 90")

    uc_offsets = list(uc_neighbor_offsets(structure.cell))
    uc_offsets[uc_offsets.index((0.0, 0.0, 0.0))] = uc_offsets[0]
    uc_offsets[0] = (0.0, 0.0, 0.0)

    s_positions = [structure.positions + uc_offset for uc_offset in uc_offsets]
    s_positions = [x for y in s_positions for x in y]

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
    return s_types_view, s_ss, index_mapper

def atoms_by_type_dict(atom_types):
    atoms_by_type = {k:[] for k in set(atom_types)}
    for i, k in enumerate(atom_types):
        atoms_by_type[k].append(i)
    return atoms_by_type

def find_pattern_in_structure(structure, pattern):
    """find pattern in structure, where both are ASE atoms objects

    Returns:
        a list of indice lists for each set of matched atoms found
    """
    p_ss = distance.cdist(pattern.positions, pattern.positions, "sqeuclidean")
    s_types_view, s_ss, index_mapper = get_types_ss_map_limited_near_uc(structure, p_ss.max(), structure.cell)
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

        match_index_tuples = remove_duplicates(match_index_tuples)
        print("round %d: (%d) " % (i, len(match_index_tuples)), match_index_tuples)

    return [[index_mapper[m] % len(structure) for m in match] for match in match_index_tuples]



def replace_pattern_in_structure(structure, search_pattern, replace_pattern):
    pass

def rotate_replace_pattern(pattern, pivot_atom_index, axis, angle):

    numatoms = len(pattern)
    c = math.cos(angle)
    s = math.sin(angle)

    x0 = pattern[pivot_atom_index].position[0]
    y0 = pattern[pivot_atom_index].position[1]
    z0 = pattern[pivot_atom_index].position[2]

    for i in range(numatoms):

        dx = pattern[i].position[0] - x0
        dy = pattern[i].position[1] - y0
        dz = pattern[i].position[2] - z0

        nX = axis[0]
        nY = axis[1]
        nZ = axis[2]

        # dxr, dyr, and dzr are the new, rotated coordinates (assuming the pivot atom is the origin)

        # We use a rotation matrix from axis and angle formula
        dxr = (nX*nX + (1 - nX*nX)*c)*dx +  (nX*nY*(1 - c) - nZ*s)*dy +  (nX*nZ*(1 - c) + nY*s)*dz
        dyr = (nX*nY*(1 - c) + nZ*s)*dx + (nY*nY + (1 - nY*nY)*c)*dy +  (nY*nZ*(1 - c) - nX*s)*dz
        dzr = (nX*nZ*(1 - c) - nY*s)*dx +  (nY*nZ*(1 - c) + nX*s)*dy + (nZ*nZ + (1 - nZ*nZ)*c)*dz

        pattern[i].position[0] = x0 + dxr
        pattern[i].position[1] = y0 + dyr
        pattern[i].position[2] = z0 + dzr

    return pattern

def translate_molecule_origin(pattern):
    pattern.translate(-pattern.positions[0])
    return pattern

def translate_replace_pattern(replace_pattern, search_instance):

    # Translate the molecule so that the 0,0,0 coordinate in the replace pattern
    # is at the position of the first atom found in the search pattern

    first_atom_position = replace_pattern[0].position
    search_position = search_instance[0].position

    dx = search_position[0] - first_atom_position[0]
    dy = search_position[1] - first_atom_position[1]
    dz = search_position[2] - first_atom_position[2]

    for position in replace_pattern.positions:
        position[0] += dx
        position[1] += dy
        position[2] += dz

    return replace_pattern

def replace_pattern_orient(search_instance, replace_pattern):

    rt = 0.01; # rotation tolerance error, in radians
    pi = math.pi

    r_pivot_atom_index = replace_pattern[0].index

    s_numatoms = len(search_instance)
    r_numatoms = len(replace_pattern)

    s_first_atom_pos = search_instance[0].position
    s_last_atom_pos = search_instance[s_numatoms-1].position
    s_second_atom_pos = search_instance[1].position

    r_first_atom_pos = replace_pattern[0].position
    r_last_atom_pos = replace_pattern[r_numatoms-1].position
    r_second_atom_pos = replace_pattern[1].position

    # Define the vectors

    # first - f, second - c, last - l, s - search, r - replace
    # sf_sl - vector between the first atom and the last atom
    # in the instance of the search pattern in the structure

    sf_sl = np.empty([3])
    sf_ss = np.empty([3])
    rf_rl = np.empty([3])
    rf_rs = np.empty([3])

    # Vectors on the search structure
    sf_sl[0] = s_last_atom_pos[0] - s_first_atom_pos[0]
    sf_sl[1] = s_last_atom_pos[1] - s_first_atom_pos[1]
    sf_sl[2] = s_last_atom_pos[2] - s_first_atom_pos[2]

    sf_ss[0] = s_second_atom_pos[0] - s_first_atom_pos[0]
    sf_ss[1] = s_second_atom_pos[1] - s_first_atom_pos[1]
    sf_ss[2] = s_second_atom_pos[2] - s_first_atom_pos[2]

    # Vectors on the replace
    rf_rl[0] = r_last_atom_pos[0] - r_first_atom_pos[0]
    rf_rl[1] = r_last_atom_pos[1] - r_first_atom_pos[1]
    rf_rl[2] = r_last_atom_pos[2] - r_first_atom_pos[2]

    rf_rs[0] = r_second_atom_pos[0] - r_first_atom_pos[0]
    rf_rs[1] = r_second_atom_pos[1] - r_first_atom_pos[1]
    rf_rs[2] = r_second_atom_pos[2] - r_first_atom_pos[2]

    # Use the dot-product formula to find the angle between the vectors: SF-SL & RF-RL

    arg = np.dot(sf_sl, rf_rl) / (np.linalg.norm(sf_sl) * np.linalg.norm(rf_rl))

    if arg > 1:
        arg = 1
    if arg < -1:
        arg = -1

    theta = math.acos(arg) # Angle beteen two vectors: SF-SL & RF-RL in Radians

    if theta > rt and theta < pi - rt: # Vectors are not parallel or anti-parallel

        # Find the axis of rotation by taking the cross-product of: SF-SL & RF-RL

        crs = np.cross(sf_sl, rf_rl)
        mag = np.linalg.norm(crs)
        mag *= -1
        crs[0] *= (1.0 / mag)
        crs[1] *= (1.0 / mag)
        crs[2] *= (1.0 / mag)
        rotate_replace_pattern(replace_pattern, r_pivot_atom_index, crs, theta)

    elif theta < rt:
        # Vectors are parallel, do nothing
        pass

    else: # Vectors are anti-parallel - rotate by 180 degrees
        # Now we can rotate by an arbitary normal vector. We can generate an arbitrary normal vector
        # by taking the cross-product of rf_rs with rf_rl
        crs = np.cross(rf_rs, rf_rl)
        mag = np.linalg.norm(crs)
        crs[0] *= (-1.0 / mag)
        crs[1] *= (-1.0 / mag)
        crs[2] *= (-1.0 / mag)
        # Now rotate by 180 degrees
        rotate_replace_pattern(replace_pattern, r_pivot_atom_index, crs, theta)

    # Update fgroup vectors after rotation
    r_first_atom_pos = replace_pattern[0].position
    r_last_atom_pos = replace_pattern[r_numatoms-1].position
    r_second_atom_pos = replace_pattern[1].position

    rf_rl[0] = r_last_atom_pos[0] - r_first_atom_pos[0]
    rf_rl[1] = r_last_atom_pos[1] - r_first_atom_pos[1]
    rf_rl[2] = r_last_atom_pos[2] - r_first_atom_pos[2]

    rf_rs[0] = r_second_atom_pos[0] - r_first_atom_pos[0]
    rf_rs[1] = r_second_atom_pos[1] - r_first_atom_pos[1]
    rf_rs[2] = r_second_atom_pos[2] - r_first_atom_pos[2]

    # Next - twist rotation

    normS = np.cross(sf_sl, sf_ss)
    normR = np.cross(rf_rl, rf_rs)

    arg = np.dot(normS, normR) / (np.linalg.norm(normS) * np.linalg.norm(normR))

    if arg > 1:
        arg = 1
    if arg < -1:
        arg = -1

    theta = math.acos(arg)

    if theta > rt and theta < pi - rt: # Vectors are not parallel or anti-parallel

        # Find the axis of rotation by taking the cross-product of: SF-SL & RF-RL

        crs = np.cross(normS, normR)
        mag = np.linalg.norm(crs)
        mag *= -1
        crs[0] *= (1.0 / mag)
        crs[1] *= (1.0 / mag)
        crs[2] *= (1.0 / mag)
        rotate_replace_pattern(replace_pattern, r_pivot_atom_index, crs, theta)

    elif theta < rt:
        # Vectors are parallel, do nothing
        pass

    else: # Rotate around sf_sl vector

        crs = sf_sl
        mag = np.linalg.norm(crs)
        crs[0] *= (1.0 / mag)
        crs[1] *= (1.0 / mag)
        crs[2] *= (1.0 / mag)
        # Now rotate by 180 degrees
        rotate_replace_pattern(replace_pattern, r_pivot_atom_index, crs, theta)
