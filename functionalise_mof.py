import math

import ase as ase
from ase import Atoms, io
import numpy as np
from numpy.linalg import norm
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as R

def atoms_of_type(types, element):
    """ returns all atom indices in types that match the symbol element """
    return [i for i, t in enumerate(types) if t == element]

def uc_neighbor_offsets(uc_vectors):
    multipliers = np.array(np.meshgrid([-1, 0, 1],[-1, 0, 1],[-1, 0, 1])).T.reshape(-1, 3)
    return {tuple((uc_vectors * m).sum(axis=1)) for m in multipliers}

def remove_duplicates(match_indices):
    found_tuples = set()
    new_match_indices = []
    for m in match_indices:
        mkey = tuple(sorted(m))
        if mkey not in found_tuples:
            new_match_indices.append(m)
            found_tuples.add(mkey)
    return new_match_indices

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


        print("round %d: (%d) " % (i, len(match_index_tuples)), match_index_tuples)

    match_index_tuples = remove_duplicates(match_index_tuples)
    return [tuple([index_mapper[m] % len(structure) for m in match]) for match in match_index_tuples]


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
    replace_pattern.translate((search_instance[0].position - replace_pattern[0].position))
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

def position_index_farthest_from_axis(axis, atoms):
    q = quaternion_from_two_axes(axis, [1., 0., 0.])
    ratoms = q.apply(atoms.positions)
    ss = (ratoms[:,1:3] ** 2).sum(axis=1)
    return np.nonzero(ss==ss.max())[0][0]


def quaternion_from_two_axes(axis1, axis2):
    """ returns the quaternion necessary to rotate ax1 to ax2

    returns None if ax1 == ax2
    """
    ax1 = np.array(axis1)
    ax2 = np.array(axis2)
    ax1 /= np.sqrt(np.dot(ax1, ax1))
    ax2 /= np.sqrt(np.dot(ax2, ax2))

    axis = np.cross(ax1, ax2)
    if np.isclose(axis, 0.0).all():
        return None

    angle = np.arccos(np.dot(ax1, ax2))
    if np.isnan(angle):
        return None
    print(axis1, axis2, axis, angle)
    return quaternion_from_axis_angle(axis, angle)

def quaternion_from_axis_angle(axis, angle):
    # from https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToQuaternion/index.htm
    # normalize axis to unit form:
    print(axis, angle)
    xyz = axis * np.sin(angle / 2) / np.sqrt(np.dot(axis, axis))
    print(axis, angle, xyz)
    return R.from_quat([*xyz, np.cos(angle/2)])

def replace_pattern_in_structure(structure, search_pattern, replace_pattern):
    search_pattern = search_pattern.copy()
    replace_pattern = replace_pattern.copy()

    match_indices = find_pattern_in_structure(structure, search_pattern)

    # translate both search and replace patterns so that first atom of search pattern is at the origin
    replace_pattern.translate(-search_pattern.positions[0])
    search_pattern.translate(-search_pattern.positions[0])
    search_axis = search_pattern.positions[-1]

    indices_to_delete = [idx for match in match_indices for idx in match]

    if len(indices_to_delete) > len(set(indices_to_delete)):
        raise Exception("There is an atom that is matched in two distinct patterns. Each atom can only be matched in one atom.")

    new_structure = structure.copy()
    if len(replace_pattern) > 0:
        for match in match_indices:
            atoms = structure[match]
            new_atoms = replace_pattern.copy()
            if len(atoms) > 1:
                found_axis = atoms.positions[-1] - atoms.positions[0]
                q1 = quaternion_from_two_axes(search_axis, found_axis)
                if q1 is not None:
                    new_atoms.positions = q1.apply(new_atoms.positions)

                if len(atoms) > 2:
                    pass
                    # do orient

            # move replacement atoms into correct position
            new_atoms.translate(atoms.positions[0])
            new_structure.extend(new_atoms)

    del(new_structure[indices_to_delete])

    return new_structure
