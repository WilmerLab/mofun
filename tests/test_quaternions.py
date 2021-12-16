import warnings

import numpy as np
from pytest import approx

from mofun import quaternion_from_two_vectors, quaternion_from_two_vectors_around_axis


def test_quaternion_from_two_axes__with_antiparallel_z_axes_inverts_z_coord():
    q = quaternion_from_two_vectors((0.,0.,2.),(0.,0.,-2.))
    assert q.apply([1., 1., 1.])[2] == approx(-1., 1e-5)

def test_quaternion_from_two_vectors__with_almost_antiparallel_z_axes_mostly_inverts_yz_coords():
    q = quaternion_from_two_vectors((0.,0.,1.),(0.,0.00001,-0.99999))
    assert np.isclose(q.as_quat(), [-1., 0., 0., 0.], atol=1e-5).all()
    assert np.isclose(q.apply([1., 1., 1.]), [1., -1., -1.]).all()

def test_quaternion_from_two_vectors__with_parallel_axes_is_donothing_0001():
    q = quaternion_from_two_vectors((0., 0., 2.),(0., 0., 2.))
    assert np.isclose(q.as_quat(), [0., 0., 0., 1.]).all()
    assert (q.apply([1., 1., 1.]) == [1., 1., 1.]).all()

def test_quaternion_from_two_vectors__with_almost_parallel_axes_is_almost_donothing_0001():
    q = quaternion_from_two_vectors((0., 0., 1.),(0., 0.00001, 0.99999))
    assert np.isclose(q.as_quat(), [0., 0., 0., 1.], atol=1e-5).all()
    assert np.isclose(q.apply([1., 1., 1.]), [1., 1., 1.]).all()

def test_quaternion_from_two_vectors__with_perpendicular_xy_axes_rotates_90_degrees_around_z_axis():
    q = quaternion_from_two_vectors((1., 0., 0.),(0., 1., 0.))
    assert np.isclose(q.as_quat(), [0., 0., 2**0.5/2, 2**0.5/2.], atol=1e-5).all()
    assert np.isclose(q.apply([1., 1., 1.]), [-1., 1., 1.]).all()
    assert np.isclose(q.apply([-1., 1., 1.]), [-1., -1., 1.]).all()
    assert np.isclose(q.apply([-1., -1., 1.]), [1., -1., 1.]).all()
    assert np.isclose(q.apply([1., -1., 1.]), [1., 1., 1.]).all()

def test_quaternion_from_two_vectors_around_axis__with_parallel_vectors_is_donothing_0001():
    q = quaternion_from_two_vectors_around_axis((0., 0., 2.),(0., 0., 2.), axis=(1., 0., 0.))
    assert np.isclose(q.as_quat(), [0., 0., 0., 1.]).all()
    assert (q.apply([1., 1., 1.]) == [1., 1., 1.]).all()

def test_quaternion_from_two_vectors_around_axis__with_antiparallel_about_x_reverses_yz():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = quaternion_from_two_vectors_around_axis((-5, -1, 0.),(5, 1, 0.), axis=(1., 0., 0.))

    assert np.isclose(q.apply([1., 1., 1.]), [1., -1., -1.]).all()
    assert np.isclose(q.apply([1., -1., 1.]), [1., 1., -1.]).all()
    assert np.isclose(q.apply([1., 1., -1.]), [1., -1., 1.]).all()

def test_quaternion_from_two_vectors_around_axis__perp():
    q = quaternion_from_two_vectors_around_axis((0., 0., 1.), (0., 1., 0.), (1., 0., 0.))
    assert np.isclose(q.apply([0., 0., 1.]), [0., 1., 0.]).all()

    q = quaternion_from_two_vectors_around_axis((1., 0., 1.), (1., 1., 0.), (1., 0., 0.))
    assert np.isclose(q.apply([1., 0., 1.]), [1., 1., 0.]).all()
