#
# This file is part of PLCC.
#
# Copyright 2016 Johannes Graeter <johannes.graeter@kit.edu (Karlsruhe Institute of Technology)
#
# PLCC comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
#
# PLCC is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PLCC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import numpy as np

from numpy import dot

from FunctionsForCalibration import convert_vector_rot_trans_to_homogenous


def calc_distance_point_line(p, l):
    a = np.cross(l, p)
    n = np.cross(a, l)
    n /= np.linalg.norm(n)
    return dot(n, p)  # no norm needed since leastsq calculates the sum of squares


def calculate_error_3d_point_to_ray(trans_vec_laser_cam, point_correspondences_3d_2d, intrin_inv):
    """calculate back projection error  for least square problem

    :trans_laser_cam: argument of minimization -> transformation from laser scanner coordinates to cam coordinates
    :intrin_inv: inverse __intrinsics of camera
    :point_correspondences_3d_2d: list with corresponding points(arrays) -> 3D,2D
    :returns: back projection error (vector)

    """
    residuals = []

    trans_laser_cam = convert_vector_rot_trans_to_homogenous(trans_vec_laser_cam)

    for Corr in point_correspondences_3d_2d:
        # transform laser point into camera frame
        local_point_3d = dot(trans_laser_cam, np.hstack([Corr[0][:3], 1.]))

        # calc ray of measured point
        ray = dot(intrin_inv, np.hstack([Corr[1][:2], 1.]))

        # calc distance to point to ray
        val = calc_distance_point_line(local_point_3d[:3], ray)
        # val=val/local_point_3d[2]

        residuals.append(val)

    # loss_function(residuals)

    return np.array(residuals)