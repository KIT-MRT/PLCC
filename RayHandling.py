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


def get_ray_of_point(point):
    p = np.array([point[0], point[1], point[2]])
    p_ray = p/np.linalg.norm(p)
    p_ray *= np.sign(dot(p_ray, p))

    return p_ray


def reduce_ray_in_rays(rays, p_ray):
    a1 = np.isclose(rays, p_ray, atol=0.00001)
    a2 = np.array([x.all() for x in a1])
    return a2


def is_ray_in_rays(rays, p_ray):
    return reduce_ray_in_rays(rays, p_ray).any()


def get_index_of_ray_in_rays(rays, p_ray):
    return reduce_ray_in_rays(rays, p_ray).tolist().index(True)


def get_all_rays(pcl_dict, yaw_dir="z"):
    rays = [get_ray_of_point(pcl_dict.values()[0][0])]

    for PCL in pcl_dict.values():
        for Point in PCL:
            p_ray = get_ray_of_point(Point)
            # Occurences=filter(lambda x: np.allclose(x,p_ray,atol=0.0000001),rays)
            # pdb.set_trace()
            # if len(Occurences)>0:

            # pdb.set_trace()
            if not is_ray_in_rays(rays, p_ray):
                rays.append(p_ray)

    if yaw_dir == "y":
        sort_func = lambda p: np.arctan2(p[2], p[0])  # sort by yaw angle
    elif yaw_dir == "z":
        sort_func = lambda p: np.arctan2(p[1], p[0])  # sort by yaw angle
    elif yaw_dir == "x":
        sort_func = lambda p: np.arctan2(p[2], p[1])  # sort by yaw angle
    else:
        assert 0

    rays = sorted(rays, key=sort_func)
    return rays