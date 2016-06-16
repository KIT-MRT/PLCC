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
import random

from matplotlib import pyplot as plt


def plot_histogram(pcl, bins):
    # x = [p[2] for p in pcl]
    x = [(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) ** 0.5 for p in pcl]
    # the histogram of the data
    n, bins, patches = plt.hist(x, bins, facecolor='green', alpha=0.5)
    plt.xlabel('Depth-Bins in meter')
    plt.ylabel('Number of points')
    plt.title('Histogram of laserscanner points')

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    print "please close the window"
    plt.ioff()
    plt.show()


def equalize_point_histogram(pcl, cam_points, bins, max_num_elements_in_bin, depth_func=lambda p: p[2]):
    zipped = zip(pcl, cam_points)
    # sort by x
    # zipped=sorted(zipped,key=lambda x:x[0][2])
    aranged = [[] for x in bins]

    # arange them
    for z in zipped:
        mindiffz = bins - depth_func(z[0])
        index = np.argmin(abs(mindiffz))
        if mindiffz[index] < 0 and index < len(mindiffz) - 1:
            index += 1
        aranged[index].append(z)

    aranged_equ = []
    for x in aranged:
        if len(x) > max_num_elements_in_bin:
            aranged_equ.append(random.sample(x, max_num_elements_in_bin))
        else:
            aranged_equ.append(x)

    assert (len(aranged) == len(bins))

    pcl_new = [z[0] for Arr in aranged_equ for z in Arr]
    cam_points_new = [z[1] for Arr in aranged_equ for z in Arr]

    return pcl_new, cam_points_new