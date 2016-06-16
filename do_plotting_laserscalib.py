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

import optparse
import pickle
import Plotting
import os
from FunctionsForCalibration import convert_vector_rot_trans_to_homogenous


def main():
    usage = "usage: %prog [options]"

    # Handle the commands line options
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-i", "--input", dest="input",
                      help="pickle with results data")
    parser.add_option("-o", "--output", dest="output",
                      help="output directory")
    parser.add_option("-e", "--extension", dest="extension", default=".svg",
                      help="file extension")

    (options, argv) = parser.parse_args()

    dir_to_save = options.output

    pickle_file = options.input
    if not os.path.isfile(pickle_file):
        raise Exception("invalid pickle_file " + pickle_file, "in main")
    print("pickle_file to read from= " + pickle_file)

    if not os.path.isdir(dir_to_save):
        raise Exception("invalid output directory " + dir_to_save, "in main")
    print "output path= " + dir_to_save

    (pcl, cam_points, intrin, result, cov_x, infodict) = pickle.load(open(pickle_file, 'rb'))

    trans_laser_cam = convert_vector_rot_trans_to_homogenous(result)

    def depth_func(p):
        return (p[0] ** 2 + p[1] ** 2 + p[2] ** 2) ** 0.5

    print "plotting depth over image row"
    save_name = dir_to_save + "/Row_over_depth" + options.extension
    Plotting.plot_depth_over_row(pcl, cam_points, trans_laser_cam, intrin, save_name, depth_func)

    print "plotting back projection error over depth"
    save_name = dir_to_save + "/BackProjectionErrorWithOutliers" + options.extension
    Plotting.plot_back_projection_error(pcl, cam_points, trans_laser_cam, intrin, save_name,
                                        depth_func, 100000, (1., 8.), 20, 2.0, 1.5)
    save_name = dir_to_save + "/BackProjectionErrorWithoutOutliers" + options.extension
    Plotting.plot_back_projection_error(pcl, cam_points, trans_laser_cam, intrin, save_name,
                                        depth_func, 5, (1., 8.), 10, 2.0, 1.5)


if __name__ == '__main__':
    main()
