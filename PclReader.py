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

def read_pcl_data(scan_files, min_max_depth=(0.1, 30), num_lines=-1):
    pcl_data = {}
    for scan_file in scan_files:
        f = open(scan_file, "r")
        # read all lines or only a certain number of lines
        lines = []
        if num_lines == -1:
            lines = f.readlines()
        else:
            for i in range(0, num_lines + 1):
                lines.append(f.readline())
        f.close()
        del (lines[0])

        pcl = []
        for l in lines:
            if l.strip().split(" ")[0] == "" or l == "\n":
                continue  # line is empty
            p = [float(p) for p in l.split(", ")]  # read the point
            assert (len(p) == 5)
            # if point is valid(reflectance>0.) push to pcl
            depth = (p[0] ** 2 + p[1] ** 2 + p[2] ** 2) ** 0.5
            if p[3] > 0. and min_max_depth[0] < depth < min_max_depth[1]:
                pcl.append(p)

        if len(pcl) > 0:
            pcl_data[scan_file.split("/")[-1]] = pcl  # laser-scanner scans from left to right
    # assert (len(pcl_data) == num_lines)
    return pcl_data
