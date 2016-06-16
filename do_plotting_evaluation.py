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
from evaluate_calibration_box import plot_results
import cPickle as pickle
import os
from matplotlib import pyplot as plt


def main():
    usage = "usage: %prog [options]"

    # handle the commands line options
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-i", "--input", dest="input",
                 help="Description of input" )
    parser.add_option("-o", "--output", dest="output",
                 help="Descripion of output")

    (options, argv) = parser.parse_args()

    input_file = options.input
    output_dir = options.output

    #if not os.path.isfile(input_file) or input_file.split(".")[-1] == "p":
    #    raise Exception("no file "+input_file+" found")
    # calculate value of variable
    # dump it
    result = pickle.load(open(input_file, "r"))

    plt.ioff()
    plot_results(result)
    plt.show()

if __name__ == '__main__':
    main()
