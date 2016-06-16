#!/usr/bin/env python
# encoding: utf-8

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

import cPickle as pickle
import glob
import numpy as np
import optparse
import os

import cv2
from matplotlib import pyplot as plt

from MeasurementAssociation import assign_img_names_to_pcl_names_unsynched
from PclReader import read_pcl_data
from BoxEvaluationClasses import PclBoxSegmentor, BoxSegmentEvaluator, calc_mean_depth_segments, plot_results


def main():
    usage = "usage: %prog [options]"

    # handle the commands line options
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-i", "--input", dest="input",
                      help="input directory")
    parser.add_option("-o", "--output", dest="output",
                      help="output directory")
    parser.add_option("-r", "--results", dest="results",
                      help="pickle file in which results from calibration are stored")

    (options, argv) = parser.parse_args()

    input_dir = options.input
    output_dir = options.output

    do_pickle = True
    max_jitter_sec = 10e-03

    # test directories and files
    if not os.path.isdir(input_dir):
        raise Exception("no directory " + input_dir + " found")
    if not os.path.isdir(output_dir):
        raise Exception("no directory " + output_dir + " found")
    if not os.path.isdir(output_dir + "/debug/"):
        os.mkdir(output_dir + "/debug/")
        print("created output dir " + output_dir + "/debug/")
    if not os.path.isfile(options.results) and options.results.split(".")[-1] == "cur_var":
        raise Exception("no file " + options.results + " found")

    print("Loading calib results from " + options.results)
    (_, _, intrin, result, cov_x, infodict) = pickle.load(open(options.results, 'rb'))

    print("Reading data from " + input_dir + "/*.csv")
    scan_files = sorted(glob.glob(input_dir + "/*.csv"))

    # get correspondance frame->scan
    img_files = sorted(glob.glob(input_dir + "/*.png"))

    print "assign image names to pcl names..."
    if do_pickle:
        pickle_file = output_dir + "/" + "evaluation_box_names_img_to_pcl" + ".p"
        if not os.path.isfile(pickle_file):
            # calculate value of variable
            dict_img_names_to_pcl_names = assign_img_names_to_pcl_names_unsynched(scan_files, img_files,
                                                                                  max_jitter_sec=max_jitter_sec)
            pickle.dump(dict_img_names_to_pcl_names, open(pickle_file, "wb"))
            print "buffered dict_img_names_to_pcl to " + pickle_file
        else:
            dict_img_names_to_pcl_names = pickle.load(open(pickle_file, "rb"))
            print "loaded dict_img_names_to_pcl from " + pickle_file
    else:
        dict_img_names_to_pcl_names = assign_img_names_to_pcl_names_unsynched(scan_files, img_files,
                                                                              max_jitter_sec=max_jitter_sec)

    s_pcl_box = PclBoxSegmentor(0.1)
    s_box_eval = BoxSegmentEvaluator(intrin, result)

    result = []
    for cur_img_name, cur_pcl_name in dict_img_names_to_pcl_names.items():
        print("reading " + cur_pcl_name)
        pcl_dict = read_pcl_data([cur_pcl_name], (0.3, 7.))
        assert (len(pcl_dict) < 2)
        cur_pcl = np.array(pcl_dict.values()[0])[:, :3]
        # cur_pcl = sorted(cur_pcl, key=lambda x: np.arctan2(x[0], x[1]))
        calc_angle = [[np.arctan2(x[0], x[1]) * 180 / np.pi, x[-2]] for x in pcl_dict.values()[0]]
        # calc_angle=sorted(calc_angle, key=lambda x: x[0])
        box_segment = s_pcl_box.process(cur_pcl)

        if not box_segment == []:
            img = cv2.imread(cur_img_name, 0)
            errs = s_box_eval.process(img, box_segment, debug_dir=output_dir + "/debug/")
            if not errs == []:
                result.append((errs, calc_mean_depth_segments(box_segment)))
                print("current result=")
                print(result[-1])

    pickle_file = output_dir + "/" + "box_evaluation_results" + ".p"

    # calculate value of variable
    # dump it
    pickle.dump(result, open(pickle_file, "wb"))

    plt.ioff()
    plot_results(result)
    plt.show()


if __name__ == '__main__':
    main()
