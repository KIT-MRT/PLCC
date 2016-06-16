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
import os
import pickle

import cv2

from FunctionsForCalibration import locate_laser_points_image_all_epipolarlines
from RayHandling import get_index_of_ray_in_rays, get_ray_of_point
from PclReader import read_pcl_data


def strip_to_num(s, extension):
    return s.split("/")[-1].split("_")[-1].split(extension)[0]


def assign_img_names_to_pcl(pcl_data, img_name_list, max_jitter_sec=100e-03):
    max_jitter_nanosec = max_jitter_sec * 1e9
    dict_img_name_pcl = {}
    for key, val in pcl_data.items():
        for n in img_name_list:
            ts_frame = long(strip_to_num(key, ".csv"))
            if ts_frame == long(strip_to_num(n, ".png")):
                if len(val) > 0:
                    jitter = abs(val[0][-1] - ts_frame)
                    if jitter < max_jitter_nanosec:
                        dict_img_name_pcl[n] = val
                else:
                    dict_img_name_pcl[n] = []
    return dict_img_name_pcl


def assign_img_names_to_pcl_names(pcl_names, img_name_list, max_jitter_sec=100e-03):
    max_jitter_nanosec = max_jitter_sec * 1e9
    dict_img_name_pcl = {}
    for key in pcl_names:
        for n in img_name_list:
            ts_frame = long(strip_to_num(key, ".csv"))
            if ts_frame == long(strip_to_num(n, ".png")):
                p = read_pcl_data([key], num_lines=1)
                if len(p) > 0:
                    cur_pcl = p.values()[0]
                    jitter = abs(cur_pcl[0][-1] - ts_frame)
                    if jitter < max_jitter_nanosec:
                        dict_img_name_pcl[n] = key
                        # else:
                        #     dict_img_name_pcl[n] = ""

    return dict_img_name_pcl


def assign_img_names_to_pcl_names_unsynched(pcl_names, img_name_list, max_jitter_sec=100e-03):
    max_jitter_nanosec = max_jitter_sec * 1e9
    dict_img_name_pcl = {}
    ts_images = np.array([long(strip_to_num(n, ".png")) for n in img_name_list])
    for key in pcl_names:
        ts_scan = long(strip_to_num(key, ".csv"))
        diff = np.abs(ts_images - ts_scan)
        min_diff = min(diff)
        img_index = diff.tolist().index(min_diff)

        if min_diff < max_jitter_nanosec:
            # print("jitter=" + str(min_diff*1e-9))
            dict_img_name_pcl[img_name_list[img_index]] = key

    return dict_img_name_pcl


def find_and_assign_cam_measurements_synched(pickle_file, dict_img_names_to_pcl, num_valid_images, epipolarlines_lr,
                                             rays):
    valid = 0
    cam_points = []
    pcl = []
    if not os.path.isfile(pickle_file):
        print "Extracting correspondences\n..."
        for im_name, CurPCL in dict_img_names_to_pcl.items():
            if valid < num_valid_images and os.path.isfile(im_name):
                img = cv2.imread(im_name, 0)
                img = cv2.GaussianBlur(img, (7, 7), 0.8)

                centroids = locate_laser_points_image_all_epipolarlines(img, epipolarlines_lr, 10, False)

                # assign point to uv measurement from camera
                for P in CurPCL:
                    ind = get_index_of_ray_in_rays(rays, get_ray_of_point(P))

                    if ind < len(centroids):
                        cur_uv = centroids[ind]
                        if not cur_uv == -1:
                            pcl.append(P)
                            cam_points.append(cur_uv)
                            valid += 1

        print "Number of valid measurements=" + str(valid)
        pickle.dump((cam_points, pcl), open(pickle_file, "wb"))
    else:
        print "!!!!!!!!!!!!!! loading correspondaces !!!!!!!!!!!!!!!!!"
        cam_points, pcl = pickle.load(open(pickle_file, "rb"))

    pcl = np.array(pcl)
    cam_points = np.array([[float(cp['u']), float(cp['v'])] for cp in cam_points])

    return pcl, cam_points
