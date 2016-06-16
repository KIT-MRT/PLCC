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

import glob
import optparse
from scipy.optimize import leastsq

import math3d

from ErrorFunctions import calculate_error_3d_point_to_ray
from FunctionsForCalibration import *
from HistogramEqualization import plot_histogram, equalize_point_histogram
from MeasurementAssociation import assign_img_names_to_pcl, find_and_assign_cam_measurements_synched
from PclReader import read_pcl_data
from Plotting import plot_measured_points_and_back_projection, plot_depth_over_row, plot_back_projection_error, \
    show_backprojected_points
from RayHandling import get_all_rays


def write_and_print_results(file_to_save, trans_start, result, info_dict, cov_x):
    f = open(file_to_save, 'w')
    f.write("StartVec\t\t=" + str(trans_start) + "\n")
    f.write("result\t\t\t=" + str(result) + "\n")
    f.write("SquaredNorm(Res)/len(Res)=" + str((linalg.norm(info_dict['fvec'])) ** 2 / len(info_dict['fvec'])) + "\n")
    f.write("NumberIterations=" + str(info_dict['nfev']) + "\n")
    f.write("Cov_x=" + str(cov_x) + "\n")
    f.write(str(info_dict) + "\n")

    print "Results"
    print "StartVec\t\t=" + str(trans_start)
    print "result\t\t\t=" + str(result)
    print "SquaredNorm(Res)/len(Res)=" + str((linalg.norm(info_dict['fvec'])) ** 2 / len(info_dict['fvec']))
    print "NumberIterations=" + str(info_dict['nfev'])


def main():
    usage = "usage: %prog [options]"

    # Handle the commands line options
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-i", "--input_dir", dest="input_dir",
                      help="Description of input")
    parser.add_option("-o", "--output_dir", dest="output_dir",
                      help="Descripion of output")
    parser.add_option("-n", "--__intrinsics", dest="intrin_dir",
                      help="path to __intrinsics(one line: F,cu,cv)")
    parser.add_option("-a", "--acc", dest="acc_image_path", default="input_dir/AccumulatedImg.png",
                      help="path to accumulated image")

    (options, argv) = parser.parse_args()

    dir_to_save = options.output_dir
    print "saving stuff to " + dir_to_save

    input_dir = options.input_dir
    if not os.path.isdir(input_dir):
        raise Exception("no image directory input_dir", "in main")
    print "Path to images= " + input_dir

    accumulated_img_path = input_dir + "/AccumulatedImg.png"
    if not options.acc_image_path == "input_dir/AccumulatedImg.png":
        accumulated_img_path = options.acc_image_path
    print "Accumulated image path=" + accumulated_img_path

    # set true if scanner scans clockwise
    scan_clockwise = False
    yaw_dir = "z"

    # parameters
    # __intrinsics
    path_intrin = options.intrin_dir
    if not os.path.isfile(path_intrin):
        print "Intrinsics file needed"
        return

    ifile = open(path_intrin, "rb")
    l = ifile.readlines()
    assert (len(l) < 4)
    s = l[1].split(", ")
    assert (len(s) == 3)
    f = float(s[0])
    cu = float(s[1])
    cv = float(s[2])
    intrin = np.array([[f, 0, cu], [0, f, cv], [0, 0, 1.]])
    print "Intrinsics="
    print intrin

    # get pcl data from scan and corresponding timestamps
    print 'reading pcl Data from ' + input_dir
    print "..."
    file_list = sorted(glob.glob(input_dir + "/*.csv"))
    pcl_dict = read_pcl_data(file_list)
    print "done"

    cur_pickle_dir_rays = dir_to_save + "/rays.p"
    if not os.path.isfile(cur_pickle_dir_rays):
        print "get all laser rays\n..."
        rays = get_all_rays(pcl_dict, yaw_dir)

        print "found " + str(len(rays)) + " laser rays"
        print "done"
        # assert(len(rays)==45)
        pickle.dump(rays, open(cur_pickle_dir_rays, "wb"))
        print "saved rays to " + cur_pickle_dir_rays
    else:
        rays = pickle.load(open(cur_pickle_dir_rays, "rb"))
        print "loaded rays" + cur_pickle_dir_rays

    cur_pickle_dir_dict_img_names = dir_to_save + "/dict_img_names_to_pcl.p"
    if not os.path.isfile(cur_pickle_dir_dict_img_names):
        # get correspondance frame->scan (one to many, because framerate scanner<framerate Camera)
        print "establish correspondances\n..."
        img_files = sorted(glob.glob(input_dir + "/*.png"))
        dict_img_names_to_pcl = assign_img_names_to_pcl(pcl_dict, img_files)
        print "done"
        pickle.dump(dict_img_names_to_pcl, open(cur_pickle_dir_dict_img_names, "wb"))
        print "saved rays to " + cur_pickle_dir_dict_img_names
    else:
        dict_img_names_to_pcl = pickle.load(open(cur_pickle_dir_dict_img_names, "rb"))
        print "loaded dict_img_names_to_pcl" + cur_pickle_dir_dict_img_names

    if scan_clockwise:
        rays = [x for x in reversed(rays)]
        print "reversed rays"

    # accumulated_img_path=dir_to_save+"/LaserAccu.png"
    num_lines = int(input("please type number of rays"))
    epipolar_lines_lr, debug_img = get_epipolarlines_user_input(accumulated_img_path, dir_to_save, num_lines)
    # draw_lines(epipolar_lines_lr,debug_img)

    # find measurements in image
    num_valid_points = 500000
    print "find measurements in image\n..."
    pcl, cam_points = find_and_assign_cam_measurements_synched(dir_to_save + "/Correspondances.p",
                                                               dict_img_names_to_pcl,
                                                               num_valid_points, epipolar_lines_lr, rays)
    print "done"

    bins = np.arange(0.5, 2., 0.3)
    bins = np.hstack([bins, np.arange(2., 6., 0.5)])
    bins = np.hstack([bins, np.arange(6., 10., 1.0)])

    max_num_elements_in_bin = 1500
    pcl, cam_points = equalize_point_histogram(pcl, cam_points, bins, max_num_elements_in_bin,
                                               lambda p: (p[0] ** 2 + p[1] ** 2 + p[2] ** 2) ** 0.5)
    plot_histogram(pcl, bins)
    print("number of measurements used for optimization="+str(len(cam_points)))

    # #debug
    # print debug_calculate_back_projection_error(trans_start, zip(pcl,cam_points), intrin, accumulated_img_path)
    # show back projected points
    print "show back-projected points"
    img = cv2.imread(accumulated_img_path, 0)
    show_backprojected_points(img.copy(), cam_points)
    print "done"

    # Back projection minimization
    # trans_start=np.array(1.209199010985[-0.051,0.5*np.pi/180,0,-0.02,-0.13,0])
    # trans_start = np.array([0.01996929, -0.01193875, -0.01405786, -0.02036318, 0.12714316, -0.04107516])

    # t = [0., 0.08, 0.]
    t = [0., 0.08, 0.]
    # rot_to_cam = math3d.Orientation.new_rot_z(np.pi/4.) * math3d.Orientation.new_rot_z(-np.pi/2.)
    # * math3d.Orientation.new_rot_x(-np.pi/2.) * math3d.Orientation.new_rot_z(np.pi)
    rot_to_cam = math3d.Orientation.new_rot_x(-np.pi/2.)
    # rot_to_cam = rot_to_cam.inverse * math3d.Orientation.new_rot_z(-np.pi/4.)
    # rot_to_cam = math3d.Orientation.new_rot_z(-np.pi / 2.) * math3d.Orientation.new_rot_x(-np.pi/2.)
    # * math3d.Orientation.new_rot_z(np.pi)
    rot_vec = rot_to_cam.get_rotation_vector().tolist()
    rot_vec.extend(t)
    trans_start = np.array(rot_vec)

    # print("showing back projection of pcl before optimization")
    # save_name = dir_to_save + "/pcl_back_projected_before_optimization.pdf"
    # plot_measured_points_and_back_projection(img.copy(), cam_points, rays,
    #                                          convert_vector_rot_trans_to_homogenous(trans_start), save_name, intrin)
    # print("done")
    # put debug in extra script and load pickled stuff?
    # debug_rays(rays, zip(pcl, cam_points), trans_start, intrin)

    intrin_inv = np.linalg.inv(intrin)
    start_diff = calculate_error_3d_point_to_ray(trans_start, zip(pcl, cam_points), intrin_inv)
    print start_diff

    print "Minimizing error point to line"
    print "..."
    # more info about leastsq:http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html 
    # result, cov_x,infodict,mesg,ler=leastsq(calculate_back_projection_error,trans_start,
    # args=(zip(pcl,cam_points),intrin),full_output=True)
    result, cov_x, infodict, mesg, ler = leastsq(calculate_error_3d_point_to_ray, trans_start,
                                                 args=(zip(pcl, cam_points), intrin_inv), full_output=True)
    write_and_print_results(dir_to_save + "/Results.dat", trans_start, result, infodict, cov_x)
    print "done\n"

    ###########################################################################################################
    print "Showing and saving debugging images"
    # declarations
    trans_laser_cam = convert_vector_rot_trans_to_homogenous(result)

    pickle.dump((pcl, cam_points, intrin, result, cov_x, infodict), open(dir_to_save + "/Results.p", 'wb'))
    print("saved results to " + dir_to_save + "/Results.p")

    # debug_img_corr(pcl,cam_points,result,intrin)

    print "plotting backprojected points after optim"
    save_name = dir_to_save + "/ShowBackProjectionScreenshot.pdf"
    plot_measured_points_and_back_projection(img.copy(), cam_points, rays, trans_laser_cam, save_name, intrin)

    print "plotting depth over image row"
    save_name = dir_to_save + "/Row_over_depth.pdf"
    plot_depth_over_row(pcl, cam_points, trans_laser_cam, intrin, save_name,
                        lambda p: (p[0] ** 2 + p[1] ** 2 + p[2] ** 2) ** 0.5)

    print "plotting back projection error over depth"
    save_name = dir_to_save + "/BackProjectionError.pdf"
    plot_back_projection_error(pcl, cam_points, trans_laser_cam, intrin, save_name,
                               lambda p: (p[0] ** 2 + p[1] ** 2 + p[2] ** 2) ** 0.5)


if __name__ == "__main__":
    main()
