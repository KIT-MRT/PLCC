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

import cv2
import pylab as pl
from matplotlib import pyplot as plt

from RayHandling import get_ray_of_point, get_index_of_ray_in_rays
from FunctionsForCalibration import project_laser_to_cam, convert_vector_rot_trans_to_homogenous

from mpl_toolkits.mplot3d import Axes3D


def get_color(num):
    modus = np.mod(num, 7)
    if modus == 0:
        return "b"
    elif modus == 1:
        return "g"
    elif modus == 2:
        return "r"
    elif modus == 3:
        return "c"
    elif modus == 4:
        return "m"
    elif modus == 5:
        return "y"
    elif modus == 6:
        return "k"
    else:
        print "could not get color"
        raise Exception("could not get color from index")


def get_color_cv(num):
    modus = np.mod(num, 7)
    if modus == 0:
        return (255, 0, 0)
    elif modus == 1:
        return (0, 255, 0)
    elif modus == 2:
        return (0, 0, 255)
    elif modus == 3:
        return (0, 255, 255)
    elif modus == 4:
        return (255, 0, 255)
    elif modus == 5:
        return (255, 255, 0)
    elif modus == 6:
        return (125, 0, 125)
    else:
        print "could not get color"
        raise Exception("could not get color from index")


def plot_measured_points_and_back_projection(img, cam_points, pcl, trans_laser_cam, save_name, intrin):
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # showing measured points
    for CP in cam_points:
        cimg[CP[1], CP[0]] = (0, 0, 255)

    # ProjPoint0=project_laser_to_cam(np.array([0.,0.,0.01]),trans_laser_cam,intrin)
    # for Point in pcl:
    #     proj_point=project_laser_to_cam(100.*Point,trans_laser_cam,intrin)
    #     cv2.line(cimg,(int(ProjPoint0[0]),int(ProjPoint0[1])),(int(proj_point[0]),int(proj_point[1])),(0,255,0),2)

    for Point in pcl:
        for Scale in np.arange(0.01, 10., 0.01):
            proj_point = project_laser_to_cam(Scale * Point, trans_laser_cam, intrin)
            if 0 < proj_point[1] < cimg.shape[0] and 0 < proj_point[0] < cimg.shape[1]:
                cimg[proj_point[1], proj_point[0]] = (255, 0, 0)

    # for Point in pcl:
    #     proj_point = project_laser_to_cam(Point, trans_laser_cam, intrin)
    #     cv2.circle(cimg, (int(proj_point[0]), int(proj_point[1])), 5, (0, 255, 0), 1)

    # create dummy colors for legend
    blue = plt.Rectangle((0, 0), 1, 1, facecolor=(0, 0, 1))
    red = plt.Rectangle((0, 0), 1, 1, facecolor=(1, 0, 0))
    # green = plt.Rectangle((0, 0), 1, 1, facecolor=(0, 1, 0))
    # show with legend maximize and save
    # show
    plt.figure("Measurements plot")
    plt.imshow(cimg)
    plt.legend([blue, red], ["Valid points measured with camera", "Reprojected laser measurements"])
    # full screen
    # figManager = plt.get_current_fig_manager()
    # figManager.resize(*figManager.window.maxsize())
    # save
    pl.savefig(save_name)
    print "Saved image to " + save_name

    print "close window"
    plt.show()


def plot_depth_over_row(pcl, cam_points, trans_laser_cam, intrin,
                        save_name_2, depth_func=lambda p: p[2], marker_size=1.5):
    plotting_data = {"zCam": [], "zProj": [], "Proj": [], "Cam": []}
    for [Point, CamMeas] in zip(pcl, cam_points):
        plotting_data["zCam"].append(depth_func(Point))
        plotting_data["Cam"].append(CamMeas[1])

        proj_point = project_laser_to_cam(Point, trans_laser_cam, intrin)
        if abs(proj_point[1]) < 2. * intrin[1][2]:
            plotting_data["Proj"].append(proj_point[1])
            plotting_data["zProj"].append(depth_func(Point))

    plt.plot(plotting_data["zCam"], plotting_data["Cam"], 'b.', label="measured points", ms=marker_size)
    plt.plot(plotting_data["zProj"], plotting_data["Proj"], 'r.', label="projected points", ms=marker_size)
    plt.legend()
    plt.ylabel("image row in px")
    plt.xlabel("measured depth of corresponding laser measurement in m")
    pl.savefig(save_name_2)
    print "Saved row_over_depth to " + save_name_2

    print "close window to enter debugging"
    plt.show()


def plot_mean_standard_dev_bins(x, y, nbins, lwidth=1, linestyle='r-'):
    n, _ = np.histogram(x, bins=nbins)
    sy, _ = np.histogram(x, bins=nbins, weights=y)
    sy2, _ = np.histogram(x, bins=nbins, weights=[y_c ** 2 for y_c in y])
    mean = sy / n
    std = np.sqrt(sy2 / n - mean * mean)

    plt.errorbar((_[1:] + _[:-1]) / 2, mean, yerr=std, linewidth=lwidth, fmt=linestyle,
                 label="mean and standard deviation in bin")


def plot_back_projection_error(pcl, cam_points, trans_laser_cam, intrin, save_name, depth_func=lambda p: p[2],
                               outlier_threshold=100000.,
                               min_max_depth=(-1., 9999.), number_bins=20, line_width=2, marker_size=0.5):
    plotting_data = {"zCam": [], "BackProjError": [], "Proj": [], "Cam": []}
    for [point, cam_meas] in zip(pcl, cam_points):
        proj_point = project_laser_to_cam(point, trans_laser_cam, intrin)
        back_proj_err = (sum([(x - y) ** 2 for x, y in zip(proj_point[:2], cam_meas[:2])])) ** 0.5

        if min_max_depth[0] < depth_func(point) < min_max_depth[1] and back_proj_err < outlier_threshold:
            plotting_data["zCam"].append(depth_func(point))
            plotting_data["Cam"].append(cam_meas)

            plotting_data["Proj"].append(proj_point)

            plotting_data["BackProjError"].append(back_proj_err)

    plt.plot(plotting_data["zCam"], plotting_data["BackProjError"], 'b.', label="back-projection error", ms=marker_size)
    plot_mean_standard_dev_bins(plotting_data["zCam"], plotting_data["BackProjError"], number_bins, line_width)
    plt.legend()
    plt.ylabel("back-projection error in px")
    plt.xlabel("measured depth in m")
    plt.savefig(save_name)
    print "back-projection error" + save_name

    print "close window to enter debugging"
    plt.show()


def draw_ray(img, r, trans_laser_cam, intrin, color):
    proj_point = project_laser_to_cam(np.array(10. * r), trans_laser_cam, intrin)
    proj_point0 = project_laser_to_cam(np.array(0.05 * r), trans_laser_cam, intrin)
    cv2.line(img, (int(proj_point0[0]), int(proj_point0[1])), (int(proj_point[0]), int(proj_point[1])), color, 5, -1)


def draw_rays(img, rays, trans_laser_cam, intrin):
    for r in rays:
        draw_ray(img, r, trans_laser_cam, intrin, (255, 255, 255))


def plot3d(points):
    fig = plt.figure()
    fig.gca(projection='3d')
    plt.xlabel("x")
    plt.ylabel("y")

    plt.title("MyPlot")
    r = np.array(points)
    plt.plot(r[:, 0], r[:, 1], r[:, 2], "xr", label="MyLabel")
    plt.legend(loc="best")

    plt.show()


def debug_rays(rays, pcl_and_points, trans_vec_laser_cam, intrin):
    plot3d(rays)

    img = np.zeros((600, 800, 3), dtype=np.uint8)

    trans_laser_cam = convert_vector_rot_trans_to_homogenous(trans_vec_laser_cam)

    draw_rays(img, rays, trans_laser_cam, intrin)
    img_rays = img.copy()

    for p, cp in pcl_and_points:
        cur_img = img_rays.copy()
        ind = get_index_of_ray_in_rays(rays, get_ray_of_point(p))
        draw_ray(cur_img, rays[ind], trans_laser_cam, intrin, (255, 0, 0))

        cv2.circle(cur_img, (int(cp[0]), int(cp[1])), 5, (0, 255, 0), -1)

        proj = project_laser_to_cam(10. * rays[ind], trans_laser_cam, intrin)
        cv2.putText(cur_img, str(ind), (int(proj[0]) - 20, int(proj[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

        plt.imshow(cur_img)
        plt.show()


def debug_img_corr(pcl, cam_points, trans_vec_laser_cam, intrin):
    img = np.zeros((600, 800, 3), dtype=np.uint8)

    max_count = 45
    count = 0

    trans_laser_cam = convert_vector_rot_trans_to_homogenous(trans_vec_laser_cam)

    cur_img = img.copy()
    for p, uv in zip(pcl, cam_points):
        if count > max_count:
            count = 0
            plt.imshow(cur_img, cmap=pl.gray())
            plt.show()
            cur_img = img.copy()

        proj_point = project_laser_to_cam(p, trans_laser_cam, intrin)
        cv2.circle(cur_img, (int(proj_point[0]), int(proj_point[1])), 5, 125, -1)
        cv2.circle(cur_img, (int(uv[0]), int(uv[1])), 3, 255, -1)

        count += 1


def show_backprojected_points(img, cam_points):
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # showing measured points
    for CP in cam_points:
        cimg[CP[1], CP[0]] = (0, 0, 255)
    blue = plt.Rectangle((0, 0), 1, 1, facecolor=(0, 0, 1))
    # red = plt.Rectangle((0, 0), 1, 1, facecolor=(1, 0, 0))
    # green = plt.Rectangle((0, 0), 1, 1, facecolor=(0, 1, 0))
    # show with legend maximize and save
    # show
    plt.figure("Measurements plot")
    plt.imshow(cimg)
    plt.legend([blue], ["Valid points measured with camera"])
    print "please close the window"
    plt.ioff()
    plt.show()
