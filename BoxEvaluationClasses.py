#
# This file is part of PLCC.
#
# Copyright 2016 Johannes Graeter <johannes.graeter@kit.edu (Karlsruhe Institute of Technology)
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
import numpy as np
from scipy.optimize import leastsq

import cv2
import math3d
from matplotlib import pyplot as plt

from Plotting import get_color, get_color_cv, plot_mean_standard_dev_bins


class PclBoxSegmentor(object):
    """this class finds the card box in the scan"""

    def __init__(self, gradient_thres):
        """Constructor for PclBoxSegmentor"""
        self.gradient_thres = gradient_thres

        self.__box_depth_tol = 0.3
        self.__width_tol = 0.05

        self.__box_width_left = 0.16
        self.__box_width_middle = 0.12
        self.__box_width_right = 0.16
        self.__box_depth = 0.4

        # vr_diff: soft threshold on range value change
        # vr_ndiff: soft threshold on neighbor-relative change in range values
        # slope_gain* exp(-dist*slope_exp)+slop_offest =tangent slope at vr_niff (sigmoid function)
        # if dist<diff_min points are seen directly as one cluster
        # self.__linkage_params = {"vr_diff": 0.2, "vr_ndiff": 1.90782, "slope_gain": 2., "slope_exp": 0.14,
        #                          "slope_offset": 0.25, "diff_min": 0.1}
        self.__linkage_params = {"vr_diff": 2.0, "vr_ndiff": 1.90782, "slope_gain": 2., "slope_exp": 0.14,
                                 "slope_offset": 0.5, "diff_min": 0.07}
        self.__linkage_thres = 0.1  # between 0 and one, the bigger, the more less segments

    def __calc_err_line(self, x_y_yaw, pcl, line):
        x, y, yaw = x_y_yaw
        p0 = np.array(line[0])
        p1 = np.array(line[1])
        direction = p1 - p0

        t = math3d.Transform([0., 0., yaw], [x, y, 0.])
        p0 = np.array(t.get_pos().list) + np.array([p0[0], p0[1], 0])
        direction = np.array((t.get_orient() * math3d.Vector([direction[0], direction[1], 0.])).list)

        s1 = np.sign(np.dot(direction, np.array([0, 1, 0.]))) * np.linalg.norm(direction)
        direction /= np.linalg.norm(direction)

        s0 = 0.

        err = []
        for p in pcl:
            cur_s = np.dot(direction, p - p0)
            if s0 < cur_s < s1 or s1 < cur_s < s0:
                err.append(p - direction * cur_s + p0)

        err = [el / len(err) for el in err]
        return err

    def __calc_error_lines(self, x_y_yaw):
        total_err = []
        for l in self.__template:
            total_err.extend(self.__calc_err_line(x_y_yaw, self.__pcl, l))

    def __template_matching(self, pcl,
                            template=[((0., 0., 0.), (0., 0.16, 0.)), ((0., 0.19, 0.), (0., 0.31, 0.)),
                                      ((0., 0.34, 0.), (0., 0.5, 0.))]):
        self.__template = template
        self.__pcl = pcl
        print np.array(pcl).shape

        result, cov_x, infodict, mesg, ler = leastsq(self.__calc_error_lines, np.array([0., 0., 0]), full_output=True)

        print result

    def __calc_segments_gradient(self, pcl):
        pcl_segments = []
        cur_segment = []
        for i in range(1, len(pcl)):
            # grad = self.__double_sided_gradient(pcl[i - 1], pcl[i], pcl[i + 1])
            depth_diff = np.linalg.norm(pcl[i - 1]) - np.linalg.norm(pcl[i])

            # if difference in depth is bigger than threshold add segment and fill new one
            if abs(depth_diff) > self.gradient_thres:
                if not cur_segment == []:
                    pcl_segments.append(cur_segment)
                    cur_segment = []
            cur_segment.append(pcl[i])
        return pcl_segments

    def __calc_segments_linkage(self, pcl):
        pcl_segments = []
        cur_segment = []
        for i in range(1, len(pcl) - 2):
            linkage = self.__calc_linkage(np.array([pcl[i - 1], pcl[i], pcl[i + 1], pcl[i + 2]]))
            if abs(linkage) < self.__linkage_thres:
                if not cur_segment == []:
                    pcl_segments.append(cur_segment)
                    cur_segment = []
            cur_segment.append(pcl[i])
        return pcl_segments

    @staticmethod
    def __sigmoid_like_soft_threshold(x, theta, m):
        return 0.5 - ((0.5 * (x - theta) * m) / (1 + (x - theta) * (x - theta) * m * m) ** 0.5)

    def __compute_forward_linkage_measure(self, d_i, d_j, abs_diff_hi, abs_diff_ij, abs_diff_jk, vr_diff, vr_ndiff,
                                          slope_gain, slope_exp, slope_offset, diff_min):
        # Reject linkage if distances out of range
        # if d_i < 0.01 or d_j < 0.01:
        #     return 0.
        # Always accept a small connection, so only check further if sufficient
        # large connection (avoids singularity if diff_hi / diff_jk are also small).
        if abs_diff_ij < diff_min:
            return 1.
        # Always reject connection if it is not small but one neighboring
        # connection is (avoids division by zero)
        if abs_diff_hi < diff_min / 2. or abs_diff_jk < diff_min / 2.:
            return 0

        # Approximate distance (for normalization)
        dist = min([d_i, d_j])
        assert (dist > 0.)
        # Determine the tangential slope
        vr_nf = slope_gain * np.exp(-dist * slope_exp) + slope_offset

        # Compute exact threshold value
        keep = self.__sigmoid_like_soft_threshold(abs_diff_ij / dist, vr_diff, 2. / vr_diff)
        keep = min([keep,
                    self.__sigmoid_like_soft_threshold(abs(abs_diff_ij - abs_diff_hi) / abs_diff_hi,
                                                       vr_ndiff, vr_nf)])
        keep = min([keep,
                    self.__sigmoid_like_soft_threshold(abs(abs_diff_ij - abs_diff_jk) / abs_diff_jk,
                                                       vr_ndiff, vr_nf)]);
        return keep

    def __calc_linkage(self, point_quadruple):
        d_h = np.linalg.norm(point_quadruple[0])  # Dist for point before
        d_i = np.linalg.norm(point_quadruple[1])  # Distance of point for which linkage is calculated
        d_j = np.linalg.norm(point_quadruple[2])  # Dist for point after that
        d_k = np.linalg.norm(point_quadruple[3])  # Dist for point after that

        linkage = self.__compute_forward_linkage_measure(d_i, d_j, abs(d_h - d_i), abs(d_i - d_j), abs(d_j - d_k),
                                                         self.__linkage_params["vr_diff"],
                                                         self.__linkage_params["vr_ndiff"],
                                                         self.__linkage_params["slope_gain"],
                                                         self.__linkage_params["slope_exp"],
                                                         self.__linkage_params["slope_offset"],
                                                         self.__linkage_params["diff_min"])
        return linkage

    @staticmethod
    def __get_depth_diff(p1, p2):
        return np.linalg.norm(p1) - np.linalg.norm(p2)

    def __find_box_segment(self, s):
        candidates = []
        for i in range(4, len(s)):
            s_left = s[i - 4]
            s_hole_left = s[i - 3]
            s_middle = s[i - 2]
            s_hole_right = s[i - 1]
            s_right = s[i]

            w_left = self.__get_segment_width(s_left)
            w_middle = self.__get_segment_width(s_middle)
            w_right = self.__get_segment_width(s_right)

            s_diff_1 = self.__get_depth_diff(s_left[-1], s_hole_left[0])
            s_diff_2 = self.__get_depth_diff(s_hole_left[-1], s_middle[0])
            s_diff_3 = self.__get_depth_diff(s_middle[-1], s_hole_right[0])
            s_diff_4 = self.__get_depth_diff(s_hole_right[-1], s_right[0])

            # noinspection PyTypeChecker
            c = [np.isclose(w_left, self.__box_width_left, atol=self.__width_tol),
                 np.isclose(w_middle, self.__box_width_middle, atol=self.__width_tol),
                 np.isclose(w_right, self.__box_width_right, atol=self.__width_tol),
                 np.isclose(s_diff_1, -self.__box_depth, atol=self.__box_depth_tol),
                 np.isclose(s_diff_2, self.__box_depth, atol=self.__box_depth_tol),
                 np.isclose(s_diff_3, -self.__box_depth, atol=self.__box_depth_tol),
                 np.isclose(s_diff_4, self.__box_depth, atol=self.__box_depth_tol)]

            if all(c):
                candidates.append([s_left, s_hole_left, s_middle, s_hole_right, s_right])

        return candidates

    @staticmethod
    def __get_segment_width(s):
        return np.linalg.norm(s[0] - s[-1])

    @staticmethod
    def plot_segment(pcl_segments):
        from matplotlib import pyplot as plt

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plt.xlabel("")
        plt.ylabel("")
        plt.axis("equal")

        plt.title("")
        plt.ioff()
        for i, xyz in enumerate(pcl_segments):
            xyz_np = np.array(xyz)
            PlotHandle = plt.plot(xyz_np[:, 0], xyz_np[:, 1], xyz_np[:, 2], "." + get_color(i), label="")
        plt.legend(loc="best")
        plt.show()

    def process(self, pcl_in):
        """main of class"""
        # print "doing template matching"
        # self.__template_matching(pcl)
        # print "done"
        pcl = self.__preprocess(pcl_in, thres=0.1)
        pcl_segments = self.__calc_segments_gradient(pcl)
        # self.plot_segment(pcl_segments)
        chosen_segment = self.__find_box_segment(pcl_segments)

        out = []
        if len(chosen_segment) == 1:
            out = chosen_segment[0]
        elif len(chosen_segment) > 1:
            print("Warning: there is more than one box segment, tune parameters")

        return out

    def __preprocess(self, pcl_in, thres):
        # reject discontinuous points
        pcl = []
        for i in range(1, len(pcl_in) - 1):
            diff0 = abs(self.__get_depth_diff(pcl_in[i - 1], pcl_in[i]))
            diff1 = abs(self.__get_depth_diff(pcl_in[i], pcl_in[i + 1]))
            if diff0 > thres and diff1 > thres:
                continue
            pcl.append(pcl_in[i])
        return pcl


class BoxSegmentEvaluator(object):
    """given a segmented scan this class finds the corresponding border points
    in the image by their image gradients and evaluates the distance between them"""

    def __init__(self, intrinsics, extrinsics):
        """Constructor for ImageBoxSegmentor"""
        # self.__maxima_binning_percentage = 0.08
        # range in meters in which maxima are searched
        # (if we use back projections as priors) or rejected (whole line search)
        self.__dy = 0.03
        self.__gradient_thres = 100  # if any of the maxima is lower than this threshold, no points will be returned
        # self.__gauss_sigma = 1.0
        # self.__gauss_kernel_size = (9, 9)
        self.__gauss_sigma = 0.3
        self.__gauss_kernel_size = (3, 3)
        self.__intrinsics = intrinsics
        self.__extrinsics = math3d.Transform(extrinsics[:3], extrinsics[3:])
        # self.__ksize_gradient = 3
        self.__ksize_gradient = 5

    @staticmethod
    def __draw_box(img, box_roi):
        p_tl = (int(box_roi["left"]), int(box_roi["top"]))
        p2 = (p_tl[0] + int(box_roi["width"]), p_tl[1] + int(box_roi["height"]))
        cv2.rectangle(img, p_tl, p2, get_color_cv(5))

    def __draw_back_projection(self, img, sub_segments):
        for ind, sub_seg in enumerate(sub_segments):
            col = get_color_cv(ind)
            for p in sub_seg:
                uv = self.__transform_and_back_project(p)
                cv2.circle(img, (int(uv[0]), int(uv[1])), 1, col, -1)

    def get_debug_image_back_projection(self, img, chosen_segment):
        # box_roi = self.__get_box_roi(chosen_segment)
        #
        # v_max = min(box_roi["top"] + box_roi["height"], img.shape[0])
        # u_max = min(box_roi["left"] + box_roi["width"], img.shape[1])
        #
        # sub_img = img[box_roi["top"]:v_max, box_roi["left"]:u_max]

        plot_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # self.__draw_box(plot_img, box_roi)
        self.__draw_back_projection(plot_img, chosen_segment)

        return plot_img

    def get_debug_image_border_points(self, plot_img, border_points, back_projected_border_points):
        """calculates gradient igefroplot ie and shows extracted points on borders"""
        out = self.__get_gradient_image().copy()
        cur_min, cur_max, _, _ = cv2.minMaxLoc(out)
        out = np.uint8((out * 255.0 / (cur_max - cur_min)) - 255 * cur_min / (cur_max - cur_min))
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        for border_point, pp in zip(border_points, back_projected_border_points):
            cv2.circle(out, (int(border_point[0]), int(border_point[1])), 9, get_color_cv(0), 2)
            cv2.circle(out, (int(pp[0]), int(pp[1])), 7, get_color_cv(6), 2)

        return out

    def plot_debug_images(self, img, segment, border_points, back_projections):
        deb_img_1 = self.get_debug_image_back_projection(img, segment)
        deb_img_2 = self.get_debug_image_border_points(img, border_points, back_projections)

        plt.subplot(2, 1, 1)
        plt.imshow(deb_img_1)
        plt.xticks([]), plt.yticks([])
        plt.subplot(2, 1, 2)
        plt.imshow(deb_img_2)
        plt.xticks([]), plt.yticks([])

        return deb_img_1, deb_img_2

    def __write_image_txt(self, n, err_vec, img1, img2):
        # write down image and errors
        cv2.imwrite(n + "_img1.png", img1)
        cv2.imwrite(n + "_img2.png", img2)
        f = open(n + ".txt", "w")
        for e in err_vec:
            f.write(str(e) + "\n")

    def process(self, img, segment, outlier_thres=10, debug_dir=""):
        """main of class"""
        back_projections = self.__calc_border_projection(segment)

        border_points_image = self.__get_border_points_image(img, segment)

        if not border_points_image:
            return []

        if not len(back_projections) == len(border_points_image):
            msg = "in BoxSegmentEvaluator: features have not same length: len(image)=" + str(
                len(border_points_image)) + " len(back_proj)=" + str(len(back_projections))
            # raise Exception(msg)
            print(msg)
            return []

        err_vec = self.__calc_back_projection_errors(back_projections, border_points_image, self.__search_line)

        # for debugging
        is_outlier = any(np.array(err_vec) > outlier_thres)
        if is_outlier:
            num_png = len(glob.glob(debug_dir + "/outlier_*.png"))
            n = debug_dir + "/outlier_" + str(num_png)
            print("detected outlier writing to " + n)
        else:
            num_png = len(glob.glob(debug_dir + "/inlier_*.png"))
            n = debug_dir + "/inlier_" + str(num_png)

        if not debug_dir == "":
            img1, img2 = self.plot_debug_images(img, segment, border_points_image, back_projections)
            self.__write_image_txt(n, err_vec, img1, img2)
            # plt.ioff()
            # plt.show()

        if is_outlier:
            return []
        else:
            return err_vec

    def __get_box_roi(self, chosen_segment):
        p_left = chosen_segment[0][0]
        p_right = chosen_segment[-1][-1]

        uv_left = self.__transform_and_back_project(p_left)
        uv_right = self.__transform_and_back_project(p_right)

        width = uv_right[0] - uv_left[0]
        assert (width > 1.)
        height = width

        return {"top": max(uv_left[1] - height / 2., 0.), "left": max(uv_left[0], 0.), "width": width, "height": height}

    def __transform_and_back_project(self, p):
        p_cam = np.array((self.__extrinsics * math3d.Vector(p)).list)
        uv1 = np.dot(self.__intrinsics, p_cam)
        uv1 /= p_cam[2]
        return uv1[:2]

    @staticmethod
    def __get_border_points(s):
        out = []
        for i in range(1, len(s)):
            out.append((s[i - 1][-1], s[i][0]))
        return out

    def __get_border_points_image(self, img, segment, back_projections=[]):
        """get point with maximum grey value diff between the projections of the border points"""
        assert (len(segment) == 5)

        ps = segment[0]
        ps.extend(segment[2])
        ps.extend(segment[4])

        search_line = self.__calc_search_line(np.array(ps))
        self.__search_line=search_line
        deriv_img = self.__calc_gradient_image(img, cv2.CV_32F)

        # get and right point of segment -> search_interval
        p_left = self.__get_point_from_segment(segment[0], 0.33)
        p_right = self.__get_point_from_segment(segment[4], -0.33)
        # p_left = segment[0][1]
        # p_right = segment[-1][-2]
        uv_left = self.__transform_and_back_project(p_left)
        uv_right = self.__transform_and_back_project(p_right)

        cur_depth = np.linalg.norm(self.__get_point_from_segment(segment[2], 0.5))
        binning_range = self.__intrinsics[0, 0] * self.__dy / cur_depth
        if not back_projections:
            grad_and_point = self.__get_border_point_in_proximity(deriv_img, uv_left, uv_right,
                                                                  search_line, binning_range=binning_range)
        else:
            grad_and_point = []
            for cur_b in back_projections:
                cur_maximum = self.__get_border_point_in_proximity_of_point(deriv_img, cur_b, search_line,
                                                                            search_range=(
                                                                                -binning_range, binning_range),
                                                                            number_points=1)
                grad_and_point.extend(cur_maximum)

        if any(np.array([p[0] for p in grad_and_point]) < self.__gradient_thres):
            return []

        # don't return gradient value
        return [p[1] for p in grad_and_point]

    def __get_point_from_segment(self, segment, proportion=0.33):
        n = len(segment)
        return segment[int((n - 1) * proportion)]

    def __get_maxima_list(self, deriv_img, search_pixel_list):
        # get highest gradient in next cur_var pixels in + and - direction use set to not get double maxima
        val_spot = []
        for cur_pix in self.set_from_list(search_pixel_list, conversion_func=lambda p: (int(p[0]), int(p[1]))):
            if 0 < cur_pix[1] < deriv_img.shape[1] and deriv_img.shape[0] > cur_pix[0] > 0:
                deriv = abs(deriv_img[cur_pix[1], cur_pix[0]])
                val_spot.append((deriv, cur_pix))
        return val_spot

    def __get_border_point_in_proximity(self, deriv_img, p1, p2, search_line, binning_range=2, number_points=4):
        """we search for the border point in search direction by grey value diff"""
        # determine pixels on box and search_line
        search_pixel_list = self.__get_search_pixel_list(p1, p2, search_line)

        val_spot = self.__get_maxima_list(deriv_img, search_pixel_list)

        if not len(val_spot) >= number_points:
            return []

        # sort by gradient and return n max vals
        unique_maxima = self.__get_unqiue_maxima(val_spot, binning_range)

        return unique_maxima

    def __get_border_point_in_proximity_of_point(self, deriv_img, p1, search_line, search_range=(-5, 5),
                                                 number_points=1):
        """we search for the border point in search direction by grey value diff"""
        # determine pixels on box and search_line
        search_pixel_list = self.__get_search_pixel_list_point_range(p1, search_line, search_range)

        val_spot = self.__get_maxima_list(deriv_img, search_pixel_list)

        if not len(val_spot) >= number_points:
            return []

        # sort by gradient and return n max vals
        # unique_maxima = self.__get_unqiue_maxima(val_spot, self.__calc_maxima_binning_range(p1, p2))
        vs_sorted = sorted(val_spot, key=lambda cur_var: cur_var[0])
        vs_sorted = [x for x in reversed(vs_sorted)]

        return vs_sorted[:number_points]

    def __calc_search_line(self, ps):
        """calc dir in which we will search the border point"""
        uvs = np.array([self.__transform_and_back_project(p) for p in ps])

        # now we perform a pca to get the least square solution
        # of the line fitting problem http://sebastianraschka.com/Articles/2014_pca_step_by_step.html
        mean_uv = np.mean(uvs, 0)
        assert (len(mean_uv) == 2)
        cov_mat = np.cov(uvs.T)
        assert (cov_mat.shape == (2, 2))
        eig_val, eig_vec = np.linalg.eig(cov_mat)

        # Sort the (eigenvalue, eigenvector) tuples from low to high
        low_val, normal = sorted(zip(np.abs(eig_val), eig_vec))[0]

        normal /= np.linalg.norm(normal)
        direction = np.array([-normal[1], normal[0]])
        return direction, mean_uv

    @staticmethod
    def __get_search_pixel_list(p1, p2, line):
        """get search interval from p1 to p2 on line is given as normal and distance in image coordinates"""
        direction, p0 = line

        direction /= np.linalg.norm(direction)

        # get line vars according to extremities of box
        s1 = int(np.dot(p1 - p0, direction))
        s2 = int(np.dot(p2 - p0, direction))

        # define a search range
        search_range = range(s1, s2)
        if not search_range:
            search_range = range(s2, s1)

        # calc corresponding pixels
        out = [direction * i + p0 for i in search_range]

        return out

    @staticmethod
    def __get_search_pixel_list_point_range(p1, line, range_minmax=(-5, 5)):
        """get search interval from p1 to p2 on line is given as normal and distance in image coordinates"""
        direction, p0 = line

        direction /= np.linalg.norm(direction)

        # get line vars according to extremities of box
        s = np.dot(p1 - p0, direction)
        s1 = int(s + range_minmax[0])
        s2 = int(s + range_minmax[1])

        # define a search range
        search_range = range(s1, s2)
        if not search_range:
            search_range = range(s2, s1)

        # calc corresponding pixels
        out = [direction * i + p0 for i in search_range]

        return out

    @staticmethod
    def set_from_list(l, conversion_func=lambda p: p):
        """converts each element of list with conversion_func and generates a set from it"""
        return {conversion_func(x) for x in l}

        # @staticmethod
        # def __get_unqiue_maxima(maxima, val_spot, proximity_range=10):
        #     """go through array and delete all maxima that are in proximity range -> binning"""
        #     unique_maxima = []
        #     for val, cur_var in maxima:
        #         diff = np.array([np.linalg.norm(np.array(cur_var) - np.array(cur_var)) for _, cur_var in maxima])
        #
        #         # get points that are in the same range
        #         is_in_range = diff < proximity_range
        #         same_range = []
        #         for ir, vs in zip(is_in_range, val_spot):
        #             if ir:
        #                 same_range.append(vs)
        #
        #         # get maximum
        #         unique_maxima.append(max(same_range, key=lambda cur_var: cur_var[0]))
        #     # get rid of double values
        #     unique_maxima = list(set(unique_maxima))
        #
        #     # fill with new values that are not in proximity
        #     while len(unique_maxima)<len(maxima):
        #         unique_maxima
        #     return unique_maxima

    def __calc_border_projection(self, segment):
        """get projections of all points of interest assumes that scan is ordered"""
        assert (len(segment) == 5)
        pois = [segment[0][-1], segment[2][0], segment[2][-1], segment[4][0]]

        return [self.__transform_and_back_project(x) for x in pois]

    @staticmethod
    def __get_unqiue_maxima(val_spot, proximity_range=10, number_elements=4):
        """get binned maxima"""
        unique_maxima = []

        vs_sorted = sorted(val_spot, key=lambda cur_var: cur_var[0])

        for v, p in reversed(vs_sorted):
            # test if cur_var is in range
            diff = np.array([np.linalg.norm(np.array(p) - np.array(x)) for _, x in unique_maxima])
            is_in_range = diff < proximity_range
            # if non of them is add to maxima
            if not any(is_in_range):
                unique_maxima.append((v, p))
            # break if we have enough elements
            if len(unique_maxima) == number_elements:
                break
        if not len(unique_maxima) == number_elements:
            return []
        return unique_maxima

    @staticmethod
    def __calc_back_projection_errors(back_projections, border_points_image, line_projection=[]):
        """calculate back projection error"""
        # calc error for both and take minimum since it is possible that points are not in the same order
        # sort by v alternatively we can sort by line if the image is tilted too much
        ordered_points = sorted(border_points_image, key=lambda x: x[0])
        ordered_points_rev = [p for p in reversed(ordered_points)]

        res1 = np.array(back_projections) - np.array(ordered_points)
        res2 = np.array(back_projections) - np.array(ordered_points_rev)

        if not line_projection:
            bp_errors1 = np.linalg.norm(res1, axis=1)
            bp_errors2 = np.linalg.norm(res2, axis=1)
        else:
            direction, p0 = line_projection
            bp_errors1 = abs(np.dot(res1, direction))
            bp_errors2 = abs(np.dot(res2, direction))

        err_vec = min([bp_errors1, bp_errors2], key=lambda x: np.sum(x))

        return err_vec

    def __calc_gradient_image(self, img, image_depth=cv2.CV_8UC1):
        deriv_img = cv2.GaussianBlur(img, self.__gauss_kernel_size, self.__gauss_sigma)
        sx = cv2.Sobel(deriv_img, image_depth, 1, 0, ksize=self.__ksize_gradient)
        sy = cv2.Sobel(deriv_img, image_depth, 0, 1, ksize=self.__ksize_gradient)
        self.__deriv_img = cv2.sqrt(cv2.add(cv2.pow(sx, 2), cv2.pow(sy, 2)))
        # self.__deriv_img = cv2.Laplacian(deriv_img, image_depth, ksize=self.__ksize_gradient)
        return self.__deriv_img

    def __get_gradient_image(self):
        return self.__deriv_img

    def __calc_maxima_binning_range(self, p1, p2):
        """returns the range in which a second maxima is not allowed, shall be proportional to total search length"""
        length_line = np.linalg.norm(p2 - p1)
        return int(length_line * self.__maxima_binning_percentage)


def calc_mean_depth_segments(segment):
    depths = [np.linalg.norm(segment[0], axis=1), np.linalg.norm(segment[2], axis=1),
              np.linalg.norm(segment[4], axis=1)]
    mean = np.mean([d.mean() for d in depths])
    return mean


def plot_results(result_in):
    fig = plt.figure()
    ax = fig.gca()
    plt.xlabel("distance to object in meters")
    plt.ylabel("back-projection error in pixels")

    result = np.array([p[0] for p in result_in])
    y = np.concatenate([result[:, 0], result[:, 1], result[:, 2], result[:, 3]])
    depths = np.array([p[1] for p in result_in])
    depths = np.concatenate([depths, depths, depths, depths])
    plt.legend(loc="best")
    PlotHandle = plt.plot(depths, y, ".b")
    plot_mean_standard_dev_bins(depths, y, 10, 3)

    # plt.show()