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

import os
import cv2
import pdb
import numpy as np
import pickle
from numpy import array
from scipy import linalg, mat, dot
from matplotlib import pyplot as plt


def RotX(AngleRad):
    return array([[1., 0, 0], [0., np.cos(AngleRad), -np.sin(AngleRad)], [0, np.sin(AngleRad), np.cos(AngleRad)]])


def RotY(AngleRad):
    return array([[np.cos(AngleRad), 0, np.sin(AngleRad)], [0, 1., 0], [-np.sin(AngleRad), 0, np.cos(AngleRad)]])


def RotZ(AngleRad):
    return array([[np.cos(AngleRad), -np.sin(AngleRad), 0], [np.sin(AngleRad), np.cos(AngleRad), 0], [0, 0, 1.]])


def rotation_matrix(axis, theta):
    axis = -axis / (dot(axis, axis) ** 0.5)
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])


def convert_vector_rot_trans_to_homogenous(vec_rot_trans):
    assert (vec_rot_trans.shape == (6,))
    angle = linalg.norm(vec_rot_trans[:3])

    if not angle == 0.:
        rot = rotation_matrix(vec_rot_trans[:3] / angle, angle)
    else:
        rot = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    stack_mat = np.hstack((rot, array([vec_rot_trans[3:]]).T))

    return np.vstack((stack_mat, array([0, 0, 0, 1.])))


def project_laser_to_cam_and_get_z(Point3D, TransLaserCam, Intrin):
    """
    :LocalPoint3D: 3d point as np.array
    :TransLaserCam: 4x4 np.array homogenous transformation laser to image
    :Intrin: __intrinsics as 3x3 np.array
    :returns: Projpoint 2x1 np.array
    """
    assert (len(Point3D) > 2)
    LocalPoint3D = dot(TransLaserCam, np.hstack([Point3D[:3], 1.]))
    if not LocalPoint3D[2] == 0.:
        ProjPoint = 1 / float(LocalPoint3D[2]) * dot(Intrin, LocalPoint3D[:3])
    else:
        ProjPoint=[999999999999999999999., 999999999999999999999999.]
    return ProjPoint[:2], LocalPoint3D[2]


def project_laser_to_cam(Point3D, TransLaserCam, Intrin):
    """
    :LocalPoint3D: 3d point as np.array
    :TransLaserCam: 4x4 np.array homogenous transformation laser to image
    :Intrin: __intrinsics as 3x3 np.array
    :returns: Projpoint 2x1 np.array
    """
    P, z = project_laser_to_cam_and_get_z(Point3D, TransLaserCam, Intrin)

    return P


def threshold_residuals(Residuals, Quantile):
    Residuals = sorted(Residuals)
    assert (Quantile <= 1.0)
    Ind = int(np.floor((len(Residuals) - 1) * Quantile))

    Thres = Residuals[Ind]
    for i in range(Ind, len(Residuals) - 1):
        Residuals[i] = Thres


def loss_functino(Residuals):
    for Val in Residuals:
        Val = Tmp[0] ** 2. + Tmp[1] ** 2.
        if Val > 2. * 20. ** 2.: Val = 2. * 20. ** 2.


def calculate_back_projection_error(TransVecLaserCam, PointCorrespondences3D2D, Intrin):
    """calculate backprojectionerror  for least square problem

    :TransLaserCam: argument of minimization -> transformation from laserscanner coords to cam coords
    :Intrin: Intrinsics of camera
    :PointCorrespondences3D2D: list with corresponding points(arrays) -> 3D,2D 
    :returns: backprojection error (vector) 

    """
    Residuals = []

    TransLaserCam = convert_vector_rot_trans_to_homogenous(TransVecLaserCam)

    for Corr in PointCorrespondences3D2D:
        ProjPoint, z = project_laser_to_cam_and_get_z(Corr[0], TransLaserCam, Intrin)
        Tmp = (ProjPoint[:2] - Corr[1][:2])
        # Skalierung mit Tiefe
        # Tmp=Tmp*z**1.1
        Tmp = Tmp * z
        Residuals.extend(Tmp)
        # Residuals.append(Val)

    # loss_function(Residuals)

    return array(Residuals)


def read_file(FileName):
    with open(FileName, 'rb') as f:
        Reader = csv.reader(f, delimiter=',')
        # skip first line because is header
        next(Reader)

        AllScans = []
        for row in Reader:
            # del all whitespaces and convert to float
            row2 = [float(item.replace(' ', '')) for item in row if item.replace(' ', '')]

            if row2:
                CurIndex = int(row2[0])
                if len(AllScans) - 1 < CurIndex:
                    AllScans.append([])
                AllScans[CurIndex].append(row2[1:])
    f.close
    return AllScans


def read_pcl_data(FileName, YawAngle):
    """Docstring for read_pcl_data(FileName,YawAngle.
    :attributes: names of csv file to be read, yawangle in rad corresponding to each file
    :returns: Pointloud data in laserscanner coordinate system read from Filenames

    """
    # ATTENTION: SCANNER SCANS FROM RIGHT TO LEFT!
    # read
    AllScans = read_file(FileName)
    if not AllScans:
        print 'No information for this Layer available!'
        return 0
    # convert to xyz and store in deict by timestamp
    # Data=[[np.sin(Point[1])*Point[2],np.sin(YawAngle)*Point[2], np.cos(Point[1])*Point[2]] for Scan in AllScans for Point in Scan]
    Data = []
    for Scan in AllScans:
        CurData = []
        for Point in Scan:
            # Point[2]=Point[2]-0.065*0.065/4./Point[2]
            # Point[2]=Point[2]/(2.)**0.5
            if Point[2] < 30:
                CurData.append([np.sin(-Point[1]) * np.cos(YawAngle) * Point[2], -np.sin(YawAngle) * Point[2],
                                np.cos(-Point[1]) * np.cos(YawAngle) * Point[2]])
            else:
                CurData.append([-1, -1, -1])
        Data.append([x for x in reversed(CurData)])  # reversed because scanner scans from right to left

    ScanTs = [float(Scan[0][-1]) for Scan in AllScans]

    return Data, ScanTs


# def get_pcl_data_from_scans(Scans,YawAngle,Timestamp,TimestampTol):
#     """get data from read PCl
#     :YawAngle: YawAngle of scanner in rad corresponding to each read pcl
#     :TimeStamp: TimeStamp (unix time) for which data shall be read
#     :TimeStampTol: Tolerance in usec for Timestamps to match
#     :returns: Pointloud data in laserscanner coordinate system read from Filenames
# 
#     """
#     Data=[]
#     for Scan in Scans:
#         if Scan[0][-1]<Timestamp+TimestampTol and Scan[0][-1]>Timestamp-TimestampTol:
#             for Point in Scan:
#                 Data.append([np.sin(Point[1])*Point[2],-np.sin(YawAngle)*Point[2], np.cos(Point[1])*Point[2]])
#     return Data 

# def read_pcl_data(FileName,YawAngle,ScanNum=-1):
#     """Docstring for read_pcl_data(FileName,YawAngle.
#     :attributes: names of csv file to be read, yawangle in rad corresponding to each file
#     :returns: Pointloud data in laserscanner coordinate system read from Filenames
# 
#     """
#     #read
#     print 'reading ' + FileName
#     AllScans=read_file(FileName)
#     if not AllScans:
#         print 'No information for this Layer available!'
#         return 0
#     # convert to xyz
#     if ScanNum<0:
#         Data=[[np.sin(Point[1])*Point[2],np.sin(YawAngle)*Point[2], np.cos(Point[1])*Point[2]] for Scan in AllScans for Point in Scan]
#     else:
#         Data=[[np.sin(Point[1])*Point[2],np.sin(YawAngle)*Point[2], np.cos(Point[1])*Point[2]] for Scan in AllScans for Point in Scan if AllScan[0]==ScanNum]
#     return Data 
def get_measurement_from_line(Img, NormalVec, d, MaxDist, GrayThres, MaxMinV=(-1, -1)):
    """function to get a measurement next to a given line in max. MaxDist distance (orthogonal projection)
        :DirVec: Direction vector of line as numpy.array([u,v])
        :SupportPoint: SupportPoint of line as numpy array([u0,v0])
        :MaxDist: maximum distance to line
        :MaxMinV: minimum and maximum v compoenent specifying line segment; default=complete line
        :returns: Measurment as {"u":U,"v":V}
    """
    if MaxMinV[0] == -1 or MaxMinV[1] == -1:
        CurV = 1
        MaxV = Img.shape[0] - 1
    else:
        CurV = int(MaxMinV[1])
        MaxV = int(MaxMinV[0])

    Max = -1
    while CurV < MaxV:
        CurU = int((d - NormalVec[1] * CurV) / NormalVec[0])
        Min, CurMax, MinLoc, MaxLoc = cv2.minMaxLoc(
            Img[CurV - MaxDist / 2:CurV + MaxDist / 2, CurU - MaxDist / 2:CurU + MaxDist / 2])
        if CurMax > GrayThres and CurMax > Max:
            if Max == 255: print "Hit saturation"
            Max = CurMax
            Centroid = {"u": CurU - MaxDist / 2 + MaxLoc[0], "v": CurV - MaxDist / 2 + MaxLoc[1]}
        CurV = CurV + 1
    if Max > 0:
        return Centroid, Max
    else:
        return [], []


def get_measurement_from_three_lines(Img, NormalVecsLMR, dLMR, GrayThres, MaxMinV=(-1, -1)):
    """function to get a measurement next to a gven line in max. MaxDist distance (orthogonal projection)
        :DirVec: Direction vector of line as numpy.array([u,v])
        :SupportPoint: SupportPoint of line as numpy array([u0,v0])
        :MaxDist: maximum distance to line
        :MaxMinV: minimum and maximum v compoenent specifying line segment; default=complete line
        :returns: Measurment as {"u":U,"v":V}
    """
    if MaxMinV[0] == -1 or MaxMinV[1] == -1:
        CurV = 1
        MaxV = Img.shape[0] - 1
    else:
        CurV = int(MaxMinV[1])
        MaxV = int(MaxMinV[0])

    if CurV > MaxV:
        tmp = CurV
        CurV = MaxV
        MaxV = tmp

    Max = -1
    while CurV < MaxV:
        # get half distance to next line
        ULMR = [int((d - NormalVec[1] * CurV) / NormalVec[0]) for NormalVec, d in zip(NormalVecsLMR, dLMR)]
        ULMR[0] = (ULMR[1] + ULMR[0]) / 2
        ULMR[2] = (ULMR[1] + ULMR[2]) / 2
        # search left and right for maximum
        Min, CurMax, MinLoc, MaxLoc = cv2.minMaxLoc(Img[CurV, ULMR[0]:ULMR[2]])
        # CurMax=0
        # MaxLoc=[]
        # LocU=0
        # for px in Img[CurV,ULMR[0]:ULMR[2]]:
        #     if px>CurMax:
        #         CurMax=px
        #         MaxLoc=(0,LocU)
        #     elif px==CurMax and not px==0 and abs(LocU-ULMR[1])<abs(MaxLoc[1]-ULMR[1]):
        #         MaxLox=(0,LocU)
        #     LocU=LocU+1
        if CurMax > GrayThres and CurMax > Max:
            if Max == 255: print "Hit saturation"
            Max = CurMax
            Centroid = {"u": ULMR[0] + MaxLoc[1], "v": CurV}
        CurV = CurV + 1
    if Max > 0:
        return Centroid, Max
    else:
        return [], []


def locate_laser_points_image_epipolarlines_lr(OrigImg, LeftEpipolarLine, RightEpipolarLine, Show=False):
    """find mesurements from image in neighbourhood of epipolarlines
    :OrigImg (cv image):Input image (laser beams in dark room)
    :EpipolarLines:list of all epipolar lines ({"n":normal vector,"d":distance to image origin})
    :ValidityThres: Grey value from which measurement will be counted for as valid
    :Show (bool): if true show segmented measurements
    :returns: list of dicts({'u':? ,'v':?}) with centrois of image laser beam traces in OrigImg
    """
    Centroids = locate_laser_points_image(OrigImg, 45, Show)
    MaxDist = 10
    ValidityThres = 50

    if len(Centroids) < 45:
        # debug
        if Show:
            cImg = cv2.cvtColor(OrigImg, cv2.COLOR_GRAY2BGR)
            PlotImg = draw_lines([LeftEpipolarLine, RightEpipolarLine], cImg)
        # real stuff
        Centroids = [-1 for x in range(0, 45)]
        MaxLoc, MaxVal = get_measurement_from_line(OrigImg, LeftEpipolarLine["n"][0], LeftEpipolarLine["d"][0], MaxDist,
                                                   ValidityThres)
        if not not MaxLoc:
            Centroids[0] = MaxLoc
            if Show:
                cv2.circle(PlotImg, (Centroids[0]["u"], Centroids[0]["v"]), MaxDist / 2, (255, 0, 0), 1)
        MaxLoc, MaxVal = get_measurement_from_line(OrigImg, RightEpipolarLine["n"][0], RightEpipolarLine["d"][0],
                                                   MaxDist, ValidityThres)
        if not not MaxLoc:
            Centroids[-1] = MaxLoc
            if Show:
                cv2.circle(PlotImg, (Centroids[1]["u"], Centroids[1]["v"]), MaxDist / 2, (255, 0, 0), 1)
        # debug!
        if Show:
            plt.imshow(PlotImg)
            plt.show()
    return Centroids


def locate_laser_points_image_all_epipolarlines(OrigImg, EpipolarLines, ValidityThres = 10, Show=False):
    """find mesurements from image in neighborhood of epipolarlines
    :OrigImg (cv image):Input image (laser beams in dark room)
    :EpipolarLines:list of all epipolar lines ({"n":normal vector,"d":distance to image origin})
    :ValidityThres: Grey value from which measurement will be counted for as valid
    :Show (bool): if true show segmented measurements
    :returns: list of dicts({'u':? ,'v':?}) with centroids of image laser beam traces in OrigImg
    """

    if Show:
        cImg = cv2.cvtColor(OrigImg, cv2.COLOR_GRAY2BGR)
        PlotImg = draw_lines(EpipolarLines, cImg)

    Centroids = [-1 for x in range(0, len(EpipolarLines))]

    # assemble line in triples
    # copy last and first line to left and right
    FirstLine = EpipolarLines[0].copy()
    FirstLine["d"] = FirstLine["d"] - 10.
    LastLine = EpipolarLines[-1].copy()
    LastLine["d"] = LastLine["d"] + 10.
    EpipolarLineTriples = [(FirstLine, EpipolarLines[0], EpipolarLines[1])]
    for i in range(1, len(EpipolarLines) - 1):
        EpipolarLineTriples.append((EpipolarLines[i - 1], EpipolarLines[i], EpipolarLines[i + 1]))
    EpipolarLineTriples.append((EpipolarLines[-2], EpipolarLines[-1], LastLine))
    # assert(len(EpipolarLineTriples)==45)

    # get points
    for i, LineTriple in enumerate(EpipolarLineTriples):
        MaxLoc, MaxVal = get_measurement_from_three_lines(OrigImg, [Line["n"][0] for Line in LineTriple],
                                                          [Line["d"][0] for Line in LineTriple], ValidityThres,
                                                          LineTriple[1]["MaxMinV"])
        # MaxLoc, MaxVal = get_measurement_from_three_lines(OrigImg, [Line["n"][0] for Line in LineTriple], [Line["d"][0] for Line in LineTriple], ValidityThres)
        if MaxLoc:
            Centroids[i] = MaxLoc
            if Show:
                cv2.circle(PlotImg, (Centroids[i]["u"], Centroids[i]["v"]), 5, (255, 255, 0), 1)
    # debug!
    if Show:
        plt.imshow(PlotImg)
        plt.show()

    # assert(len(Centroids)==45)
    return Centroids


def locate_laser_points_image(OrigImg, NumPoints, Show=False):
    """Docstring for locate_laser_points_image
    :attributes: OrigImg (cv image):Input image (laser beams in dark room)
    :attributes: Show (bool): if true show contours
    :returns: list of dicts({'u':? ,'v':?}) with centrois of image laser beam traces in OrigImg

    """
    # detect edges
    edges = cv2.Canny(OrigImg, 30, 90, apertureSize=3)
    # close strucutr, so that finding contours is easier
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # # try combinations until found NumPoint contours in image
    # Quit=False
    # for BlurVal in [1,3,5,7,9]:
    #     BlurImg= cv2.medianBlur(OrigImg,BlurVal)
    #     # thresh=cv2.adaptiveThreshold(BlurImg,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    #     # ret, thresh = cv2.threshold(BlurImg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #     for ThreshVal in range(5,40,1):
    #         ret, thresh = cv2.threshold(BlurImg,ThreshVal,255,cv2.THRESH_BINARY)
    #         contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  

    #         if len(contours)==NumPoints:
    #             Quit=True
    #             break
    #     if Quit: break
    # if not Quit:
    #     for BlurVal in [1,3,5,7,9]:
    #         BlurImg= cv2.medianBlur(OrigImg,BlurVal)
    #         ret, thresh = cv2.threshold(BlurImg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #         contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #         if len(contours)==NumPoints:
    #             Quit=True
    #             break

    # if not Quit: 
    #     print "Error: wrong number of points ("+ str(len(contours)) +"/"+str(NumPoints)+")"
    #     return []

    print "Number of beam projections found: " + str(len(contours))

    if Show and NumPoints == len(contours):
        plt.imshow(closing, 'gray')
        cImg = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(cImg, contours, -1, (0, 255, 0), 1)
        plt.imshow(cImg)
        plt.show()

    Centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if not M['m00'] == 0.0:
            Centroids.append({'u': int(M['m10'] / M['m00']), 'v': int(M['m01'] / M['m00'])})

    # possible: outlier rejection by ransac
    # or all on roi found by hough line transfrom
    Centroids = sorted(Centroids, key=lambda c: c['u'])
    return Centroids


def truncate_stamps(CamTimestampsAndRest, PCLTimestampsAndRest):
    """truncate two containers, so that timestamps are in same interval, first entry has to be timestamp rest doesn' matter
        :CamTimestampsAndRest: list with timestamps on first entry,to be truncated
        :PCLTimestampsAndRest: list with timestamps on first entry, to be truncated
        :returns:nothing
    """
    # which onews first?
    if CamTimestampsAndRest[0][0] < PCLTimestampsAndRest[0][0]:
        First = CamTimestampsAndRest
        Second = PCLTimestampsAndRest
    else:
        First = PCLTimestampsAndRest
        Second = CamTimestampsAndRest
    # which one's last?
    if CamTimestampsAndRest[-1][0] > PCLTimestampsAndRest[-1][0]:
        Last = CamTimestampsAndRest
        BeforeLast = PCLTimestampsAndRest
    else:
        Last = PCLTimestampsAndRest
        BeforeLast = CamTimestampsAndRest

    # del elements of first until it is more or less equal to second
    while First[1][0] < Second[0][0]:
        del (First[0])
    # del elements of Last until it is more or less equal to beforelast
    while Last[-2][0] > BeforeLast[-1][0]:
        del (Last[-1])


def calc_corresponding_data(TimestampsAndRest1, TimestampsAndRest2, TimestampTolerance, IDString1="1", IDString2="2"):
    """calc which data corresponds to which by timestamps 
    :TimestampsAndRest1 (const):List containing Timestamps in usec, with other data. Timestamp has to be on first entry ( [x][0])
    :TimestampsAndRest2 (const):List containing Timestamps in usec, with other data. Timestamp has to be on first entry ( [x][0])
    :TimeStampTolerance (const):Tolerance for timestamps in usec, for timestamps to be accepted 
    :returns:List with corresponing data per line
    """
    if len(TimestampsAndRest1) < len(TimestampsAndRest2):
        One = TimestampsAndRest1
        OneString = IDString1
        Many = TimestampsAndRest2
        ManyString = IDString2
    else:
        Many = TimestampsAndRest1
        ManyString = IDString1
        One = TimestampsAndRest2
        OneString = IDString2

    Out = {OneString: [], ManyString: [], "TimeDiff": []}

    for el in One:
        Diff = [abs(x[0] - el[0]) for x in Many]
        MinVal = min(Diff)
        Ind = Diff.index(MinVal)
        if MinVal < TimestampTolerance:
            Out[OneString].append(el[1:])
            Out[ManyString].append(Many[Ind][1:])
            Out["TimeDiff"].append(MinVal)
    # i=0
    # for el in One:
    #     #find optimal match
    #     CurVal=abs(Many[i][0]-el[0])
    #     ValAfter=abs(Many[i+1][0]-el[0])
    #     while CurVal>ValAfter:
    #         CurVal=ValAfter
    #         i=i+1
    #         ValAfter=abs(Many[i][0]-el[0])
    #     # is valid?
    #     if min(ValAfter,CurVal)<TimestampTolerance:
    #         Out[OneString].append(el[1:])
    #         Out[ManyString].append(Many[i][1:])
    #         Out["TimeDiff"].append(min(CurVal,ValAfter))
    #     i=max(i-10,0)
    return Out


# 
#         AllMatchDiffs=np.array([[i,PCLStamp-Val] for i,PCLStamp in enumerate(PCLTimestamps) if (PCLStamp-TimeStampTol<Val and PCLStamp+TimeStampTol>Val)])
#         if AllMatchDiffs.shape==(0,): continue
#         ##############HIER WEITER MACHEN!!!!!!##################
#         MinIndex=AllMatchDiffs.argmin(1)
#         print MinIndex
# assert(not not FirstMatch)
# return 1
def get_linepoints_left_and_right_user_input(ImgName):
    # user input to select lines for enhanced measurement detection
    Img = cv2.imread(ImgName, 0)
    cOrigImg = cv2.cvtColor(Img, cv2.COLOR_GRAY2BGR)

    Lines = []
    for i in range(0, 2):
        LineAcceptedStr = "n"
        while not LineAcceptedStr == "y":
            cImg = cOrigImg.copy()
            TryAgain = []
            plt.imshow(cImg)
            LineUV = plt.ginput(2)
            # convert ot int
            LineUVShow = [(int(x[0]), int(x[1])) for x in LineUV]
            cv2.line(cImg, LineUVShow[0], LineUVShow[1], (255, 0, 0), 2)
            plt.imshow(cImg)
            print "if you want to accept the line please close the window otherwise click on the image"
            # click on image to do anew
            plt.draw()
            LineAcceptedStr = raw_input("Do you want to accept the line? y or n ")

        Lines.append(LineUV)
        cv2.line(cOrigImg, LineUVShow[0], LineUVShow[1], (255, 0, 0), 2)
    # sort left and right
    return sorted(Lines, key=lambda x: x[0]), cImg


def get_all_linepoints_user_input(ImgName, DirToSave, NumPointVecs=45):
    # user input to select lines for enhanced measurement detection
    Img = cv2.imread(ImgName, 0)
    if not os.path.isfile(ImgName):
        raise Exception("Accumulated image not defined or not on right path", "raised in get_all_linepoints_user_input")
    cOrigImg = cv2.cvtColor(Img, cv2.COLOR_GRAY2BGR)
    if not os.path.isfile(DirToSave + "/LinePoints.p"):
        print "pick lines"
        plt.ion()
        Lines = []
        for i in range(0, NumPointVecs / 3):
            while True:
                cImg = cOrigImg.copy()
                TryAgain = []
                plt.title("Please click 3 lines")
                plt.imshow(cImg)
                RawLineUV = plt.ginput(6)
                LineUV = [RawLineUV[i:i + 2] for i in range(0, len(RawLineUV), 2)]
                if len(LineUV) == 3:
                    # convert ot int
                    for CurLineUV in LineUV:
                        LineUVShow = [(int(x[0]), int(x[1])) for x in CurLineUV]
                        cv2.line(cImg, LineUVShow[0], LineUVShow[1], (255, 0, 0), 2)
                    plt.title("click on lower half of image to accept, on upper to neglect")
                    plt.imshow(cImg)
                    # click on image to do anew
                    UV = plt.ginput(1)
                    if not not UV and UV[0][1] > Img.shape[0] / 2:
                        break

            Lines.extend(LineUV)
            cv2.line(cOrigImg, LineUVShow[0], LineUVShow[1], (255, 0, 0), 2)
        pickle.dump(Lines, open(DirToSave + "/LinePoints.p", "wb"))
        print "saved lines as " + DirToSave + "/LinePoints.p"
    else:
        print "!!!!!!!!!!!!!! loading lines !!!!!!!!!!!!!!!!!"
        Lines = pickle.load(open(DirToSave + "/LinePoints.p", "rb"))
        cImg = cOrigImg.copy()

    for Line in Lines:
        CurLine = [(int(x[0]), int(x[1])) for x in Line]
        cv2.line(cOrigImg, CurLine[0], CurLine[1], (255, 0, 0), 2)
    plt.imshow(cOrigImg)
    plt.ioff()

    print "please close the window"
    plt.show()
    # sort left and right
    return sorted(Lines, key=lambda x: x[0][0]), cImg


def calc_line_from_points(PointsOnLine):
    du = PointsOnLine[1][0] - PointsOnLine[0][0]
    dv = PointsOnLine[1][1] - PointsOnLine[0][1]
    Vec = np.array([du, dv])
    Vec = Vec / linalg.norm(Vec)
    n0 = np.array([[dv, -du]])
    n0 = n0 / linalg.norm(n0)
    d = dot(n0, np.array([PointsOnLine[0][0], PointsOnLine[0][1]]))

    Line = {"n": n0, "d": d, "DirVec": Vec, "MaxMinV": [PointsOnLine[1][1], PointsOnLine[0][1]]}

    return Line


def get_epipolarlines_user_input(ImgPath, DirToSave, NumLines=45):
    """ get all epipolar lines by user input. left and right epipolarlines have to be specified, rest will be distributed automatically"""
    # LinePoints,Img=get_linepoints_left_and_right_user_input(ImgPath)
    LinePoints, Img = get_all_linepoints_user_input(ImgPath, DirToSave, NumLines)

    # DEBUG!!!!
    # Img=cv2.cvtColor(cv2.imread(ImgPath,0),cv2.COLOR_GRAY2BGR)
    # LinePoints=[[(95.883540372670836, 230.15838509316771), (84.962215320910957, 386.87939958592131)], [(474.85351966873708, 229.06625258799173), (457.92546583850941, 331.18064182194621)]]

    AllLines = []
    for LinePoint in LinePoints:
        AllLines.append(calc_line_from_points(LinePoint))

    # du=LinePoints[1][1][0]-LinePoints[1][0][0]
    # dv=LinePoints[1][1][1]-LinePoints[1][0][1]
    # Vec=np.array([du,dv])
    # Vec=Vec/linalg.norm(Vec)
    # n1=np.array([[dv,-du]])
    # n1=n1/linalg.norm(n1)
    # d=dot(n1,np.array([LinePoints[1][0][0],LinePoints[1][0][1]]))

    # Lines.append({"n":n1,"d":d ,"DirVec":Vec,"SupportPoint":np.array([d,0])})

    # m=np.vstack([Lines[0]["n"],Lines[1]["n"]])
    # # Intersection=dot(linalg.inv(m),np.array([Lines[0]["d"],Lines[1]["d"]]))

    # # TotalAngle=np.arccos(dot(n0,n1.transpose()))
    # dn=(n1-n0)/44.
    # AllLines=[]
    # PhiMax=44.*np.pi/180.
    # dPhi=2*np.pi/180.
    # fVirt=(Lines[1]["d"]-Lines[0]["d"])/2./np.tan(PhiMax)
    # d0=(Lines[1]["d"]+Lines[0]["d"])/2.
    # #Annahme: d = abstand zu 0 auf u-Achse
    # n2=n0.copy()
    # for phi in np.arange(-PhiMax,PhiMax+dPhi,dPhi):
    #     d=d0+np.tan(phi)*fVirt
    #     AllLines.append({"n":n2,"d":d,"DirVec":np.array([-n2[0][1],n2[0][0]]),"SupportPoint":np.array([d,0])})
    #     n2=n2+dn

    return AllLines, Img


def sparsify_meas_with_binning(ZippedPCLAndCamMeas, BinSize):
    return []


def draw_lines(AllLines, Img):
    ######draw lines#########
    SecPoints = []
    AxisU = {"n": np.array([[0, 1]]), "d": np.array([[0]])}
    AxisU2 = {"n": np.array([[0, 1]]), "d": np.array([[Img.shape[0]]])}
    for Line in AllLines:
        m = np.vstack([AxisU["n"], Line["n"]])
        Intersec = dot(linalg.inv(m), np.array([AxisU["d"][0], Line["d"][0]]))
        m = np.vstack([AxisU2["n"], Line["n"]])
        Intersec2 = dot(linalg.inv(m), np.array([AxisU2["d"][0], Line["d"][0]]))
        # SecPoints.append({"Top":Intersec,"Bottom":Intersec2})
        cv2.line(Img, (int(Intersec[0][0]), int(Intersec[1][0])), (int(Intersec2[0][0]), int(Intersec2[1][0])),
                 (0, 255, 0), 1)
    plt.imshow(Img)
    print "Showed lines"
    plt.show()
    return Img
