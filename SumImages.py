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

import sys
import os
import getopt
import cv2
import glob
from matplotlib import pyplot as plt

def main(argv=None):
    #option handling
    if argv is None:
        argv=sys.argv
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
    except getopt.error, msg:
        print msg
        print "for help use --help"
        sys.exit(2)
    # process options
    for o, a in opts:
        if o in ("-h", "--help"):
            print __doc__
            sys.exit(0) 

    # interpret input data
    FolderName=argv[1]
    print "Image folder="+FolderName
    OutputFile=argv[2]
    print "OutputFile="+OutputFile
    if (not os.path.isdir(FolderName) ):
        print "not a directory"
        return

    AllFileNames=glob.glob(FolderName+"/*.png")
    #read first image to add rest up
    MaxImg=cv2.imread(AllFileNames[0],0).astype("double")
    del(AllFileNames[0])

    for ImName in AllFileNames:
        img=cv2.imread(ImName,0).astype("double")
        MaxImg+=img
    
    #Threshold image
    MaxVal=len(AllFileNames)*8.

    Max=0.
    Min=999999.
    for line in MaxImg:
        for pixel in line:
            if pixel>MaxVal:
                pixel=MaxVal

            if pixel>Max:
                Max=pixel
            if pixel<Min:
                Min=pixel

    MaxImg=255./(Max-Min)*(MaxImg-Min)
    cv2.imwrite(OutputFile,MaxImg.astype("uint8"))

    plt.imshow(MaxImg)
    plt.show()

if __name__ == '__main__':
    main()
                
