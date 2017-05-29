#!/usr/bin/env python

import cv2
import glob
import sys
import os
import numpy as np

calib_fname, input_dir, output_dir = sys.argv[1:]

test_images = glob.glob(input_dir + "/*.jpg")

arr = np.load(calib_fname)
cameraMatrix=arr['cameraMatrix']
distCoeffs=arr['distCoeffs']
newcameramtx=arr['newcameramtx']


win='window'
#cv2.namedWindow(win)
#cv2.startWindowThread()
for idx, f in enumerate(test_images):
    img = cv2.imread(f)
    assert(img is not None)

    fname=os.path.basename(f)
    undist = cv2.undistort(img,cameraMatrix,distCoeffs,newcameramtx)
    #cv2.imshow(win, undist)
    cv2.imwrite(output_dir+"/"+fname, undist)
    #cv2.waitKey(-1)
