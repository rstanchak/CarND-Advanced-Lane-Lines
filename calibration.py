import cv2
import numpy as np

def findcorners(img, nx, ny):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    success, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if success:
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    return success, corners

import glob
camera_cal_filenames=glob.glob("camera_cal/*.jpg")

NX = 9
NY = 6
image_points=[]
objp=np.zeros((NX*NY, 3), np.float32)
objp[:,0:2]=np.mgrid[0:NX,0:NY].T.reshape(NX*NY,2)
image_size = None

import os
image_points=[]
valid_images=[]
for f in camera_cal_filenames:
    img = cv2.imread(f)
    assert(img is not None)
    if image_size is None:
        image_size = (img.shape[1], img.shape[0])
    success, corners = findcorners(img, NX, NY)
    if success:
        image_points.append( corners )
        valid_images.append(f)
    print("CORNERS\t{}\t{}".format(os.path.basename(f), success))
    break

object_points=[]

for i in range(len(valid_images)):
    object_points.append( objp )

print("Calibrating with {} object_points, {} image_points on image of size {}".format(len(object_points), len(image_points), image_size))
retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera( object_points, image_points, image_size, None, None)
print(cameraMatrix)
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(cameraMatrix,distCoeffs,image_size,1,image_size)
print(newcameramtx)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,10))

ndisp = 3
for idx, f in enumerate(valid_images):
    if idx>=ndisp:
        break
    img = cv2.imread(f)
    assert(img is not None)
    # undistort
    #mapx,mapy = cv2.initUndistortRectifyMap(cameraMatrix,distCoeffs,None,newcameramtx,image_size,5)
    #undist = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
    undist=cv2.undistort(img, cameraMatrix, distCoeffs)
    ax = fig.add_subplot(ndisp,3,idx*3+1)
    ax.imshow(img)
    ax.set_title("{} (Distorted)".format(os.path.basename(f)))
    ax = fig.add_subplot(ndisp,3,idx*3+2)
    ax.imshow(undist)
    ax.set_title("{} (Undistorted)".format(os.path.basename(f)))
    src_pts = image_points[idx][[0, (NX-1), (NX*NY-NX),(NX*NY-1)]]
    dst_pts = object_points[idx][[0, (NX-1), (NX*NY-NX), (NX*NY-1)]]
    dist=np.zeros(shape=(1,5),dtype=np.float64)
    imgpoints2, _ = cv2.projectPoints(dst_pts, 
                                      rvecs[idx], tvecs[idx], cameraMatrix, dist)
    dst_pts2=dst_pts[:,0:2]*100+110
    M = cv2.getPerspectiveTransform(imgpoints2, dst_pts2)
    persp=cv2.warpPerspective(undist,M,image_size)
    sq = ((dst_pts2[0][0], dst_pts2[0][1]), (dst_pts2[3][0], dst_pts2[3][1]))
    np.array(dst_pts2[0], np.int32)
    cv2.rectangle(persp,  sq[0], sq[1], (0,128,255), 5)
    ax = fig.add_subplot(ndisp,3,idx*3+3)
    ax.imshow(persp)
    ax.set_title("{} (Perspective)".format(os.path.basename(f)))


#fig.show()
#plt.pause(2**31-1)


