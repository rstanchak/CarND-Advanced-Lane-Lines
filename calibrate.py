import sys
import cv2
import numpy as np

if __name__=='__main__':

    valid_corners=[]
    image_points=[]
    image_points=[]
    object_points=[]
    objp=None
    image_size=None
    outfname = sys.argv[-1]

    for corner_fname in sys.argv[1:-1]:
        arr = np.load( corner_fname )
        if not arr['success']:
            continue

        valid_corners.append(corner_fname)
        image_points.append( arr['corners'] )

        if objp==None:
            NX,NY = arr['patternSize']
            objp=np.zeros((NX*NY, 3), np.float32)
            objp[:,0:2]=np.mgrid[0:NX,0:NY].T.reshape(NX*NY,2)
        image_size = tuple(arr['image_size'][0:2])
        object_points.append( objp )

    print(np.shape(object_points))
    print (np.shape(image_points))
    print("Calibrating with {} object_points, {} image_points on image of size {}".format(len(object_points), len(image_points), image_size))
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera( object_points, image_points, image_size, None, None)
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(cameraMatrix,distCoeffs,image_size,1,image_size)
    np.savez( outfname, image_size=image_size, retval=retval, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, newcameramtx=newcameramtx, roi=roi, rvecs=rvecs, tvecs=tvecs, corners=valid_corners )
