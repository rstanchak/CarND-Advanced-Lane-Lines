from lanefinder import PerspectiveTransform, roipoly3, destpoly
import sys
import numpy as np
import cv2

if __name__ == "__main__":
    infname = sys.argv[1]
    calibfname = sys.argv[2]
    outfname1 = sys.argv[3]
    outfname2 = sys.argv[4]

    img=cv2.imread(infname)
    arr=np.load(calibfname)
    src_pts=roipoly3()
    dst_pts=destpoly(tuple(arr['image_size']))
    transform=PerspectiveTransform(
            src_pts,
            dst_pts,
            cameraMatrix=arr['cameraMatrix'],
            distCoeffs=arr['distCoeffs'],
            newcameramtx=arr['newcameramtx'],
            image_size=tuple(arr['image_size']))
    persp = transform(img, False)
    cv2.polylines(img,[np.asarray(src_pts,dtype=np.int32)],isClosed=True,color=(255,0,64),thickness=5)
    cv2.polylines(persp,[np.asarray(dst_pts,dtype=np.int32)],isClosed=True,color=(255,0,64),thickness=5)

    cv2.imwrite(outfname1,img)
    cv2.imwrite(outfname2,persp)

