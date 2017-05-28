import sys
import cv2
import numpy as np
import pickle

class ChessboardCorners:

    def __init__(self, infname, nx, ny):
        self.infname = infname
        self.patternSize=(nx,ny)

    def imread(self):
        bgr = cv2.imread( self.infname )
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return bgr, gray

    def find(self):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        bgr, gray = self.imread()
        self.image_size = (bgr.shape[1], bgr.shape[0])
        success, corners = cv2.findChessboardCorners(gray, self.patternSize, None)
        self.success = success
        if success:
            cv2.cornerSubPix(gray, corners,(11,11),(-1,-1),criteria)
        self.corners = corners


if __name__ == '__main__':
    infname = sys.argv[1]
    NX = int(sys.argv[2])
    NY = int(sys.argv[3])
    outfname = sys.argv[4]

    corners = ChessboardCorners( infname, NX, NY )
    corners.find()
    np.savez(outfname, corners=corners.corners, patternSize=(NX,NY), infname=infname, success=corners.success, image_size=corners.image_size)
