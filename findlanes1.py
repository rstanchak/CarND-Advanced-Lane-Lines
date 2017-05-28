from lanefinder import LaneFinder
import sys
import numpy as np
import cv2

if __name__ == "__main__":
    infname = sys.argv[1]
    calibfname = sys.argv[2]
    outfname = sys.argv[3]

    arr = np.load(calibfname)
    lanefinder = LaneFinder( image_size=tuple(arr['image_size']), cameraMatrix=arr['cameraMatrix'], distCoeffs=arr['distCoeffs'], newcameramtx=arr['newcameramtx'] )

    img = cv2.imread(infname)
    result = lanefinder.process_image(img)
    cv2.imwrite(outfname, result)
