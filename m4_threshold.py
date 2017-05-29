from lanefinder import Thresholder
import sys
import numpy as np
import cv2

if __name__ == "__main__":
    infname = sys.argv[1]
    outfname = sys.argv[2]

    img = cv2.imread(infname)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    thresholder = Thresholder( rgb )
    result = thresholder.threshold()*255
    cv2.imwrite(outfname, result)
