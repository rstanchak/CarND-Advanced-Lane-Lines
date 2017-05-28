import sys
import cv2
import numpy as np

if __name__ == '__main__':
    cornerfname = sys.argv[1]
    outfname = sys.argv[2]

    arr = np.load(cornerfname)
    infname = str(arr['infname'])
    patternSize=tuple(arr['patternSize'])
    corners = np.asarray(arr['corners'], dtype=np.float32)
    img = cv2.imread( infname )
    cv2.drawChessboardCorners(img, patternSize, np.squeeze(corners), arr['success'])
    cv2.imwrite(outfname, img)
