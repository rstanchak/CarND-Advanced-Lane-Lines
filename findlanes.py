from lanefinder import LaneFinder
import sys
import numpy as np
import cv2
from moviepy.editor import VideoFileClip

def plot( img, y0, y, color):
    x = np.mgrid[0:len(y)]
    y1 = 10*y/max(y) + y0
    poly=np.vstack([x, y1]).T
    cv2.polylines( img, [np.asarray(poly, dtype=np.int32)], isClosed=False, color=color, thickness=2)

def preview( lanefinder ):
    def f( img ):
        cv2.imwrite('tmp.bmp', img)
        result = lanefinder.process_image( img )
        cv2.imshow("result", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        persp = cv2.cvtColor(lanefinder.viz['perspective'], cv2.COLOR_RGB2BGR)
        for idx, yslice in enumerate(lanefinder.viz['slice']):
            #plot( persp, (yslice[1]+yslice[0])/2, lanefinder.viz['hist'][idx], (0,0,255) )
            plot( persp, (yslice[1]+yslice[0])/2, lanefinder.viz['dhist'][idx], (0,255,0) )
            plot( persp, (yslice[1]+yslice[0])/2, lanefinder.viz['dilated'][idx], (128,0,255) )
        cv2.imshow("perspective", persp )
        #dI = lanefinder.viz['gradient']
        #minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dI)
        #cv2.imshow("gradient", (dI-minVal)/(maxVal-minVal))

        cv2.waitKey(10)
        return result
    return f

if __name__ == "__main__":
    infname = sys.argv[1]
    calibfname = sys.argv[2]
    outfname = sys.argv[3]

    arr = np.load(calibfname)
    lanefinder = LaneFinder( image_size=tuple(arr['image_size']), cameraMatrix=arr['cameraMatrix'], distCoeffs=arr['distCoeffs'], newcameramtx=arr['newcameramtx'] )
    cv2.namedWindow("result")
    cv2.namedWindow("perspective")

    input_clip = VideoFileClip( infname )
    output_clip = input_clip.fl_image( preview(lanefinder) )
    output_clip.write_videofile(outfname, audio=False)
