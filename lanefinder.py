import cv2
import numpy as np

LANE_WIDTH_METERS = 3.7
LANE_WIDTH_FEET = 12

def rgb2s(img):
    """Convert the RGB image to HSL and return the 'saturation' 
    channel """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    return np.asarray(hls[:,:,2], dtype=np.uint8)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def gradient(im):
    """compute the x,y components of the image gradient"""
    blurimage2=gaussian_blur(im,3)
    Ix = cv2.Sobel(blurimage2, cv2.CV_32F,1,0)
    Iy = cv2.Sobel(blurimage2, cv2.CV_32F,0,1)
    return Ix,Iy

def roipoly3():
    return np.array([(534,490),(751,490),(1120,720),(199,720)], dtype=np.float32)

def destpoly(imsize):
    w = imsize[0]
    h = imsize[1]
    x0 = w/4
    x1 = 3*w/4
    return np.array([(x0,h/4),(x1,h/4),(x1,h),(x0,h)], dtype=np.float32)

class PerspectiveTransform:
    def __init__(self, src_pts, dst_pts, cameraMatrix,distCoeffs,newcameramtx, image_size):
        self.M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.invM = cv2.getPerspectiveTransform(dst_pts, src_pts)
        self.calib = cameraMatrix, distCoeffs, newcameramtx
        self.image_size = image_size
    def __call__(self, img, undistort=True):
        if undistort:
            undist = cv2.undistort(img, self.calib[0], self.calib[1], self.calib[2])
        else:
            undist = img
        persp = cv2.warpPerspective(undist, self.M, self.image_size, borderMode=cv2.BORDER_REPLICATE)
        if undistort:
            return undist, persp
        else:
            return persp
        
    def backproject(self, overlay, dst_size):
        return cv2.warpPerspective(overlay, self.invM, dst_size)

class LaneFinder:
    def __init__(self, image_size, cameraMatrix, distCoeffs, newcameramtx):
        self.image_size = image_size
        self.perspective_transform = PerspectiveTransform(
                roipoly3(),
                destpoly(image_size),
                cameraMatrix,
                distCoeffs,
                newcameramtx,
                image_size)
        self.ctr = image_size[0]/2
        w = image_size[0]
        self.lane_center = [0.,0.,w/2.]
        self.lane_width = w/2
        self.viz={}
        
    def drawlanes( self, rgb ):
        mask = np.zeros( (self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        #C = int(self.lane_center[2])
        #C_L=int(self.lane_center[2]-self.lane_width/2)
        #C_R=int(self.lane_center[2]+self.lane_width/2)
        #cv2.rectangle(mask, (C_L,0), (C_R,h), (0,128,0), -1)
        #cv2.line(mask, (C_L,0), (C_L, h), (255, 0, 0), 10)
        #cv2.line(mask, (C_R,0), (C_R, h), (0, 0, 255), 10)
        #cv2.line(mask, (C,0), (C, h), (255,255,255),15)
        self.drawcurve(mask, self.lane_center, (0,128,0), int(self.lane_width))
        self.drawcurve(mask, self.lane_center, (255,255,255), 3)


        bp=self.perspective_transform.backproject(mask, (rgb.shape[1], rgb.shape[0]))
        return cv2.addWeighted(bp, 0.8, rgb, 1.0, 0.0)
    def drawtextline(self, rgb, text, lineno):
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 3.0
        thickness = 3
        color = (255,255,255)
        sz = cv2.getTextSize(text, font, font_scale, thickness)
        pt = (20, (sz[1]+20)*lineno)
        cv2.putText(rgb, text, pt, font, font_scale, color, thickness)

    def drawtext( self, rgb ):
        h = rgb.shape[0]
        x = np.dot(self.lane_center, [h*h, h, 1])
        ctr = (x - rgb.shape[1]/2)/self.lane_width*LANE_WIDTH_METERS
        if ctr==0:
            direction = 'centered'
        elif ctr < 0:
            direction = '{:.2f}m right of center'.format(-ctr)
        else:
            direction = '{:.2f}m left of center'.format(ctr)
        A,B,C = self.lane_center
        radius = ((1 + (2*A*h+B)**2)**(3/2))/abs(2*A) * (LANE_WIDTH_METERS/self.lane_width)
        self.drawtextline(rgb, "Radius of curvature: {}".format(int(radius)), lineno=1)
        self.drawtextline(rgb, "Vehicle is {}".format(direction), lineno=2)
        return rgb
    
    def process_slice(self, Ix, y0, y1, win=100):
        hist = np.sum(Ix[y0:y1,:], axis=0)
        dhist = np.gradient(hist)
        dilated = np.reshape(cv2.dilate(dhist,np.ones((win,1))),dhist.shape)
        dilated = np.where(dhist==dilated, dilated, 0)
        candidates = np.argwhere(dilated>0)
        # center
        y = (y1-y0)/2
        C = np.dot( self.lane_center, [y*y, y, 1])
        # left and right
        C_L=int(C-self.lane_width/2)
        C_R=int(C+self.lane_width/2)

        self.viz['slice'].append((y0,y1))
        self.viz['hist'].append(hist)
        self.viz['dhist'].append(dhist)
        self.viz['dilated'].append(dilated)
        assert(C!=0)

        # find best match in candidates
        f_L=dilated[candidates]/(1+((candidates-C_L)**2))
        self.viz['f_L'].append(f_L)
        C_L_new = candidates[np.argmax(f_L, axis=0)][0,0]
    
        f_R=dilated[candidates]/(1+((candidates-C_R)**2))
        self.viz['f_R'].append(f_R)
        C_R_new = candidates[np.argmax(f_R, axis=0)][0,0]
        return C_L_new, C_R_new
  
    def drawcurve(self, rgb, curve, color, thickness):
        y = np.arange(rgb.shape[0])
        x = curve[0]*y**2 + curve[1]*y + curve[2]
        cv2.polylines(rgb, [np.array([x,y], dtype=np.int32).T], isClosed=False, color=color, thickness=thickness)

    def process_image(self, rgb):
        
        undist, persp = self.perspective_transform(rgb)
        self.viz['undistorted'] = undist
        self.viz['perspective'] = persp
        self.viz['hist'] = []
        self.viz['dhist'] = []
        self.viz['dilated'] = []
        self.viz['f_L'] = []
        self.viz['f_R'] = []
        self.viz['slice'] = []
        s = np.asarray(rgb2s(persp), dtype=np.float32)
        Ix,Iy = gradient(s)
        dI = Ix**2 + Iy**2
        self.viz['gradient'] = dI
        h=rgb.shape[0]
        w=rgb.shape[1]
        step = int(h/9)
        left_pts=[]
        right_pts=[]
        y_array=[]
        for i in range(int(h/step)):
            y0 = i*step
            y1 = y0+step
            y = (y1+y0)/2
            C_L_new, C_R_new = self.process_slice(dI, i*step, (i+1)*step)
            cv2.line(persp, (int(C_L_new), int(y)), (int(C_R_new), int(y)), (255,255,255), 5)
            y_array.append(y)
            left_pts.append(C_L_new)
            right_pts.append(C_R_new)

        left_poly = np.polyfit(y_array, left_pts,2)
        self.drawcurve(persp, left_poly, (255,255,0), 5)
        right_poly = np.polyfit(y_array, right_pts,2)
        self.drawcurve(persp, right_poly, (255,255,0), 5)
        ctr_pts = np.hstack(
            [np.asarray(right_pts)-self.lane_width/2,
            np.asarray(left_pts)+self.lane_width/2])
        self.lane_center = np.polyfit(y_array+y_array, ctr_pts,2)
        #self.drawcurve(persp, ctr_poly, (255,0,255), 5)

        #self.lane_width = np.average( np.subtract(right_pts,left_pts))
        #self.lane_center[2] = C_L_new + self.lane_width/2
        processed_image = self.drawlanes(rgb)
        return self.drawtext(processed_image)
