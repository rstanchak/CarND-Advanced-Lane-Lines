# File: lanefinder
# Author: Roman Stanchak
# Description: methods for detecting the active lane in first-person road images

import cv2
import numpy as np

# constants based on U.S. regulations
LANE_WIDTH_FEET = 12.
LANE_LENGTH_FEET = 10.

# empirical measurements using a perspective_transform from
# roipoly3 to destpoly below
LANE_WIDTH_PIXELS = 657
LANE_LENGTH_PIXELS = 180
X_FT_PER_PX = LANE_WIDTH_FEET/LANE_WIDTH_PIXELS
Y_FT_PER_PX = LANE_LENGTH_FEET/LANE_LENGTH_PIXELS


def roipoly3():
    """hard-coded trapezoidal region of interest for the test videos"""
    return np.array([
        (534, 490),
        (751, 490),
        (1120, 720),
        (199, 720)], dtype=np.float32)


def destpoly(imsize):
    """
    destination polygon for perspective transformation to given image size
    """
    w = imsize[0]
    h = imsize[1]
    x0 = w/4
    x1 = 3*w/4
    return np.array([(x0, h/4), (x1, h/4), (x1, h), (x0, h)], dtype=np.float32)


class Thresholder:
    """
    container for intermediate images used for processing road image to produce
    lane pixels
    """
    def __init__(self, rgb):
        """create Thresholder with 3-channel image in RGB format"""
        self.hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
        self.hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        self.s_channel = self.hls[:, :, 2]
        self.v_channel = self.hsv[:, :, 2]
        self.s_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        self.v_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        self.s_x = cv2.Sobel(self.s_channel, cv2.CV_32F, 1, 0)
        self.s_y = cv2.Sobel(self.s_channel, cv2.CV_32F, 0, 1)
        self.sx_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        self.sy_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        self.mask = np.zeros_like(self.s_channel)

    def threshold(self, sx_thresh=192, sy_thresh=192,
                  v_thresh=192, s_thresh=128):
        """apply thresholds as provided in args and return combined result"""
        self.sx_mask = np.where((np.abs(self.s_x) >= sx_thresh),
                                np.uint8(1), np.uint8(0))
        self.sy_mask = np.where((np.abs(self.s_y) >= sy_thresh),
                                np.uint8(1), np.uint8(0))
        self.s_mask = np.where((self.s_channel >= s_thresh),
                               np.uint8(1), np.uint8(0))
        self.v_mask = np.where((self.v_channel >= v_thresh),
                               np.uint8(1), np.uint8(0))
        self.mask = np.where(((self.sx_mask & self.sy_mask) |
                              (self.s_mask & self.v_mask)),
                             np.uint8(1), np.uint8(0))
        return self.mask


class PerspectiveTransform:
    """container for computing perspective transformation and inverse"""
    def __init__(self, src_pts, dst_pts, cameraMatrix, distCoeffs,
                 newcameramtx, image_size):
        """
        compute perspective transform for quadrilateral defined by src_pts
        to that defined by dst_pts with to a destination image of size
        'image_size'.
        cameraMatrix, distCoeffs, newcameramtx are used to optionally
        undistort the input image
        """
        self.M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.invM = cv2.getPerspectiveTransform(dst_pts, src_pts)
        self.calib = cameraMatrix, distCoeffs, newcameramtx
        self.image_size = image_size

    def __call__(self, img, undistort=True):
        """
        apply the perspective transformation to the input image, optionally
        undistorting it first
        """
        if undistort:
            undist = cv2.undistort(img, self.calib[0], self.calib[1],
                                   self.calib[2])
        else:
            undist = img
        persp = cv2.warpPerspective(undist, self.M, self.image_size,
                                    borderMode=cv2.BORDER_REPLICATE)
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
        self.lane_center = [0., 0., w/2.]
        self.lane_width = w/2
        self.viz = {}

    def drawlanes(self, rgb):
        mask = np.zeros((self.image_size[1], self.image_size[0], 3),
                        dtype=np.uint8)
        self.drawcurve(mask, self.lane_center, (0, 128, 0),
                       int(self.lane_width))
        self.drawcurve(mask, self.lane_center, (255, 255, 255), 3)
        bp = self.perspective_transform.backproject(
                mask,
                (rgb.shape[1], rgb.shape[0]))
        return cv2.addWeighted(bp, 0.8, rgb, 1.0, 0.0)

    def drawtextline(self, rgb, text, lineno):
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 3.0
        thickness = 3
        color = (255, 255, 255)
        sz = cv2.getTextSize(text, font, font_scale, thickness)
        pt = (20, (sz[1]+20)*lineno)
        cv2.putText(rgb, text, pt, font, font_scale, color, thickness)

    def drawtext(self, rgb):
        h = rgb.shape[0]
        x = np.dot(self.lane_center, [h*h, h, 1])
        # assume image center is center of car
        ctr = (x - rgb.shape[1]/2)*X_FT_PER_PX
        if ctr == 0:
            direction = 'centered'
        elif ctr < 0:
            direction = '{:.2f}ft right of center'.format(-ctr)
        else:
            direction = '{:.2f}ft left of center'.format(ctr)
        A, B, C = self.lane_center
        # the polynomial coefficients A,B,C are in warped image coordinates
        # x = Ay^2 + By + C
        # need corresponding A_w, B_w, C_w in world coordinates x_w, y_w
        # where x_w = A_w*y_w^2 + B_w*y_w + C

        # transform to world coordinates using scaling factors sx and sy
        # x_w = sx * x
        # y_w = sy * y
        sx = X_FT_PER_PX
        sy = Y_FT_PER_PX

        # substituting x_w/sx for x and y_w/sy for y in the polynomial gives
        # x_w/sx = A*(y_w/sy)^2 + B*(y_w/sy) + C
        # rearranging terms gives
        # x_w = (sx*A/sy^2)*y_w^2 + (sx*B/sy)*y_w + sx*C
        # so the corresponding coefficients in world coordinates are...
        A_w = sx*A/(sy*sy)
        B_w = sx*B/sy

        if A_w != 0:
            radius = ((1 + (2*A_w*h*Y_FT_PER_PX+B_w)**2)**(3/2))/abs(2*A_w)
            self.drawtextline(rgb,
                              "Radius of curvature: {}ft".format(int(radius)),
                              lineno=1)
        else:
            self.drawtextline(rgb,
                              "Radius of curvature: straight road",
                              lineno=1)
        self.drawtextline(rgb, "Vehicle is {}".format(direction), lineno=2)
        return rgb

    def drawcurve(self, rgb, curve, color, thickness):
        y = np.arange(rgb.shape[0])
        x = curve[0]*y**2 + curve[1]*y + curve[2]
        cv2.polylines(rgb, [np.array([x, y], dtype=np.int32).T],
                      isClosed=False, color=color, thickness=thickness)

    def lanepoints(self, thresh, x0, x1, ksize=49):
        blurimage = cv2.GaussianBlur(thresh[:, x0:x1].astype(np.float32),
                                     (ksize, 3), 0)
        self.viz['blurimage'] = blurimage
        h = thresh.shape[0]
        X = x0 + np.argmax(blurimage, axis=1)
        Y = np.argwhere(blurimage[(np.arange(h), X-x0)] > 0)
        if len(Y) == 0:
            return None
        X = X[Y]
        return X, Y

    def process_image(self, rgb):
        win = 100
        undist, persp = self.perspective_transform(rgb)
        self.viz['undistorted'] = undist
        self.viz['perspective'] = persp
        self.viz['hist'] = []
        self.viz['dhist'] = []
        self.viz['dilated'] = []
        self.viz['f_L'] = []
        self.viz['f_R'] = []
        self.viz['slice'] = []
        self.thresholder = Thresholder(persp)
        threshold = self.thresholder.threshold()
        self.viz['threshold'] = threshold

        h = persp.shape[0]
        w = persp.shape[1]

        XY_l = self.lanepoints(threshold, int(w/2-self.lane_width/2-win),
                               int(w/2))
        XY_r = self.lanepoints(threshold, int(w/2),
                               int(w/2+self.lane_width/2+win))

        # seed lane fitter with lane lines from previous estimate
        Y = np.arange(0, h, 16).reshape(-1, 1)
        # generate Y matrix
        Y_prev = np.vstack([
            np.hstack((Y**2, Y, np.ones_like(Y), -0.5*np.ones_like(Y))),
            np.hstack((Y**2, Y, np.ones_like(Y), 0.5*np.ones_like(Y)))])
        # compute x = [ A*y^2 + B*y + C + alpha*lane_width ]
        # where alpha = -0.5 for left lane and 0.5 for right lane
        x_prev = np.matmul(Y_prev, np.hstack([self.lane_center[:],
                                             self.lane_width]))
        Y_array = [Y_prev]
        x_array = [x_prev.reshape(-1, 1)]

        # add new measurements to x, Y points
        if XY_l is not None:
            x_array.append(XY_l[0].reshape(-1, 1))
            Y_array.append(np.hstack((XY_l[1]**2,
                                      XY_l[1],
                                      np.ones_like(XY_l[1]),
                                      -0.5*np.ones_like(XY_l[1]))))

        if XY_r is not None:
            x_array.append(XY_r[0].reshape(-1, 1))
            Y_array.append(np.hstack((XY_r[1]**2,
                                      XY_r[1],
                                      np.ones_like(XY_r[1]),
                                      0.5*np.ones_like(XY_r[1]))))

        x = np.vstack(x_array)
        Y = np.vstack(Y_array)

        # solve for betaHat = [ A, B, C, w ]
        # where Ay^2 + By + C is the lane center
        # and w is the lane width
        betaHat = np.linalg.solve(Y.T.dot(Y), Y.T.dot(x))
        self.lane_center = betaHat[:3].ravel()
        self.lane_width = betaHat[3, 0]

        processed_image = self.drawlanes(rgb)
        return self.drawtext(processed_image)
