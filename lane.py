"""
Classes to do lane detection.
"""
import cv2
import numpy as np

import draw
from utils import eval_poly

# Assume image size is 1280x720
H, W = 720, 1280

# Define unwarp region source.
SRC_REGION = np.float32(
    [(570, 465),  # top-left
     (200, 720),  # bottom-left
     (710, 465),  # top-right
     (1080, 720)])  # bottom-right

# Define unwarp region destination
OFFSET = 450
DST_REGION = np.float32([
    (OFFSET, 0),
    (OFFSET, H),
    (W - OFFSET, 0),
    (W - OFFSET, H)])

# Define conversions in x and y from pixels space to world space (maters).
# Meters per pixel in y dimension, dashed lane line is 10 ft = 3.048 meters.
# 120 pixels are determined from the dashline pixel length in images/dashline.jpg.
MY = 3.048 / 120
# Meters per pixel in x dimension, lane width is 12 ft = 3.658 meters
MX = 3.658 / (W - 2 * OFFSET)


class Line(object):
    """
    A class to receive the characteristics of each line detection.
    """
    def __init__(self, n=3, h=H, mx=MX, my=MY):
        """
        Args:
            :param n: number of the most recent fits.
            :param h: image height.
            :param mx: meters per x pixel.
            :param my: meters per y pixel.
        """
        # Number of previous valid fits stored in queue.
        self.n = n
        self.h = h
        self.mx = mx
        self.my = my

        # Was the line detected in the last iteration?
        self.detected = False
        # Polynomial coefficients for recent fits. The last one is the most current fit.
        self.recent_fits = [] #[np.array([False])]
        # Difference in fit coefficients between last and new fits.
        self.diffs = np.array([0, 0, 0], dtype='float')

        # Polynomial coefficients averaged over the last n iterations.
        self.best_fit = None
        # Radius of curvature of the line in some units
        self.best_curverad = None

    def add_fit(self, fit):
        """
        Adds a polyfit coefficients and update itself.
        :param fit: poly fit parameters.
        :return: lane line indices.
        """
        self.detected = False

        # Do sanity check on |fit|, by comparing with |best_fit|
        if fit is not None:
            if self.best_fit is not None:
                # Sanity check the diffence between fit and best_fit is not too large.
                self.diffs = abs(fit - self.best_fit)
                if self.diffs[0] < 0.01 and self.diffs[1] < 1.0 and self.diffs[2] < 100.0:
                    self.detected = True
            else:
                self.detected = True

        if self.detected:
            # Append |fit| to queue and pop expired ones.
            self.recent_fits.append(fit)
            self.recent_fits = self.recent_fits[-self.n:]
        else:
            # Always pop the oldest fit, even the queue is not full.
            self.recent_fits = self.recent_fits[1:]

        if len(self.recent_fits) > 0:
            self.best_fit = np.average(self.recent_fits, axis=0)

        if self.best_fit is not None:
            self.best_curverad = self.calc_curverad(self.best_fit)

    def calc_curverad(self, fit):
        """
        Calculates curvature radius from polynomial fit.
        :param fit: poly fit parameters.
        :return: curvature radius.
        """
        if fit is None:
            return 0.0
        ploty = np.linspace(0, self.h - 1, self.h)
        ymax = np.max(ploty) * self.my
        mfit = [(MX / (MY ** 2)) * fit[0], (self.mx / self.my) * fit[1], fit[2]]
        return ((1 + (2 * mfit[0] * ymax + mfit[1]) ** 2) ** 1.5) / np.absolute(2 * mfit[0])


def lab_b_thresh(img, thresh):
    """
    Binary thresholds on Lab b-channel.
    :param img: input image.
    :param thresh: (min, max) tuple
    :return: binary image.
    """
    lab_b = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:, :, 2]
    # If there are yellow (b=175) pixels, normalize the highest value to 255
    # for preciser threshold.
    if np.max(lab_b) > 175:
        lab_b = lab_b * (255 / np.max(lab_b))

    min_th, max_th = thresh
    binary_output = np.zeros_like(lab_b)
    binary_output[(lab_b > min_th) & (lab_b <= max_th)] = 1
    return binary_output


def hls_l_thresh(img, thresh):
    """
    Binary thresholds on HLS L-channel.
    :param img: input image.
    :param thresh: (min, max) tuple.
    :return: binary image.
    """
    hls_l = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:, :, 1]
    binary_output = np.zeros_like(hls_l)
    min_th, max_th = thresh
    binary_output[(hls_l > min_th) & (hls_l <= max_th)] = 1
    return binary_output


def curverad_center_dist_texts(left_curverad, right_curverad, center_dist):
    """
    Generates texts for curvature radius and center distance.
    :param left_curverad: left curvature radius.
    :param right_curverad: right curvature radius.
    :param center_dist: center distance.
    :return: list of strings.
    """
    curverad = (left_curverad + right_curverad) / 2
    curverad_text = 'Curvature radius: {:05.3f}m'.format(curverad)
    direction = ''
    if center_dist > 0.0:
        direction = 'right'
    elif center_dist < 0.0:
        direction = 'left'
    center_dist_text = 'Center distance: {} {:05.3f}m'.format(direction, abs(center_dist))
    return [curverad_text, center_dist_text]


class LaneDetector(object):
    """
    Class to detect left and right lane lines.
    """
    def __init__(self, width=W, height=H, mx=MX, my=MY,
                 offset=OFFSET, src=SRC_REGION, dst=DST_REGION,
                 hls_l_threshold=(207, 255), lab_b_threshold=(190, 255),
                 nwindows=10, margin=80, minpix=40):
        """
        :param width: image width.
        :param height: image height.
        :param src: source region for perspective transform.
        :param dst: destination region for perspective transform.
        :param hls_l_threshold: HLS L-channel threshold.
        :param lab_b_threshold: Lab b-channel threshold.
        :param nwindows: number of sliding windows in search_lanes().
        :param margin: width of the windows +/- margin.
        :param minpix: minimum number of pixels found to recenter window.
        """
        self.w = width
        self.h = height
        self.mx = mx
        self.my = my
        self.offset = offset
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        self.hls_l_threshold = hls_l_threshold
        self.lab_b_threshold = lab_b_threshold
        self.nwindows = nwindows
        self.margin = margin
        self.minpix = minpix

        self.left_line = Line()
        self.right_line = Line()

    def unwarp(self, img):
        """
        Transforms image from vehicle-view to bird-view.
        """
        return cv2.warpPerspective(img, self.M, (self.w, self.h))

    def warp(self, img):
        """
        Transforms image from bird-view to vehicle-view.
        :param img:
        :return:
        """
        return cv2.warpPerspective(img, self.Minv, (self.w, self.h))

    def binarize(self, img):
        """
        Generates a binary image by doing colorspace thresholding.
        """
        hls_l_bin = hls_l_thresh(img, self.hls_l_threshold)
        lab_b_bin = lab_b_thresh(img, self.lab_b_threshold)
        binary_image = np.zeros_like(hls_l_bin)
        binary_image[(hls_l_bin == 1) | (lab_b_bin == 1)] = 1
        return binary_image

    def check_lane_separation(self, left_fit, right_fit, max_diff=150):
        """
        Sanity check left lane and right lane are separted in appropriate distance.
        """
        if left_fit is None or right_fit is None:
            return None, None
        left_bottom_x = eval_poly(left_fit, self.h)
        right_bottom_x = eval_poly(right_fit, self.h)
        bottom_dist = abs(left_bottom_x - right_bottom_x)
        # Proper distance is defined by W and OFFSET in perspective transform.
        if abs(bottom_dist - (self.w - 2 * self.offset)) > max_diff:
            return None, None
        return left_fit, right_fit

    def calc_center_dist(self, left_fit, right_fit):
        """
        Calculates car distance from lane center
        :param left_fit: left lane line poly fit parameters.
        :param right_fit: right lane line poly fit parameters.
        :return: center distance in meters.
        """
        if left_fit is None or right_fit is None:
            return 0.0
        car_pos = self.w // 2
        left_bottom_x = eval_poly(left_fit, self.h)
        right_bottom_x = eval_poly(right_fit, self.h)
        lane_center = (left_bottom_x + right_bottom_x) / 2
        return (car_pos - lane_center) * self.mx

    def detect(self, img, vis=True):
        """
        Detects lanes.
        :param img: input image.
        :param vis: returns binary image an lane image additionally.
        :return:
            left line fit: left lane line polyfit.
            right line fit: right lane line polyfit.
            texts: curvature radius and center distance info.
            binary image: binary image for color-thresholding reuslt.
            lane image: image for lane search result.
        """
        unwarp_img = self.unwarp(img)
        binary_img = self.binarize(unwarp_img)

        left_fit, right_fit = None, None
        rectangles, margin = None, None
        if self.left_line.detected and self.right_line.detected:
            left_fit, right_fit, left_lane_inds, right_lane_inds, margin = self.search_lanes_with_prev(
                binary_img, self.left_line.best_fit, self.right_line.best_fit)

        if left_fit is None or right_fit is None:
            margin = None
            left_fit, right_fit, left_lane_inds, right_lane_inds, rectangles = self.search_lanes(binary_img)

        # Sanity check
        left_fit, right_fit = self.check_lane_separation(left_fit, right_fit)

        # Update lines
        self.left_line.add_fit(left_fit)
        self.right_line.add_fit(right_fit)

        texts = ["Lane detection error"]

        if self.left_line.best_fit is not None and self.right_line.best_fit is not None:
            center_dist = self.calc_center_dist(self.left_line.best_fit, self.right_line.best_fit)
            texts = curverad_center_dist_texts(self.left_line.best_curverad, self.right_line.best_curverad,
                                               center_dist)
        if vis:
            lane_img = draw.draw_lane_fits(
                binary_img, self.left_line.best_fit, self.right_line.best_fit,
                left_lane_inds, right_lane_inds, rectangles=rectangles, margin=margin)
            binary_img = draw.draw_binary(binary_img)  # Make it RGB
            return self.left_line.best_fit, self.right_line.best_fit, texts, binary_img, lane_img
        return self.left_line.best_fit, self.right_line.best_fit, texts

    def search_lanes(self, binary_img):
        """
        Searches left and right lane lines from a binary image by sliding window and polynomial fitting.

        :param binary_img: binary image.
        :return:
            left_fit: left lane poly fit parameters.
            right_fit: right lane poly fit parameters.
            left_lane_inds: left lane indices.
            right_lane_inds: right lane indices.
            rectangles: rectangles as sliding windows.
        """
        # Find the base points of the left lane line in [quarter_point, mid_point] and
        # right lane line in [mid_point, mid_point + quarter_point], based on assumption
        # that lane line should not deviate from image middle too much.
        histogram = np.sum(binary_img[binary_img.shape[0] // 2:, :], axis=0)
        mid_point = np.int(histogram.shape[0] // 2)
        quarter_point = np.int(histogram.shape[0] // 4)
        leftx_base = np.argmax(histogram[quarter_point:mid_point]) + quarter_point
        rightx_base = np.argmax(histogram[mid_point:(mid_point + quarter_point)]) + mid_point

        # Set height of windows - based on nwindows above and image shape.
        window_height = np.int(binary_img.shape[0] // self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image.
        nonzero = binary_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows.
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Create empty lists to receive left and right lane pixel indices.
        left_lane_inds = []
        right_lane_inds = []
        rectangles = []

        # Step through the windows one by one.
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left).
            win_y_low = binary_img.shape[0] - (window + 1) * window_height
            win_y_high = binary_img.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            # Save rectangles.
            rectangles.append((win_xleft_low, win_y_low, win_xleft_high, win_y_high))
            rectangles.append((win_xright_low, win_y_low, win_xright_high, win_y_high))
            # Identify the nonzero pixels in x and y within the window.
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append indices to the lists.
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit lane line points by polynomial, make sure they are not empty.
        left_fit, right_fit = None, None
        if len(leftx) > 0:
            left_fit = np.polyfit(lefty, leftx, 2)
        if len(rightx) > 0:
            right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit, left_lane_inds, right_lane_inds, rectangles

    def search_lanes_with_prev(self, binary_img, prev_left_fit, prev_right_fit):
        """
        Searches left and right lane lines from a binary image, by using previous fitting lanes and polynomial fitting.

        :param binary_img: binary image.
        :param prev_left_fit: previous left polyfit parameters.
        :param prev_right_fit: previous right polyfit parameters.
        :return:
            left_fit: left lane line poly fit parameters.
            right_fit: right lane line poly fit parameters.
            left_lane_inds: left lane line indices.
            right_lane_inds: right lane line indices.
            margin: margin +/- from the previous lanes.
        """
        nonzero = binary_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        prev_left_fit_x = eval_poly(prev_left_fit, nonzeroy)
        prev_right_fit_x = eval_poly(prev_right_fit, nonzeroy)

        left_lane_inds = ((nonzerox > prev_left_fit_x - self.margin) & (nonzerox < prev_left_fit_x + self.margin))
        right_lane_inds = ((nonzerox > prev_right_fit_x - self.margin) & (nonzerox < prev_right_fit_x + self.margin))

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit, right_fit = None, None
        if len(leftx) > 0:
            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
        if len(rightx) > 0:
            right_fit = np.polyfit(righty, rightx, 2)
        return left_fit, right_fit, left_lane_inds, right_lane_inds, self.margin