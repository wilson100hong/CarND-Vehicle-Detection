"""
Camera correction for image distortion.
"""
import cv2
import glob
import numpy as np

_CALIB_IMG_DIR = 'camera_cal/calibration*.jpg'


class CameraCorrector(object):
    """
    A class to correct camera image distortion.
    """
    def __init__(self, y_corners=6, x_corners=9):
        self.y_corners = y_corners
        self.x_corners = x_corners
        self.mtx = None
        self.dist = None

    def calibrate(self, image_dir=_CALIB_IMG_DIR):
        """
        Calibrates from chessboard images and get distortion coefficients.

        :param image_dir: camera calibration images.
        :return: a tuple of (mtx, dist) representing distortion coefficients.
        """
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(9,6,0)
        objp = np.zeros((self.y_corners * self.x_corners, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.x_corners, 0:self.y_corners].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Make a list of calibration images
        calib_images = glob.glob(image_dir)

        # Step through the list and search for chessboard corners
        for fname in calib_images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.x_corners, self.y_corners), None)
            # If found, add object points, image points
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        self.mtx = mtx
        self.dist = dist
        return mtx, dist

    def correct(self, img):
        """
        Corrects the image by applying undistortion.

        :param img: input image
        :return: undistorted image
        """
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
