"""Vehicle detector"""
import collections
import cv2
import glob
import numpy as np
import os.path
import time

from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from scipy.ndimage.measurements import label
from utils import (read_image, convert_color, get_color_features, get_spatial_features,
                   get_hog_features, get_channel_hog_features)

SEARCH_OPTIONS = [
    # scale, ystart, ystop, cells_per_step, min_confidence
    (1.0, 384, 640, 2, 0.1),
    (1.5, 376, 640, 2, 0.1),
    (2.0, 368, 640, 2, 0.1),
]

HEATMAP_DECAY = 0.8
BBOX_CONFIDENCE_THRESHOLD = 0.8
HEATMAP_THRESHOLD = 0.8

# Class to holds feature parameters.
class FeatureParams(collections.namedtuple('FeatureParams', ' '.join([
    'color_space',
    'spatial_size',
    'window_size',
    'color_nbins',
    'orient',
    'pix_per_cell',
    'cell_per_block'
]))):
    pass


class Trainer(object):
    """
    Class to train car classifier and vector scaler.
    """
    def __init__(self, feature_params, car_dir='vehicles', noncar_dir='non-vehicles'):
        """
        Initialize Trainer.
        """
        self.P = feature_params
        self.car_dir = car_dir
        self.noncar_dir = noncar_dir

        # Loads car and non-car images.
        self.car_images = []
        for fpath in glob.glob(os.path.join(self.car_dir, '*', '*.png')):
            self.car_images.append(read_image(fpath))

        self.noncar_images = []
        for fpath in glob.glob(os.path.join(self.noncar_dir, '*', '*.png')):
            self.noncar_images.append(read_image(fpath))

        self.car_features = []
        self.noncar_features = []
        self.scaler = None
        self.clf = svm.LinearSVC()

    def extract_image_features(self, img):
        """
        Extract features from single image
        """
        features = []
        cvt_img = convert_color(img, self.P.color_space)

        spatial_features = get_spatial_features(cvt_img, size=self.P.spatial_size)
        features.append(spatial_features)

        color_features = get_color_features(cvt_img, size=self.P.window_size,
                                            nbins=self.P.color_nbins)
        features.append(color_features)

        if self.P.window_size != (cvt_img.shape[0], cvt_img.shape[1]):
            cvt_img = cv2.resize(cvt_img, self.P.window_size)
        hog_features = get_hog_features(cvt_img, orient=self.P.orient,
                                        pix_per_cell=self.P.pix_per_cell,
                                        cell_per_block=self.P.cell_per_block)
        features.append(hog_features)
        return np.concatenate(features)

    def extract_features(self):
        """
        Extracts features from images.
        """
        t = time.time()
        print('Extracting features...')
        for image in self.car_images:
            self.car_features.append(self.extract_image_features(image))

        for image in self.noncar_images:
            self.noncar_features.append(self.extract_image_features(image))
        print(round(time.time() - t, 2), 'Seconds to extract features.')

    def train(self):
        """
        Trains classifier and set scaler and clf.
        """
        if not self.car_features or not self.noncar_features:
            print("Features not extract, run extract_feature() first.")
            return

        # Train classifier.
        # Create an array stack of feature vectors
        x = np.vstack((self.car_features, self.noncar_features)).astype(np.float64)
        # Define the labels vector
        y = np.hstack((np.ones(len(self.car_features)), np.zeros(len(self.noncar_features))))
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=rand_state)

        # Fit a per-column scaler
        self.scaler = StandardScaler().fit(x_train)
        # Apply the scaler to X
        x_train = self.scaler.transform(x_train)
        x_test = self.scaler.transform(x_test)
        # Shuffle
        x_train, y_train = shuffle(x_train, y_train)

        print('Feature vector length:', len(x_train[0]))
        # Use linear SVC

        t = time.time()
        print('Training linear SVC...')

        self.clf.fit(x_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train.')
        print('Test Accuracy of linear SVC = ', round(self.clf.score(x_test, y_test), 4))
        return self.clf, self.scaler


class VehicleDetector(object):
    """Class to detect vehicles."""
    def __init__(self, clf, scaler, feature_params,
                 search_options=SEARCH_OPTIONS, threshold=HEATMAP_THRESHOLD, decay=HEATMAP_DECAY):
        self.clf = clf
        self.scaler = scaler
        self.P = feature_params
        self.search_options = search_options
        self.threshold = threshold
        self.decay = decay
        self.scale_bbox_confs = None
        self.history_heatmap = None
        self.unfiltered_heatmap = None

    def search_cars_with_option(self, img, scale, cells_per_step, ystart, ystop, conf_thresh):
        """
        Detects car bboxes of image with given scale in region of img[ystart:ystop:,:,:]
        :param img: input image
        :param scale: window scale.
        :param cells_per_step: cells per step.
        :param ystart: y-range start.
        :param ystop: y-range stop.
        :param conf_thresh: classifier confidence threshold.
        :return: list of (bbox, confidence)
        """
        cvt_img = convert_color(img, self.P.color_space)

        # Crop image on in y-region
        ystart = 0 if ystart is None else ystart
        ystop = img.shape[1] if ystop is None else ystop
        cvt_img = cvt_img[ystart:ystop,:,:]

        # Scale the image.
        if scale != 1:
            cvt_img = cv2.resize(cvt_img, (np.int(cvt_img.shape[1] / scale), np.int(cvt_img.shape[0] / scale)))

        # Define blocks and steps as above
        nxblocks = (cvt_img.shape[1] // self.P.pix_per_cell) - self.P.cell_per_block + 1
        nyblocks = (cvt_img.shape[0] // self.P.pix_per_cell) - self.P.cell_per_block + 1
        nblocks_per_window = (self.P.window_size[0] // self.P.pix_per_cell) - self.P.cell_per_block + 1
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

        # Compute individual channel HOG features for the entire image
        hogs = []
        for ch in range(cvt_img.shape[2]):
            hogs.append(get_channel_hog_features(
                img=cvt_img[:,:,ch], orient=self.P.orient,
                pix_per_cell=self.P.pix_per_cell, cell_per_block=self.P.cell_per_block,
                feature_vec=False, vis=False))

        bbox_confs = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step

                hog_features = []
                for ch in range(cvt_img.shape[2]):
                    hog_features.append(
                        hogs[ch][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel())
                hog_features = np.hstack((hog_features[0], hog_features[1], hog_features[2]))

                # Extract the image patch
                xleft = xpos * self.P.pix_per_cell
                ytop = ypos * self.P.pix_per_cell
                subimg = cv2.resize(cvt_img[ytop:ytop + self.P.window_size[0],
                                    xleft:xleft + self.P.window_size[0]],
                                    self.P.window_size)
                # Get spatial features
                spatial_features = get_spatial_features(subimg, self.P.spatial_size)
                # Get color features
                color_features = get_color_features(subimg, size=self.P.window_size, nbins=self.P.color_nbins)
                window_features = self.scaler.transform(np.hstack(
                    (spatial_features, color_features, hog_features)).reshape(1, -1))

                if self.clf.predict(window_features) == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    box_draw = np.int(self.P.window_size[0] * scale)
                    confidence = self.clf.decision_function(window_features)[0]
                    if confidence < conf_thresh:
                        # Only consider window with confidence score >= threshold.
                        continue

                    bbox = [(xbox_left, ytop_draw+ystart), (xbox_left+box_draw,ytop_draw+ystart+box_draw)]
                    bbox_conf = (bbox, confidence)
                    bbox_confs.append(bbox_conf)
        return bbox_confs

    def search_cars(self, img, search_options):
        """
        Find cars by all scale-region sets provided.
        """
        scale_bbox_confs = {}
        for (scale, ystart, ystop, cells_per_step, conf_thresh) in search_options:
            bbox_confs = self.search_cars_with_option(
                img=img, cells_per_step=cells_per_step, scale=scale, ystart=ystart, ystop=ystop, conf_thresh=conf_thresh)
            scale_bbox_confs[scale] = bbox_confs
        return scale_bbox_confs

    def get_heatmap(self, img, scale_bbox_confs):
        """
        Gets heat map from list of bounding box-confidence.

        :param img: input image.
        :param scale_bbox_confs: a map of scale to list of (bbox, confidence).
        """
        heatmap = np.zeros_like(np.zeros_like(img[:, :, 0]).astype(np.float))
        for scale, bbox_confs in scale_bbox_confs.items():
            for bbox, confidence in bbox_confs:
                # Add += confidence for all pixels inside each bbox
                # Assuming each "box" takes the form ((x1, y1), (x2, y2))
                heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += confidence
        return heatmap

    def remove_fp_bboxes(self, scale_bbox_confs, heatmap, bbox_conf_thresh=BBOX_CONFIDENCE_THRESHOLD):
        """
        Remove false positive bounding boxes.
        :param scale_bbox_confs: a map of scale to list of (bbox, confidence).
        :param heatmap:
        :param bbox_conf_thresh:
        :return:
        """
        res = dict()
        for scale, bbox_confs in scale_bbox_confs.items():
            tp_bbox_confs = []
            for bbox, confidence in bbox_confs:
                x1, y1 = bbox[0]
                x2, y2 = bbox[1]
                # TODO: use area sum
                if np.max(heatmap[y1:y2, x1:x2]) >= bbox_conf_thresh:
                    tp_bbox_confs.append( (bbox, confidence))
            res[scale] = tp_bbox_confs
        return res

    def update_heatmap(self, heatmap):
        """
        Updates heatmap
        :param heatmap: current heatmap.
        """
        if self.history_heatmap is None:
            self.history_heatmap = np.copy(heatmap)
            return

        self.history_heatmap = self.history_heatmap * (1 - self.decay) + heatmap * self.decay
        self.history_heatmap = np.clip(self.history_heatmap, 0, 255)
        self.history_heatmap[self.history_heatmap <= self.threshold] = 0

    def detect(self, img):
        """
        Detects vehicles in image.
        :param img: input image
        :return:
            scale_bbox_confs: a map of scale to list of (bbox, confidence).
            unfiltered_heatmap = heatmap before filtering.
            heatmap: heatmap after filtered.
            labels: detected labels.
        """
        # Get car bounding box with confidence scores.
        scale_bbox_confs = self.search_cars(img, self.search_options)
        naive_heatmap = self.get_heatmap(img, scale_bbox_confs)
        # Remove false positive cars.
        filtered_scale_bbox_confs = self.remove_fp_bboxes(scale_bbox_confs, naive_heatmap)
        # Recompute heatmap and derive labels.
        conf_heatmap = self.get_heatmap(img, filtered_scale_bbox_confs)
        self.update_heatmap(conf_heatmap)
        labels = label(self.history_heatmap)
        return scale_bbox_confs, filtered_scale_bbox_confs, naive_heatmap, self.history_heatmap, labels
