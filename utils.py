"""Utility functions"""
import cv2
import numpy as np
from skimage.feature import hog


def read_image(filepath):
    """
    Reads image from file.

    :param filepath: image file path.
    :return: RGB image.
    """
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


COLOR_SPACES = [
    'RGB',
    'HSV',
    'LUV',
    'HLS',
    'YUV',
    'YCrCb'
]


def convert_color(img, color_space):
    """
    Converts image color space.

    :param img: RGB image.
    :param color_space: color space in COLOR_SPACES.
    :return: converted image.
    """
    if color_space != 'RGB':
        if color_space == 'HSV':
            cvt_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            cvt_img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            cvt_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            cvt_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            cvt_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        cvt_img = np.copy(img)
    return cvt_img


def get_spatial_features(img, size):
    """
    Gets spatial bins features.

    :param img: input image.
    :param size: spatial bins size, a tuple of (height, width).
    :return: 1-d vector of sptial bins features.
    """
    features = cv2.resize(img, size).ravel()
    return features


def get_color_features(img, size, nbins, bins_range=(0, 256)):
    """
    Gets color histogram features.
    :param img: input image.
    :param size: window size (height, width) for getting color histogram. If input image is in difference size,
                 it will be scaled to window size first.
    :param nbins: number of bins.
    :param bins_range: value range in bins
    :return: 1-d vector of color histogram features.
    """
    window_img = img

    if size != (img.shape[0], img.shape[1]):
        window_img = cv2.resize(img, size)
    # Compute the histogram of the color channels separately
    channel_hists = []
    for ch in range(window_img.shape[2]):
        hist = np.histogram(window_img[:, :, ch], bins=nbins, range=bins_range)
        channel_hists.append(hist[0])
    features = np.concatenate(channel_hists)
    return features


def get_channel_hog_features(img, orient, pix_per_cell, cell_per_block, feature_vec, vis):
    """
    Gets single channel HOG features.
    :param img: input image.
    :param orient: number of orientation.
    :param pix_per_cell: pixels per cells.
    :param cell_per_block: cells per block.
    :param feature_vec: flatten feature into 1-d vector.
    :param vis: also return visualized image.
    :return: HOG features and (optional) HOG image.
    """
    # Call with two outputs if vis==True
    if vis:
        features, hog_img = hog(
            img, orientations=orient,
            pixels_per_cell=(pix_per_cell, pix_per_cell),
            block_norm='L2-Hys',
            cells_per_block=(cell_per_block, cell_per_block),
            transform_sqrt=False,
            feature_vector=feature_vec,
            visualize=vis)
        return features, hog_img
    # Otherwise call with one output
    else:
        features = hog(
            img, orientations=orient,
            pixels_per_cell=(pix_per_cell, pix_per_cell),
            cells_per_block=(cell_per_block, cell_per_block),
            block_norm='L2-Hys',
            transform_sqrt=False,
            feature_vector=feature_vec,
            visualize=vis)
        return features


def get_hog_features(img, orient, pix_per_cell, cell_per_block):
    """
    Gets all channels' HOG features.
    :param img: input image.
    :param orient: number of orientation.
    :param pix_per_cell: pixels per cells.
    :param cell_per_block: cells per block.
    :return: 1-d vector of HOG features for all channels.
    """
    hog_features = []
    for channel in range(img.shape[2]):
        features = get_channel_hog_features(
            orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
            img=img[:, :, channel], feature_vec=True, vis=False)
        hog_features.append(features)
    hog_features = np.ravel(hog_features)
    return hog_features


def eval_poly(poly, y):
    """
    Evaluate value of x of a polynomial with given |y|.
    """
    return poly[0]*y**2 + poly[1]*y + poly[2]