"""
Utilities to draw image.
"""

import cv2
import numpy as np

from utils import eval_poly

BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)

COLORS = [
    BLUE,
    RED,
    GREEN,
    BLACK,
]

FONT = cv2.FONT_HERSHEY_DUPLEX


def draw_labeles(img, labels, color=BLUE, thick=6):
    """
    Draws labels as bounding boxes on original image.
    :param img: input image.
    :param labels: list of labels.
    :param color: color for bounding box.
    :param thick: bounding box thickness.
    :return: created image.
    """
    draw_img = np.copy(img)
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # Return the image
    return draw_img


def draw_bbox_conf_list(img, bbox_conf_list, show_conf=False, thick=2):
    """
    Generates a new image by overlaying bbox confidence on source image.
    :param img: image image.
    :param bbox_conf_list: list of tuple (bbox, confidence). Each bbox is [(x1,y1), (x2,y2)].
    :param show_conf: show confidence score
    :param thick: bounding box thickness.
    :return: created image.
    """
    draw_img = np.copy(img)
    idx = 0
    for bbox_confs in bbox_conf_list:
        for bbox, conf in bbox_confs:
            cv2.rectangle(draw_img, bbox[0], bbox[1], COLORS[idx], thick)
            if show_conf:
                cv2.putText(draw_img, '{0:.2f}'.format(conf),
                            ((bbox[0][0] + bbox[1][0]) // 2, bbox[1][1] - 10), FONT, 0.5,
                            WHITE, 1, cv2.LINE_AA)
        idx += 1
    return draw_img


def draw_heatmap(heatmap, threshold):
    """
    Generate heatmap image with threshold legend.
    :param heatmap: heatmap as binary image.
    :param threshold: threshold showed in legend.
    :return: created image.
    """
    # First convert to HLS colorspace.
    h_coeff = 0.5
    l_coeff = 10.0
    heatmap_img = np.dstack((np.clip(heatmap * h_coeff, 0, 255),  # H
                             np.clip(heatmap * l_coeff, 0, 255),  # L
                             np.full(heatmap.shape, 255.0))).astype(np.uint8)  # S
    if threshold is not None:
        color = (threshold*h_coeff, threshold*l_coeff, 255.0)
        cv2.circle(heatmap_img, (100, 100), 50, color, -1)
        cv2.putText(heatmap_img, 'Threshold', (200, 125), FONT, 2, WHITE, 2, cv2.LINE_AA)
    return cv2.cvtColor(heatmap_img, cv2.COLOR_HLS2RGB)


def draw_texts(img, texts, y):
    """
    Draw texts on image.
    :param img: input image.
    :param texts: list of string.
    :param y: y position of texts
    :return: updated image.
    """
    draw_img = np.copy(img)
    w, h = draw_img.shape[1], draw_img.shape[0]
    # Add a transparent black overlay, so white texts are more obvious.
    texts_overlay = np.copy(draw_img)
    cv2.rectangle(texts_overlay, (0, y), (w, y+10 + len(texts)*30), BLACK, thickness=-1)
    for idx, text in enumerate(texts):
        cv2.putText(texts_overlay, text, (30, y+30 + idx*30), FONT, 1.0, WHITE, 2, cv2.LINE_AA)
    alpha = 0.5
    cv2.addWeighted(texts_overlay, alpha, draw_img, 1 - alpha, 0, draw_img)
    return draw_img


def draw_binary(img):
    """
    Generates a black-white RGB image from a binary image.
    :param img: binary image.
    :return: created image (RGB colorspace)
    """
    return np.dstack((img*255, img*255, img*255))


def draw_lane_fits(img, left_fit, right_fit, left_lane_inds, right_lane_inds, rectangles, margin):
    """
    Generates lane line fits.
    :param img: binary image
    :param left_fit: left line polyfit.
    :param right_fit: right line polyfit.
    :param left_lane_inds: left line indices.
    :param right_lane_inds: right line indices.
    :param rectangles: rectangles used in sliding window search.
    :param margin: margin used in marginal search.
    :return: created (RGB colorspace).
    """
    w, h = img.shape[1], img.shape[0]
    ploty = np.linspace(0, h-1, h)
    left_fitx, right_fitx = eval_poly([1, 1, 0], ploty), eval_poly([1, 1, 0], ploty)
    if left_fit is not None:
        left_fitx = eval_poly(left_fit, ploty)
    if right_fit is not None:
        right_fitx = eval_poly(right_fit, ploty)

    out_img = np.dstack((img, img, img))
    # Draw the rectangles on the visualization image, if any.
    if rectangles:
        for rect in rectangles:
            cv2.rectangle(out_img, (rect[0], rect[1]), (rect[2], rect[3]), GREEN, 2)

    # Identify the x and y positions of all nonzero pixels in the image.
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = RED
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = BLUE

    # Draw the lane onto the warped blank image
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    cv2.polylines(out_img, np.int32([pts_left]), isClosed=False, color=YELLOW, thickness=2)
    cv2.polylines(out_img, np.int32([pts_right]), isClosed=False, color=YELLOW, thickness=2)

    # Draw margin on image, if any.
    if margin:
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        window_img = np.zeros_like(out_img)
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), GREEN)
        cv2.fillPoly(window_img, np.int_([right_line_pts]), GREEN)
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    return out_img


def draw_lanes(img, left_fit, right_fit, warper):
    """
    Draws lanes on image.
    :param img: input image.
    :param left_fit: left line polyfit.
    :param right_fit: right line polyfit
    :param warper: a callable warps bird-view to vehicle-view.
    :return: created image.
    """
    draw_img = np.copy(img)
    if left_fit is None or right_fit is None:
        return draw_img

    color_warp = np.zeros_like(draw_img)

    # to cover same y-range as image
    w, h = draw_img.shape[1], draw_img.shape[0]
    ploty = np.linspace(0, h - 1, h)
    left_fitx = eval_poly(left_fit, ploty)
    right_fitx = eval_poly(right_fit, ploty)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), GREEN)
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=RED, thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=BLUE, thickness=15)

    # Warp the blank back to original image space using |warper|
    newwarp = warper(color_warp)
    # Combine the result with the original image
    return cv2.addWeighted(draw_img, 1, newwarp, 0.5, 0)


def borderline(img, color, thick):
    """
    Draws borderline of image.
    :return: updated image.
    """
    img[:, :thick] = color
    img[:, -thick:] = color
    img[:thick, :] = color
    img[-thick:, :] = color
    return img


def overlay(canvas, img, x, y, scale):
    """
    Overlays image on canvas. Images can be scaled down and a borderline will be added.
    :param canvas: image on bottom.
    :param img: image on top.
    :param x: location of overlay.
    :param y: location of overlay.
    :param scale: image scaling factor.
    :return: updated image.
    """
    scaled_img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
    borderline(scaled_img, RED, thick=2)
    w, h = scaled_img.shape[1], scaled_img.shape[0]
    x = int(x)
    y = int(y)
    canvas[y:y+h, x:x+w] = scaled_img
    return canvas