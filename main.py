import argparse
import cv2
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
from sklearn.externals import joblib

from camera import CameraCorrector
import draw
from lane import LaneDetector
from vehicle import FeatureParams, Trainer, VehicleDetector, HEATMAP_THRESHOLD


def main(args):
    feature_params = FeatureParams(
        color_space='YUV',
        spatial_size=(16, 16),
        window_size=(64, 64),
        color_nbins=16,
        orient=9,
        pix_per_cell=8,
        cell_per_block=2)

    # # Train classifier and store model and scaler.
    # # Uncomment this if classifier not existed.
    # trainer = vehicle.Trainer(feature_params=feature_params, car_dir='vehicles', noncar_dir='non-vehicles')
    # trainer.extract_features()
    # trainer.train()
    # joblib.dump(trainer.scaler, 'scaler.p')
    # joblib.dump(trainer.clf, 'clf.p')

    print("Loading classifier and scaler...")
    clf = joblib.load('clf.p')
    scaler = joblib.load('scaler.p')
    print("Classifier loaded")

    print("Calibrating camera...")
    camera_corrector = CameraCorrector()
    camera_corrector.calibrate()
    print("Camera calibrated.")

    vehicle_detector = VehicleDetector(clf, scaler, feature_params)
    lane_detector = LaneDetector()

    def process_image(input_img):
        img = camera_corrector.correct(input_img)

        h, w = img.shape[0], img.shape[1]
        # Do vehicle detection.
        scale_bbox_confs, heatmap, labels = vehicle_detector.detect(img)
        bbox_confs_img = draw.draw_bbox_conf_list(img, scale_bbox_confs.values(), show_conf=False)
        heatmap_img = draw.draw_heatmap(heatmap, HEATMAP_THRESHOLD)

        # Draw labeled vehicles.
        canvas_img = draw.draw_labeles(img, labels)
        cv2.imwrite('temp/label_{}.png'.format(temp.I), cv2.cvtColor(canvas_img, cv2.COLOR_RGB2BGR))

        # Do lane finding.
        l_fit, r_fit, texts, binary_img, lane_img = lane_detector.detect(img, vis=True)
        # Draw lane lines.
        canvas_img = draw.draw_lanes(canvas_img, l_fit, r_fit, lane_detector.warp)

        # Overlay image as sub-windows.
        draw.overlay(canvas_img, bbox_confs_img, 0, 0, 0.25)
        draw.overlay(canvas_img, heatmap_img, w//4, 0, 0.25)
        draw.overlay(canvas_img, binary_img, w//2, 0, 0.25)
        draw.overlay(canvas_img, lane_img, 3*w//4, 0, 0.25)

        # Draw info texts.
        canvas_img = draw.draw_texts(canvas_img, texts, h//4)

        return canvas_img

    input_clip = VideoFileClip(args.input_video)
    # input_clip = input_clip.subclip(19.0, 21.0)
    print("Processing video...")
    processed_clip = input_clip.fl_image(process_image)
    processed_clip.write_videofile(args.output_video, audio=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vehicle detector main pipeline')
    parser.add_argument('-i', '--input-video', help='Input video file', required=True)
    parser.add_argument('-o', '--output-video', help='Output video file', default='output.mp4')
    args = parser.parse_args()
    main(args)
