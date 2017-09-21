from lane_finder import LaneFinder
from car_finder import CarFinder
from sklearn.externals import joblib
import numpy as np

lane_finder = LaneFinder()
car_finder = CarFinder()

# Configurations
MODEL_FOLDER = 'gen'

ystart = 400
ystop = 656
scales = [1.0, 1.5, 2.0]

orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins

# load svm classifiers
svc = joblib.load('svc.pkl')
X_scaler = joblib.load('x_scaler.pkl')


class DetectionPipeline():
    def process_frame(self, img):
        """Detect and draw lane lines and cars"""
        frame_copy = np.copy(img)

        frame_lane = lane_finder.process_frame(frame_copy)

        car_finder.find_cars(img, ystart, ystop, scales, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                             spatial_size, hist_bins)

        frame_cars = car_finder.process_frame(frame_lane)

        return frame_cars
