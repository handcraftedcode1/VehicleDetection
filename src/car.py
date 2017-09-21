import cv2
import numpy as np
from feature_extractor import FeatureExtractor
from sklearn.preprocessing import normalize

class Car():
    def __init__(self, center=(0, 0)):
        self.x_center = center[0]
        self.y_center = center[1]
        self.past_x_centers = []
        self.past_y_centers = []
        self.x_center_history = []
        self.y_center_history = []
        self.max_center_history = 300
        self.avg_window = 5
        self.best_poly_fit = []
        self.feature_extractor = FeatureExtractor()
        self.confidence_score = 100  # a measurement of how confident we are about the position of this car

    def update(self, center=(0, 0)):

        # save previously measured values
        if len(self.past_x_centers) < self.avg_window:
            self.past_x_centers.append(center[0])

        if len(self.past_y_centers) < self.avg_window:
            self.past_y_centers.append(center[1])

        # do avg update on the center
        if len(self.past_x_centers) == self.avg_window:

            # calculate avg with new value
            self.x_center = np.mean(self.past_x_centers)
            self.y_center = np.mean(self.past_y_centers)

            self.past_x_centers.pop(0)  # shift first item off the window
            self.past_x_centers.append(self.x_center)
            self.past_y_centers.pop(0)
            self.past_y_centers.append(self.y_center)

        # save history of centers
        self.x_center_history.append(self.x_center)
        self.y_center_history.append(self.y_center)

        if len(self.x_center_history) == self.max_center_history:
            self.x_center_history.pop(0)
            self.y_center_history.pop(0)

    def draw_trailing_path(self, img):
        """Draws a series of circles representing the past path of the car"""
        for i in range(0, len(self.x_center_history)):
            cv2.circle(img, center=(int(self.x_center_history[i]), int(self.y_center_history[i])), radius=12, color=(0, 255, 0))
        return img
