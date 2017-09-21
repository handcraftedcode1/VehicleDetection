from scipy.ndimage.measurements import label

import cv2
import numpy as np

from feature_extractor import FeatureExtractor

#from car_tracker import CarTracker
#car_tracker = CarTracker()

# Finds cars in the given image by applying pretrained svm classifier over image in sliding window fashion on image pyramid
class CarFinder:
    """Finds cars in a given image by sliding a window across the image and using a SVM to classify existence of a car"""

    def __init__(self):

        self.feature_extractor = FeatureExtractor()
        self.frames = 10  # num of frames to aggregate heatmaps over
        self.heatmaps = []  # collection of heatmaps over past 10 frames
        self.cummulative_heatmap = np.zeros((720, 1280)).astype(np.float64)  # cummulative heat map over 10 frames

        self.cars_detected = 0  # count of cars detected in this frame
        self.contours_detected = []

    def find_cars(self, img, ystart, ystop, scales, svc, X_scaler, orient, pix_per_cell,
                  cell_per_block, spatial_size, hist_bins):

        bbox_list = self.predict_bboxes(img, ystart, ystop, scales, svc, X_scaler, orient, pix_per_cell,
                                        cell_per_block, spatial_size, hist_bins)

        heatmap = self.predict_heatmap(img, bbox_list)

        draw_img = self.predict_contours(img, heatmap)

        #car_tracker.identify_cars(img, self.contours_detected)

        return draw_img

    def draw_detections(self, img):

        for bbox in self.contours_detected:
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 3)

        cv2.putText(img, 'Cars detected: %s' % self.cars_detected, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

        return img

    def process_frame(self, img):
        car_boundaries = self.draw_detections(img)
        #car_paths = car_tracker.draw_cars_past_path(car_boundaries)
        return car_boundaries

    def predict_heatmap(self, image, bbox_list, threshold=1):

        heat = np.zeros_like(image[:, :, 0]).astype(np.float)

        # add heat to each box in box list
        heat = self.add_heat(heat, bbox_list)

        # apply threshold to help remove false positives
        heat = self.apply_threshold(heat, threshold)

        # visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # accumulate heatmaps for 10 frames. Makes smooth/accurate bounding boxes
        self.heatmaps.append(heatmap)
        self.cummulative_heatmap = self.cummulative_heatmap + heatmap

        if len(self.heatmaps) > self.frames:
            self.cummulative_heatmap = self.cummulative_heatmap - self.heatmaps.pop(0)
            self.cummulative_heatmap = np.clip(self.cummulative_heatmap, 0.0, 9999999.0)

        return self.cummulative_heatmap

    def predict_contours(self, image, heatmap):
        """Draws a contour around a detected heat blob"""

        # find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = self.draw_labeled_bboxes(np.copy(image), labels)

        return draw_img

    def add_heat(self, heatmap, bbox_list):
        """Creates a heatmap given a list of bounding boxes"""

        # iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        return heatmap

    def apply_threshold(self, heatmap, threshold):
        """Threshold the heatmap to filter weak heat blobs"""
        heatmap[heatmap <= threshold] = 0
        return heatmap

    def predict_bboxes(self, img, y_start, y_stop, scales, svc, X_scaler, orient, pix_per_cell,
                       cell_per_block, spatial_size, hist_bins, vis=False):
        """Find the bounding boxes around cars given an image using a SVC"""

        x_start = img.shape[1] // 2
        x_stop = img.shape[1]

        bbox_list = []  # box" takes the form ((x1, y1), (x2, y2))
        for scale in scales:
            bboxes = self.scan_image(img, y_start, y_stop, x_start,
                                     x_stop, scale, svc, X_scaler, orient, pix_per_cell,
                                     cell_per_block, spatial_size, hist_bins)
            bbox_list.extend(bboxes)

        if vis:
            draw_img = np.copy(img)
            for bbox in bbox_list:
                cv2.rectangle(draw_img, (bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[1][1]), (0, 0, 255), 6)
            return draw_img, bbox_list
        else:
            return bbox_list

    def scan_image(self, img, y_start, y_stop, x_start, x_stop, scale, svc, X_scaler, orient, pix_per_cell,
                   cell_per_block, spatial_size, hist_bins, vis=False):
        """"""

        draw_img = np.copy(img)
        img = img.astype(np.float32) / 255

        img_tosearch = img[y_start:y_stop, x_start:x_stop, :]
        ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)

        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - 1

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nx_steps = (nxblocks - nblocks_per_window) // cells_per_step
        ny_steps = (nyblocks - nblocks_per_window) // cells_per_step

        # compute individual channel HOG features for the entire image
        hog1 = self.feature_extractor.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = self.feature_extractor.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = self.feature_extractor.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        bbox_list = []  # box" takes the form ((x1, y1), (x2, y2))
        for xb in range(nx_steps):
            for yb in range(ny_steps):

                ypos = yb * cells_per_step
                xpos = xb * cells_per_step

                # extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                x_left = xpos * pix_per_cell
                y_top = ypos * pix_per_cell

                # extract the image patch
                sub_img = cv2.resize(ctrans_tosearch[y_top:y_top + window, x_left:x_left + window], (64, 64))

                # get color features
                spatial_features = self.feature_extractor.bin_spatial(sub_img, size=spatial_size)
                hist_features = self.feature_extractor.color_hist(sub_img, nbins=hist_bins)

                # scale features and make a prediction
                test_features = X_scaler.transform(
                    np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                test_prediction = svc.predict(test_features)

                if test_prediction == 1:
                    x_box_left = np.int(x_left * scale)
                    y_top_draw = np.int(y_top * scale)
                    win_draw = np.int(window * scale)
                    bbox_list.append(((x_box_left + x_start, y_top_draw + y_start),
                                      (x_box_left + win_draw + x_start, y_top_draw + win_draw + y_start)))

                if vis:
                    x_box_left = np.int(x_left * scale)
                    y_top_draw = np.int(y_top * scale)
                    win_draw = np.int(window * scale)
                    cv2.rectangle(draw_img, (x_box_left + x_start, y_top_draw + y_start),
                                  (x_box_left + win_draw + x_start, y_top_draw + win_draw + y_start), (0, 0, 255), 2)

        if vis:
            return draw_img, bbox_list
        else:
            return bbox_list

    def draw_labeled_bboxes(self, img, labels):

        self.cars_detected = labels[1]  # count of cars detected in this frame
        self.contours_detected = []

        # iterate through all detected cars
        for car_number in range(1, labels[1] + 1):

            # find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()

            # identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            # define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

            # draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 3)

            self.contours_detected.append(bbox)

        return img
