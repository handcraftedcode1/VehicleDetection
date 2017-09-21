import numpy as np
from car import Car


class CarTracker():
    def __init__(self):
        self.cars = []

    def identify_cars(self, img, car_bboxes):

        cars_found = []

        # check if any new cars were found
        for bbox in car_bboxes:

            center_x = ((bbox[1][0] - bbox[0][0]) / 2) + bbox[0][0]
            center_y = ((bbox[1][1] - bbox[0][1]) / 2) + bbox[0][1]

            if len(self.cars) == 0:
                self.cars.append(Car(center=(center_x, center_y)))
            else:
                # if there isn't a car that doesn't have a similar bounding box as something already tracked
                # then consider it a new car
                new_car = False
                for c in self.cars:
                    if np.sqrt(abs((c.x_center - center_x)**2 + (c.y_center - center_y)**2)) > 50:
                        new_car = True
                    else:
                        new_car = False
                        break

                if new_car:
                    self.cars.append(Car(center=(center_x, center_y)))
                else:
                    c.update(center=(center_x, center_y))
                    cars_found.append(c)

        # adjust confidence scores for already tracked cars (found or not found)
        for index, c in enumerate(self.cars):

            if not c in cars_found:
                c.confidence_score -= 5
            else:
                c.confidence_score += 5
                c.confidence_score = np.clip(c.confidence_score + 10, a_min=0, a_max=100)

            # stop tracking cars with a low confidence score
            if c.confidence_score <= 0:
                self.cars.pop(index)

    def draw_cars_past_path(self, img):

        #print('\nCar founds: ' + str(len(self.cars)))
        paths_img = img.copy()
        for c in self.cars:
            paths_img = c.draw_trailing_path(paths_img)
        return paths_img
