import numpy as np

class AccelerationCalculator:
    def __init__(self):
        self.prev_center = None

    def compute_euclidean_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def compute_acceleration(self, bbox):
        current_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

        if self.prev_center is None:
            self.prev_center = current_center
            return 0

        velocity = self.compute_euclidean_distance(self.prev_center, current_center)

        acceleration = velocity / 1

        self.prev_center = current_center

        return acceleration
