import numpy as np

class AccelerationCalculator:
    def __init__(self):
        # 이전 프레임의 keypoint 위치를 저장할 변수
        self.prev_point = None

    def compute_euclidean_distance(self, point1, point2):
        # 유클리드 거리 계산
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def compute_acceleration(self, kpts):
        # 1번 포인트만 추출
        current_point = (kpts[1][0], kpts[1][1]) if kpts.shape[0] > 1 else None

        # 이전 프레임의 keypoint가 없거나 현재 프레임의 1번 포인트가 없으면 가속도는 0으로 설정
        if self.prev_point is None or current_point is None:
            self.prev_point = current_point
            return 0

        # 이전 프레임과 현재 프레임의 keypoint 사이의 거리(속도)를 계산
        velocity = self.compute_euclidean_distance(self.prev_point, current_point)

        # 가속도 = 속도의 변화량 / 시간 (여기서는 프레임으로 나누었습니다.)
        acceleration = velocity / 1  # 1은 프레임 간격입니다. 적절히 조정할 수 있습니다.

        # 현재 keypoint 위치를 저장
        self.prev_point = current_point

        return acceleration
