"""Distance-from-floor estimation (original logic preserved).

The estimate is the pixel gap between the bottom of the object and a
horizontal floor line, scaled by a calibrated cm-per-pixel value. It is only
approximate: it assumes the camera is fixed and level, and the object sits at
roughly the same depth as the plane used for calibration.
"""


class HeightEstimator:

    def __init__(self, floor_y: int = 430, pixel_to_cm: float = 0.25):
        self.floor_y = int(floor_y)            # floor line position (pixel)
        self.pixel_to_cm = float(pixel_to_cm)  # 1 pixel = pixel_to_cm cm

    def calculate_height(self, bbox) -> float:
        """Distance (cm) between the object's bottom edge and the floor line."""
        _, _, _, y2 = bbox
        pixel_distance = max(self.floor_y - y2, 0)
        return round(pixel_distance * self.pixel_to_cm, 2)
