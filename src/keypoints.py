import cv2
import dlib
import numpy as np
from typing import Tuple, Any, List


def get_detector_and_predictor(predictor_path: str) -> Tuple[Any, Any]:
    """
    Gets detector and predictor objects
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    return detector, predictor


def get_keypoints(
    img: np.array, detector: Any, predictor: Any, cutoff: int = 47
) -> List[Tuple[int, int]]:
    """
    Gets face keypoints for given image
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = detector(img_gray)
    
    if len(faces) > 1:
        assert "There is more than one face on the image!"
    
    face = faces[0]
    landmarks = predictor(img_gray, face)

    landmarks_points = _get_points(landmarks)

    return landmarks_points[:cutoff]


def _get_points(landmarks: Any) -> List[Tuple[int, int]]:
    """
    Gets landmarks points
    """
    landmarks_points = []

    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))
    
    return landmarks_points
