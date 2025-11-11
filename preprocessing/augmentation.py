import numpy as np
from config import cfg
from preprocessing.utils import generate_edge_bank  # ✅ only import existing function


# -------------------- #
# Utility transforms
# -------------------- #
def identity(image):
    """No-op transformation."""
    return image


def horizontal_flip(image, landmarks):
    """Simple horizontal flip."""
    image_height, image_width = image.shape[:2]
    flipped_image = np.ascontiguousarray(image[:, ::-1, :])
    flipped_landmarks = landmarks.copy()
    flipped_landmarks[:, 0] = image_width - landmarks[:, 0]
    return flipped_image, flipped_landmarks


# -------------------- #
# Main Augmentation class
# -------------------- #
class Augmentation(object):
    """
    Minimal augmentation wrapper for cephalometric dataset.
    Includes optional geometric jitter and edge-map fusion compatibility.
    """

    def __init__(
        self,
        random_flip: bool = True,
        landmark_shift: bool = False,
        add_edges: bool = True
    ):
        self.random_flip = random_flip
        self.landmark_shift = landmark_shift
        self.add_edges = add_edges

    def apply(self, image: np.ndarray, landmarks: np.ndarray):
        # optional random flip
        if self.random_flip and np.random.rand() < 0.5:
            image, landmarks = horizontal_flip(image, landmarks)

        # optional landmark perturbation
        if self.landmark_shift and np.random.rand() < 0.25:
            noise = np.random.randint(-10, 10, landmarks.shape)
            landmarks = landmarks + noise

        # ✅ return only image and landmarks
        return image, landmarks
