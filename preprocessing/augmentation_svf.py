import numpy as np
from preprocessing.utils import horizontal_flip

class AugmentationSVF:
    """
    Minimal safe augmentations for Diffusion + SVF alignment.
    """

    def __init__(self, random_flip=True, landmark_shift=False):
        self.random_flip = random_flip
        self.landmark_shift = landmark_shift

    def apply(self, image, landmarks):
        if self.random_flip and np.random.rand() < 0.5:
            image, landmarks = horizontal_flip(image, landmarks)

        if self.landmark_shift and np.random.rand() < 0.25:
            noise = np.random.randint(-10, 10, landmarks.shape)
            landmarks = landmarks + noise

        return image, landmarks
