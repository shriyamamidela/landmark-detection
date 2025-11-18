import numpy as np
from config import cfg
from preprocessing.utils import (
    identity, high_pass_filtering, solarization, unsharp_masking, inversion,
    contrast_limited_histogram_equalization, horizontal_flip,
    random_cropping, vertical_shift
)

class AugmentationBackbone:
    """
    Full augmentation for Stage-1 backbone training
    (same as research paper)
    """

    def __init__(
        self,
        highpass_filter=True,
        unsharp_mask=True,
        solarize=True,
        invert=True,
        clahe=True,
        random_flip=True,
        random_shift=True,
        random_crop=True,
        landmark_shift=True
    ):
        # Photometric transforms
        self.photometric = [identity]
        if highpass_filter: self.photometric.append(high_pass_filtering)
        if solarize: self.photometric.append(solarization)
        if unsharp_mask: self.photometric.append(unsharp_masking)
        if invert: self.photometric.append(inversion)
        if clahe: self.photometric.append(contrast_limited_histogram_equalization)

        # Geometric transforms
        self.geometric = []
        if random_flip: self.geometric.append(horizontal_flip)
        if random_shift: self.geometric.append(vertical_shift)
        if random_crop: self.geometric.append(random_cropping)

        self.landmark_shift = landmark_shift

    def apply(self, image, landmarks):
        # photometric
        fn = np.random.choice(self.photometric)
        image = fn(image)

        # geometric
        np.random.shuffle(self.geometric)
        for fn in self.geometric:
            if np.random.rand() < 0.5:
                image, landmarks = fn(image, landmarks)

        # jitter
        if self.landmark_shift and np.random.rand() < 0.25:
            noise = np.random.randint(-10, 10, landmarks.shape)
            landmarks = landmarks + noise

        return image, landmarks
