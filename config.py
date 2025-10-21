import torch
from easydict import EasyDict as edict

# ---------------------------------------------------------------------------- #
# Cephalometric Landmark Detection Configuration (HRNet-W48)
# ---------------------------------------------------------------------------- #

cfg = edict()

# -------------------------------
# Image parameters
# -------------------------------
cfg.ORIGINAL_HEIGHT = 2400
cfg.ORIGINAL_WIDTH  = 1935

# Network input size
cfg.HEIGHT = 800          # you can increase to 864 or 1024 for higher precision
cfg.WIDTH  = 640
cfg.IMAGE_INPUT_SHAPE = (3, cfg.HEIGHT, cfg.WIDTH)

# Image resolution in mm/pixel
cfg.IMAGE_RESOLUTION = 0.1

# -------------------------------
# Landmark parameters
# -------------------------------
cfg.NUM_LANDMARKS = 19

cfg.ANATOMICAL_LANDMARKS = {
    "0": "Sella",
    "1": "Nasion",
    "2": "Orbitale",
    "3": "Porion",
    "4": "A-point",
    "5": "B-point",
    "6": "Pogonion",
    "7": "Menton",
    "8": "Gnathion",
    "9": "Gonion",
    "10": "Lower Incisal Incision",
    "11": "Upper Incisal Incision",
    "12": "Upper Lip",
    "13": "Lower Lip",
    "14": "Subnasale",
    "15": "Soft Tissue Pogonion",
    "16": "Posterior Nasal Spine",
    "17": "Anterior Nasal Spine",
    "18": "Articulare",
}

# -------------------------------
# Device configuration
# -------------------------------
cfg.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Backbone configuration
# -------------------------------
cfg.BACKBONE_NAME = "hrnet_w48"  # ✅ switched to HRNet-W48
cfg.PRETRAINED_BACKBONE_WEIGHTS = "pretrained_weights/hrnet_w48_imagenet.pth"  # ✅ correct pretrained path

# Feature block mappings (for non-HRNet backbones)
cfg.BACKBONE_BLOCKS_INFO = {
    "vgg16": {
        "C1": "features.1",
        "C2": "features.6",
        "C3": "features.11",
        "C4": "features.16",
        "C5": "features.21"
    },
    "vgg19": {
        "C1": "features.1",
        "C2": "features.6",
        "C3": "features.11",
        "C4": "features.16",
        "C5": "features.21"
    },
    "resnet18": {
        "C2": "layer1",
        "C3": "layer2",
        "C4": "layer3",
        "C5": "layer4"
    },
    "resnet34": {
        "C2": "layer1",
        "C3": "layer2",
        "C4": "layer3",
        "C5": "layer4"
    },
    "resnet50": {
        "C2": "layer1",
        "C3": "layer2",
        "C4": "layer3",
        "C5": "layer4"
    },
    "darknet19": {
        "C1": "layer1",
        "C2": "layer2",
        "C3": "layer3",
        "C4": "layer4",
        "C5": "layer5"
    },
    "darknet53": {
        "C1": "layer1",
        "C2": "layer2",
        "C3": "layer3",
        "C4": "layer4",
        "C5": "layer5"
    }
}

# -------------------------------
# ROI and region proposal parameters
# -------------------------------
cfg.ROI_POOL_SIZE = (5, 5)
cfg.BOX_MARGIN = 32

# -------------------------------
# Training hyperparameters
# -------------------------------
cfg.TRAIN = edict()
cfg.TRAIN.EPOCHS = 100
cfg.TRAIN.OPTIMIZER = "adam"
cfg.TRAIN.LEARNING_RATE = 1e-4
cfg.TRAIN.BATCH_SIZE = 4
cfg.TRAIN.WEIGHT_DECAY = 1e-5
cfg.TRAIN.CHECKPOINT_DIR = "checkpoints"
cfg.TRAIN.LOG_DIR = "logs"
cfg.TRAIN.SAVE_INTERVAL = 10

# -------------------------------
# Miscellaneous
# -------------------------------
cfg.DEBUG = False
cfg.SEED = 42

# -------------------------------
# Convenience aliases
# -------------------------------
config = cfg  # alias for backward compatibility
