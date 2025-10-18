from network.landmark_refinement_network import LandmarkRefinementNetwork
from network.landmark_detection_network import LandmarkDetectionNetwork
from network.semantic_fusion_block import SemanticFusionBlock
from network.pooling import ROIAlign2D
from models.backbone import Backbone
import torch
import torch.nn as nn
from config import cfg


class Network(nn.Module):

    def __init__(
        self,
        backbone_name: str,
        freeze_backbone: bool = False,
        backbone_weights: str = None,
    ):
        super(Network, self).__init__()
        
        # Backbone feature extractor
        self.backbone = Backbone(name=backbone_name, pretrained=False, weights_root_path=backbone_weights)
        if freeze_backbone:
            self.backbone.freeze()
        
        # Get intermediate features from backbone
        self.backbone_features = {}
        self._register_backbone_hooks(backbone_name)
        
        # Calculate input size for detection network (this will be set after first forward pass)
        self.detection_input_size = None
        
        # Landmark detection module (will be initialized after first forward pass)
        self.landmark_detection_module = None
        
        # Semantic fusion block
        self.fusion_block = None  # will be constructed lazily with correct in_channels
        
        # Craniofacial feature extraction & rescaling
        self.roi_align = ROIAlign2D(
            crop_size=cfg.ROI_POOL_SIZE,
            name="craniofacial_feature_extraction"
        )
        
        # Landmark refinement module (will be initialized after first forward pass)
        self.landmark_refinement_module = None
        
        # Flag to track if modules are initialized
        self.modules_initialized = False

    def _register_backbone_hooks(self, backbone_name):
        """Register hooks to extract intermediate features from backbone"""
        def get_features(name):
            def hook(module, input, output):
                self.backbone_features[name] = output
            return hook
        
        # Register hooks for the layers we need
        if backbone_name in ["vgg16", "vgg19"]:
            self.backbone.features[11].register_forward_hook(get_features("C3"))  # block3
            self.backbone.features[16].register_forward_hook(get_features("C4"))  # block4
            self.backbone.features[21].register_forward_hook(get_features("C5"))  # block5
        elif backbone_name in ["resnet18", "resnet34", "resnet50"]:
            self.backbone.base_model.layer1.register_forward_hook(get_features("C2"))
            self.backbone.base_model.layer2.register_forward_hook(get_features("C3"))
            self.backbone.base_model.layer3.register_forward_hook(get_features("C4"))
            self.backbone.base_model.layer4.register_forward_hook(get_features("C5"))
        else:
            # HRNet or others: no hooks; features will be fetched from backbone feature dict
            pass

    def _initialize_modules(self, x):
        """Initialize modules after first forward pass when we know the feature dimensions"""
        if self.modules_initialized:
            return
            
        # Get backbone output to determine input size for detection network
        backbone_output = self.backbone(x)
        
        # If HRNet, populate backbone_features from its feature dict
        if getattr(self.backbone, 'is_hrnet', False):
            feat_dict = self.backbone.get_feature_dict() or {}
            for k in ["C3", "C4", "C5"]:
                if k in feat_dict:
                    self.backbone_features[k] = feat_dict[k]
        
        # Calculate input size for detection network
        if len(backbone_output.shape) == 4:
            # If backbone outputs a feature map, flatten it
            self.detection_input_size = backbone_output.numel() // backbone_output.shape[0]
        else:
            # If backbone outputs a vector
            self.detection_input_size = backbone_output.shape[1]
        
        # Initialize landmark detection module
        self.landmark_detection_module = LandmarkDetectionNetwork(input_size=self.detection_input_size)
        
        # Initialize fusion block now that we know feature channels
        c3_ch = self.backbone_features["C3"].shape[1]
        c4_ch = self.backbone_features["C4"].shape[1]
        c5_ch = self.backbone_features["C5"].shape[1]
        self.fusion_block = SemanticFusionBlock(num_filters=256, in_channels=(c3_ch, c4_ch, c5_ch), name="semantic_fusion_block").to(x.device)

        # Initialize landmark refinement module
        # Calculate input channels for refinement network
        # This depends on the number of feature maps and ROI pool size
        total_channels = 0
        for feature_map in [self.backbone_features["C3"], self.backbone_features["C4"], self.backbone_features["C5"]]:
            total_channels += feature_map.shape[1]
        
        self.landmark_refinement_module = LandmarkRefinementNetwork(input_channels=total_channels)
        
        # Move modules to device
        self.landmark_detection_module = self.landmark_detection_module.to(x.device)
        self.landmark_refinement_module = self.landmark_refinement_module.to(x.device)
        
        self.modules_initialized = True

    def forward(self, images, proposals):
        # Initialize modules if this is the first forward pass
        if not self.modules_initialized:
            self._initialize_modules(images)
        
        # Backbone forward pass (hooks will capture intermediate features, or HRNet dict will be used)
        backbone_output = self.backbone(images)
        if getattr(self.backbone, 'is_hrnet', False):
            feat_dict = self.backbone.get_feature_dict() or {}
            for k in ["C3", "C4", "C5"]:
                if k in feat_dict:
                    self.backbone_features[k] = feat_dict[k]
        
        # Get intermediate features
        C3 = self.backbone_features["C3"]
        C4 = self.backbone_features["C4"]
        C5 = self.backbone_features["C5"]
        
        # Landmark detection
        detection_output = self.landmark_detection_module(backbone_output)
        
        # Semantic fusion
        P3, P4, P5 = self.fusion_block([C3, C4, C5])
        
        # ROI alignment
        region_proposal_map = self.roi_align([P3, P4, P5], proposals)
        
        # Landmark refinement
        refinement_outputs = self.landmark_refinement_module(region_proposal_map)
        
        return detection_output, refinement_outputs

    def get_detection_module(self):
        """Get the landmark detection module for separate inference"""
        return self.landmark_detection_module

    def get_refinement_module(self):
        """Get the landmark refinement module for separate inference"""
        return self.landmark_refinement_module