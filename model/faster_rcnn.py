from torch import nn
from typing import Tuple
import torch
import torchvision
from .rpn import RegionProposalNetwork 
from .roi_head import ROIHead
from .utils import transform_boxes_to_original_size

class FasterRCNN(nn.Module):
    def __init__(self, model_config, num_classes):
        """
        Initializes the Faster R-CNN model.

        This module integrates a backbone feature extractor (VGG16), a Region Proposal Network (RPN)
        to generate object proposals, and an ROI Head to classify these proposals and refine their
        bounding boxes.

        Args:
            num_classes (int): The number of classes for the classifier, including the background.
                            Defaults to 21.
        """
        super(FasterRCNN, self).__init__()
        self.model_config = model_config
        
        #Backbone is vgg16 without last max pooling layers and classification layers
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.backbone = vgg16.features[:-1] # removing last max pooling layers and cls layers
        
        self.rpn = RegionProposalNetwork(in_channels=model_config["backbone_out_channels"],
                                         scales=model_config["scales"],
                                         aspect_ratios=model_config["aspect_ratios"],
                                         model_config=model_config)
        
        self.roi_head = ROIHead(model_config, num_classes, in_channels=model_config["backbone_out_channels"])
        
        # Freeze the first few layers of VGG16
        for layer in self.backbone[:10]:
            for p in layer.parameters():
                p.requires_grad = False
        
        #ImageNet mean and standard deviation used to normalize the input 
        #Hyperparam???
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        
        # resize the smaller dimension to min_size (600 pixels) and also cap the larger dimension to max_size (1000 pixels)
        self.min_size = model_config["min_im_size"]
        self.max_size = model_config["max_im_size"]
        
    def normalize_resize_image_and_boxes(self, image: torch.Tensor, bboxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalizes and resizes an image and its corresponding bounding boxes.

        The image is normalized using ImageNet's mean and std. It is then resized such that
        its smaller dimension is `self.min_size` (e.g., 600 pixels), while ensuring its
        larger dimension does not exceed `self.max_size` (e.g., 1000 pixels).
        The bounding boxes are scaled accordingly.

        Args:
            image (torch.Tensor): The input image tensor of shape [C, H, W].
            bboxes (torch.Tensor): A tensor of ground-truth bounding boxes of shape [1, N, 4].
                                   Can be None if not in training mode.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the processed image
                                               and the resized bounding boxes.
        """
        # Normalize
        mean = torch.as_tensor( self.image_mean, dtype=image.dtype, device=image.device)
        std = torch.as_tensor( self.image_std, dtype=image.dtype, device=image.device)
        image = (image - mean[:, None, None]) / std[:, None, None]
        ######
        
        # Resize such that lower dim is scaled to 600 but larger dim not more than 1000
        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])
        min_size = torch.min(im_shape).to(dtype=torch.float32)
        max_size = torch.max(im_shape).to(dtype=torch.float32)
        
        # scale ensures that if resizing the smaller dimension to 600, the larger dimentsion will be less than 1000 
        # eg.: scale = 1.6
        scale = torch.min(
            float(self.min_size) / min_size,
            float(self.max_size) / max_size
        ) 
        
        scale_factor = scale.item()
        #Resize image based on scale
        image = torch.nn.functional.interpolate(
            image,
            size=None,
            scale_factor=scale_factor,
            mode="bilinear",
            recompute_scale_factor=True,
            align_corners=False
        )
        
        # Resize boxes
        # As the image was resized, the gt boxes need also to be resized
        if bboxes is not None:
            # Calculate h and w scales [resized_image_h/original_image_h, resized_image_w/original_image_w]
            ratios = [
                torch.tensor(s, dtype=torch.float32, device=bboxes.device) / torch.tensor(s_orig, dtype=torch.float32, device=bboxes.device) 
                for s, s_orig in zip(image.shape[-2:], (h,w))
            ]
            
            # resize the box coordinates accordingly to image resize scale
            ratio_height, ratio_width = ratios
            xmin, ymin, xmax, ymax = bboxes.unbind(2)
            xmin = xmin * ratio_width
            xmax = xmax * ratio_width
            ymin = ymin * ratio_height
            ymax = ymax * ratio_height

            bboxes = torch.stack((
                xmin,
                ymin,
                xmax, 
                ymax
            ), dim=2)
        return image, bboxes 
        
    def forward(self, image: torch.Tensor, target = None):
        """
        Defines the forward pass of the Faster R-CNN model.

        Args:
            image (torch.Tensor): The input image tensor of shape [1, C, H, W].
            target (dict, optional): A dictionary containing ground-truth data, primarily
                                     "bboxes" and "labels". Required for training. Defaults to None.

        Returns:
            Tuple[dict, dict]: A tuple containing two dictionaries:
                               - rpn_output: Losses and proposals from the RPN.
                               - frcnn_output: Losses, boxes, labels, and scores from the ROI Head.
        """
        old_shape = image.shape[-2:] # eg.: (375, 500)
        
        # image -> eg.: (1,3, 600, 800)
        if self.training:    
            # Normalize and resize boxes  
            image, bboxes = self.normalize_resize_image_and_boxes(image, target["bboxes"])
            target["bboxes"] = bboxes
        else:
            image, _ = self.normalize_resize_image_and_boxes(image, None)
            
        #Call backbone
        feat = self.backbone(image) # eg.: (1,512,37,50)
        
        #Call RPN and get proposals
        rpn_output = self.rpn(image, feat, target)
        proposals = rpn_output["proposals"]
        
        # Call ROI head and convert proposals to boxes
        frcnn_output = self.roi_head(feat, proposals, image.shape[-2:], target)
        
        if not self.training:
            # Transform boxes back to original image dimension
            frcnn_output["boxes"] = transform_boxes_to_original_size(frcnn_output["boxes"], image.shape[-2:], old_shape)

        return rpn_output, frcnn_output
        