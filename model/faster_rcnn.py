from torch import nn
import torch
import torchvision
from .rpn import RegionProposalNetwork 
from .roi_head import ROIHead

def transform_boxes_to_original_size(boxes, new_size, original_size):
    ratios = [
            torch.tensor(s_orig, dtype=torch.float32, device=boxes.device) / torch.tensor(s, dtype=torch.float32, device=boxes.device) 
            for s, s_orig in zip(new_size, original_size)
        ]
    
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)

class FasterRCNN(nn.Module):
    def __init__(self, num_classes = 21):
        super(FasterRCNN, self).__init__()
        
        #Backbone is vgg16 without last max pooling layers and classification layers
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.backbone = vgg16.features[:-1] # removing last max pooling layers and cls layers
        
        self.rpn = RegionProposalNetwork(in_channels=512)
        self.roi_head = ROIHead(num_classes=num_classes, in_channels=512)
        
        # Freeze the first few layers of VGG16
        for layer in self.backbone[:10]:
            for p in layer.parameters():
                p.requires_grad = False
        
        #ImageNet mean and standard deviation used to normalize the input 
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        
        # resize the smaller dimension to min_size (600 pixels) and also cap the larger dimension to max_size (1000 pixels)
        self.min_size = 600
        self.max_size = 1000
        
    def normalize_resize_image_and_boxes(self, image: torch.Tensor, bboxes: torch.Tensor):
        # Normalize
        mean = torch.as_tensor( self.image_mean, dtype=image.dtype, device=image.device)
        std = torch.as_tensor( self.image_std, dtype=image.dtype, device=image.device)
        image = (image - mean[:, None, None]) / std[:, None, None]
        ######
        
        # Resize such that lower dim is scaled to 600
        # but larger dim not more than 1000
        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])
        min_size = torch.min(im_shape).to(dtype=torch.float32)
        max_size = torch.max(im_shape).to(dtype=torch.float32)
        
        # A partir daqui fui no automatico
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
            align_corners=True
        )
        
        # Resize boxes
        # As the image was resized, the gt boxes need also to be resized
        if bboxes is not None:
            ratios = [
                torch.tensor(s, dtype= torch.float32, device=bboxes.device) / torch.tensor(s_orig, dtype= torch.float32, device=bboxes.device) 
                for s, s_orig in zip(image.shape[-2:], (h,w))
            ]
            
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
        old_shape = image.shape[-2:] # eg.: (375, 500)
        if self.training:
            # Normalize and resize boxes
            # image -> eg.: (1,3, 600, 800)
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
            # Transform boxes to original image dimension
            frcnn_output["boxes"] = transform_boxes_to_original_size(frcnn_output["boxes"], image.shape[-2:], old_shape)

        return rpn_output, frcnn_output
        