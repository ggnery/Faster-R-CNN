import torch
import torch.nn as nn
import torchvision
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512):
        """
        Args:
            in_channels (int, optional): in_channels = input channels present in feature map. Defaults to 512.
        """
        
        super(RegionProposalNetwork, self).__init__()
        self.scales = [128, 256, 512] # scales/areas for anchor boxes in feature map (128^2, 256^2 and 512^2)
        self.aspect_ratios = [0.5, 1, 2] # aspect_ratios for anchor boxes in feature map (1:2, 1:1, 2:1)
        
        self.num_anchors = len(self.scales, self.aspect_ratios) #Each feature map cell will have 3x3 = 9 anchor boxes  
        
        # 3x3 conv to get feature representation map
        self.rpn_conv = nn.Conv2d(in_channels=in_channels, 
                                  out_channels=in_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
        
        # 1x1 classification layer to get classification scores (background or foreground)
        self.classification_layer = nn.Conv2d(in_channels=in_channels,
                                   out_channels=self.num_anchors, # K = number of anchors for each feature map cell (p0fg, p1fg, ..., pKfg)
                                   kernel_size=1,
                                   stride=1)
        #Why not 2*K? It is fine to have just one value representing the foreground probability  
        
        # 1x1 regression layer to get proposal transformations (tx, ty, th, tw)
        self.bbox_regressor_layer = nn.Conv2d(in_channels=in_channels,
                                        out_channels=self.num_anchors * 4,# (p0_tx, p0_ty, p0_th, p0_tw, ..., pk_tx, pk_ty, pk_th, pk_tw)
                                        kernel_size=1,
                                        stride=1) 
    
    def generate_anchors(self, image: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:        
        """Generate all anchors boxes in the image.\n
        The base anchors will be zero centered and are blueprint of the defined "scales" and "aspect_ratios" (in our case 9 anchor boxes around base anchors).\n
        The base anchors will be multiplicated, accordingly to scale and aspect ratios, and shifted using feature map stride to crate anchors for entire image.\n
        Args:
            image (Tensor): original image (eg: 1x3x600x800)
            feat (Tensor): feature map (eg: 1x512x37x50)
        """
        grid_h, grid_w = feature_map.shape[-2:] # feature map height and width
        image_h, image_w = image.shape[-2:] # image height and width
        
        #Calculating strides in h and w dimensions
        stride_h = torch.tensor(image_h//grid_h, dtype= torch.int64, device=feature_map.device) # stride_h is a ratio between image_h and grid_h
        stride_w = torch.tensor(image_w//grid_w, dtype= torch.int64, device=feature_map.device) # stride_w is a ratio between image_w and grid_w
        
        scales = torch.tensor(self.scales, dtype=feature_map.dtype, device=feature_map.device)
        aspect_ratios = torch.tensor(self.aspect_ratios, dtype=feature_map.dtype, device=feature_map.device)
        
        #Getting anchor boxes of area 1 (h*w = 1)
        #The bellow code is only valid if h/w = aspect_ratios. Therefore, h = sqrt(aspect_ratios) and w = 1/h
        h = torch.sqrt(aspect_ratios) # height of unit area anchor boxes
        w = 1/h # width of unit area anchor boxes
        
        # Getting 1 area anchors with different scales
        ws =(w[:, None] * scales[None, :]).view(-1) # get width of unit area anchors in all different scales (eg: 3x3 where lines are aspect ratios and columns are scales)
        hs =(h[:, None] * scales[None, :]).view(-1) # get height of unit area anchors in all different scales (eg: 3x3 where lines are aspect ratios and columns are scales)
        
        # calculate width and height of all 9 anchor boxes 
        # All those boxes are centered in the same point (0,0)
        # eg: shape 9x4 where each line is an anchor and each colums is an edge
        # [-91., -45., 91., 45.]
        # [-181., -91., 181., 91.]
        # [-362., -182., 362., 181.]
        # [-64., -64., 64., 64.]
        # [-128., -128., 128., 128.]
        # [-256., -256, 256., 256.]
        # [-45., -91., 45., 91.]
        # [-91., -181. 91., 181.]
        # [-181., -362., 181., 362.]
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1)/2  
        base_anchors =base_anchors.round()
        
        # Get shifts of (0,0) anchors to all locations in feature map in x axis -> (0, 1, ..., w_feat-1) * stride_w
        # obs: we shift the anchors in feature map proportionally to the original image sizes, this explains the stride_w
        shifts_x = torch.arange(0, grid_w, dtype=torch.int32, device=feature_map.device) * stride_w
    
        # Get shifts of (0,0) anchors to all locations in feature map in y axis -> (0, 1, ..., h_feat-1) * stride_h
        # obs: we shift the anchors in feature map proportionally to the original image sizes, this explains the stride_h
        shifts_y = torch.arange(0, grid_h, dtype=torch.int32, device=feature_map.device) * stride_h

        # Do cartesian product between shifts to get anchor coordinates in feature map
        # torch.meshgrid reference: https://docs.pytorch.org/docs/stable/generated/torch.meshgrid.html
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

        #flatten shifts coordinates
        shifts_y = shifts_y.reshape(-1)
        shifts_x = shifts_x.reshape(-1)
        
        # shift is 4 dimesion because the shift is applied in the 4 points of the box
        #                   bottom_left top_left bottom_right top_right
        shifts = torch.stack([shifts_x, shifts_y, shifts_x,  shifts_y], dim=1) # shifts -> (h_feat * w_feat, 4)
        
        # Adding shifts to original (0,0) base anchors and creating all anchor boxes to entire image
        # base_anchors -> (num_anchors_per_location, 4) eg: 9x4
        anchors = (shifts.view(-1,1,4) + base_anchors.view(1,-1,4)) # anchors -> (h_feat * w_feat, num_anchors_per_location, 4) 
        anchors = anchors.reshape(-1, 4) # anchors -> (h_feat * w_feat * num_anchors_per_location, 4) eg: 16650x4
        
        #Note: In this implementation, the center of anchors is not in the middle of the cell, instead they are in top left corner of each cell
        return anchors
        
    def forward(self, image, feature_map, target):
        """Forward method for RPN

        Args:
            image (Tensor): original image (eg: 1x3x600x800)
            feature_map (Tensor): feature map (eg: 1x512x37x50)
            target (Dict[str, Tensor]): Dict with "bboxes" and "labels" keys (eg: target["bboxes"] - 1x6x4  target["labels"]- 1x6)
        """
        rpn_feat_representation_map = nn.ReLU()(self.rpn_conv(feature_map)) # Apply ReLu activation function in the 3x3 conv opertation 
        classification_scores = self.classification_layer(rpn_feat_representation_map) # Apply 1x1 conv in feature representation map to get anchor scores (eg: 1x9x37x50)
        box_transformation_preds = self.bbox_regressor_layer(rpn_feat_representation_map) # Apply 1x1 conv in feature representation map to get anchor transformation values (eg: 1x36x37x50)
        
        #Generate anchors
        anchors = self.generate_anchors(image, feature_map)