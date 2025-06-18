import torch
import torch.nn as nn
import torchvision
from typing import Tuple
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def apply_regression_pred_to_anchors_or_proposals(box_trasnform_pred: torch.Tensor, anchors_or_proposals: torch.Tensor) -> torch.Tensor:
    """Apply regression transformations in anchors to get proposal boxes or in proposals to get final box predictions 

    Args:
        box_trasnform_pred (torch.Tensor): (num_anchors_or_proposals, num_classes, 4)
        anchors_or_proposals (torch.Tensor): (num_anchors_or_proposals, 4)
        return: pred_boxes (num_anchors_or_proposals, num_classes, 4)
    """
    box_trasnform_pred = box_trasnform_pred.reshape(
        box_trasnform_pred.size(0), -1, 4
    )
    
    #get xs, cy, w, h from x1, y1, x2, y2 of anchors/proposals
    w = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0] # x2 - x1
    h = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1] # y2 - y1
    center_x = anchors_or_proposals[:, 0] + 0.5 * w # x1 + w/2
    center_y = anchors_or_proposals[:, 1] + 0.5 * h # y1 + h/2
    
    #Transformation predictions (tx, ty, tw, th)
    # all bellow have dimension (num_anchors_or_proposals, num_classes, 1) 
    tx = box_trasnform_pred[..., 0]  
    ty = box_trasnform_pred[..., 1]
    tw = box_trasnform_pred[..., 2]
    th = box_trasnform_pred[..., 3]
    
    # Applying trasnformations
    # P -> Anchor box
    # G -> Ground truth
    pred_center_x = tx * w[:, None] + center_x[:, None] # Gx = tx * Pw + Px
    pred_center_y = ty * h[:, None] + center_y[:, None] # Gy = ty * Ph + Py
    pred_w = torch.exp(tw) * w[:, None] # Gw = exp(tw) * Pw
    pred_h = torch.exp(th) * h[:, None] # Gh = exp(th) * Ph
    
    # Calculating regressed box coordinates
    # all bellow have dimension -> (num_anchors_or_proposals, num_classes, 1)
    pred_box_x1 = pred_center_x - 0.5 * pred_w
    pred_box_y1 = pred_center_y - 0.5 * pred_h
    pred_box_x2 = pred_center_x + 0.5 * pred_w
    pred_box_y2 = pred_center_y + 0.5 * pred_h
    
    # get predicted boxes
    # pred_boxes -> (num_anchors_or_proposals, num_classes, 4) 
    pred_boxes = torch.stack((pred_box_x1,
                              pred_box_y1,
                              pred_box_x2,
                              pred_box_y2), dim = 2)
    
    return pred_boxes
    

def clamp_boxes_to_image_boundary(boxes: torch.Tensor, image_shape) -> torch.Tensor:
    """filter boxes only inside image boundary 

    Args:
        boxes (torch.Tensor): proposal boxes
        image_shape (_type_): original image shape

    Returns:
        torch.Tensor: boxes within image
    """
    boxes_x1 = boxes[..., 0]
    boxes_y1 = boxes[..., 1]
    boxes_x2 = boxes[..., 2]
    boxes_y2 = boxes[..., 3]
    height, width = image_shape[-2:]
    
    boxes_x1 = boxes_x1.clamp(min=0, max=width)
    boxes_x2 = boxes_x2.clamp(min=0, max=width)
    boxes_y1 = boxes_y1.clamp(min=0, max=height)
    boxes_y2 = boxes_y2.clamp(min=0, max=height)
    
    boxes = torch.cat((
        boxes_x1[..., None],
        boxes_y1[..., None],
        boxes_x2[..., None],
        boxes_y2[..., None]
    ), dim=-1)
    
    return boxes

def filter_proposals(proposals: torch.Tensor, classification_scores: torch.Tensor, image_shape) -> Tuple[torch.Tensor, torch.Tensor]:
    """filter proposals based on following criterias
    1- top 10000 classification scores
    2- boxes inside image boundaries
    3- NMS with 0.7 threshold on objectness (with top 2000 classification scores)

    Args:
        proposals (torch.Tensor): proposals 
        classification_scores (torch.Tensor): classification scores for porposal
        image_shape (_type_): original image shape

    Returns:
        torch.Tensor: filtered proposals
    """
    #Pre NMS filtering
    classification_scores = classification_scores.reshape(-1) # flatten proposal classification scores
    classification_scores = torch.sigmoid(classification_scores)
    _, top_n_idx = classification_scores.topk(10000) # get top 10000 proposal based on classification scores (foreground or background)
    classification_scores = classification_scores[top_n_idx]
    proposals = proposals[top_n_idx] # filter only top 10000 proposals
    
    # Clamp boxes to image boundary
    proposals = clamp_boxes_to_image_boundary(proposals, image_shape)
    
    #NMS based on objectness
    keep_mask = torch.zeros_like(classification_scores, dtype=torch.bool) # mask with proposals to keep
    keep_indices = torchvision.ops.nms(proposals, classification_scores, 0.7) # apply nms with 0.7 threshold
    
    #sort keep_indices by classification scores
    post_nms_keep_indices = keep_indices[
        classification_scores[keep_indices].sort(descending=True)[1]
    ]
    
    # Post NMS topk = 2000 filtering
    proposals = proposals[post_nms_keep_indices[:2000]]
    classification_scores = classification_scores[post_nms_keep_indices[:2000]]
    
    return proposals, classification_scores

def get_iou(boxes1 :torch.Tensor, boxes2 :torch.Tensor) -> torch.Tensor:
    """from boxes1 (Nx4) and boxes2 (Mx4) compute IoU matrix between then (NxM)

    Args:
        boxes1 (torch.Tensor): (Nx4)
        boxes2 (torch.Tensor): (Mx4)

    Returns:
        torch.Tensor: IoU matrix of shape (NxM) of all boxes combinations
    """
    #Compute area of boxes (x2 - x1) * (y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    #Get top left x1, y1 in each possible box pair intersection
    x_left = torch.max(boxes1[:, None, 0], boxes2[:, None, 0]) # Get top left x (NxM)
    y_top = torch.max(boxes1[:, None, 1], boxes2[:, None, 1]) # Get top left y (NxM)
    
    #Get bottom right x2, y2 in each possible box pair intersection
    x_right = torch.min(boxes1[:, None, 2], boxes2[:, None, 2]) # Get bottom right x (NxM)
    y_bottom = torch.min(boxes1[:, None, 3], boxes2[:, None, 3]) # Get bottom right y (NxM)
    
    # Since intersection area cannot be negative, clamp(min=0) ensures that:
    # If boxes don't overlap → width/height becomes 0 → intersection area = 0
    # If boxes do overlap → positive width/height → correct intersection area
    intersection_area = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)
    union = area1[:,None] + area2 - intersection_area
    
    return intersection_area/union #(NxM)
    
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
            return:  Anchors boxes in the feature map
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
    
    def assign_targets_to_anchors(self, anchors: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        """Assigns ground truth boxes to anchor boxes for training object detectors.
    
        This function implements the anchor assignment strategy commonly used in object
        detection frameworks (e.g., Faster R-CNN). It matches each anchor box with the
        most suitable ground truth box based on IoU overlap, following these rules:
        
        1. High-quality matches (IoU >= 0.7): Assigned as positive/foreground
        2. Low-quality matches (IoU < 0.3): Assigned as negative/background  
        3. Medium-quality matches (0.3 <= IoU < 0.7): Ignored during training
        4. Special case: For each ground truth box, ensures at least one anchor is 
        assigned to it (even if IoU < 0.7) to guarantee all objects are detected
        
        Args:
            anchors (torch.Tensor): Anchor boxes for the image. Shape: (N, 4) where N is
                the number of anchors (e.g., 16650) and each row contains [x1, y1, x2, y2].
            gt_boxes (torch.Tensor): Ground truth bounding boxes. Shape: (M, 4) where M
                is the number of objects (e.g., 6) and each row contains [x1, y1, x2, y2].
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - labels (torch.Tensor): Shape (N,). Classification labels for each anchor:
                    * 1.0: Foreground/positive (has object)
                    * 0.0: Background/negative (no object)
                    * -1.0: Ignored (ambiguous, excluded from loss)
                - matched_gt_boxes (torch.Tensor): Shape (N, 4). Ground truth box 
                    coordinates assigned to each anchor. For background/ignored anchors,
                    these are placeholder values (copies of gt_box[0]) that won't be
                    used during training.
        
        Example:
            >>> anchors = torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40], ...])  # Shape: (16650, 4)
            >>> gt_boxes = torch.tensor([[15, 15, 25, 25], [35, 35, 45, 45], ...])  # Shape: (6, 4)
            >>> labels, matched_boxes = assign_targets_to_anchors(anchors, gt_boxes)
            >>> # labels[i] = 1.0 if anchor[i] matches an object (IoU >= 0.7)
            >>> # labels[i] = 0.0 if anchor[i] is background (IoU < 0.3)
            >>> # labels[i] = -1.0 if anchor[i] is ignored (0.3 <= IoU < 0.7)
        """
        
        # get IoU matrix with shape (gt_boxes, num_anchors)
        iou_matrix = get_iou(gt_boxes, iou_matrix) # eg -> 6 x 16650
        
        # For each anchor get best ground truth box index
        # Goes column by column (each column is one anchor)
        # Finds the maximum IoU value in that column
        # Records which row (gt_box) had that maximum
        # The matrix looks like:
        #          anchor0  anchor1  anchor2 ... anchor16649
        # gt_box0   0.1      0.3      0.05         0.2
        # gt_box1   0.7      0.2      0.15         0.1
        # gt_box2   0.2      0.8      0.03         0.4
        # gt_box3   0.05     0.1      0.9          0.3
        # gt_box4   0.15     0.05     0.2          0.6
        # gt_box5   0.3      0.15     0.1          0.15
        
        # best_match_iou: Tensor of shape (16650,) containing the highest IoU for each anchor
        # Example: [0.7, 0.8, 0.9, ..., 0.6]

        # best_match_gt_index: Tensor of shape (16650,) containing the index of the best matching gt_box
        # Example: [1, 2, 3, ..., 4]
        best_match_iou, best_match_gt_index = iou_matrix.max(dim=0)
        
        #This copy will be needed later to add low quality boxes 
        best_match_gt_idx_pre_threshold = best_match_gt_index.clone()
        
        bellow_low_threshold = best_match_iou < 0.3 # background anchors
        between_threshold = (best_match_iou >= 0.3) & (best_match_iou < 0.7) # ignored anchors
        
        best_match_gt_index[bellow_low_threshold] = -1 # -1 is backfround
        best_match_gt_index[between_threshold] = -2 # -2 is ignored
        
        # *LOW QUALITY POSITIVE anchor boxes process
        # Those low quality positive anchors are between 0.3 and 0.7 but have high threashold 
        #
        # for each gt box we are finding the maximum IoU value in which this gt box has with any anchor
        best_anchor_iou_for_gt, _ = iou_matrix.max(dim=1) # Get the best anchor for each gt_box -> (6,)
        
        # Get all the anchor boxes which have the same overlap value with ground truth i'th box
        # In a nutshell, we are getting ground truth box and anchor box pairs that have maximum IoU between them
        gt_pred_pair_with_highest_iou = torch.where(iou_matrix == best_anchor_iou_for_gt[:, None]) 
        
        # Get all box indexes which are low quality positive anchor canditades 
        # and update the ground truth box indexes of those
        pred_inds_to_update = gt_pred_pair_with_highest_iou[1]
        best_match_gt_index[pred_inds_to_update] = best_match_gt_idx_pre_threshold[pred_inds_to_update] # keep low quality positive anchors in best_match_gt_index
        
        # *Seting regression targets for the anchors 
        # due to clamp, all -1 or -2 anchors will be assigned as background (background is treated as 0'th anchor)
        # Example
        # If gt_boxes is:
        # [[10, 20, 30, 40],   # gt_box 0
        #  [50, 60, 70, 80],   # gt_box 1
        #  [90, 100, 110, 120], # gt_box 2
        #  ...]

        # And best_match_gt_index (after clamp) is [1, 0, 2, 0, 1, ...]

        # Then matched_gt_boxes will be:
        # [[50, 60, 70, 80],    # anchor 0 -> gt_box 1
        #  [10, 20, 30, 40],    # anchor 1 -> gt_box 0
        #  [90, 100, 110, 120], # anchor 2 -> gt_box 2
        #  [10, 20, 30, 40],    # anchor 3 -> gt_box 0 (was background)
        #  [50, 60, 70, 80],    # anchor 4 -> gt_box 1
        #  ...]                 # ... for all 16650 anchors
        matched_gt_boxes = gt_boxes[best_match_gt_index.clamp(min=0)] #  size 16650x4
        
        # set all foreground anchor labels as 1
        labels = best_match_gt_index >= 0
        labels = labels.to(dtype=torch.float32)
        
        # set all background labels as 0
        background_anchors = best_match_gt_index == -1
        labels[background_anchors] = 0.0
        
        # Set all ignored anchors labels as -1
        ignored_anchors = best_match_gt_index == -2
        labels[ignored_anchors] = -1.0
        
        # Later for classification we pick labels which are >= 0
        return labels, matched_gt_boxes 
        
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
        
        # reshaping classification_scores to be the same shape as anchors
        # classification_scores -> (batch, Numer of anchors per location, h_feat, w_feat)
        number_of_anchors_per_location = classification_scores.size(1)
        classification_scores = classification_scores.permute(0,2,3,1) # classification_scores -> (batch, h_feat, w_feat, Numer of anchors per location)
        classification_scores.reshape(-1, 1) # classification_scores -> (batch * h_feat * w_feat * Numer of anchors per location, 1)
        
        # reshaping box_transformation_preds scores to be the same shape as anchors
        # box_transformation_preds -> (Batch, Number of Anchors per location * 4, h_feat, w_feat)
        box_transformation_preds = box_transformation_preds.view(
            box_transformation_preds.size(0), #Batch
            number_of_anchors_per_location, # Number of Anchors per location
            4, # 4 predicted bbox_coordinates(p_tx, p_ty, p_tz, p_tw)
            rpn_feat_representation_map.shape[-2], # h_feat
            rpn_feat_representation_map.shape[-1] # w_feat
        ) # box_transformation_preds -> (Batch, Number of Anchors per location, 4, h_feat, w_feat)
        
        box_transformation_preds = box_transformation_preds.permute(0, 3, 4, 1, 2)
        box_transformation_preds = box_transformation_preds.reshape(-1, 4) # box_transformation_preds -> (Batch * Number of Anchors per location * h_feat * w_feat, 4)
        
        # Transforming generated anchors according to box_transform_pred
        proposals = apply_regression_pred_to_anchors_or_proposals(
            box_transformation_preds.detach().reshape(-1, 1, 4) # (Batch * Number of Anchors per location * h_feat * w_feat, 1, 4) where the 1 here is class (background or foregroung)
        )
        
        #Filtering the proposals
        proposals = proposals.reshape(proposals.size(0), 4) # (Batch * Number of Anchors per location * h_feat * w_feat, 4)
        proposals, scores = filter_proposals(proposals, classification_scores, image.shape)

        rpn_output = {
            "proposals": proposals, # eg: proposals -> (2000 x 4)
            "scores": scores # eg: scores -> (2000 x 4)
        }
        
        
        if not self.train or target is None:
            return rpn_output
        else:
        # In train mode, assign ground truth values to anchors and compute
            labels_for_anchors, matched_gt_boxes_for_anchors = self.assign_targets_to_anchors(
                anchors,
                target["bboxes"][0]
            ) 