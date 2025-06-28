import torch
import torchvision

from torch import nn
from typing import Tuple
from .rpn import get_iou, sample_positive_negative, boxes_to_transformation_targets

class ROIHead(nn.Module):
    def __init__(self, num_classes = 21, in_channels = 512):
        super(ROIHead, self).__init__()
        self.num_classes = num_classes
        self.pool_size = 7 # output size of pooling layer
        self.fc_inner_dim = 1024 # dimension of FC layers
        
        #1st fc after pooling
        self.fc6 = nn.Linear(
            in_channels*self.pool_size*self.pool_size, # input example (512*7*7, )   
            self.fc_inner_dim # output example (1024, )
        )
        
        #2nd fc after pooling
        self.fc7 = nn.Linear(
            self.fc_inner_dim, # input example (1024, )  
            self.fc_inner_dim # input example (1024, )
        )
        
        # classification layer
        self.cls_layer = nn.Linear(
            self.fc_inner_dim,
            self.num_classes
        )       
        
        #bbox regression layer
        self.bbox_reg_layer = nn.Linear(
            self.fc_inner_dim, # (1024, )
            self.num_classes * 4 # (21*4 ,)
        )
        
    def assign_target_to_proposals(self, proposals: torch.Tensor, gt_boxes: torch.Tensor, gt_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:        
        """Assign ground truth boxes and labels to region proposals based on IoU overlap.
    
        This function is a critical component of the training process for object detection models.
        It determines which proposals correspond to actual objects (positive samples) and which
        are background (negative samples) by computing IoU overlap with ground truth boxes.
        
        Algorithm:
        1. Compute IoU matrix between all proposals and ground truth boxes
        2. For each proposal, find the ground truth box with highest IoU
        3. Apply IoU threshold (0.5) to classify proposals as foreground/background
        4. Assign corresponding labels and target boxes for regression
        
        Args:
            proposals (torch.Tensor): Region proposals from RPN 
                Shape: (N, 4) where N is number of proposals (e.g., 2000)
                Format: [x1, y1, x2, y2] in absolute coordinates
                
            gt_boxes (torch.Tensor): Ground truth bounding boxes
                Shape: (M, 4) where M is number of objects in image (e.g., 6)
                Format: [x1, y1, x2, y2] in absolute coordinates
                
            gt_labels (torch.Tensor): Ground truth class labels  
                Shape: (M,) where M matches number of gt_boxes
                Values: Class indices (1-based, where 0 reserved for background)
        
        Returns:
            tuple: (labels, matched_gt_boxes_for_proposals)
            
            labels (torch.Tensor): Assigned class labels for each proposal
                Shape: (N,) matching number of proposals
                Values: 
                    - 0 for background proposals (IoU < 0.5)
                    - ith-num_class for foreground proposals (IoU >= 0.5)
                    
            matched_gt_boxes_for_proposals (torch.Tensor): Target boxes for regression
                Shape: (N, 4) matching number of proposals  
                Contains: Ground truth box coordinates for each proposal
                Note: Background proposals get arbitrary gt box (not used in loss)
        
        IoU Threshold Logic:
            - IoU >= 0.5: Proposal considered positive (foreground)
            - IoU < 0.5: Proposal considered negative (background, label=0)
            
        Training Usage:
            - Labels used for classification loss (cross-entropy)
            - Matched gt boxes used for bbox regression loss (only for positive samples)
            - Background proposals (label=0) don't contribute to regression loss
        """
        iou_matrix = get_iou(gt_boxes, proposals) # compute IoU bettwen proposals and gt
        best_match_iou, best_match_gt_idx = iou_matrix.max(dim = 0) # get the gt box with best IoU in respect to all proposals
            
        below_low_threshold = best_match_iou < 0.5 # mask with proposals that have IoU below 0.5
        
        best_match_gt_idx[below_low_threshold] = -1 # background proposals (IoU less than threshold) have -1 index
        matched_gt_boxes_for_proposals = gt_boxes[best_match_gt_idx.clamp(min = 0)] # assign a gt box to each proposal based in IoU
        
        labels = gt_labels[best_match_gt_idx.clamp(min=0)]# assing a gt label to each proposal based in IoU 
        labels = labels.to(dtype=torch.int64)
        
        # the  for background proposals is 0
        background_proposal = best_match_gt_idx == -1
        labels[background_proposal] = 0
        
        return labels, matched_gt_boxes_for_proposals # both have shape (2000,) and (2000, 4)
        
    def forward(self, feature_map: torch.Tensor, proposals: torch.Tensor, image_shape: Tuple, target):
        """

        Args:
            feature_map (torch.Tensor): feature map that came from image feature extractor (1x512x37x50)
            proposals (torch.Tensor): proposals that came from rpn (2000x4)
            image_shape (Tuple): original image shape
            target (Dict): gt for training
                target["labels"] - gt labels (1x6)
                target["bboxes"] - gt bbox (1x6x4)
        """
        if self.training and target is not None:
            gt_boxes = target["bboxes"][0] # shape example (6x4)
            gt_labels = target["labels"][0] # shape example (6,)

            # Assign labels and gt boxes to proposals
            labels, matched_gt_boxes_for_proposals = self.assign_target_to_proposals(proposals, gt_boxes, gt_labels)

            # Sample positive and negative proposals
            sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_negative(labels, positive_count=32, total_count=128)
            sampled_idxs = torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)[0] # union of positive and negative samples
            
            # Get only the sampled proposals for training
            proposals = proposals[sampled_idxs]
            labels = labels[sampled_idxs] # labels -> (128,)
            matched_gt_boxes_for_proposals = matched_gt_boxes_for_proposals[sampled_idxs] # matched_gt_boxes_for_proposals -> (128, 4)
            
            # get transformation targets (tx, ty, tw, th)
            # regression_targets -> (sampled_training_proposals, 4) eg.: (128, 4)
            regression_targets = boxes_to_transformation_targets(matched_gt_boxes_for_proposals, proposals)
        
        # ROI pooling
        # spatial scale for ROI pooling (how much downscaled the feature map is related to image)
        # for vgg this is 1/16
        spatial_scale = 0.0625
        
        # proposal_roi_pool_feats -> (number_proposals, 512, 7, 7)
        proposal_roi_pool_feats = torchvision.ops.roi_pool(
            feature_map, # (1, 512, 37, 50)
            [proposals], # (number_proposals, 4)
            output_size=self.pool_size, # 7
            spatial_scale=spatial_scale
        )
        
        # flatten starting by 1st dimension therefore the proposal_roi_pool_feats shape will be (number_proposals, 512*7*7)
        proposal_roi_pool_feats = proposal_roi_pool_feats.flatten(start_dim=1)
        
        #FC layers 
        box_fc_6 = torch.nn.functional.relu(self.fc6(proposal_roi_pool_feats))
        box_fc_7 = torch.nn.functional.relu(self.fc7(box_fc_6))
        
        # Classification layer
        cls_scores = self.cls_layer(box_fc_7) # cls_scores -> (number_proposals, num_classes) eg.: (128, 21)
        
        # Bbox regression layer
        box_transform_pred = self.bbox_reg_layer(box_fc_7) # box_transform_pred -> (number_proposals, num_classes * 4) eg.: (128, 84)
        
    
        num_boxes, num_classes = cls_scores.shape
        box_transform_pred = box_transform_pred.reshape(num_boxes, num_classes, 4) # Reshape box transformation predictions to (number_proposals, number_classes, 4)
        
        frcnn_output = {}
        if self.training and target is not None:
            #Computing classification loss
            classification_loss = torch.nn.functional.cross_entropy(
                cls_scores,
                labels
            )
            
            