import torch
import torchvision

from torch import nn
from typing import Tuple
from .rpn import get_iou, sample_positive_negative, boxes_to_transformation_targets, apply_regression_pred_to_anchors_or_proposals, clamp_boxes_to_image_boundary

class ROIHead(nn.Module):
    def __init__(self, num_classes = 21, in_channels = 512):
        """
        Initializes the Region of Interest (ROI) Head for a Faster R-CNN model.

        The ROIHead is responsible for taking region proposals from the Region Proposal Network (RPN)
        and refining them into final object detections by performing classification and bounding box regression.

        Args:
            num_classes (int, optional): The total number of object classes, including the background class.
                                         Defaults to 21.
            in_channels (int, optional): The number of input channels from the feature map, which is
                                         typically the output channel dimension of the backbone network
                                         (e.g., VGG). Defaults to 512.

        Attributes:
            num_classes (int): Stores the number of classes.
            pool_size (int): The target output size (width and height) of the ROI pooling layer.
                             This determines the fixed size of features extracted for each proposal.
                             A common value is 7, resulting in a 7x7 feature map.
            fc_inner_dim (int): The dimension of the inner fully connected layers (fc6 and fc7).
                                 These layers process the pooled features before the final
                                 classification and regression layers.

            fc6 (nn.Linear): The first fully connected layer.
                             Input features: `in_channels * pool_size * pool_size` (flattened pooled features).
                             Output features: `fc_inner_dim`.
            fc7 (nn.Linear): The second fully connected layer.
                             Input features: `fc_inner_dim`.
                             Output features: `fc_inner_dim`.
            cls_layer (nn.Linear): The classification layer.
                                   Input features: `fc_inner_dim`.
                                   Output features: `num_classes` (scores for each class, including background).
            bbox_reg_layer (nn.Linear): The bounding box regression layer.
                                        Input features: `fc_inner_dim`.
                                        Output features: `num_classes * 4` (4 regression offsets for each class).
        """
        super(ROIHead, self).__init__()
        self.num_classes = num_classes
        self.pool_size = 7 # output size of pooling layer
        self.fc_inner_dim = 1024 # dimension of FC layers
        
        #1st fc after pooling
        self.fc6 = nn.Linear(
            in_channels*self.pool_size*self.pool_size, # input example (512*7*7, )   
            self.fc_inner_dim 
        )
        
        #2nd fc after pooling
        self.fc7 = nn.Linear(
            self.fc_inner_dim,  
            self.fc_inner_dim 
        )
        
        # classification layer
        self.cls_layer = nn.Linear(
            self.fc_inner_dim,
            self.num_classes
        )       
        
        #bbox regression layer
        self.bbox_reg_layer = nn.Linear(
            self.fc_inner_dim, 
            self.num_classes * 4 
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
    
    def filter_predictions(self, pred_boxes: torch.Tensor, pred_scores: torch.Tensor, pred_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:    
        """Refines raw predictions by applying a series of filtering steps.

        This function processes the output of the detection head to produce a clean,
        final set of bounding boxes. It filters out low-confidence and small boxes,
        removes redundant overlapping boxes using class-wise Non-Maximum Suppression (NMS),
        and returns only the top-scoring predictions.

        Algorithm:
        1.  **Score Thresholding:** Removes all predictions with a confidence score
            below a fixed threshold (0.05).
        2.  **Size Filtering:** Discards boxes that are smaller than a minimum
            size (1x1 pixels).
        3.  **Class-wise NMS:** For each class independently, it applies NMS
            with an IoU threshold of 0.5. This eliminates redundant, overlapping
            boxes for the same object, keeping only the one with the highest score.
        4.  **Sorting:** Sorts all surviving boxes from all classes by their
            confidence score in descending order.
        5.  **Top-K Selection:** Returns the top 100 highest-scoring predictions
            across all classes for the final output.

        Args:
            pred_boxes (torch.Tensor): The predicted bounding boxes for all classes.
                Shape: (C * P, 4), where C is the number of classes and P is the
                number of proposals. Format: [x1, y1, x2, y2].
            pred_scores (torch.Tensor): The confidence scores for each predicted box.
                Shape: (C * P,).
            pred_labels (torch.Tensor): The class label for each predicted box.
                Shape: (C * P,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the
            filtered and refined boxes, scores, and labels.
                - boxes (torch.Tensor): Shape (K, 4), where K <= 100.
                - scores (torch.Tensor): Shape (K,).
                - labels (torch.Tensor): Shape (K,).
        """
        #remove low scoring boxes (less than 0.05)
        keep = torch.where(pred_scores > 0.05)[0]
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]
        
        # remove small boxes (keep boxes with height and width greater than 1)
        min_size = 1
        ws, hs = pred_boxes[:, 2] - pred_boxes[:, 0], pred_boxes[:, 3] - pred_boxes[:, 1] 
        keep = torch.where((ws >= min_size) & (hs <= 1))[0]
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]

        # Class wise NMS
        #1) for each class we pick boxes with that class label
        #2) call nms (threshold of 0.5)
        #3) keep only boxes which are filtered by nms  
        keep_mask = torch.zeros_like(pred_scores, dtype=torch.bool)
        for class_id in torch.unique(pred_labels):
            curr_indices = torch.where(pred_labels == class_id)[0]
            curr_keep_indices = torchvision.ops.nms(
                pred_boxes[curr_indices],
                pred_scores[curr_indices],
                0.5
            )
            keep_mask[curr_indices[curr_keep_indices]] = True
        keep_indices = torch.where(keep_mask)[0]   
        
        # Sort boxes based on their classification scores
        post_nms_keep_indices = keep_indices[pred_scores[keep_indices].sort(
            descending=True
            )[1]]
        
        #return only top 100 detections for each image
        keep = post_nms_keep_indices[:100]
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]
        return pred_boxes, pred_scores, pred_labels

    def forward(self, feature_map: torch.Tensor, proposals: torch.Tensor, image_shape: Tuple, target):
        """Processes region proposals to produce final object detections or training losses.

        This method acts as the second stage of the Faster R-CNN detector. It takes the
        feature map from the backbone and region proposals from the RPN. Depending on
        whether the model is in training or inference mode, its behavior changes.

        TRAINING
            - Assign ground truth boxes to proposals
            - Sample positive and negative proposals
            - Get classification and regression targets for proposals
            - ROI pooling to get proposal features
            - Call classification and regression layers
            - Compute classification and localization loss
            
        INFERENCE
            - ROI pooling to get proposal features
            - Call classification and regression layers
            - Convert proposals to predictions using box transformation prediction
            - Filter boxes

        Args:
            feature_map (torch.Tensor): The feature map from the backbone network.
                Example shape: `(1, 512, 37, 50)`.
            proposals (torch.Tensor): The region proposals generated by the RPN.
                Example shape: `(2000, 4)`.
            image_shape (Tuple): A tuple representing the original image dimensions
                as `(height, width)`.
            target (Dict): A dictionary containing ground truth information, used only
                during training. It contains:
                - `labels` (torch.Tensor): Ground truth labels for each object.
                - `bboxes` (torch.Tensor): Ground truth bounding boxes.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing either losses (during training)
            or final predictions (during inference).
            - During training: `{"frcnn_classification_loss": Tensor, "frcnn_localization_loss": Tensor}`
            - During inference: `{"boxes": Tensor, "scores": Tensor, "labels": Tensor}`
        """
        if self.training and target is not None:
            gt_boxes = target["bboxes"][0] # shape example (6, 4)
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
            feature_map, # eg.: (1, 512, 37, 50)
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
        box_transform_pred = self.bbox_reg_layer(box_fc_7) # box_transform_pred -> (number_proposals, num_classes * 4) eg.: (128, 21 * 4)
        
        num_boxes, num_classes = cls_scores.shape
        box_transform_pred = box_transform_pred.reshape(num_boxes, num_classes, 4) # Reshape box transformation predictions to (number_proposals, number_classes, 4)
        
        frcnn_output = {}
        if self.training and target is not None:
            #Computing classification loss
            classification_loss = torch.nn.functional.cross_entropy(
                cls_scores,
                labels
            )
            
            #Computing localization loss only for foreground proposals
            fg_proposals_idxs = torch.where(labels > 0)[0] # get only foreground proposals
            fg_class_labels = labels[fg_proposals_idxs] # get foreground class labels 
            localization_loss = torch.nn.functional.smooth_l1_loss(
                box_transform_pred[fg_proposals_idxs, fg_class_labels],
                regression_targets[fg_proposals_idxs],
                beta=1/9,
                reduction='sum'
            )
            localization_loss = localization_loss/labels.numel() # normalizing localization loss
            
            frcnn_output["frcnn_classification_loss"] = classification_loss
            frcnn_output["frcnn_localization_loss"] = localization_loss
            return frcnn_output
        else:
            # Apply transformation predictions to proposals
            # pred_boxes -> (number_proposals, number_classes, 4) eg.: (128, 21, 4)
            pred_boxes = apply_regression_pred_to_anchors_or_proposals(
                box_transform_pred,
                proposals
            )
            
            # pred_scores -> (number_proposals, number_classes) eg.: (128, 21)
            pred_scores = torch.nn.functional.softmax(cls_scores, dim=-1) # get predicted classification scores probabilities
            
            # Filtering proposals as predictions boxes
            pred_boxes = clamp_boxes_to_image_boundary(pred_boxes, image_shape) # Clamp boxes to be inside image boundary

            # Create labels for each prediction (class indexes)
            pred_labels = torch.arange(num_classes, device=cls_scores.device) # 0..21
            pred_labels = pred_labels.view(1, -1).expand_as(pred_scores) # expand pred_labels to be the same shape as pred_scores (number_proposals, number_classes) eg.:  (128, 21)
            
            # remove background class predictions
            pred_boxes = pred_boxes[:, 1:] # pred_boxes -> (num_proposals, num_classes-1, 4) eg.: (128, 20, 4)
            pred_scores = pred_scores[:, 1:] # pred_scores -> (num_proposals, num_classes-1) eg.: (128, 20)
            pred_labels = pred_labels[:, 1:] # pred_labels -> (num_proposals, num_classes-1) eg.: (128, 20)
            
            # Each of 128 proposals have prediction boxes for all classes. 
            # That is the reason why we have pred_boxes as dim (num_proposals, num_classes-1, 4) 
            # So instead of having, for example, 128 prediction boxes, we would have 128 * 20 = 2560 boxes
            pred_boxes = pred_boxes.reshape(-1, 4) # pred_boxes -> (num_proposals * (num_classes-1), 4) eg.: (2560, 4)
            pred_scores = pred_scores.reshape(-1) # pred_scores -> (num_proposals * (num_classes-1), ) eg.: (2560, )
            pred_labels = pred_labels.reshape(-1) # pred_labels -> (num_proposals * (num_classes-1), ) eg.: (2560, )
            
            pred_boxes, pred_scores, pred_labels  = self.filter_predictions(pred_boxes, pred_scores, pred_labels )
            frcnn_output["boxes"] = pred_boxes
            frcnn_output["scores"] = pred_scores
            frcnn_output["labels"] = pred_labels
            
            return frcnn_output
            