import torch
import math
from typing import Tuple

def apply_regression_pred_to_anchors_or_proposals(box_transform_pred: torch.Tensor, anchors_or_proposals: torch.Tensor) -> torch.Tensor:
    """Applies predicted regression transformations to reference boxes to generate refined bounding boxes.
    
    This function implements the standard bounding box regression used in object detection
    models like Faster R-CNN and RetinaNet. It converts relative transformations (deltas)
    into absolute box coordinates using a parameterization that is scale-invariant and
    more stable for training than direct coordinate regression.
    
    The transformation follows the equations from the original Faster R-CNN paper:
    - Given anchor/proposal P = (Px, Py, Pw, Ph) in center-size format
    - Given ground truth G = (Gx, Gy, Gw, Gh) in center-size format
    - The network learns to predict transformations t = (tx, ty, tw, th) where:
        * tx = (Gx - Px) / Pw  (normalized x-offset)
        * ty = (Gy - Py) / Ph  (normalized y-offset)
        * tw = log(Gw / Pw)    (log-space width scaling)
        * th = log(Gh / Ph)    (log-space height scaling)
    - This function applies the inverse transformation to get predicted boxes
    
    Args:
        box_transform_pred (torch.Tensor): Predicted regression deltas from the network.
            Shape: (N, K*4) or (N, K, 4) where:
            - N = number of anchors/proposals
            - K = number of classes (1 for RPN binary classification, C for RCNN)
            - 4 = regression parameters (tx, ty, tw, th) per class
            The function automatically reshapes to (N, K, 4) if needed.
            
        anchors_or_proposals (torch.Tensor): Reference boxes to transform.
            Shape: (N, 4) where each row contains [x1, y1, x2, y2] in corner format.
            For RPN: these are anchor boxes
            For RCNN: these are region proposals from RPN
    
    Returns:
        torch.Tensor: Predicted bounding boxes after applying transformations.
            Shape: (N, K, 4) where each box is [x1, y1, x2, y2] in corner format.
            Each anchor/proposal gets K predicted boxes (one per class).
    
    Mathematical Details:
        Given predictions (tx, ty, tw, th) and reference box (Px, Py, Pw, Ph):
        1. Predicted center: 
           - Gx = tx * Pw + Px
           - Gy = ty * Ph + Py
        2. Predicted size:
           - Gw = exp(tw) * Pw
           - Gh = exp(th) * Ph
        3. Convert to corners:
           - x1 = Gx - Gw/2, y1 = Gy - Gh/2
           - x2 = Gx + Gw/2, y2 = Gy + Gh/2
     
    """
    box_transform_pred = box_transform_pred.reshape(
        box_transform_pred.size(0), -1, 4
    )
    
    #get xs, cy, w, h from x1, y1, x2, y2 of anchors/proposals
    w = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0] # x2 - x1
    h = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1] # y2 - y1
    center_x = anchors_or_proposals[:, 0] + 0.5 * w # x1 + w/2
    center_y = anchors_or_proposals[:, 1] + 0.5 * h # y1 + h/2
    
    #Transformation predictions (tx, ty, tw, th)
    # all bellow have dimension (num_anchors_or_proposals, num_classes, 1) 
    tx = box_transform_pred[..., 0]  
    ty = box_transform_pred[..., 1]
    tw = box_transform_pred[..., 2]
    th = box_transform_pred[..., 3]
    
    tw = torch.clamp(tw, max=math.log(1000.0 / 16))
    th = torch.clamp(th, max=math.log(1000.0 / 16))
    
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
    """Clips bounding box coordinates to ensure they lie within image boundaries.
    
    This function performs coordinate clipping to handle boxes that extend beyond
    the image boundaries, which commonly occurs after bounding box regression or
    data augmentation. It ensures all box coordinates are valid for downstream
    processing without removing any boxes.
    
    Args:
        boxes (torch.Tensor): Bounding boxes to clip. Shape: (..., 4) where the last
            dimension contains [x1, y1, x2, y2] in corner format. The function handles
            arbitrary batch dimensions, e.g.:
            - (N, 4) for simple box lists
            - (N, K, 4) for K boxes per sample
            - (B, N, K, 4) for batched multi-class predictions
            
        image_shape (Tuple[int, ...]): Image dimensions. The last two elements must be
            (height, width). Common formats:
            - (H, W) for grayscale images
            - (C, H, W) for single images
            - (B, C, H, W) for batched images
            The function only uses the last two dimensions.
    
    Returns:
        torch.Tensor: Clipped boxes with the same shape as input. All coordinates
            are guaranteed to satisfy:
            - 0 <= x1, x2 <= width
            - 0 <= y1, y2 <= height
            
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

def get_iou(boxes1 :torch.Tensor, boxes2 :torch.Tensor) -> torch.Tensor:
    """Computes the Intersection over Union (IoU) between all pairs of bounding boxes.
    
    IoU is a fundamental metric in object detection that measures the overlap between
    two bounding boxes. It's defined as:
        IoU = Area of Intersection / Area of Union
    
    Values range from 0 (no overlap) to 1 (perfect overlap). Common thresholds:
    - IoU > 0.5: Generally considered a "good" match
    - IoU > 0.7: High-quality match (often used for positive training samples)
    - IoU < 0.3: Poor match (often used for negative training samples)
    
    This function efficiently computes IoU for all possible pairs between two sets
    of boxes using broadcasting, avoiding explicit loops.
    
    Args:
        boxes1 (torch.Tensor): First set of bounding boxes. Shape: (N, 4)
            Each row contains [x1, y1, x2, y2] where:
            - (x1, y1): top-left corner coordinates
            - (x2, y2): bottom-right corner coordinates
            - Assumes x2 > x1 and y2 > y1
        boxes2 (torch.Tensor): Second set of bounding boxes. Shape: (M, 4)
            Same format as boxes1.
    
    Returns:
        torch.Tensor: IoU matrix of shape (N, M) where element [i, j] contains
            the IoU between boxes1[i] and boxes2[j]. Values are in range [0, 1].
    
    Algorithm:
        1. Compute areas of all boxes in both sets
        2. Find intersection rectangle for each pair:
           - Left edge: max(box1_left, box2_left)
           - Top edge: max(box1_top, box2_top)
           - Right edge: min(box1_right, box2_right)
           - Bottom edge: min(box1_bottom, box2_bottom)
        3. Compute intersection area (0 if boxes don't overlap)
        4. Compute union area: area1 + area2 - intersection
        5. Return intersection / union
    
    """
    #Compute area of boxes (x2 - x1) * (y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    #Get top left x1, y1 in each possible box pair intersection
    x_left = torch.max(boxes1[:, None, 0], boxes2[:, 0]) # Get top left x (NxM)
    y_top = torch.max(boxes1[:, None, 1], boxes2[:, 1]) # Get top left y (NxM)
    
    #Get bottom right x2, y2 in each possible box pair intersection
    x_right = torch.min(boxes1[:, None, 2], boxes2[:, 2]) # Get bottom right x (NxM)
    y_bottom = torch.min(boxes1[:, None, 3], boxes2[:, 3]) # Get bottom right y (NxM)
    
    # Since intersection area cannot be negative, clamp(min=0) ensures that:
    # If boxes don't overlap → width/height becomes 0 → intersection area = 0
    # If boxes do overlap → positive width/height → correct intersection area
    intersection_area = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)
    union = area1[:,None] + area2 - intersection_area
    
    return intersection_area/union #(NxM)

def boxes_to_transformation_targets(ground_truth_boxes: torch.Tensor, anchors_or_proposals: torch.Tensor) -> torch.Tensor:  
    """Converts ground truth boxes and reference boxes into regression targets for training.
    
    This function implements the inverse transformation of `apply_regression_pred_to_anchors_or_proposals`.
    It computes the regression targets (tx, ty, tw, th) that the network should learn to predict
    in order to transform anchor/proposal boxes into ground truth boxes. This parameterization
    is crucial for stable training and good performance in object detection models.
    
    The transformation targets are computed as:
    - tx = (Gx - Px) / Pw  (normalized x-center offset)
    - ty = (Gy - Py) / Ph  (normalized y-center offset)  
    - tw = log(Gw / Pw)    (log-space width ratio)
    - th = log(Gh / Ph)    (log-space height ratio)
    
    Where:
    - P = (Px, Py, Pw, Ph): anchor/proposal center and size
    - G = (Gx, Gy, Gw, Gh): ground truth center and size
    
    Args:
        ground_truth_boxes (torch.Tensor): Target boxes to regress to. Shape: (N, 4)
            where each row contains [x1, y1, x2, y2] in corner format.
            In RPN training, these are the matched GT boxes for each anchor.
            
        anchors_or_proposals (torch.Tensor): Reference boxes to transform from. Shape: (N, 4)
            where each row contains [x1, y1, x2, y2] in corner format.
            Must have the same number of boxes as ground_truth_boxes.
    
    Returns:
        torch.Tensor: Regression targets for training. Shape: (N, 4)
            Each row contains [tx, ty, tw, th] representing the transformations
            needed to convert the corresponding anchor/proposal to its ground truth.
    
    """
    #get center x, center y, w, h from anchors (x1, y1, x2, y2) coordinates
    widths = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    heights = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
    center_x = anchors_or_proposals[:, 0] + 0.5 * widths
    center_y = anchors_or_proposals[:, 1] + 0.5 * heights
    
    #get center x, center y, w, h from gt boxes (x1, y1, x2, y2) coordinates
    gt_widths = ground_truth_boxes[:, 2] - ground_truth_boxes[:, 0]
    gt_heights = ground_truth_boxes[:, 3] - ground_truth_boxes[:, 1]
    gt_center_x = ground_truth_boxes[:, 0] + 0.5 * gt_widths
    gt_center_y = ground_truth_boxes[:, 1] + 0.5 * gt_heights
    
    target_tx = (gt_center_x - center_x) / widths # tx = (Gx - Px) / Pw
    target_ty = (gt_center_y - center_y) / heights # ty = (Gy - Py) / Ph
    target_tw = torch.log(gt_widths / widths) # tw = log(Gw/Pw)
    target_th = torch.log(gt_heights / heights) # th = log(Gh/Ph)
    
    regression_targets = torch.stack((
        target_tx,
        target_ty,
        target_tw,
        target_th
    ), dim=1)
    return regression_targets
def sample_positive_negative(labels: torch.Tensor, positive_count: int, total_count: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Samples a balanced subset of positive and negative anchors for training.
    
    This function implements the sampling strategy used in Faster R-CNN to address the
    extreme class imbalance in object detection. Since most anchors are background
    (negative), training on all anchors would bias the model. Instead, we sample a
    small, balanced subset for each training batch.
    
    The sampling follows these rules:
    1. Sample up to `positive_count` positive anchors (foreground)
    2. Sample remaining slots with negative anchors (background)  
    3. If fewer positive anchors exist than `positive_count`, sample more negatives
    4. Random sampling ensures different anchors are seen across iterations
    
    Args:
        labels (torch.Tensor): Classification labels for all anchors. Shape: (N,)
            where N is the total number of anchors (e.g., 16650). Values:
            - 1.0: Positive/foreground anchor (matched to object)
            - 0.0: Negative/background anchor (no object)
            - -1.0: Ignored anchor (not used in training)
            
        positive_count (int): Maximum number of positive anchors to sample.
            Typically 128 in Faster R-CNN (50% of mini-batch).
            
        total_count (int): Total number of anchors to sample (positive + negative).
            Typically 256 in Faster R-CNN (mini-batch size).
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Two boolean masks with shape (N,):
            - sampled_neg_idx_mask: True for sampled negative anchors
            - sampled_pos_idx_mask: True for sampled positive anchors
            
    Sampling Strategy:
        1. **Identify candidates**: Find all positive (label >= 1) and negative (label == 0) anchors
        2. **Sample positives**: Randomly select min(num_positive_available, positive_count)
        3. **Fill with negatives**: Use remaining budget for negative samples
        4. **Random selection**: Use random permutation for unbiased sampling
    """
    positive = torch.where(labels >= 1)[0] # foreground anchors
    negative = torch.where(labels == 0)[0] # background anchors
    
    # First sample as many positive as we can
    # if the count of positives can't be reached, sample extra negative anchors
    num_pos = min(positive.numel(), positive_count) 
    num_neg = total_count - num_pos
    num_neg = min(negative.numel(), num_neg)
    
    #Select randomly the desired number of positive and negative anchor indexes
    permuted_positive_idx = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    permuted_negative_idx = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
    
    pos_idxs = positive[permuted_positive_idx]
    neg_idxs = negative[permuted_negative_idx]
    
    #Returns 2 boolean masks with same shape as labels
    # 1 - indexes that belong to negative anchors are true
    # 2 - indexes that belong to positive anchors are true
    sampled_neg_idx_mask = torch.zeros_like(labels, dtype=torch.bool, device=labels.device)
    sampled_pos_idx_mask = torch.zeros_like(labels, dtype=torch.bool, device=labels.device)
    
    sampled_neg_idx_mask[neg_idxs] = True
    sampled_pos_idx_mask[pos_idxs] = True
    
    return sampled_neg_idx_mask, sampled_pos_idx_mask

def transform_boxes_to_original_size(boxes: torch.Tensor, new_size: torch.Tensor, original_size: torch.Tensor) -> torch.Tensor:
    """
    Transforms bounding box coordinates from a resized image scale back to the original image scale.

    Args:
        boxes (torch.Tensor): A tensor of shape [N, 4] containing the bounding boxes
                              in (xmin, ymin, xmax, ymax) format, corresponding to the resized image.
        new_size (torch.Tensor): A tensor representing the size of the resized image [height, width].
        original_size (torch.Tensor): A tensor representing the size of the original image [height, width].

    Returns:
        torch.Tensor: A tensor of shape [N, 4] with bounding boxes scaled to the original image dimensions.
    """
    # Calculate h and w scales [original_image_h/resized_image_h, original_image_w/resized_image_w]
    ratios = [
            torch.tensor(s_orig, dtype=torch.float32, device=boxes.device) / torch.tensor(s, dtype=torch.float32, device=boxes.device) 
            for s, s_orig in zip(new_size, original_size)
        ]
    
    # resize the box coordinates accordingly to image resize scale
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)