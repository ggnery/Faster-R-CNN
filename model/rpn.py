import torch
import torch.nn as nn
import torchvision
from typing import Tuple
from .utils import get_iou, clamp_boxes_to_image_boundary, apply_regression_pred_to_anchors_or_proposals, boxes_to_transformation_targets, sample_positive_negative

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels, scales, aspect_ratios, model_config):
        """Initializes the Region Proposal Network (RPN) module for object detection.
    
        The RPN is a fully convolutional network that generates object proposals by
        sliding a small network over the convolutional feature map. At each sliding
        window location, it simultaneously predicts multiple region proposals and
        their "objectness" scores (probability of containing an object).
        
        Architecture:
        1. Shared 3x3 conv layer for feature extraction at each location
        2. Two parallel 1x1 conv heads:
        - Classification head: Predicts objectness score for each anchor
        - Regression head: Predicts bounding box refinements (Δx, Δy, Δw, Δh)
        
        For each spatial location in the feature map, the network predicts:
        - K objectness scores (one per anchor)
        - Kx4 regression parameters (4 per anchor: tx, ty, tw, th)
        where K = len(scales) x len(aspect_ratios) = 9 anchors per location
        
        Args:
            in_channels (int, optional): Number of input channels from the backbone
                feature extractor (e.g., ResNet, VGG). Defaults to 512.
        
        Attributes:
            scales (list): Anchor box sizes in pixels [128, 256, 512]. These define
                the square root of anchor areas (16384, 65536, 262144 pixels²).
            aspect_ratios (list): Height/width ratios [0.5, 1, 2] corresponding to
                rectangular anchors (1:2, 1:1, 2:1).
            num_anchors (int): Number of anchors per location (should be 9).
            rpn_conv (Conv2d): 3x3 conv layer that processes the feature map to
                extract spatial features for proposal generation.
            classification_layer (Conv2d): 1x1 conv that outputs K objectness scores
                per location. Uses K outputs instead of 2K because:
                - Output represents P(foreground) directly
                - P(background) = 1 - P(foreground) is implicit
                - Reduces parameters and computation
            bbox_regressor_layer (Conv2d): 1x1 conv that outputs 4K values per
                location for refining anchor boxes into proposals.
        
        Output shapes (for feature map of size HxW):
            - Classification: (B, K, H, W) - objectness scores
            - Regression: (B, Kx4, H, W) - box refinements
        
        """
        
        super(RegionProposalNetwork, self).__init__()
        self.scales = scales # scales/areas for anchor boxes in feature map (128^2, 256^2 and 512^2)
        self.aspect_ratios = aspect_ratios # aspect_ratios for anchor boxes in feature map (1:2, 1:1, 2:1)
        
        self.low_iou_threshold =  model_config["rpn_bg_threshold"] # background threshold for rpn (default = 0.3)
        self.high_iou_threshold = model_config["rpn_fg_threshold"] # foreground threshold for rpn (default = 0.7)
        
        self.rpn_nms_threshold = model_config["rpn_nms_threshold"] # threshold for NMS in filter_proposals function (default = 0.7)
        
        self.rpn_batch_size = model_config["rpn_batch_size"] # total batch for sampling positive and negative anchors (default = 256)
        self.rpn_pos_count = int(model_config["rpn_pos_fraction"] * self.rpn_batch_size) # total of positive anchors in sampling (default = 256 * 0.5 = 128)
        #Note: rpn_pos_fraction is the fraction of positive anchors in rpn_batch_size
        
        #Pre NMS topk anchors in filter_proposals function (default = 12000 in train and 6000 in test)
        self.rpn_prenms_topk = model_config["rpn_train_prenms_topk"] if self.training else model_config["rpn_test_prenms_topk"]
        #Post NMS topk anchors in filter_proposals function (default = 2000 in train and 300 in test)
        self.rpn_topk = model_config["rpn_train_topk"] if self.training else model_config["rpn_test_topk"] 
        
        self.min_size = model_config["rpn_min_size"] # min h and w in filter_proposals function (default = 16)
        
        self.num_anchors = len(self.scales) * len(self.aspect_ratios) #Each feature map cell will have 3x3 = 9 anchor boxes  
        
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
        
        #Initializing weights and bias 
        for layer in [self.rpn_conv, self.classification_layer, self.bbox_regressor_layer]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)
    
    def generate_anchors(self, image: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:        
        """Generates anchor boxes across the entire image based on the feature map grid.
    
        This function implements the anchor generation strategy used in object detection
        models like Faster R-CNN. It creates a dense set of anchor boxes at each spatial
        location in the feature map, with multiple scales and aspect ratios per location.
        
        Algorithm:
        1. Creates base anchors (centered at origin) with different scales and aspect ratios
        2. Computes the stride between feature map and original image
        3. Creates a grid of anchor center points based on feature map dimensions
        4. Shifts base anchors to each grid point to cover the entire image
        
        The anchors are generated such that:
        - Each feature map cell corresponds to multiple anchors (e.g., 9 anchors)
        - Anchors are placed at regular intervals determined by the stride
        - All anchors maintain constant area at each scale (width x height = scale²)
        
        Args:
            image (torch.Tensor): Original input image. Shape: (B, C, H, W), e.g., (1, 3, 600, 800)
                Used only to compute the stride relative to the feature map.
            feature_map (torch.Tensor): CNN feature map. Shape: (B, C, H_feat, W_feat), 
                e.g., (1, 512, 37, 50). Determines the grid density of anchors.
        
        Returns:
            torch.Tensor: All anchor boxes in [x1, y1, x2, y2] format. 
                Shape: (num_anchors, 4) where num_anchors = H_feat x W_feat x num_base_anchors
                Example: (37 x 50 x 9, 4) = (16650, 4)
        
        Attributes used from self:
            scales (list): Anchor scales (e.g., [128, 256, 512])
            aspect_ratios (list): Height/width ratios (e.g., [0.5, 1.0, 2.0])
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

        # Create a grid using these shifts
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
        
        """
        
        # get IoU matrix with shape (gt_boxes, num_anchors)
        iou_matrix = get_iou(gt_boxes, anchors) # eg -> 6 x 16650
        
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
        
        below_low_threshold = best_match_iou < self.low_iou_threshold # background anchors
        between_threshold = (best_match_iou >= self.low_iou_threshold) & (best_match_iou < self.high_iou_threshold) # ignored anchors
        
        best_match_gt_index[below_low_threshold] = -1 # -1 is background
        best_match_gt_index[between_threshold] = -2 # -2 is ignored
        
        # LOW QUALITY POSITIVE anchor boxes process
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
        
        # Seting regression targets for the anchors 
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
    
    def filter_proposals(self, proposals: torch.Tensor, classification_scores: torch.Tensor, image_shape) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filters and refines region proposals generated by the RPN for downstream processing.
        
        This function implements a multi-stage filtering pipeline to reduce the number of
        proposals from potentially millions to a manageable set of high-quality candidates.
        The filtering process balances computational efficiency with detection quality by
        removing redundant, low-confidence, and out-of-bounds proposals.
        
        The filtering follows the standard Faster R-CNN approach:
        1. Pre-NMS filtering: Reduces computational load for NMS
        2. Boundary clipping: Ensures valid proposals within image bounds
        3. NMS (Non-Maximum Suppression): Removes duplicate detections
        4. Post-NMS filtering: Final selection of best proposals
        
        Args:
            proposals (torch.Tensor): Raw proposal boxes from RPN regression. 
                Shape: (N, 4) where N = H_feat x W_feat x num_anchors (e.g., 16650)
                Each row contains [x1, y1, x2, y2] in image coordinates.
                These are the transformed anchor boxes after applying predicted deltas.
            
            classification_scores (torch.Tensor): Raw objectness scores from RPN classification.
                Shape: (N,) matching the number of proposals.
                Higher scores indicate higher probability of containing an object.
                Note: These are raw logits (pre-sigmoid) from the network.
            
            image_shape (Tuple[int, int]): Original input image dimensions as (height, width).
                Used to clip proposals to valid image boundaries.
                Example: (600, 800) for a 600x800 image.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - proposals (torch.Tensor): Filtered proposal boxes. Shape: (K, 4) where
                K ≤ 2000. Each row contains [x1, y1, x2, y2] in image coordinates.
                Sorted by objectness score in descending order.
                - scores (torch.Tensor): Objectness scores for the filtered proposals.
                Shape: (K,) matching the number of filtered proposals.
                Values are in range [0, 1] after sigmoid activation.
        
        """
        #Pre NMS filtering
        classification_scores = classification_scores.reshape(-1) # flatten proposal classification scores
        classification_scores = torch.sigmoid(classification_scores)
        
        # get top k<rpn_prenms_topk proposal boxes based on classification scores (foreground or background)
        _, top_n_idx = classification_scores.topk(min(self.rpn_prenms_topk, len(classification_scores)))
        
        classification_scores = classification_scores[top_n_idx]
        proposals = proposals[top_n_idx] # filter only top 10000 proposals
        
        # Clamp boxes to image boundary
        proposals = clamp_boxes_to_image_boundary(proposals, image_shape)
        
        # Small boxes boxes based on width and height filtering (h and w >= 16)
        min_size = self.min_size
        ws, hs = proposals[:, 2] - proposals[:, 0], proposals[:, 3] - proposals[:, 1] 
        keep = (ws >= min_size) & (hs >= min_size)
        keep = torch.where(keep)[0]
        proposals = proposals[keep]
        classification_scores = classification_scores[keep] 
        
        #NMS based on objectness
        keep_mask = torch.zeros_like(classification_scores, dtype=torch.bool) # mask with proposals to keep
        keep_indices = torchvision.ops.nms(proposals, classification_scores, self.rpn_nms_threshold) # apply nms with 0.7 threshold
        keep_mask[keep_indices] = True
        keep_indices = torch.where(keep_mask)[0]
        
        #sort keep_indices by classification scores
        post_nms_keep_indices = keep_indices[
            classification_scores[keep_indices].sort(descending=True)[1]
        ]
        
        # Post NMS topk = 2000 filtering
        proposals = proposals[post_nms_keep_indices[:self.rpn_topk]]
        classification_scores = classification_scores[post_nms_keep_indices[:self.rpn_topk]]
        
        return proposals, classification_scores
     
    def forward(self, image, feature_map, target):
        """Executes the Region Proposal Network forward pass for object proposal generation.
    
        This method implements the complete RPN pipeline, which serves as the first stage
        in two-stage object detectors like Faster R-CNN. It processes feature maps to
        generate object proposals that are later refined by the second stage (RCNN).
        
        Inference
            - Call RPN layers
            - Generate anchors
            - Convert anchors to proposals using box transformation prediction
            - Filter proposals
        Training only
            - All steps done in inference
            - Assign ground truth boxes to anchors
            - Compute labels and regression targets for anchors
            - Sample positive and negative anchors
            - Compute classification loss using sampled anchors
            - Compute localization loss using sampled positive anchors

        Detailed Algorithm Flow:
    
        1. **Feature Extraction** (3x3 conv + ReLU):
            
        2. **Parallel Predictions** (1x1 convs):

            
        3. **Anchor Generation**:
            - Creates K anchors at each of H_feat x W_feat locations
            - Total anchors = H_feat x W_feat x K (e.g., 37x50x9 = 16,650)
            - Anchors have 3 scales x 3 aspect ratios = 9 variants per location
            
        4. **Proposal Generation**:
            - Applies predicted deltas to anchors: proposals = transform(anchors, deltas)
            - No gradient flow through proposals during training (uses .detach())
            
        5. **Proposal Filtering** (inference and training):
            - Top-K by score (K=10,000) → Boundary clipping → NMS (IoU=0.7) → Top-K (K=2,000)
            - Typically reduces from ~16,650 raw proposals to ≤2,000 high-quality ones
            
        6. **Loss Computation** (training only):
            - Assigns ground truth to anchors based on IoU overlap
            - Samples 256 anchors (128 positive, 128 negative) for efficiency
            - Classification loss: BCE on all sampled anchors
            - Localization loss: Smooth L1 on positive anchors only

        Args:
            image (torch.Tensor): Original input image tensor. Shape: (B, C, H, W)
                - B: Batch size (typically 1 for object detection)
                - C: Number of channels (typically 3 for RGB)
                - H, W: Image height and width (e.g., 600, 800)
                Used only for computing anchor positions relative to feature map stride.
                
            feature_map (torch.Tensor): CNN feature map from backbone network.
                Shape: (B, C_feat, H_feat, W_feat)
                - B: Batch size (must match image batch size)
                - C_feat: Feature channels (e.g., 512 for VGG, 256/512/1024/2048 for ResNet)
                - H_feat, W_feat: Spatial dimensions (e.g., 37x50 for 600x800 image with stride 16)
                This is the main input that RPN processes to generate proposals.
                
            target (Dict[str, torch.Tensor], optional): Ground truth annotations for training.
                Required keys:
                - "bboxes": Ground truth boxes. Shape: (B, N_gt, 4) where N_gt is the number
                of objects. Each box is [x1, y1, x2, y2] in image coordinates.
                - "labels": Object class labels. Shape: (B, N_gt). Not used by RPN but
                required for consistency with the full model interface.
                If None, the network runs in inference mode (no loss computation).        
        
        Returns:
            Dict[str, torch.Tensor]: Output dictionary containing:
                
            **Always included (inference and training)**:
                - "proposals": Filtered region proposals. Shape: (N_proposals, 4)
                where N_proposals ≤ 2000. Each row is [x1, y1, x2, y2] in image coords.
                Sorted by objectness score in descending order.
                - "scores": Objectness scores for proposals. Shape: (N_proposals,)
                Values in range [0, 1] after sigmoid. Higher = more likely to contain object.
                
            **Training mode only** (when target is provided):
                - "rpn_classification_loss": Binary cross-entropy loss for objectness.
                Scalar value averaged over sampled anchors (256 per image).
                - "rpn_localization_loss": Smooth L1 loss for bounding box regression.
                Scalar value averaged over positive anchors only (~128 per image).
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
        classification_scores = classification_scores.reshape(-1, 1) # classification_scores -> (batch * h_feat * w_feat * Numer of anchors per location, 1)
        
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
            box_transformation_preds.detach().reshape(-1, 1, 4), # (Batch * Number of Anchors per location * h_feat * w_feat, 1, 4) where the 1 here is class (background or foregroung)
            anchors
        )
        
        #Filtering the proposals
        proposals = proposals.reshape(proposals.size(0), 4) # (Batch * Number of Anchors per location * h_feat * w_feat, 4)
        proposals, scores = self.filter_proposals(proposals, classification_scores.detach(), image.shape)

        rpn_output = {
            "proposals": proposals, # eg: proposals -> (2000 x 4)
            "scores": scores # eg: scores -> (2000,)
        }
            
        if not self.training or target is None:
            return rpn_output
        else:
            # In train mode, assign ground truth values to anchors
            # matched_gt_boxes_for_anchors -> (Number of anchors in image, 4)
            labels_for_anchors, matched_gt_boxes_for_anchors = self.assign_targets_to_anchors(
                anchors,
                target["bboxes"][0]
            ) 
            
            # Based on gt assignment above, get groud truth regression targets (tx, ty, tz, tw) for anchors
            regression_targets = boxes_to_transformation_targets(matched_gt_boxes_for_anchors, anchors) # 16650x4
            
            # Sample positive (foreground) and negative anchors (background) for training
            sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_negative(
                labels_for_anchors, 
                positive_count=self.rpn_pos_count, 
                total_count=self.rpn_batch_size)
            sampled_idxs = torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)[0] #get positive and negative anchor idxs
            
            #Smooth L1 loss for proposal regression (positive only)
            localization_loss = (
                torch.nn.functional.smooth_l1_loss(
                    box_transformation_preds[sampled_pos_idx_mask], # box_transformation_preds -> predicted (tx, ty, tz, tw) from 1x1 conv layer
                    regression_targets[sampled_pos_idx_mask], # regression_targets -> gt regression targets (tx, ty, tz, tw) for each anchor
                    beta=1/9,
                    reduction="sum"
                ) / (sampled_idxs.numel())
            )
            
            # Binary cross entropy loss between classification scores and labels of sampled indexes (positive and negative)
            classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                classification_scores[sampled_idxs].flatten(), # classification_scores -> predicted classification scores for anchors after 1x1 conv layer
                labels_for_anchors[sampled_idxs].flatten() # labels_for_anchors -> ground truth classification scores for anchors
            )
            
            rpn_output["rpn_classification_loss"] = classification_loss
            rpn_output["rpn_localization_loss"] = localization_loss
            
            return rpn_output