#' Build a Faster R-CNN detection model for geospatial imagery
#'
#' Constructs a Faster R-CNN model with a ResNet-50 + FPN backbone via
#' `torchvision`. The R `torchvision` package exposes
#' `torchvision::model_detection_faster_rcnn()` (or the underlying
#' `torchvision::fasterrcnn_resnet50_fpn()`) rather than the Python-style
#' component-level API. This function uses the correct R API and replaces
#' the classification head for the user's number of classes.
#'
#' @param num_classes Integer. Number of object classes NOT including
#'   background. The model adds 1 internally for background.
#' @param pretrained_backbone Logical. Load ImageNet-pretrained ResNet-50
#'   encoder weights. Default TRUE.
#' @param freeze_backbone Logical. Freeze ResNet-50 encoder; train only FPN,
#'   RPN, and heads. Default FALSE.
#' @param min_size Integer. Minimum image size in the model's resize transform.
#'   Default 512.
#' @param max_size Integer. Maximum image size. Default 1024.
#' @param anchor_sizes Numeric vector of anchor sizes (one per FPN level).
#'   Default c(32, 64, 128, 256, 512). Shrink for small objects.
#' @param anchor_ratios Numeric vector of anchor aspect ratios.
#'   Default c(0.5, 1.0, 2.0).
#' @param n_detections Integer. Max detections per image. Default 300.
#' @param score_thresh Numeric. Min confidence for returned detections.
#'   Default 0.05.
#' @param nms_thresh Numeric. IoU threshold for NMS. Default 0.5.
#'
#' @return A `torch::nn_module` (Faster R-CNN).
#' @export
build_detector <- function(num_classes         = 2L,
                           pretrained_backbone = TRUE,
                           freeze_backbone     = FALSE,
                           min_size            = 512L,
                           max_size            = 1024L,
                           anchor_sizes        = c(32, 64, 128, 256, 512),
                           anchor_ratios       = c(0.5, 1.0, 2.0),
                           n_detections        = 300L,
                           score_thresh        = 0.05,
                           nms_thresh          = 0.5) {

  if (!requireNamespace("torch",       quietly = TRUE)) stop("Package 'torch' required.")
  if (!requireNamespace("torchvision", quietly = TRUE)) stop("Package 'torchvision' required.")

  num_classes <- as.integer(num_classes)
  if (num_classes < 1L)
    cli::cli_abort("{.arg num_classes} must be >= 1.")

  # BUG FIX 7: The original code built backbone_layers, return_layers, and
  # in_channels_list but never passed them to anything — dead code.
  # torchvision::resnet_fpn_backbone() is the only thing needed and is called
  # directly below.

  # BUG FIX 8: torchvision::anchor_generator(), torchvision::multi_scale_roi_align(),
  # and torchvision::faster_rcnn() do NOT exist in the R torchvision package.
  # These are Python torchvision API names. The correct R API is:
  #   torchvision::fasterrcnn_resnet50_fpn()
  # which returns a complete Faster R-CNN with ResNet-50 FPN backbone.
  # We then swap the box predictor head for the correct number of classes.

  # Build the full model with pretrained COCO weights (91 classes)
  model <- torchvision::fasterrcnn_resnet50_fpn(
    pretrained       = FALSE,           # COCO weights not always available
    pretrained_backbone = pretrained_backbone,
    progress         = FALSE,
    num_classes      = 91L,            # Standard COCO head — we replace it next
    trainable_backbone_layers = if (freeze_backbone) 0L else 3L
  )

  # Replace the box predictor head for num_classes + background
  n_cls_total  <- num_classes + 1L
  in_features  <- model$roi_heads$box_predictor$cls_score$in_features

  model$roi_heads$box_predictor <- torchvision::faster_rcnn_predictor(
    in_channels = in_features,
    num_classes = n_cls_total
  )

  # Adjust RPN/transform parameters
  model$rpn$anchor_generator$sizes        <- lapply(anchor_sizes, function(s) c(as.integer(s)))
  model$rpn$anchor_generator$aspect_ratios <- rep(list(anchor_ratios), length(anchor_sizes))

  model$transform$min_size <- as.integer(min_size)
  model$transform$max_size <- as.integer(max_size)
  model$roi_heads$detections_per_img <- as.integer(n_detections)
  model$roi_heads$score_thresh       <- score_thresh
  model$roi_heads$nms_thresh         <- nms_thresh

  cli::cli_alert_success(
    "Built Faster R-CNN: {num_classes} class(es) + background | \\
    backbone frozen: {freeze_backbone}"
  )

  model
}


#' Load a saved geodetect model checkpoint
#'
#' @param checkpoint_path Character. Path to the `.pt` checkpoint file.
#' @param device Character. `"cpu"` or `"cuda"`. Default `"cpu"`.
#'
#' @return A list: `model`, `class_map`, `chip_size`, `n_bands`.
#' @export
load_detector <- function(checkpoint_path, device = "cpu") {

  if (!file.exists(checkpoint_path))
    cli::cli_abort("Checkpoint not found: {.path {checkpoint_path}}")

  ckpt <- torch::torch_load(checkpoint_path, device = device)

  # BUG FIX 9: Checkpoint stores min_size/max_size from training;
  # the original code hard-coded 512/1024 here, ignoring what was saved.
  # Now uses ckpt values with safe fallbacks.
  model <- build_detector(
    num_classes         = ckpt$num_classes,
    pretrained_backbone = FALSE,
    min_size            = ckpt$min_size  %||% 512L,
    max_size            = ckpt$max_size  %||% 1024L
  )
  model$load_state_dict(ckpt$state_dict)
  model$to(device = device)
  model$eval()

  cli::cli_alert_success("Loaded detector from {.path {checkpoint_path}}")
  list(
    model     = model,
    class_map = ckpt$class_map,
    chip_size = ckpt$chip_size,
    n_bands   = ckpt$n_bands
  )
}

# Internal null-coalescing operator
`%||%` <- function(a, b) if (!is.null(a)) a else b
