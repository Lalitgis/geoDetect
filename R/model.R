#' Build a Faster R-CNN detection model for geospatial imagery
#'
#' Constructs a Faster R-CNN model with a ResNet-50 + FPN backbone via
#' `torchvision`. The classification head is replaced with one sized for the
#' user's number of classes. Pretrained ImageNet weights can be loaded for
#' the backbone to enable transfer learning — critical when labelled geospatial
#' training data is limited.
#'
#' @param num_classes Integer. Number of object classes **not including
#'   background**. The model internally adds 1 for the background class.
#' @param pretrained_backbone Logical. Load ImageNet-pretrained weights for the
#'   ResNet-50 encoder. Strongly recommended unless training data is very large.
#'   Default TRUE.
#' @param freeze_backbone Logical. Freeze all ResNet-50 encoder weights and
#'   train only the FPN, RPN, and detection heads. Useful for small datasets.
#'   Default FALSE.
#' @param min_size Integer. Minimum image size passed to the model's transform
#'   pipeline. Larger values increase accuracy on small objects at higher compute
#'   cost. Default 512.
#' @param max_size Integer. Maximum image size. Default 1024.
#' @param anchor_sizes List of anchor size tuples per FPN level. NULL uses
#'   torchvision defaults ((32,), (64,), (128,), (256,), (512,)).
#'   Adjust smaller (e.g. ((8,),(16,),(32,),(64,),(128,))) for very small
#'   objects like vehicles in high-resolution aerial imagery.
#' @param anchor_ratios List of anchor aspect ratio tuples. Default
#'   ((0.5, 1.0, 2.0),) repeated for each level.
#' @param n_detections Integer. Maximum number of detections returned per image.
#'   Default 300.
#' @param score_thresh Numeric. Minimum confidence score for a detection to be
#'   returned during inference. Default 0.05.
#' @param nms_thresh Numeric. IoU threshold for non-maximum suppression.
#'   Default 0.5.
#'
#' @return A `torch::nn_module` (Faster R-CNN) ready for training or inference.
#'
#' @details
#' **Architecture overview:**
#' ```
#' Input image(s)
#'   └─> ResNet-50 backbone (pretrained ImageNet)
#'         └─> Feature Pyramid Network (FPN)
#'               └─> Region Proposal Network (RPN)
#'                     └─> RoI Align
#'                           └─> Box regression head
#'                           └─> Classification head  ← replaced for n_classes
#' ```
#'
#' @examples
#' \dontrun{
#' model <- build_detector(num_classes = 3)  # e.g. building, vehicle, tree
#' model$eval()
#' }
#'
#' @export
build_detector <- function(num_classes         = 2L,
                           pretrained_backbone = TRUE,
                           freeze_backbone     = FALSE,
                           min_size            = 512L,
                           max_size            = 1024L,
                           anchor_sizes        = NULL,
                           anchor_ratios       = NULL,
                           n_detections        = 300L,
                           score_thresh        = 0.05,
                           nms_thresh          = 0.5) {

  if (!requireNamespace("torch",       quietly = TRUE)) stop("Package 'torch' required.")
  if (!requireNamespace("torchvision", quietly = TRUE)) stop("Package 'torchvision' required.")

  num_classes <- as.integer(num_classes)
  if (num_classes < 1L)
    cli::cli_abort("{.arg num_classes} must be >= 1.")

  # Total classes including background
  n_cls_total <- num_classes + 1L

  # ---- Backbone: ResNet-50 with FPN ----
  backbone <- torchvision::model_resnet50(pretrained = pretrained_backbone)

  # Remove the final average pool and fully connected layers —
  # we only need the convolutional feature extractor layers 1-4
  backbone_layers <- torch::nn_module_list(list(
    backbone$conv1,
    backbone$bn1,
    backbone$relu,
    backbone$maxpool,
    backbone$layer1,
    backbone$layer2,
    backbone$layer3,
    backbone$layer4
  ))

  # Feature map channel sizes for ResNet-50 layers 1-4
  return_layers <- list("4" = "0", "5" = "1", "6" = "2", "7" = "3")
  in_channels_list <- c(256L, 512L, 1024L, 2048L)
  out_channels      <- 256L

  fpn_backbone <- torchvision::resnet_fpn_backbone(
    backbone_name    = "resnet50",
    pretrained       = pretrained_backbone,
    trainable_layers = if (freeze_backbone) 0L else 3L
  )

  # ---- Anchor generator ----
  if (is.null(anchor_sizes)) {
    anchor_sizes <- list(
      c(32L),  c(64L), c(128L), c(256L), c(512L)
    )
  }
  if (is.null(anchor_ratios)) {
    anchor_ratios <- rep(list(c(0.5, 1.0, 2.0)), length(anchor_sizes))
  }

  anchor_gen <- torchvision::anchor_generator(
    sizes        = anchor_sizes,
    aspect_ratios = anchor_ratios
  )

  # ---- RoI pooler ----
  roi_pooler <- torchvision::multi_scale_roi_align(
    featmap_names = c("0", "1", "2", "3"),
    output_size   = 7L,
    sampling_ratio = 2L
  )

  # ---- Full Faster R-CNN model ----
  model <- torchvision::faster_rcnn(
    backbone           = fpn_backbone,
    num_classes        = n_cls_total,
    rpn_anchor_generator = anchor_gen,
    box_roi_pool       = roi_pooler,
    min_size           = as.integer(min_size),
    max_size           = as.integer(max_size),
    box_detections_per_img = as.integer(n_detections),
    box_score_thresh   = score_thresh,
    box_nms_thresh     = nms_thresh
  )

  cli::cli_alert_success(
    "Built Faster R-CNN: {num_classes} class(es) + background | backbone frozen: {freeze_backbone}"
  )

  model
}


#' Load a saved geodetect model checkpoint
#'
#' Restores a model and its metadata saved by [train_detector()].
#'
#' @param checkpoint_path Character. Path to the `.pt` checkpoint file.
#' @param device Character. `"cpu"` or `"cuda"`. Default `"cpu"`.
#'
#' @return A list with elements `model` (the `nn_module`), `class_map`
#'   (named integer vector), `chip_size`, and `n_bands`.
#'
#' @export
load_detector <- function(checkpoint_path, device = "cpu") {

  if (!file.exists(checkpoint_path))
    cli::cli_abort("Checkpoint not found: {.path {checkpoint_path}}")

  ckpt <- torch::torch_load(checkpoint_path, device = device)

  model <- build_detector(
    num_classes         = ckpt$num_classes,
    pretrained_backbone = FALSE,
    min_size            = ckpt$min_size,
    max_size            = ckpt$max_size
  )
  model$load_state_dict(ckpt$state_dict)
  model$to(device = device)
  model$eval()

  cli::cli_alert_success("Loaded detector from {.path {checkpoint_path}}")
  list(
    model      = model,
    class_map  = ckpt$class_map,
    chip_size  = ckpt$chip_size,
    n_bands    = ckpt$n_bands
  )
}
