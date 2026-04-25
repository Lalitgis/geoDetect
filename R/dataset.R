#' Create a torch Dataset for geospatial object detection
#'
#' Wraps a list of image chip paths and their bounding-box annotations into a
#' `torch::dataset` that returns `(image_tensor, target)` pairs ready for
#' training a Faster R-CNN model via `torchvision`.
#'
#' @param chips_df Data frame from [make_chips()].
#' @param labels_list List of data frames from [assign_labels_to_chips()].
#'   Must be the same length as `nrow(chips_df)`.
#' @param chip_size Integer. Pixel dimension. Default 512.
#' @param n_bands Integer. Number of raster bands to use. If the chip has
#'   more bands, the first `n_bands` are selected. Default 3.
#' @param augment Logical. Apply random horizontal/vertical flipping and
#'   brightness jitter during training. Default FALSE.
#' @param normalize Logical. Normalize image tensors using per-dataset mean and
#'   standard deviation if provided, or ImageNet defaults. Default TRUE.
#' @param mean Numeric vector of length `n_bands`. Channel means for
#'   normalization. If NULL and `normalize = TRUE`, uses ImageNet means
#'   (only valid for n_bands = 3).
#' @param sd Numeric vector of length `n_bands`. Channel standard deviations
#'   for normalization. If NULL and `normalize = TRUE`, uses ImageNet sds.
#'
#' @return A `torch::dataset` object.
#'
#' @details
#' Each item returned is a named list:
#' \describe{
#'   \item{image}{Float tensor of shape \[C, H, W\] scaled to \[0, 1\].}
#'   \item{target}{Named list with:
#'     \describe{
#'       \item{boxes}{Float tensor of shape \[N, 4\] in \[xmin, ymin, xmax, ymax\] format.}
#'       \item{labels}{Integer tensor of shape \[N\] with 1-indexed class IDs.}
#'     }
#'   }
#'   \item{chip_idx}{Integer index into `chips_df` (for back-projection).}
#' }
#'
#' @export
geo_detection_dataset <- function(chips_df,
                                  labels_list,
                                  chip_size  = 512L,
                                  n_bands    = 3L,
                                  augment    = FALSE,
                                  normalize  = TRUE,
                                  mean       = NULL,
                                  sd         = NULL) {

  if (!requireNamespace("torch",       quietly = TRUE)) stop("Package 'torch' required.")
  if (!requireNamespace("torchvision", quietly = TRUE)) stop("Package 'torchvision' required.")

  if (nrow(chips_df) != length(labels_list))
    cli::cli_abort("Length of {.arg labels_list} must equal nrow({.arg chips_df}).")

  # Default to ImageNet normalization for 3-band (RGB) imagery
  if (is.null(mean)) mean <- c(0.485, 0.456, 0.406)[seq_len(n_bands)]
  if (is.null(sd))   sd   <- c(0.229, 0.224, 0.225)[seq_len(n_bands)]

  n_bands   <- as.integer(n_bands)
  chip_size <- as.integer(chip_size)

  torch::dataset(
    name = "GeoDetectionDataset",

    initialize = function(chips_df, labels_list, chip_size, n_bands,
                          augment, normalize, mean, sd) {
      self$chips_df    <- chips_df
      self$labels_list <- labels_list
      self$chip_size   <- chip_size
      self$n_bands     <- n_bands
      self$augment     <- augment
      self$normalize   <- normalize
      self$mean        <- mean
      self$sd          <- sd
    },

    .length = function() nrow(self$chips_df),

    .getitem = function(idx) {
      chip_row <- self$chips_df[idx, ]
      label_df <- self$labels_list[[idx]]

      # ------ Load raster chip ------
      r <- terra::rast(chip_row$path)

      # Select bands
      nb <- min(self$n_bands, terra::nlyr(r))
      if (nb < self$n_bands) {
        cli::cli_warn(
          "Chip {.path {chip_row$path}} has {nb} band(s), expected {self$n_bands}. Padding with zeros."
        )
      }
      r <- r[[seq_len(nb)]]

      # To numeric matrix [bands, rows, cols]
      arr <- terra::as.array(r)           # [rows, cols, bands]
      arr <- aperm(arr, c(3L, 1L, 2L))   # [bands, rows, cols]
      arr[is.na(arr)] <- 0.0

      # Scale to [0, 1]
      rmin <- min(arr, na.rm = TRUE)
      rmax <- max(arr, na.rm = TRUE)
      if (rmax > rmin) arr <- (arr - rmin) / (rmax - rmin)

      # Pad if needed so tensor is exactly [n_bands, chip_size, chip_size]
      if (nb < self$n_bands) {
        pad <- array(0.0, dim = c(self$n_bands - nb, self$chip_size, self$chip_size))
        arr <- abind::abind(arr, pad, along = 1)
      }

      img_tensor <- torch::torch_tensor(arr, dtype = torch::torch_float32())

      # ------ Bounding boxes ------
      if (nrow(label_df) > 0) {
        boxes  <- torch::torch_tensor(
          as.matrix(label_df[, c("xmin", "ymin", "xmax", "ymax")]),
          dtype = torch::torch_float32()
        )
        labels <- torch::torch_tensor(
          label_df$class_id,
          dtype = torch::torch_long()
        )
      } else {
        # Empty chip — no objects. Faster R-CNN can handle this.
        boxes  <- torch::torch_zeros(c(0L, 4L), dtype = torch::torch_float32())
        labels <- torch::torch_zeros(0L,         dtype = torch::torch_long())
      }

      # ------ Augmentation ------
      if (self$augment) {
        # Horizontal flip
        if (runif(1) > 0.5) {
          img_tensor <- torch::torch_flip(img_tensor, dims = 3L)
          if (nrow(label_df) > 0) {
            new_xmin <- self$chip_size - boxes[, 3]
            new_xmax <- self$chip_size - boxes[, 1]
            boxes[, 1] <- new_xmin
            boxes[, 3] <- new_xmax
          }
        }
        # Vertical flip
        if (runif(1) > 0.5) {
          img_tensor <- torch::torch_flip(img_tensor, dims = 2L)
          if (nrow(label_df) > 0) {
            new_ymin <- self$chip_size - boxes[, 4]
            new_ymax <- self$chip_size - boxes[, 2]
            boxes[, 2] <- new_ymin
            boxes[, 4] <- new_ymax
          }
        }
        # Brightness jitter (simple additive)
        jitter <- runif(1, -0.1, 0.1)
        img_tensor <- torch::torch_clamp(img_tensor + jitter, 0, 1)
      }

      # ------ Normalize ------
      if (self$normalize) {
        mean_t <- torch::torch_tensor(self$mean)$view(c(-1L, 1L, 1L))
        sd_t   <- torch::torch_tensor(self$sd)$view(c(-1L, 1L, 1L))
        img_tensor <- (img_tensor - mean_t) / sd_t
      }

      list(
        image    = img_tensor,
        target   = list(boxes = boxes, labels = labels),
        chip_idx = idx
      )
    },

    initialize = initialize,
    .length    = .length,
    .getitem   = .getitem
  )(
    chips_df    = chips_df,
    labels_list = labels_list,
    chip_size   = chip_size,
    n_bands     = n_bands,
    augment     = augment,
    normalize   = normalize,
    mean        = mean,
    sd          = sd
  )
}


#' Custom collate function for detection DataLoaders
#'
#' Faster R-CNN requires each image to have its own target list (boxes and
#' labels). This collate function assembles a batch as a list of images and
#' a list of target lists instead of stacking tensors — the format expected
#' by `torchvision`'s detection models.
#'
#' @param batch A list of items from [geo_detection_dataset].
#'
#' @return A list with two elements:
#'   \describe{
#'     \item{images}{List of image tensors.}
#'     \item{targets}{List of target lists, each with `boxes` and `labels`.}
#'   }
#'
#' @export
detection_collate_fn <- function(batch) {
  images   <- lapply(batch, function(x) x$image)
  targets  <- lapply(batch, function(x) x$target)
  chip_ids <- sapply(batch, function(x) x$chip_idx)
  list(images = images, targets = targets, chip_ids = chip_ids)
}
