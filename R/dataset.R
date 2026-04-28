#' Create a torch Dataset for geospatial object detection
#'
#' Wraps chip paths and bounding-box annotations into a `torch::dataset` that
#' returns `(image_tensor, target)` pairs for Faster R-CNN training.
#'
#' @param chips_df Data frame from [make_chips()].
#' @param labels_list List of data frames from [assign_labels_to_chips()].
#' @param chip_size Integer. Pixel dimension. Default 512.
#' @param n_bands Integer. Number of raster bands to use. Default 3.
#' @param augment Logical. Random horizontal/vertical flips + brightness jitter.
#'   Default FALSE.
#' @param normalize Logical. Normalize using `mean`/`sd`. Default TRUE.
#' @param mean Numeric vector. Channel means. NULL = ImageNet defaults.
#' @param sd Numeric vector. Channel standard deviations. NULL = ImageNet defaults.
#'
#' @return A `torch::dataset` object.
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

  if (is.null(mean)) mean <- c(0.485, 0.456, 0.406)[seq_len(n_bands)]
  if (is.null(sd))   sd   <- c(0.229, 0.224, 0.225)[seq_len(n_bands)]

  n_bands   <- as.integer(n_bands)
  chip_size <- as.integer(chip_size)

  # BUG FIX 4: The original dataset() call defined initialize/.length/.getitem
  # as local functions then passed them again as named arguments at the end —
  # this is invalid torch::dataset() syntax and causes an "unused argument"
  # error or silently produces a broken dataset with no methods.
  # The correct pattern is to define methods *inside* the dataset() call only.

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

      # Load raster chip
      r  <- terra::rast(chip_row$path)
      nb <- min(self$n_bands, terra::nlyr(r))
      if (nb < self$n_bands) {
        cli::cli_warn("Chip {basename(chip_row$path)}: {nb} band(s) found, expected {self$n_bands}. Padding with zeros.")
      }
      r <- r[[seq_len(nb)]]

      arr <- terra::as.array(r)         # [rows, cols, bands]
      arr <- aperm(arr, c(3L, 1L, 2L)) # [bands, rows, cols]
      arr[is.na(arr)] <- 0.0

      # Scale to [0, 1]
      rmin <- min(arr)
      rmax <- max(arr)
      if (rmax > rmin) arr <- (arr - rmin) / (rmax - rmin)

      # BUG FIX 5: abind::abind() was used to pad bands but `abind` is not
      # listed in DESCRIPTION Imports. Replaced with base-R array operations.
      if (nb < self$n_bands) {
        pad <- array(0.0, dim = c(self$n_bands - nb, self$chip_size, self$chip_size))
        arr <- array(
          c(arr, pad),
          dim = c(self$n_bands, self$chip_size, self$chip_size)
        )
      }

      img_tensor <- torch::torch_tensor(arr, dtype = torch::torch_float32())

      # Bounding boxes
      if (nrow(label_df) > 0) {
        boxes_mat <- as.matrix(label_df[, c("xmin", "ymin", "xmax", "ymax")])
        boxes     <- torch::torch_tensor(boxes_mat, dtype = torch::torch_float32())
        labels    <- torch::torch_tensor(label_df$class_id, dtype = torch::torch_long())
      } else {
        boxes  <- torch::torch_zeros(c(0L, 4L), dtype = torch::torch_float32())
        labels <- torch::torch_zeros(0L, dtype = torch::torch_long())
      }

      # BUG FIX 6: Augmentation modified `label_df$...` (R data frame rows)
      # when it should be modifying the `boxes` *tensor*.  The `nrow(label_df)`
      # guard is also wrong after boxes tensor creation — use `boxes$shape[1]`.
      if (self$augment) {
        n_boxes <- boxes$shape[1]

        # Horizontal flip
        if (runif(1) > 0.5) {
          img_tensor <- torch::torch_flip(img_tensor, dims = 3L)
          if (n_boxes > 0) {
            new_xmin    <- self$chip_size - boxes[, 3]
            new_xmax    <- self$chip_size - boxes[, 1]
            boxes[, 1]  <- new_xmin
            boxes[, 3]  <- new_xmax
          }
        }

        # Vertical flip
        if (runif(1) > 0.5) {
          img_tensor <- torch::torch_flip(img_tensor, dims = 2L)
          if (n_boxes > 0) {
            new_ymin    <- self$chip_size - boxes[, 4]
            new_ymax    <- self$chip_size - boxes[, 2]
            boxes[, 2]  <- new_ymin
            boxes[, 4]  <- new_ymax
          }
        }

        # Brightness jitter — apply BEFORE normalization, clamp to [0, 1]
        jitter     <- runif(1, -0.1, 0.1)
        img_tensor <- torch::torch_clamp(img_tensor + jitter, 0, 1)
      }

      # Normalize
      if (self$normalize) {
        mean_t     <- torch::torch_tensor(self$mean)$view(c(-1L, 1L, 1L))
        sd_t       <- torch::torch_tensor(self$sd)$view(c(-1L, 1L, 1L))
        img_tensor <- (img_tensor - mean_t) / sd_t
      }

      list(
        image    = img_tensor,
        target   = list(boxes = boxes, labels = labels),
        chip_idx = idx
      )
    }
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
#' Faster R-CNN requires each image to have its own target list. This collate
#' function builds a list of images and a list of target lists rather than
#' stacking tensors.
#'
#' @param batch A list of items from [geo_detection_dataset()].
#' @return A list: `images` (list of tensors), `targets` (list of lists),
#'   `chip_ids` (integer vector).
#' @export
detection_collate_fn <- function(batch) {
  list(
    images   = lapply(batch, `[[`, "image"),
    targets  = lapply(batch, `[[`, "target"),
    chip_ids = vapply(batch, `[[`, integer(1), "chip_idx")
  )
}
