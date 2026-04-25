#' Run object detection on a large geospatial raster
#'
#' Applies a trained Faster R-CNN detector to an arbitrarily large raster using
#' a sliding-window tiling strategy. Detections from overlapping chips are
#' merged using non-maximum suppression (NMS) in geographic coordinate space.
#' Results are returned as an `sf` object with geographic bounding boxes,
#' class labels, and confidence scores.
#'
#' @param model A trained Faster R-CNN `nn_module` in eval mode. Can be the
#'   `model` element from [train_detector()] or [load_detector()].
#' @param rast A `SpatRaster` (terra) to run inference on.
#' @param chip_size Integer. Must match the value used during training.
#'   Default 512.
#' @param overlap Numeric in [0, 1). Overlap between sliding-window chips.
#'   Higher overlap reduces missed detections near chip boundaries but
#'   increases computation. Default 0.25.
#' @param n_bands Integer. Number of bands to use. Default 3.
#' @param score_thresh Numeric. Minimum confidence score. Default 0.5.
#' @param nms_thresh Numeric. IoU threshold for global NMS across chips.
#'   Default 0.5.
#' @param class_map Named integer vector (class name → integer ID). Used to
#'   attach class names to results. If NULL, class IDs are returned as-is.
#' @param normalize Logical. Normalize image chips before inference (must match
#'   training settings). Default TRUE.
#' @param mean Numeric vector. Channel means. NULL = ImageNet defaults.
#' @param sd Numeric vector. Channel standard deviations. NULL = ImageNet defaults.
#' @param batch_size Integer. Chips processed per forward pass. Default 4.
#' @param device Character. `"cpu"` or `"cuda"`. Auto-detected.
#' @param out_dir Character or NULL. If supplied, chip `.tif` files are written
#'   here and deleted after inference. If NULL, chips are processed in memory.
#'   Default NULL.
#' @param verbose Logical. Show progress bar. Default TRUE.
#'
#' @return An `sf` object (POLYGON geometry) in the raster's CRS, with columns:
#'   \describe{
#'     \item{class_id}{Integer. 1-indexed class identifier.}
#'     \item{class_name}{Character. Human-readable class name (NA if no class_map).}
#'     \item{score}{Numeric. Model confidence in [0, 1].}
#'     \item{geometry}{POLYGON bounding box in the raster's CRS.}
#'   }
#'
#' @examples
#' \dontrun{
#' ckpt   <- load_detector("my_detector.pt")
#' result <- predict_raster(
#'   model      = ckpt$model,
#'   rast       = terra::rast("scene.tif"),
#'   class_map  = ckpt$class_map,
#'   chip_size  = ckpt$chip_size,
#'   score_thresh = 0.6
#' )
#' plot(result["class_name"])
#' }
#'
#' @export
predict_raster <- function(model,
                           rast,
                           chip_size    = 512L,
                           overlap      = 0.25,
                           n_bands      = 3L,
                           score_thresh = 0.5,
                           nms_thresh   = 0.5,
                           class_map    = NULL,
                           normalize    = TRUE,
                           mean         = NULL,
                           sd           = NULL,
                           batch_size   = 4L,
                           device       = NULL,
                           out_dir      = NULL,
                           verbose      = TRUE) {

  if (!requireNamespace("torch", quietly = TRUE)) stop("Package 'torch' required.")
  if (!inherits(rast, "SpatRaster"))
    cli::cli_abort("{.arg rast} must be a {.cls SpatRaster}.")

  if (is.null(device))
    device <- if (torch::cuda_is_available()) "cuda" else "cpu"

  model$to(device = device)
  model$eval()

  if (is.null(mean)) mean <- c(0.485, 0.456, 0.406)[seq_len(n_bands)]
  if (is.null(sd))   sd   <- c(0.229, 0.224, 0.225)[seq_len(n_bands)]

  # Build chip index (don't write to disk — process from memory)
  chip_size  <- as.integer(chip_size)
  nr         <- terra::nrow(rast)
  nc         <- terra::ncol(rast)
  step       <- max(1L, as.integer(chip_size * (1 - overlap)))
  ext0       <- terra::ext(rast)
  res        <- terra::res(rast)

  row_starts <- seq(1L, nr, by = step)
  col_starts <- seq(1L, nc, by = step)

  n_chips <- length(row_starts) * length(col_starts)
  if (verbose) cli::cli_alert_info("Scanning {n_chips} chips over raster...")

  # Accumulate detections in geo-coordinates across all chips
  det_boxes   <- matrix(numeric(0), ncol = 4)  # [xmin, ymin, xmax, ymax] geo
  det_scores  <- numeric(0)
  det_cls_ids <- integer(0)

  chip_batch_imgs   <- list()
  chip_batch_extents <- list()  # list of c(xmin, ymin, xmax, ymax) per chip

  flush_batch <- function() {
    if (length(chip_batch_imgs) == 0) return()

    # Stack to tensor list and forward pass
    with(torch::no_grad(), {
      img_list <- lapply(chip_batch_imgs, function(arr) {
        t <- torch::torch_tensor(arr, dtype = torch::torch_float32())$to(device = device)
        t
      })
      preds <- model(img_list)
    })

    for (i in seq_along(preds)) {
      boxes_px  <- as.matrix(preds[[i]]$boxes$cpu())
      scores_i  <- as.numeric(preds[[i]]$scores$cpu())
      labels_i  <- as.integer(preds[[i]]$labels$cpu())
      ext_i     <- chip_batch_extents[[i]]

      keep <- scores_i >= score_thresh
      if (!any(keep)) next

      boxes_px  <- boxes_px[keep, , drop = FALSE]
      scores_i  <- scores_i[keep]
      labels_i  <- labels_i[keep]

      # Back-project pixel coords → geographic coords
      chip_w <- ext_i[3] - ext_i[1]
      chip_h <- ext_i[4] - ext_i[2]

      geo_xmin <- ext_i[1] + (boxes_px[, 1] / chip_size) * chip_w
      geo_xmax <- ext_i[1] + (boxes_px[, 3] / chip_size) * chip_w
      # y is flipped: pixel row 0 = geo ymax
      geo_ymax  <- ext_i[4] - (boxes_px[, 2] / chip_size) * chip_h
      geo_ymin  <- ext_i[4] - (boxes_px[, 4] / chip_size) * chip_h

      geo_boxes <- cbind(geo_xmin, geo_ymin, geo_xmax, geo_ymax)

      det_boxes   <<- rbind(det_boxes, geo_boxes)
      det_scores  <<- c(det_scores, scores_i)
      det_cls_ids <<- c(det_cls_ids, labels_i)
    }

    # Reset batch buffers
    chip_batch_imgs    <<- list()
    chip_batch_extents <<- list()
  }

  if (verbose) cli::cli_progress_bar("Running inference", total = n_chips)

  for (ri in seq_along(row_starts)) {
    for (ci in seq_along(col_starts)) {
      if (verbose) cli::cli_progress_update()

      r0 <- row_starts[ri]; r1 <- min(r0 + chip_size - 1L, nr)
      c0 <- col_starts[ci]; c1 <- min(c0 + chip_size - 1L, nc)

      chip_xmin <- ext0$xmin + (c0 - 1) * res[1]
      chip_xmax <- ext0$xmin + c1        * res[1]
      chip_ymax <- ext0$ymax - (r0 - 1) * res[2]
      chip_ymin <- ext0$ymax - r1        * res[2]

      chip_rast <- terra::crop(rast, terra::ext(chip_xmin, chip_xmax, chip_ymin, chip_ymax))

      # Pad to chip_size x chip_size if needed
      if (terra::nrow(chip_rast) != chip_size || terra::ncol(chip_rast) != chip_size) {
        tmpl <- terra::rast(
          nrows = chip_size, ncols = chip_size,
          xmin = chip_xmin, xmax = chip_xmin + chip_size * res[1],
          ymin = chip_ymax - chip_size * res[2], ymax = chip_ymax,
          crs  = terra::crs(rast)
        )
        chip_rast <- terra::resample(chip_rast, tmpl, method = "near")
      }

      nb   <- min(n_bands, terra::nlyr(chip_rast))
      r_use <- chip_rast[[seq_len(nb)]]
      arr  <- aperm(terra::as.array(r_use), c(3L, 1L, 2L))
      arr[is.na(arr)] <- 0.0

      rmin <- min(arr); rmax <- max(arr)
      if (rmax > rmin) arr <- (arr - rmin) / (rmax - rmin)

      if (nb < n_bands) {
        pad <- array(0.0, dim = c(n_bands - nb, chip_size, chip_size))
        arr <- abind::abind(arr, pad, along = 1)
      }

      if (normalize) {
        for (b in seq_len(n_bands)) arr[b, , ] <- (arr[b, , ] - mean[b]) / sd[b]
      }

      chip_batch_imgs[[length(chip_batch_imgs) + 1]]      <- arr
      chip_batch_extents[[length(chip_batch_extents) + 1]] <- c(chip_xmin, chip_ymin, chip_xmax, chip_ymax)

      if (length(chip_batch_imgs) >= batch_size) flush_batch()
    }
  }
  flush_batch()  # Remaining chips

  if (verbose) cli::cli_progress_done()

  if (nrow(det_boxes) == 0) {
    cli::cli_alert_warning("No detections above score_thresh = {score_thresh}.")
    return(sf::st_sf(
      class_id   = integer(),
      class_name = character(),
      score      = numeric(),
      geometry   = sf::st_sfc(crs = terra::crs(rast))
    ))
  }

  # ------ Global NMS in geographic space ------
  keep_idx <- .nms_geo(det_boxes, det_scores, nms_thresh)
  det_boxes   <- det_boxes[keep_idx, , drop = FALSE]
  det_scores  <- det_scores[keep_idx]
  det_cls_ids <- det_cls_ids[keep_idx]

  # ------ Build sf output ------
  raster_crs <- terra::crs(rast)

  polys <- lapply(seq_len(nrow(det_boxes)), function(i) {
    b <- det_boxes[i, ]
    sf::st_polygon(list(matrix(
      c(b[1], b[2],  b[3], b[2],  b[3], b[4],  b[1], b[4],  b[1], b[2]),
      ncol = 2, byrow = TRUE
    )))
  })

  cls_names <- if (!is.null(class_map)) {
    id_to_name <- setNames(names(class_map), class_map)
    id_to_name[as.character(det_cls_ids)]
  } else {
    rep(NA_character_, length(det_cls_ids))
  }

  result <- sf::st_sf(
    class_id   = det_cls_ids,
    class_name = cls_names,
    score      = round(det_scores, 4),
    geometry   = sf::st_sfc(polys, crs = raster_crs)
  )

  cli::cli_alert_success(
    "{nrow(result)} detection(s) in {length(unique(det_cls_ids))} class(es) after NMS."
  )
  result
}


# Internal: greedy NMS on geographic bounding boxes
.nms_geo <- function(boxes, scores, iou_thresh) {
  if (nrow(boxes) == 0) return(integer(0))

  ord  <- order(scores, decreasing = TRUE)
  keep <- integer(0)

  while (length(ord) > 0) {
    i    <- ord[1]
    keep <- c(keep, i)
    if (length(ord) == 1) break

    rest <- ord[-1]
    ious <- .box_iou_vec(boxes[i, ], boxes[rest, , drop = FALSE])
    ord  <- rest[ious < iou_thresh]
  }
  keep
}


#' Visualise detections overlaid on a raster
#'
#' Plots a `SpatRaster` (RGB or single band) and overlays detection
#' bounding boxes coloured by class.
#'
#' @param rast A `SpatRaster`.
#' @param detections An `sf` object from [predict_raster()].
#' @param r_band,g_band,b_band Integer. Band indices for RGB display.
#'   Default 1, 2, 3.
#' @param alpha Numeric. Transparency for box fills. Default 0.2.
#' @param title Character. Plot title.
#'
#' @return Invisibly returns the ggplot2 object.
#'
#' @export
plot_detections <- function(rast,
                            detections,
                            r_band = 1L, g_band = 2L, b_band = 3L,
                            alpha  = 0.2,
                            title  = "Detections") {

  if (!requireNamespace("ggplot2", quietly = TRUE))
    cli::cli_abort("Package {.pkg ggplot2} required for plotting.")

  # Convert raster to data frame for ggplot
  nb <- terra::nlyr(rast)
  if (nb >= 3) {
    rgb_rast <- rast[[c(r_band, g_band, b_band)]]
    names(rgb_rast) <- c("r", "g", "b")
    df <- as.data.frame(rgb_rast, xy = TRUE)
    # Stretch to 0-1
    for (col in c("r", "g", "b")) {
      rng <- range(df[[col]], na.rm = TRUE)
      if (diff(rng) > 0) df[[col]] <- (df[[col]] - rng[1]) / diff(rng)
    }
    df$fill_rgb <- rgb(pmax(0, pmin(1, df$r)),
                       pmax(0, pmin(1, df$g)),
                       pmax(0, pmin(1, df$b)))
    p <- ggplot2::ggplot() +
      ggplot2::geom_raster(data = df, ggplot2::aes(x = x, y = y, fill = fill_rgb)) +
      ggplot2::scale_fill_identity()
  } else {
    df <- as.data.frame(rast[[1]], xy = TRUE)
    names(df)[3] <- "value"
    p <- ggplot2::ggplot() +
      ggplot2::geom_raster(data = df, ggplot2::aes(x = x, y = y, fill = value)) +
      ggplot2::scale_fill_viridis_c()
  }

  if (nrow(detections) > 0) {
    det_df <- as.data.frame(sf::st_coordinates(
      sf::st_cast(detections, "POLYGON")
    ))
    det_full <- cbind(det_df,
                      class_name = rep(detections$class_name, each = 5),
                      score      = rep(round(detections$score, 2), each = 5))

    p <- p +
      ggplot2::geom_polygon(
        data    = det_full,
        mapping = ggplot2::aes(x = X, y = Y,
                               group = interaction(L2, class_name),
                               colour = class_name,
                               fill   = class_name),
        alpha     = alpha,
        linewidth = 0.6
      )
  }

  p <- p +
    ggplot2::coord_equal() +
    ggplot2::labs(title = title, x = "Easting", y = "Northing") +
    ggplot2::theme_minimal()

  print(p)
  invisible(p)
}
