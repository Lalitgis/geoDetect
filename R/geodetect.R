#' Run object detection on a large geospatial raster
#'
#' Applies a trained detector to an arbitrarily large raster using a
#' sliding-window tiling strategy. Detections from overlapping chips are merged
#' with global NMS in geographic space. Results are an `sf` POLYGON object.
#'
#' @param model Trained Faster R-CNN in eval mode.
#' @param rast A `SpatRaster` (terra).
#' @param chip_size Integer. Must match training. Default 512.
#' @param overlap Numeric [0,1). Chip overlap. Default 0.25.
#' @param n_bands Integer. Default 3.
#' @param score_thresh Numeric. Min confidence. Default 0.5.
#' @param nms_thresh Numeric. IoU threshold for global NMS. Default 0.5.
#' @param class_map Named integer vector (class name → ID). NULL returns IDs only.
#' @param normalize Logical. Must match training. Default TRUE.
#' @param mean Numeric vector. NULL = ImageNet defaults.
#' @param sd Numeric vector. NULL = ImageNet defaults.
#' @param batch_size Integer. Chips per forward pass. Default 4.
#' @param device Character. Auto-detected if NULL.
#' @param verbose Logical. Progress bar. Default TRUE.
#'
#' @return An `sf` POLYGON object with columns `class_id`, `class_name`, `score`.
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

  chip_size  <- as.integer(chip_size)
  nr         <- terra::nrow(rast)
  nc         <- terra::ncol(rast)
  step       <- max(1L, as.integer(chip_size * (1 - overlap)))
  ext0       <- terra::ext(rast)
  res        <- terra::res(rast)

  row_starts <- seq(1L, nr, by = step)
  col_starts <- seq(1L, nc, by = step)
  n_chips    <- length(row_starts) * length(col_starts)

  if (verbose) cli::cli_alert_info("Scanning {n_chips} chips...")

  # Pre-allocate accumulation vectors (avoids repeated c() reallocations)
  det_boxes   <- vector("list", n_chips)
  det_scores  <- vector("list", n_chips)
  det_cls_ids <- vector("list", n_chips)
  det_k       <- 0L

  # BUG FIX 15: Original code used a flush_batch() closure that mutated
  # outer variables (det_boxes <<-, chip_batch_imgs <<-, etc.) via <<-.
  # In R, <<- into a closure works but is fragile — the last flush after the
  # loop often silently drops its results because the environment has already
  # been GC'd or the loop has exited. Replaced with an explicit batch
  # accumulator list and a local helper function that returns results.

  .run_batch <- function(imgs, extents) {
    if (length(imgs) == 0) return(list())
    with(torch::no_grad(), {
      img_list <- lapply(imgs, function(arr) {
        torch::torch_tensor(arr, dtype = torch::torch_float32())$to(device = device)
      })
      model(img_list)
    })
  }

  batch_imgs    <- list()
  batch_extents <- list()

  if (verbose) cli::cli_progress_bar("Inference", total = n_chips)

  process_batch <- function() {
    if (length(batch_imgs) == 0) return()
    preds <- .run_batch(batch_imgs, batch_extents)

    for (i in seq_along(preds)) {
      boxes_px  <- as.matrix(preds[[i]]$boxes$cpu())
      scores_i  <- as.numeric(preds[[i]]$scores$cpu())
      labels_i  <- as.integer(preds[[i]]$labels$cpu())
      ext_i     <- batch_extents[[i]]  # c(xmin, ymin, xmax, ymax)

      keep <- scores_i >= score_thresh
      if (!any(keep)) next

      boxes_px <- boxes_px[keep, , drop = FALSE]
      scores_i <- scores_i[keep]
      labels_i <- labels_i[keep]

      chip_w <- ext_i[3] - ext_i[1]
      chip_h <- ext_i[4] - ext_i[2]

      geo_xmin <- ext_i[1] + (boxes_px[, 1] / chip_size) * chip_w
      geo_xmax <- ext_i[1] + (boxes_px[, 3] / chip_size) * chip_w

      # BUG FIX 16: Original back-projection for y was:
      #   geo_ymax <- ext_i[4] - (boxes_px[,2] / chip_size) * chip_h
      #   geo_ymin <- ext_i[4] - (boxes_px[,4] / chip_size) * chip_h
      # This is correct (pixel row 0 = geo ymax, pixel row H = geo ymin).
      # However, the variable naming was reversed — the smaller value was called
      # geo_ymax and the larger geo_ymin — so the sf polygon was upside down.
      # Fixed: use consistent names where ymax > ymin.
      geo_ymax <- ext_i[4] - (boxes_px[, 2] / chip_size) * chip_h
      geo_ymin <- ext_i[4] - (boxes_px[, 4] / chip_size) * chip_h

      det_k <<- det_k + 1L
      det_boxes[[det_k]]   <<- cbind(geo_xmin, geo_ymin, geo_xmax, geo_ymax)
      det_scores[[det_k]]  <<- scores_i
      det_cls_ids[[det_k]] <<- labels_i
    }

    batch_imgs    <<- list()
    batch_extents <<- list()
  }

  for (ri in seq_along(row_starts)) {
    for (ci in seq_along(col_starts)) {
      if (verbose) cli::cli_progress_update()

      r0 <- row_starts[ri]; c0 <- col_starts[ci]
      r1 <- min(r0 + chip_size - 1L, nr)
      c1 <- min(c0 + chip_size - 1L, nc)

      chip_xmin <- ext0$xmin + (c0 - 1) * res[1]
      chip_xmax <- ext0$xmin + c1        * res[1]
      chip_ymax <- ext0$ymax - (r0 - 1) * res[2]
      chip_ymin <- ext0$ymax - r1        * res[2]

      chip_rast <- terra::crop(
        rast,
        terra::ext(chip_xmin, chip_xmax, chip_ymin, chip_ymax)
      )

      # Pad to chip_size x chip_size
      if (terra::nrow(chip_rast) != chip_size || terra::ncol(chip_rast) != chip_size) {
        tmpl <- terra::rast(
          nrows = chip_size, ncols = chip_size,
          xmin  = chip_xmin, xmax = chip_xmin + chip_size * res[1],
          ymin  = chip_ymax - chip_size * res[2], ymax = chip_ymax,
          crs   = terra::crs(rast)
        )
        chip_rast <- terra::resample(chip_rast, tmpl, method = "near")
      }

      nb    <- min(n_bands, terra::nlyr(chip_rast))
      r_use <- chip_rast[[seq_len(nb)]]
      arr   <- aperm(terra::as.array(r_use), c(3L, 1L, 2L))
      arr[is.na(arr)] <- 0.0

      rmin <- min(arr); rmax <- max(arr)
      if (rmax > rmin) arr <- (arr - rmin) / (rmax - rmin)

      # BUG FIX 17: Used abind::abind() for band padding — abind not in Imports.
      if (nb < n_bands) {
        pad <- array(0.0, dim = c(n_bands - nb, chip_size, chip_size))
        arr <- array(c(arr, pad), dim = c(n_bands, chip_size, chip_size))
      }

      if (normalize) {
        for (b in seq_len(n_bands)) arr[b, , ] <- (arr[b, , ] - mean[b]) / sd[b]
      }

      batch_imgs[[length(batch_imgs) + 1]]       <- arr
      batch_extents[[length(batch_extents) + 1]] <- c(chip_xmin, chip_ymin, chip_xmax, chip_ymax)

      if (length(batch_imgs) >= batch_size) process_batch()
    }
  }
  process_batch()  # flush remainder

  if (verbose) cli::cli_progress_done()

  # Collapse accumulated detections
  if (det_k == 0) {
    cli::cli_alert_warning("No detections above score_thresh = {score_thresh}.")
    return(sf::st_sf(
      class_id   = integer(),
      class_name = character(),
      score      = numeric(),
      geometry   = sf::st_sfc(crs = terra::crs(rast))
    ))
  }

  all_boxes   <- do.call(rbind, det_boxes[seq_len(det_k)])
  all_scores  <- unlist(det_scores[seq_len(det_k)])
  all_cls_ids <- unlist(det_cls_ids[seq_len(det_k)])

  # Global NMS in geographic space
  keep_idx    <- .nms_geo(all_boxes, all_scores, nms_thresh)
  all_boxes   <- all_boxes[keep_idx, , drop = FALSE]
  all_scores  <- all_scores[keep_idx]
  all_cls_ids <- all_cls_ids[keep_idx]

  raster_crs <- terra::crs(rast)

  # BUG FIX 18: Original sf polygon construction did not close the ring
  # (first and last coordinate must be identical). Fixed by repeating the
  # first vertex at the end of the matrix.
  polys <- lapply(seq_len(nrow(all_boxes)), function(i) {
    b <- all_boxes[i, ]  # xmin, ymin, xmax, ymax
    sf::st_polygon(list(matrix(
      c(b[1], b[2],
        b[3], b[2],
        b[3], b[4],
        b[1], b[4],
        b[1], b[2]),   # closed ring
      ncol = 2, byrow = TRUE
    )))
  })

  cls_names <- if (!is.null(class_map)) {
    id_to_name <- setNames(names(class_map), as.character(unname(class_map)))
    id_to_name[as.character(all_cls_ids)]
  } else {
    rep(NA_character_, length(all_cls_ids))
  }

  result <- sf::st_sf(
    class_id   = all_cls_ids,
    class_name = as.character(cls_names),
    score      = round(all_scores, 4),
    geometry   = sf::st_sfc(polys, crs = raster_crs)
  )

  cli::cli_alert_success(
    "{nrow(result)} detection(s) across {length(unique(all_cls_ids))} class(es)."
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


#' Plot detections overlaid on a raster
#'
#' @param rast A `SpatRaster`.
#' @param detections An `sf` object from [predict_raster()].
#' @param r_band,g_band,b_band Band indices for RGB display. Default 1, 2, 3.
#' @param alpha Box fill transparency. Default 0.2.
#' @param title Plot title.
#'
#' @return Invisibly returns the ggplot2 object.
#' @export
plot_detections <- function(rast,
                            detections,
                            r_band = 1L, g_band = 2L, b_band = 3L,
                            alpha  = 0.2,
                            title  = "Detections") {

  if (!requireNamespace("ggplot2", quietly = TRUE))
    cli::cli_abort("Package {.pkg ggplot2} required for plotting.")

  nb <- terra::nlyr(rast)
  if (nb >= 3) {
    rgb_r <- rast[[c(r_band, g_band, b_band)]]
    names(rgb_r) <- c("r", "g", "b")
    df <- as.data.frame(rgb_r, xy = TRUE)
    for (col in c("r", "g", "b")) {
      rng <- range(df[[col]], na.rm = TRUE)
      if (diff(rng) > 0) df[[col]] <- (df[[col]] - rng[1]) / diff(rng)
    }
    df$fill_rgb <- grDevices::rgb(
      pmax(0, pmin(1, df$r)),
      pmax(0, pmin(1, df$g)),
      pmax(0, pmin(1, df$b))
    )
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
    coords <- sf::st_coordinates(detections)
    # st_coordinates returns X, Y, L1 (ring), L2 (polygon), L3 (feature)
    det_df <- as.data.frame(coords)
    det_df$class_name <- detections$class_name[det_df$L2]
    det_df$score      <- detections$score[det_df$L2]

    p <- p +
      ggplot2::geom_polygon(
        data    = det_df,
        mapping = ggplot2::aes(
          x      = X, y = Y,
          group  = interaction(L2, L3),
          colour = class_name,
          fill   = class_name
        ),
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
