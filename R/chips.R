#' Create image chips from a large geospatial raster
#'
#' Tiles a large raster (satellite image, aerial photo) into fixed-size,
#' optionally overlapping chips suitable for object detection training or
#' inference. Each chip retains its geographic extent so that detected
#' bounding boxes can be back-projected to real-world coordinates.
#'
#' @param rast A `SpatRaster` object (terra). Must have a defined CRS.
#' @param chip_size Integer. Height and width of each chip in pixels. Default 512.
#' @param overlap Numeric in [0, 1). Fractional overlap between adjacent chips.
#'   For example, 0.25 means chips overlap by 25% of `chip_size`. Default 0.25.
#' @param out_dir Character. Directory to write chip `.tif` files. Created if
#'   absent.
#' @param prefix Character. File name prefix for chip files. Default `"chip"`.
#' @param min_valid Numeric in [0, 1]. Chips where the fraction of valid
#'   (non-NA) pixels is below this threshold are skipped. Default 0.5.
#' @param scale_to_byte Logical. If TRUE, stretch each chip's values linearly
#'   to 0-255 and write as byte (UINT8). Useful for RGB imagery.
#'   Default FALSE.
#'
#' @return A `data.frame` with columns:
#'   \describe{
#'     \item{path}{Absolute path to the chip `.tif`.}
#'     \item{row_idx}{Row index of the chip in the tiling grid.}
#'     \item{col_idx}{Column index of the chip in the tiling grid.}
#'     \item{xmin, ymin, xmax, ymax}{Geographic extent of the chip in the
#'       raster's CRS.}
#'     \item{valid_frac}{Fraction of non-NA pixels in the chip.}
#'   }
#'
#' @details
#' The stride between chip origins is `chip_size * (1 - overlap)`. The last
#' row/column of chips may extend beyond the raster boundary; these are padded
#' with NA values so every chip has exactly `chip_size x chip_size` pixels.
#'
#' @examples
#' \dontrun{
#' library(terra)
#' r <- rast(system.file("ex/elev.tif", package = "terra"))
#' chips <- make_chips(r, chip_size = 64, overlap = 0.25, out_dir = tempdir())
#' head(chips)
#' }
#'
#' @export
make_chips <- function(rast,
                       chip_size   = 512L,
                       overlap     = 0.25,
                       out_dir     = "chips",
                       prefix      = "chip",
                       min_valid   = 0.5,
                       scale_to_byte = FALSE) {

  if (!inherits(rast, "SpatRaster"))
    cli::cli_abort("{.arg rast} must be a {.cls SpatRaster}.")
  if (!is.numeric(chip_size) || chip_size < 16)
    cli::cli_abort("{.arg chip_size} must be a positive integer >= 16.")
  if (!is.numeric(overlap) || overlap < 0 || overlap >= 1)
    cli::cli_abort("{.arg overlap} must be in [0, 1).")
  if (!is.numeric(min_valid) || min_valid < 0 || min_valid > 1)
    cli::cli_abort("{.arg min_valid} must be in [0, 1].")

  chip_size <- as.integer(chip_size)
  fs::dir_create(out_dir)

  nr   <- terra::nrow(rast)
  nc   <- terra::ncol(rast)
  step <- max(1L, as.integer(chip_size * (1 - overlap)))

  row_starts <- seq(1L, nr, by = step)
  col_starts <- seq(1L, nc, by = step)

  ext0 <- terra::ext(rast)
  res  <- terra::res(rast)  # c(x_res, y_res)

  records <- vector("list", length(row_starts) * length(col_starts))
  k <- 0L

  cli::cli_progress_bar(
    "Generating chips",
    total = length(row_starts) * length(col_starts)
  )

  for (ri in seq_along(row_starts)) {
    for (ci in seq_along(col_starts)) {
      k <- k + 1L
      cli::cli_progress_update()

      r0 <- row_starts[ri]
      c0 <- col_starts[ci]
      r1 <- min(r0 + chip_size - 1L, nr)
      c1 <- min(c0 + chip_size - 1L, nc)

      # Crop the chip
      chip_rast <- terra::crop(
        rast,
        terra::ext(
          ext0$xmin + (c0 - 1) * res[1],
          ext0$xmin + c1 * res[1],
          ext0$ymax - r1 * res[2],
          ext0$ymax - (r0 - 1) * res[2]
        )
      )

      # Compute valid fraction
      vals      <- terra::values(chip_rast[[1]])
      valid_frac <- mean(!is.na(vals))

      if (valid_frac < min_valid) {
        records[[k]] <- NULL
        next
      }

      # Pad to exactly chip_size x chip_size if edge chip
      if (terra::nrow(chip_rast) != chip_size ||
          terra::ncol(chip_rast) != chip_size) {
        template <- terra::rast(
          nrows  = chip_size,
          ncols  = chip_size,
          xmin   = ext0$xmin + (c0 - 1) * res[1],
          xmax   = ext0$xmin + (c0 - 1) * res[1] + chip_size * res[1],
          ymin   = ext0$ymax - (r0 - 1) * res[2] - chip_size * res[2],
          ymax   = ext0$ymax - (r0 - 1) * res[2],
          crs    = terra::crs(rast)
        )
        chip_rast <- terra::resample(chip_rast, template, method = "near")
      }

      if (scale_to_byte) {
        chip_rast <- terra::app(chip_rast, fun = function(x) {
          rng <- range(x, na.rm = TRUE)
          if (diff(rng) == 0) return(x * 0)
          ((x - rng[1]) / diff(rng)) * 255
        })
        chip_rast <- terra::clamp(chip_rast, 0, 255)
      }

      ext_chip <- terra::ext(chip_rast)
      fname    <- file.path(
        out_dir,
        sprintf("%s_r%04d_c%04d.tif", prefix, ri, ci)
      )
      terra::writeRaster(chip_rast, fname, overwrite = TRUE, datatype = "FLT4S")

      records[[k]] <- data.frame(
        path       = normalizePath(fname),
        row_idx    = ri,
        col_idx    = ci,
        xmin       = ext_chip$xmin,
        ymin       = ext_chip$ymin,
        xmax       = ext_chip$xmax,
        ymax       = ext_chip$ymax,
        valid_frac = valid_frac,
        stringsAsFactors = FALSE
      )
    }
  }

  cli::cli_progress_done()
  out <- do.call(rbind, Filter(Negate(is.null), records))
  rownames(out) <- NULL
  cli::cli_alert_success("Created {nrow(out)} chips in {.path {out_dir}}")
  out
}


#' Convert an sf bounding-box annotation layer to per-chip label data frames
#'
#' Takes a vector layer of rectangular annotations (bounding boxes around
#' objects of interest) and returns, for each chip listed in a chips data
#' frame, the set of boxes that fall within that chip. The box coordinates are
#' expressed in pixel space (0-indexed, origin top-left) as expected by the
#' detection model.
#'
#' @param chips_df A data frame produced by [make_chips()].
#' @param labels_sf An `sf` object with polygon or bbox geometries representing
#'   annotated objects. Must contain a column `class` (character or factor)
#'   with the class name for each object.
#' @param chip_size Integer. Pixel size of chips. Must match the value used in
#'   [make_chips()]. Default 512.
#' @param iou_threshold Numeric. Minimum IoU between the annotation box and the
#'   chip extent required for the annotation to be assigned to that chip.
#'   Default 0.3.
#' @param class_map Named integer vector mapping class names to integer IDs
#'   (1-indexed; 0 is reserved for background). If NULL, classes are assigned
#'   alphabetically.
#'
#' @return A list of data frames, one per chip row in `chips_df`. Each data
#'   frame contains columns:
#'   \describe{
#'     \item{xmin, ymin, xmax, ymax}{Pixel coordinates (0-indexed).}
#'     \item{class_name}{Character class label.}
#'     \item{class_id}{Integer class ID (1+).}
#'   }
#'   An empty data frame is returned for chips with no annotations.
#'
#' @export
assign_labels_to_chips <- function(chips_df,
                                   labels_sf,
                                   chip_size       = 512L,
                                   iou_threshold   = 0.3,
                                   class_map       = NULL) {

  if (!inherits(labels_sf, "sf"))
    cli::cli_abort("{.arg labels_sf} must be an {.cls sf} object.")
  if (!"class" %in% names(labels_sf))
    cli::cli_abort("{.arg labels_sf} must have a column named {.field class}.")

  chip_size <- as.integer(chip_size)
  classes   <- sort(unique(as.character(labels_sf$class)))

  if (is.null(class_map)) {
    class_map <- setNames(seq_along(classes), classes)
  }

  # Convert labels to bounding boxes in the raster's coordinate system
  label_bboxes <- sf::st_bbox  # we'll use this per polygon below
  labels_geom  <- sf::st_geometry(labels_sf)
  labels_class <- as.character(labels_sf$class)

  result <- lapply(seq_len(nrow(chips_df)), function(i) {
    chip <- chips_df[i, ]

    chip_xmin <- chip$xmin
    chip_ymin <- chip$ymin
    chip_xmax <- chip$xmax
    chip_ymax <- chip$ymax

    # Width and height in geo units
    chip_w <- chip_xmax - chip_xmin
    chip_h <- chip_ymax - chip_ymin

    chip_poly <- sf::st_as_sfc(
      sf::st_bbox(c(xmin = chip_xmin, ymin = chip_ymin,
                    xmax = chip_xmax, ymax = chip_ymax))
    )

    rows <- lapply(seq_along(labels_geom), function(j) {
      lbb <- sf::st_bbox(labels_geom[[j]])

      # Intersection over union with chip
      inter_xmin <- max(lbb["xmin"], chip_xmin)
      inter_ymin <- max(lbb["ymin"], chip_ymin)
      inter_xmax <- min(lbb["xmax"], chip_xmax)
      inter_ymax <- min(lbb["ymax"], chip_ymax)

      if (inter_xmin >= inter_xmax || inter_ymin >= inter_ymax) return(NULL)

      inter_area <- (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
      ann_area   <- (lbb["xmax"] - lbb["xmin"]) * (lbb["ymax"] - lbb["ymin"])
      chip_area  <- chip_w * chip_h
      union_area <- ann_area + chip_area - inter_area
      iou        <- inter_area / union_area

      if (iou < iou_threshold) return(NULL)

      # Clip annotation to chip extent
      clipped_xmin <- max(lbb["xmin"], chip_xmin)
      clipped_ymin <- max(lbb["ymin"], chip_ymin)
      clipped_xmax <- min(lbb["xmax"], chip_xmax)
      clipped_ymax <- min(lbb["ymax"], chip_ymax)

      # Convert to pixel coordinates (0-indexed, top-left origin)
      px_xmin <- ((clipped_xmin - chip_xmin) / chip_w) * chip_size
      px_xmax <- ((clipped_xmax - chip_xmin) / chip_w) * chip_size
      # Note: y flipped because raster row 0 is top, geo y increases up
      px_ymin <- ((chip_ymax - clipped_ymax) / chip_h) * chip_size
      px_ymax <- ((chip_ymax - clipped_ymin) / chip_h) * chip_size

      cls_name <- labels_class[[j]]
      cls_id   <- as.integer(class_map[cls_name])

      data.frame(
        xmin       = px_xmin,
        ymin       = px_ymin,
        xmax       = px_xmax,
        ymax       = px_ymax,
        class_name = cls_name,
        class_id   = cls_id,
        stringsAsFactors = FALSE
      )
    })

    rows_valid <- Filter(Negate(is.null), rows)
    if (length(rows_valid) == 0) {
      return(data.frame(
        xmin = numeric(), ymin = numeric(),
        xmax = numeric(), ymax = numeric(),
        class_name = character(), class_id = integer(),
        stringsAsFactors = FALSE
      ))
    }
    do.call(rbind, rows_valid)
  })

  result
}
