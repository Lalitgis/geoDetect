#' geodetect: Geospatial Object Detection Using Deep Learning in R
#'
#' @description
#' `geodetect` provides a complete, Python-free workflow for training and
#' applying deep learning object detectors to geospatial raster imagery
#' (satellite, aerial, drone). It is built on the `torch` and `torchvision`
#' R packages and the `terra` raster engine.
#'
#' ## Workflow
#'
#' ```
#' 1. make_chips()               Tile a large raster into chips
#' 2. assign_labels_to_chips()   Map vector annotations → per-chip labels
#' 3. geo_detection_dataset()    Wrap chips + labels as a torch Dataset
#' 4. build_detector()           Construct Faster R-CNN model
#' 5. train_detector()           Train with early stopping + checkpointing
#' 6. predict_raster()           Sliding-window inference → sf bounding boxes
#' 7. plot_detections()          Visualise results overlaid on raster
#' ```
#'
#' ## Key design decisions
#'
#' - **No Python dependency.** All deep learning runs via R's native `torch`
#'   package (libTorch bindings), installable from CRAN.
#' - **CRS-aware throughout.** Detected bounding boxes are back-projected to
#'   real-world coordinates and returned as `sf` objects.
#' - **Handles arbitrarily large rasters.** Sliding-window tiling with
#'   configurable overlap ensures large scenes are fully covered.
#' - **Global NMS.** Overlapping detections from adjacent chips are deduplicated
#'   in geographic space before returning results.
#' - **Transfer learning.** ImageNet-pretrained ResNet-50 backbone reduces the
#'   amount of labelled geospatial training data required.
#'
#' @keywords internal
"_PACKAGE"


#' Check whether a CUDA-capable GPU is available
#'
#' @return Logical.
#' @export
has_gpu <- function() {
  if (!requireNamespace("torch", quietly = TRUE)) return(FALSE)
  torch::cuda_is_available()
}


#' Summarise a chips data frame
#'
#' @param chips_df Data frame from [make_chips()].
#' @return Invisibly returns `chips_df`; prints a summary.
#' @export
chips_summary <- function(chips_df) {
  cli::cli_h2("Chips summary")
  cli::cli_bullets(c(
    "*" = "Total chips:       {nrow(chips_df)}",
    "*" = "Grid rows:         {max(chips_df$row_idx)}",
    "*" = "Grid columns:      {max(chips_df$col_idx)}",
    "*" = "Mean valid pixels: {round(mean(chips_df$valid_frac) * 100, 1)}%",
    "*" = "Output directory:  {.path {dirname(chips_df$path[1])}}"
  ))
  invisible(chips_df)
}


#' Split chips into training and validation sets
#'
#' Randomly splits a chips data frame and its associated labels list into
#' training and validation subsets, preserving the correspondence between
#' chips and labels.
#'
#' @param chips_df Data frame from [make_chips()].
#' @param labels_list List from [assign_labels_to_chips()].
#' @param val_frac Numeric in (0, 1). Fraction for validation. Default 0.2.
#' @param seed Integer. Random seed. Default 42.
#'
#' @return A list with elements `train` and `val`, each a list of
#'   `chips_df` and `labels_list`.
#'
#' @export
split_chips <- function(chips_df, labels_list, val_frac = 0.2, seed = 42L) {
  if (nrow(chips_df) != length(labels_list))
    cli::cli_abort("chips_df and labels_list must have the same length.")
  if (val_frac <= 0 || val_frac >= 1)
    cli::cli_abort("{.arg val_frac} must be in (0, 1).")

  set.seed(seed)
  n   <- nrow(chips_df)
  val_idx   <- sample(n, size = floor(n * val_frac))
  train_idx <- setdiff(seq_len(n), val_idx)

  list(
    train = list(
      chips_df    = chips_df[train_idx, ],
      labels_list = labels_list[train_idx]
    ),
    val = list(
      chips_df    = chips_df[val_idx, ],
      labels_list = labels_list[val_idx]
    )
  )
}


#' Export detections to a GeoPackage or shapefile
#'
#' @param detections An `sf` object from [predict_raster()].
#' @param path Character. Output file path. Extension determines format:
#'   `.gpkg` → GeoPackage, `.shp` → shapefile, `.geojson` → GeoJSON.
#' @param overwrite Logical. Overwrite existing file. Default FALSE.
#'
#' @return Invisibly returns `path`.
#' @export
export_detections <- function(detections, path, overwrite = FALSE) {
  if (!inherits(detections, "sf"))
    cli::cli_abort("{.arg detections} must be an {.cls sf} object.")
  if (file.exists(path) && !overwrite)
    cli::cli_abort("File exists. Set {.arg overwrite = TRUE} to replace.")

  sf::st_write(detections, path, delete_dsn = overwrite, quiet = TRUE)
  cli::cli_alert_success("Exported {nrow(detections)} detection(s) to {.path {path}}")
  invisible(path)
}
