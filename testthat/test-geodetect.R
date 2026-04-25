library(testthat)
library(terra)
library(sf)

test_that("make_chips produces correct output structure", {
  # Synthetic 3-band raster, 200x200 pixels
  r <- terra::rast(
    nrows = 200, ncols = 200, nlyr = 3,
    xmin = 0, xmax = 200, ymin = 0, ymax = 200,
    crs = "EPSG:4326"
  )
  terra::values(r) <- runif(200 * 200 * 3)

  td <- tempdir()
  chips <- make_chips(r, chip_size = 64, overlap = 0.25, out_dir = td,
                      prefix = "test", min_valid = 0.0)

  expect_s3_class(chips, "data.frame")
  expect_true(all(c("path", "row_idx", "col_idx",
                    "xmin", "ymin", "xmax", "ymax", "valid_frac") %in% names(chips)))
  expect_true(nrow(chips) > 0)
  expect_true(all(file.exists(chips$path)))
  expect_true(all(chips$valid_frac >= 0 & chips$valid_frac <= 1))

  # Each chip should be exactly chip_size x chip_size
  for (p in chips$path[seq_len(min(3, nrow(chips)))]) {
    chip_r <- terra::rast(p)
    expect_equal(terra::nrow(chip_r), 64)
    expect_equal(terra::ncol(chip_r), 64)
  }
})

test_that("assign_labels_to_chips returns correct structure", {
  r <- terra::rast(
    nrows = 100, ncols = 100, nlyr = 3,
    xmin = 0, xmax = 100, ymin = 0, ymax = 100,
    crs = "EPSG:4326"
  )
  terra::values(r) <- runif(100 * 100 * 3)

  chips <- make_chips(r, chip_size = 50, overlap = 0, out_dir = tempdir(),
                      min_valid = 0)

  # Create an annotation polygon in the middle of the raster
  box <- sf::st_sfc(sf::st_polygon(list(matrix(
    c(30, 30, 70, 30, 70, 70, 30, 70, 30, 30),
    ncol = 2, byrow = TRUE
  ))), crs = "EPSG:4326")
  ann <- sf::st_sf(class = "building", geometry = box)

  labels <- assign_labels_to_chips(
    chips_df      = chips,
    labels_sf     = ann,
    chip_size     = 50,
    iou_threshold = 0.01
  )

  expect_type(labels, "list")
  expect_equal(length(labels), nrow(chips))

  # At least one chip should have an annotation
  n_with_labels <- sum(sapply(labels, nrow) > 0)
  expect_true(n_with_labels >= 1)

  # Check column names of non-empty label data frames
  for (df in labels) {
    if (nrow(df) > 0) {
      expect_true(all(c("xmin", "ymin", "xmax", "ymax",
                        "class_name", "class_id") %in% names(df)))
      # Pixel coords should be within [0, chip_size]
      expect_true(all(df$xmin >= 0 & df$xmax <= 50))
      expect_true(all(df$ymin >= 0 & df$ymax <= 50))
      expect_true(all(df$xmin < df$xmax))
      expect_true(all(df$ymin < df$ymax))
    }
  }
})

test_that("split_chips preserves lengths", {
  chips_df <- data.frame(
    path = paste0("x", 1:100, ".tif"),
    row_idx = 1:100, col_idx = 1:100,
    xmin = 0, ymin = 0, xmax = 1, ymax = 1,
    valid_frac = 1.0
  )
  labels_list <- replicate(100, data.frame(), simplify = FALSE)

  splits <- split_chips(chips_df, labels_list, val_frac = 0.2, seed = 1)

  expect_equal(length(splits$train$labels_list), nrow(splits$train$chips_df))
  expect_equal(length(splits$val$labels_list),   nrow(splits$val$chips_df))
  expect_equal(nrow(splits$train$chips_df) + nrow(splits$val$chips_df), 100)
})

test_that("global NMS (.nms_geo) suppresses overlapping boxes correctly", {
  # Two identical boxes → should keep only one
  boxes  <- matrix(c(0, 0, 10, 10,
                     0, 0, 10, 10), ncol = 4, byrow = TRUE)
  scores <- c(0.9, 0.8)
  keep   <- geodetect:::.nms_geo(boxes, scores, iou_thresh = 0.5)
  expect_equal(length(keep), 1L)
  expect_equal(keep, 1L)  # Higher-score box kept

  # Two non-overlapping boxes → both kept
  boxes2  <- matrix(c(0, 0, 5, 5,
                      10, 10, 15, 15), ncol = 4, byrow = TRUE)
  scores2 <- c(0.9, 0.8)
  keep2   <- geodetect:::.nms_geo(boxes2, scores2, iou_thresh = 0.5)
  expect_equal(length(keep2), 2L)
})

test_that("IoU computation is correct (.box_iou_vec)", {
  box   <- c(0, 0, 10, 10)
  boxes <- matrix(c(
    0,  0, 10, 10,   # identical → IoU = 1
    5,  5, 15, 15,   # partial overlap
    20, 20, 30, 30   # no overlap → IoU = 0
  ), ncol = 4, byrow = TRUE)

  ious <- geodetect:::.box_iou_vec(box, boxes)
  expect_equal(length(ious), 3L)
  expect_equal(ious[1], 1.0, tolerance = 1e-5)
  expect_true(ious[2] > 0 && ious[2] < 1)
  expect_true(ious[3] < 1e-5)
})

test_that("detection_collate_fn assembles batch correctly", {
  fake_item <- function(n_boxes) {
    list(
      image    = torch::torch_randn(c(3L, 64L, 64L)),
      target   = list(
        boxes  = torch::torch_zeros(c(n_boxes, 4L)),
        labels = torch::torch_ones(n_boxes, dtype = torch::torch_long())
      ),
      chip_idx = 1L
    )
  }

  batch <- list(fake_item(2), fake_item(0), fake_item(3))
  out   <- detection_collate_fn(batch)

  expect_equal(length(out$images),  3L)
  expect_equal(length(out$targets), 3L)
  expect_equal(out$targets[[2]]$boxes$shape[1], 0L)
  expect_equal(out$targets[[3]]$labels$shape[1], 3L)
})
