library(testthat)
library(terra)
library(sf)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
make_synth_raster <- function(nr = 200, nc = 200, nb = 3, crs = "EPSG:4326") {
  r <- terra::rast(
    nrows = nr, ncols = nc, nlyr = nb,
    xmin = 0, xmax = nc, ymin = 0, ymax = nr,
    crs  = crs
  )
  terra::values(r) <- runif(nr * nc * nb)
  r
}

make_annotation <- function(xmin, ymin, xmax, ymax, class, crs = "EPSG:4326") {
  poly <- sf::st_polygon(list(matrix(
    c(xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, xmin, ymin),
    ncol = 2, byrow = TRUE
  )))
  sf::st_sf(class = class, geometry = sf::st_sfc(poly, crs = crs))
}

# ─────────────────────────────────────────────────────────────────────────────
# make_chips
# ─────────────────────────────────────────────────────────────────────────────
test_that("make_chips returns correct data frame structure", {
  r     <- make_synth_raster()
  chips <- make_chips(r, chip_size = 64, overlap = 0, out_dir = tempdir(),
                      min_valid = 0)
  expected_cols <- c("path", "row_idx", "col_idx", "xmin", "ymin", "xmax", "ymax", "valid_frac")
  expect_true(all(expected_cols %in% names(chips)))
  expect_true(nrow(chips) > 0)
  expect_true(all(file.exists(chips$path)))
})

test_that("make_chips chips are exactly chip_size x chip_size", {
  r     <- make_synth_raster(nr = 100, nc = 100)
  chips <- make_chips(r, chip_size = 32, overlap = 0, out_dir = tempdir(),
                      min_valid = 0)
  for (p in chips$path) {
    cr <- terra::rast(p)
    expect_equal(terra::nrow(cr), 32)
    expect_equal(terra::ncol(cr), 32)
  }
})

test_that("make_chips xmin/ymin/xmax/ymax are consistent with stored extents", {
  # Bug: original code stored ext_chip$xmin AFTER resample which could shift
  # compared to the planned chip origin. Fixed version stores the planned extents.
  r     <- make_synth_raster(nr = 50, nc = 50)
  chips <- make_chips(r, chip_size = 30, overlap = 0, out_dir = tempdir(),
                      min_valid = 0)
  # All stored xmin should be >= raster xmin
  ext0  <- terra::ext(r)
  expect_true(all(chips$xmin >= ext0$xmin))
  expect_true(all(chips$ymin >= ext0$ymin))
  expect_true(all(chips$xmax <= ext0$xmax + terra::res(r)[1] * 30))
})

# ─────────────────────────────────────────────────────────────────────────────
# assign_labels_to_chips
# ─────────────────────────────────────────────────────────────────────────────
test_that("assign_labels_to_chips basic assignment works", {
  r      <- make_synth_raster(nr = 100, nc = 100)
  chips  <- make_chips(r, chip_size = 50, overlap = 0, out_dir = tempdir(),
                       min_valid = 0)
  # Box centred in raster
  ann    <- make_annotation(30, 30, 70, 70, "building")
  labels <- assign_labels_to_chips(chips, ann, chip_size = 50,
                                   iou_threshold = 0.01,
                                   class_map = c(building = 1L))
  expect_equal(length(labels), nrow(chips))
  n_pos <- sum(sapply(labels, nrow) > 0)
  expect_true(n_pos >= 1)
})

test_that("assign_labels_to_chips pixel coords stay within [0, chip_size]", {
  r     <- make_synth_raster(nr = 100, nc = 100)
  chips <- make_chips(r, chip_size = 50, overlap = 0, out_dir = tempdir(),
                      min_valid = 0)
  ann   <- make_annotation(10, 10, 90, 90, "car")
  labs  <- assign_labels_to_chips(chips, ann, chip_size = 50,
                                  iou_threshold = 0.01,
                                  class_map = c(car = 1L))
  for (df in labs) {
    if (nrow(df) == 0) next
    expect_true(all(df$xmin >= 0 & df$xmax <= 50))
    expect_true(all(df$ymin >= 0 & df$ymax <= 50))
    expect_true(all(df$xmin < df$xmax))
    expect_true(all(df$ymin < df$ymax))
  }
})

test_that("assign_labels_to_chips errors on unknown class (BUG FIX 3)", {
  r     <- make_synth_raster(nr = 60, nc = 60)
  chips <- make_chips(r, chip_size = 30, overlap = 0, out_dir = tempdir(),
                      min_valid = 0)
  ann   <- make_annotation(5, 5, 25, 25, "unknown_class")
  expect_error(
    assign_labels_to_chips(chips, ann, chip_size = 30,
                           iou_threshold = 0.01,
                           class_map = c(building = 1L)),
    "not in.*class_map"
  )
})

# ─────────────────────────────────────────────────────────────────────────────
# detection_collate_fn
# ─────────────────────────────────────────────────────────────────────────────
test_that("detection_collate_fn assembles batch correctly (BUG FIX in sapply → vapply)", {
  skip_if_not_installed("torch")
  fake_item <- function(n) {
    list(
      image    = torch::torch_randn(c(3L, 32L, 32L)),
      target   = list(
        boxes  = torch::torch_zeros(c(n, 4L)),
        labels = torch::torch_ones(n, dtype = torch::torch_long())
      ),
      chip_idx = 1L
    )
  }
  batch <- list(fake_item(2L), fake_item(0L), fake_item(3L))
  out   <- detection_collate_fn(batch)

  expect_equal(length(out$images),  3L)
  expect_equal(length(out$targets), 3L)
  expect_equal(out$targets[[2]]$boxes$shape[1], 0L)
  expect_equal(out$targets[[3]]$labels$shape[1], 3L)
  # chip_ids should be a plain integer vector now (vapply fix)
  expect_type(out$chip_ids, "integer")
})

# ─────────────────────────────────────────────────────────────────────────────
# NMS and IoU helpers
# ─────────────────────────────────────────────────────────────────────────────
test_that(".nms_geo keeps highest-score box when two overlap (BUG FIX 18)", {
  boxes  <- matrix(c(0, 0, 10, 10,
                     1, 1, 11, 11), ncol = 4, byrow = TRUE)
  scores <- c(0.9, 0.8)
  keep   <- geodetect:::.nms_geo(boxes, scores, 0.3)
  expect_equal(length(keep), 1L)
  expect_equal(keep, 1L)
})

test_that(".nms_geo keeps non-overlapping boxes", {
  boxes  <- matrix(c(0, 0, 5, 5,
                     100, 100, 110, 110), ncol = 4, byrow = TRUE)
  scores <- c(0.9, 0.8)
  keep   <- geodetect:::.nms_geo(boxes, scores, 0.5)
  expect_equal(length(keep), 2L)
})

test_that(".box_iou_vec returns 1 for identical boxes, 0 for disjoint", {
  box   <- c(0, 0, 10, 10)
  boxes <- matrix(c(
    0,  0, 10, 10,
    50, 50, 60, 60
  ), ncol = 4, byrow = TRUE)
  ious  <- geodetect:::.box_iou_vec(box, boxes)
  expect_equal(ious[1], 1.0,  tolerance = 1e-5)
  expect_true(ious[2] < 1e-5)
})

# ─────────────────────────────────────────────────────────────────────────────
# split_chips
# ─────────────────────────────────────────────────────────────────────────────
test_that("split_chips row names are reset (BUG FIX 19)", {
  chips_df    <- data.frame(path = paste0(1:20, ".tif"), row_idx = 1:20,
                             col_idx = 1:20, xmin = 0, ymin = 0,
                             xmax = 1, ymax = 1, valid_frac = 1)
  labels_list <- replicate(20, data.frame(), simplify = FALSE)
  splits      <- split_chips(chips_df, labels_list, val_frac = 0.25, seed = 1)

  expect_equal(rownames(splits$train$chips_df), as.character(seq_len(nrow(splits$train$chips_df))))
  expect_equal(rownames(splits$val$chips_df),   as.character(seq_len(nrow(splits$val$chips_df))))
  expect_equal(nrow(splits$train$chips_df) + nrow(splits$val$chips_df), 20L)
  expect_equal(length(splits$train$labels_list), nrow(splits$train$chips_df))
})

# ─────────────────────────────────────────────────────────────────────────────
# geo_detection_dataset (smoke test — requires torch)
# ─────────────────────────────────────────────────────────────────────────────
test_that("geo_detection_dataset returns correct item structure", {
  skip_if_not_installed("torch")
  skip_if_not_installed("torchvision")

  r     <- make_synth_raster(nr = 64, nc = 64, nb = 3)
  chips <- make_chips(r, chip_size = 32, overlap = 0, out_dir = tempdir(),
                      min_valid = 0)
  labs  <- replicate(nrow(chips), data.frame(
    xmin = 2, ymin = 2, xmax = 10, ymax = 10,
    class_name = "obj", class_id = 1L,
    stringsAsFactors = FALSE
  ), simplify = FALSE)

  ds   <- geo_detection_dataset(chips, labs, chip_size = 32, n_bands = 3,
                                 augment = FALSE, normalize = FALSE)
  item <- ds$.getitem(1L)

  expect_equal(item$image$shape, c(3L, 32L, 32L))
  expect_equal(item$target$boxes$shape, c(1L, 4L))
  expect_equal(item$target$labels$shape, 1L)
})

test_that("geo_detection_dataset handles empty chips (no annotations)", {
  skip_if_not_installed("torch")
  skip_if_not_installed("torchvision")

  r     <- make_synth_raster(nr = 64, nc = 64, nb = 3)
  chips <- make_chips(r, chip_size = 32, overlap = 0, out_dir = tempdir(),
                      min_valid = 0)
  labs  <- replicate(nrow(chips), data.frame(
    xmin = numeric(), ymin = numeric(), xmax = numeric(), ymax = numeric(),
    class_name = character(), class_id = integer(),
    stringsAsFactors = FALSE
  ), simplify = FALSE)

  ds   <- geo_detection_dataset(chips, labs, chip_size = 32, n_bands = 3)
  item <- ds$.getitem(1L)

  expect_equal(item$target$boxes$shape[1], 0L)
})
