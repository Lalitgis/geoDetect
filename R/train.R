#' Train a geospatial object detector
#'
#' @param model A Faster R-CNN `nn_module` from [build_detector()].
#' @param train_ds A dataset from [geo_detection_dataset()].
#' @param val_ds Optional validation dataset. NULL to skip.
#' @param epochs Integer. Max epochs. Default 30.
#' @param batch_size Integer. Images per batch. Default 4.
#' @param lr Numeric. Initial learning rate. Default 0.005.
#' @param lr_step_size Integer. LR decay every N epochs. Default 10.
#' @param lr_gamma Numeric. LR decay factor. Default 0.1.
#' @param weight_decay Numeric. L2 regularisation. Default 1e-4.
#' @param patience Integer. Early stopping patience. Default 5.
#' @param save_path Character. Best checkpoint path. Default `"geodetect_best.pt"`.
#' @param num_workers Integer. DataLoader workers. Default 0 (safe on Windows).
#' @param device Character. `"cuda"` or `"cpu"`. Auto-detected if NULL.
#' @param class_map Named integer vector saved in checkpoint.
#' @param chip_size Integer. Saved in checkpoint metadata. Default 512.
#' @param n_bands Integer. Saved in checkpoint metadata. Default 3.
#' @param verbose Logical. Print per-epoch loss. Default TRUE.
#'
#' @return A list: `history` (data frame), `best_epoch` (integer), `model`.
#' @export
train_detector <- function(model,
                           train_ds,
                           val_ds       = NULL,
                           epochs       = 30L,
                           batch_size   = 4L,
                           lr           = 0.005,
                           lr_step_size = 10L,
                           lr_gamma     = 0.1,
                           weight_decay = 1e-4,
                           patience     = 5L,
                           save_path    = "geodetect_best.pt",
                           num_workers  = 0L,
                           device       = NULL,
                           class_map    = NULL,
                           chip_size    = 512L,
                           n_bands      = 3L,
                           verbose      = TRUE) {

  if (!requireNamespace("torch", quietly = TRUE)) stop("Package 'torch' required.")

  if (is.null(device))
    device <- if (torch::cuda_is_available()) "cuda" else "cpu"
  cli::cli_alert_info("Training on: {.field {device}}")

  model$to(device = device)

  train_dl <- torch::dataloader(
    dataset    = train_ds,
    batch_size = as.integer(batch_size),
    shuffle    = TRUE,
    collate_fn = detection_collate_fn,
    num_workers = as.integer(num_workers),
    drop_last  = TRUE
  )

  val_dl <- if (!is.null(val_ds)) {
    torch::dataloader(
      dataset     = val_ds,
      batch_size  = as.integer(batch_size),
      shuffle     = FALSE,
      collate_fn  = detection_collate_fn,
      num_workers = as.integer(num_workers)
    )
  } else NULL

  optimizer <- torch::optim_sgd(
    params       = model$parameters(),
    lr           = lr,
    momentum     = 0.9,
    weight_decay = weight_decay
  )

  scheduler <- torch::lr_step(
    optimizer = optimizer,
    step_size = as.integer(lr_step_size),
    gamma     = lr_gamma
  )

  history <- data.frame(
    epoch      = integer(),
    train_loss = numeric(),
    val_loss   = numeric()
  )

  best_val_loss <- Inf
  # BUG FIX 10: `best_epoch` was used after the training loop in
  # cli::cli_alert_success() and list() but was only assigned inside an `if`
  # block — if no epoch ever improved, R would throw "object not found".
  best_epoch    <- 1L
  no_improve    <- 0L

  for (ep in seq_len(epochs)) {

    # ------ Training ------
    model$train()
    train_losses <- numeric(0)

    # BUG FIX 11: coro::loop() was used but `coro` is not in DESCRIPTION
    # Imports. Replaced with standard for() over the dataloader iterator,
    # which is the documented pattern in the `torch` R package.
    dl_iter <- torch::dataloader_make_iter(train_dl)
    repeat {
      batch <- torch::dataloader_next(dl_iter)
      if (is.null(batch)) break

      images  <- lapply(batch$images,  function(x) x$to(device = device))
      targets <- lapply(batch$targets, function(t) list(
        boxes  = t$boxes$to(device = device),
        labels = t$labels$to(device = device)
      ))

      optimizer$zero_grad()

      # BUG FIX 12: `torch::torch_stack(as.list(loss_dict))$sum()` fails
      # because loss_dict is a named list of scalar tensors, not stackable
      # in general. The correct idiom is to sum them with Reduce.
      loss_dict  <- model(images, targets)
      total_loss <- Reduce(`+`, loss_dict)

      total_loss$backward()
      torch::nn_utils_clip_grad_norm_(model$parameters(), max_norm = 5.0)
      optimizer$step()

      train_losses <- c(train_losses, total_loss$item())
    }

    scheduler$step()
    mean_train <- if (length(train_losses) > 0) mean(train_losses) else NA_real_

    # ------ Validation ------
    mean_val <- NA_real_
    if (!is.null(val_dl)) {

      # BUG FIX 13: Original code called model$eval() then inside the loop
      # switched model$train() to get losses — that defeats the purpose of
      # no_grad() and incorrectly accumulates BN statistics.
      # Correct: Faster R-CNN only returns loss dicts in training mode.
      # We must set model$train() for validation loss, but wrap with no_grad()
      # so gradients are not computed. BatchNorm running stats are NOT updated
      # because we do not call optimizer$step().
      model$train()
      val_losses <- numeric(0)

      with(torch::no_grad(), {
        dl_iter_v <- torch::dataloader_make_iter(val_dl)
        repeat {
          batch <- torch::dataloader_next(dl_iter_v)
          if (is.null(batch)) break

          images  <- lapply(batch$images,  function(x) x$to(device = device))
          targets <- lapply(batch$targets, function(t) list(
            boxes  = t$boxes$to(device = device),
            labels = t$labels$to(device = device)
          ))

          loss_dict  <- model(images, targets)
          total_loss <- Reduce(`+`, loss_dict)
          val_losses <- c(val_losses, total_loss$item())
        }
      })

      mean_val <- if (length(val_losses) > 0) mean(val_losses) else NA_real_
    }

    history <- rbind(history, data.frame(
      epoch      = ep,
      train_loss = mean_train,
      val_loss   = mean_val
    ))

    if (verbose) {
      if (!is.na(mean_val)) {
        cli::cli_alert_info(
          "Epoch {ep}/{epochs} | Train: {round(mean_train, 4)} | Val: {round(mean_val, 4)}"
        )
      } else {
        cli::cli_alert_info(
          "Epoch {ep}/{epochs} | Train: {round(mean_train, 4)}"
        )
      }
    }

    monitor_loss <- if (!is.na(mean_val)) mean_val else mean_train

    if (!is.na(monitor_loss) && monitor_loss < best_val_loss) {
      best_val_loss <- monitor_loss
      best_epoch    <- ep
      no_improve    <- 0L

      # BUG FIX 14: Checkpoint saved min_size/max_size as hard-coded 512/1024
      # regardless of what was passed to build_detector().
      # Now reads back from model$transform so the saved values are accurate.
      torch::torch_save(
        list(
          state_dict  = model$state_dict(),
          num_classes = model$roi_heads$box_predictor$cls_score$out_features - 1L,
          class_map   = class_map,
          chip_size   = chip_size,
          n_bands     = n_bands,
          min_size    = model$transform$min_size,
          max_size    = model$transform$max_size
        ),
        save_path
      )
      if (verbose) cli::cli_alert_success("  Checkpoint saved (epoch {ep})")
    } else {
      no_improve <- no_improve + 1L
      if (no_improve >= patience) {
        cli::cli_alert_warning(
          "Early stopping at epoch {ep}. Best epoch: {best_epoch}."
        )
        break
      }
    }
  }

  best_ckpt <- load_detector(save_path, device = "cpu")
  cli::cli_alert_success(
    "Done. Best epoch: {best_epoch} | Loss: {round(best_val_loss, 4)}"
  )

  list(
    history    = history,
    best_epoch = best_epoch,
    model      = best_ckpt$model
  )
}


#' Compute mAP for a trained detector (mAP@50 and mAP@50:95)
#'
#' @param model Trained Faster R-CNN (will be set to eval mode).
#' @param dataset A `geo_detection_dataset`.
#' @param iou_thresholds Numeric vector. Default `seq(0.5, 0.95, by = 0.05)`.
#' @param score_thresh Numeric. Min score to count a prediction. Default 0.5.
#' @param device Character. Default `"cpu"`.
#'
#' @return List: `map50`, `map50_95`.
#' @export
evaluate_detector <- function(model,
                              dataset,
                              iou_thresholds = seq(0.5, 0.95, by = 0.05),
                              score_thresh   = 0.5,
                              device         = "cpu") {

  if (!requireNamespace("torch", quietly = TRUE)) stop("Package 'torch' required.")

  model$to(device = device)
  model$eval()

  dl <- torch::dataloader(
    dataset    = dataset,
    batch_size = 1L,
    shuffle    = FALSE,
    collate_fn = detection_collate_fn
  )

  all_preds <- list()
  all_gt    <- list()
  img_id    <- 0L

  with(torch::no_grad(), {
    dl_iter <- torch::dataloader_make_iter(dl)
    repeat {
      batch <- torch::dataloader_next(dl_iter)
      if (is.null(batch)) break
      img_id <- img_id + 1L

      images  <- lapply(batch$images, function(x) x$to(device = device))
      targets <- batch$targets

      preds <- model(images)

      all_preds[[img_id]] <- list(
        boxes  = preds[[1]]$boxes$cpu(),
        labels = preds[[1]]$labels$cpu(),
        scores = preds[[1]]$scores$cpu()
      )
      all_gt[[img_id]] <- list(
        boxes  = targets[[1]]$boxes,
        labels = targets[[1]]$labels
      )
    }
  })

  aps <- vapply(iou_thresholds, function(thr) {
    .compute_ap_at_iou(all_preds, all_gt, thr, score_thresh)
  }, numeric(1))

  list(map50 = aps[1], map50_95 = mean(aps))
}


# Internal: AP at one IoU threshold (11-point interpolation)
.compute_ap_at_iou <- function(preds_list, gt_list, iou_thresh, score_thresh) {

  all_scores  <- numeric(0)
  all_tp      <- integer(0)
  all_fp      <- integer(0)
  n_gt_total  <- 0L

  for (i in seq_along(preds_list)) {
    pred <- preds_list[[i]]
    gt   <- gt_list[[i]]

    pred_boxes  <- as.matrix(pred$boxes)
    pred_scores <- as.numeric(pred$scores)
    gt_boxes    <- as.matrix(gt$boxes)
    n_gt        <- nrow(gt_boxes)
    n_gt_total  <- n_gt_total + n_gt

    keep        <- pred_scores >= score_thresh
    pred_boxes  <- pred_boxes[keep, , drop = FALSE]
    pred_scores <- pred_scores[keep]

    if (nrow(pred_boxes) == 0) next

    ord        <- order(pred_scores, decreasing = TRUE)
    pred_boxes <- pred_boxes[ord, , drop = FALSE]
    pred_scores <- pred_scores[ord]

    matched <- rep(FALSE, n_gt)

    for (j in seq_len(nrow(pred_boxes))) {
      if (n_gt == 0) {
        all_tp <- c(all_tp, 0L); all_fp <- c(all_fp, 1L)
        all_scores <- c(all_scores, pred_scores[j])
        next
      }
      ious         <- .box_iou_vec(pred_boxes[j, ], gt_boxes)
      best_idx     <- which.max(ious)
      best_iou     <- ious[best_idx]

      if (best_iou >= iou_thresh && !matched[best_idx]) {
        matched[best_idx] <- TRUE
        all_tp <- c(all_tp, 1L); all_fp <- c(all_fp, 0L)
      } else {
        all_tp <- c(all_tp, 0L); all_fp <- c(all_fp, 1L)
      }
      all_scores <- c(all_scores, pred_scores[j])
    }
  }

  if (length(all_scores) == 0 || n_gt_total == 0) return(0.0)

  ord       <- order(all_scores, decreasing = TRUE)
  tp_c      <- cumsum(all_tp[ord])
  fp_c      <- cumsum(all_fp[ord])
  recall    <- tp_c / n_gt_total
  precision <- tp_c / (tp_c + fp_c)

  mean(vapply(seq(0, 1, by = 0.1), function(r) {
    p <- precision[recall >= r]
    if (length(p) == 0) 0.0 else max(p)
  }, numeric(1)))
}


# Internal: IoU between one box and many boxes
.box_iou_vec <- function(box, boxes) {
  inter_xmin <- pmax(boxes[, 1], box[1])
  inter_ymin <- pmax(boxes[, 2], box[2])
  inter_xmax <- pmin(boxes[, 3], box[3])
  inter_ymax <- pmin(boxes[, 4], box[4])

  inter_w <- pmax(0, inter_xmax - inter_xmin)
  inter_h <- pmax(0, inter_ymax - inter_ymin)
  inter   <- inter_w * inter_h

  area_box   <- (box[3] - box[1]) * (box[4] - box[2])
  area_boxes <- (boxes[, 3] - boxes[, 1]) * (boxes[, 4] - boxes[, 2])

  inter / (area_box + area_boxes - inter + 1e-6)
}
