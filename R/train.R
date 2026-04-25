#' Train a geospatial object detector
#'
#' Runs the full training loop for a Faster R-CNN model on geospatial image
#' chips. Handles loss logging, validation, early stopping, and saving the
#' best checkpoint. Returns training history and the best model.
#'
#' @param model A Faster R-CNN `nn_module` from [build_detector()].
#' @param train_ds A dataset from [geo_detection_dataset()] for training.
#' @param val_ds A dataset from [geo_detection_dataset()] for validation.
#'   Can be NULL to skip validation.
#' @param epochs Integer. Maximum training epochs. Default 30.
#' @param batch_size Integer. Images per batch. Default 4. Reduce if GPU OOM.
#' @param lr Numeric. Initial learning rate. Default 0.005.
#' @param lr_step_size Integer. Decay the learning rate every this many
#'   epochs by `lr_gamma`. Default 10.
#' @param lr_gamma Numeric. Multiplicative LR decay factor. Default 0.1.
#' @param weight_decay Numeric. L2 regularization. Default 1e-4.
#' @param patience Integer. Stop training early if validation loss does not
#'   improve for this many consecutive epochs. Default 5.
#' @param save_path Character. Path to save the best model checkpoint (`.pt`).
#'   Default `"geodetect_best.pt"`.
#' @param num_workers Integer. Parallel DataLoader workers. 0 = main process
#'   (safest on Windows). Default 0.
#' @param device Character. `"cuda"` or `"cpu"`. Auto-detected if not supplied.
#' @param class_map Named integer vector (class → ID). Saved with the checkpoint
#'   so [predict_raster()] can reconstruct class names.
#' @param chip_size Integer. Saved in checkpoint metadata. Default 512.
#' @param n_bands Integer. Saved in checkpoint metadata. Default 3.
#' @param verbose Logical. Print per-epoch loss. Default TRUE.
#'
#' @return A list with:
#'   \describe{
#'     \item{history}{Data frame with columns `epoch`, `train_loss`,
#'       `val_loss` (NA if no validation set).}
#'     \item{best_epoch}{Integer. Epoch with lowest validation loss.}
#'     \item{model}{The model loaded from the best checkpoint.}
#'   }
#'
#' @details
#' Faster R-CNN training uses the model's own internal loss functions:
#' classification loss (cross-entropy), box regression loss (smooth L1),
#' and RPN losses. The total loss is their sum. No custom loss needs to be
#' defined.
#'
#' @examples
#' \dontrun{
#' model  <- build_detector(num_classes = 2)
#' result <- train_detector(
#'   model      = model,
#'   train_ds   = train_ds,
#'   val_ds     = val_ds,
#'   epochs     = 20,
#'   batch_size = 4,
#'   class_map  = c(building = 1L, vehicle = 2L),
#'   save_path  = "my_detector.pt"
#' )
#' plot(result$history$train_loss, type = "l", ylab = "Loss", xlab = "Epoch")
#' }
#'
#' @export
train_detector <- function(model,
                           train_ds,
                           val_ds      = NULL,
                           epochs      = 30L,
                           batch_size  = 4L,
                           lr          = 0.005,
                           lr_step_size = 10L,
                           lr_gamma    = 0.1,
                           weight_decay = 1e-4,
                           patience    = 5L,
                           save_path   = "geodetect_best.pt",
                           num_workers = 0L,
                           device      = NULL,
                           class_map   = NULL,
                           chip_size   = 512L,
                           n_bands     = 3L,
                           verbose     = TRUE) {

  if (!requireNamespace("torch", quietly = TRUE)) stop("Package 'torch' required.")

  # Auto-detect device
  if (is.null(device)) {
    device <- if (torch::cuda_is_available()) "cuda" else "cpu"
  }
  cli::cli_alert_info("Training on: {.field {device}}")

  model$to(device = device)
  model$train()

  # DataLoader — uses custom collate for variable-size targets
  train_dl <- torch::dataloader(
    dataset     = train_ds,
    batch_size  = as.integer(batch_size),
    shuffle     = TRUE,
    collate_fn  = detection_collate_fn,
    num_workers = as.integer(num_workers),
    drop_last   = TRUE
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

  # Optimizer — SGD with momentum, standard for Faster R-CNN
  optimizer <- torch::optim_sgd(
    params       = model$parameters(),
    lr           = lr,
    momentum     = 0.9,
    weight_decay = weight_decay
  )

  # Step LR scheduler
  scheduler <- torch::lr_step(
    optimizer  = optimizer,
    step_size  = as.integer(lr_step_size),
    gamma      = lr_gamma
  )

  history <- data.frame(
    epoch      = integer(),
    train_loss = numeric(),
    val_loss   = numeric()
  )

  best_val_loss <- Inf
  no_improve    <- 0L

  for (ep in seq_len(epochs)) {

    # ------ Training phase ------
    model$train()
    train_losses <- numeric(0)

    coro::loop(for (batch in train_dl) {
      images  <- lapply(batch$images,  function(x) x$to(device = device))
      targets <- lapply(batch$targets, function(t) list(
        boxes  = t$boxes$to(device = device),
        labels = t$labels$to(device = device)
      ))

      optimizer$zero_grad()

      # Faster R-CNN forward pass in training mode returns a loss dict
      loss_dict <- model(images, targets)
      total_loss <- torch::torch_stack(as.list(loss_dict))$sum()

      total_loss$backward()
      # Gradient clipping for stability
      torch::nn_utils_clip_grad_norm_(model$parameters(), max_norm = 5.0)
      optimizer$step()

      train_losses <- c(train_losses, total_loss$item())
    })

    scheduler$step()
    mean_train <- mean(train_losses)

    # ------ Validation phase ------
    mean_val <- NA_real_
    if (!is.null(val_dl)) {
      model$eval()
      val_losses <- numeric(0)

      with(torch::no_grad(), {
        coro::loop(for (batch in val_dl) {
          images  <- lapply(batch$images,  function(x) x$to(device = device))
          targets <- lapply(batch$targets, function(t) list(
            boxes  = t$boxes$to(device = device),
            labels = t$labels$to(device = device)
          ))
          # Call in train() mode temporarily to get loss from Faster R-CNN
          model$train()
          loss_dict  <- model(images, targets)
          model$eval()
          total_loss <- torch::torch_stack(as.list(loss_dict))$sum()
          val_losses <- c(val_losses, total_loss$item())
        })
      })
      mean_val <- mean(val_losses)
    }

    history <- rbind(history, data.frame(
      epoch      = ep,
      train_loss = mean_train,
      val_loss   = mean_val
    ))

    if (verbose) {
      if (!is.na(mean_val)) {
        cli::cli_alert_info(
          "Epoch {ep}/{epochs} | Train loss: {round(mean_train, 4)} | Val loss: {round(mean_val, 4)}"
        )
      } else {
        cli::cli_alert_info(
          "Epoch {ep}/{epochs} | Train loss: {round(mean_train, 4)}"
        )
      }
    }

    # ------ Early stopping & checkpointing ------
    monitor_loss <- if (!is.na(mean_val)) mean_val else mean_train

    if (monitor_loss < best_val_loss) {
      best_val_loss <- monitor_loss
      best_epoch    <- ep
      no_improve    <- 0L

      # Save checkpoint with metadata for reloading
      torch::torch_save(
        list(
          state_dict  = model$state_dict(),
          num_classes = model$roi_heads$box_predictor$cls_score$out_features - 1L,
          class_map   = class_map,
          chip_size   = chip_size,
          n_bands     = n_bands,
          min_size    = 512L,
          max_size    = 1024L
        ),
        save_path
      )
      if (verbose) cli::cli_alert_success("  Saved best model (epoch {ep})")
    } else {
      no_improve <- no_improve + 1L
      if (no_improve >= patience) {
        cli::cli_alert_warning(
          "Early stopping: no improvement for {patience} epochs. Best epoch: {best_epoch}."
        )
        break
      }
    }
  }

  # Reload best model
  best_ckpt <- load_detector(save_path, device = "cpu")

  cli::cli_alert_success("Training complete. Best epoch: {best_epoch} | Loss: {round(best_val_loss, 4)}")

  list(
    history    = history,
    best_epoch = best_epoch,
    model      = best_ckpt$model
  )
}


#' Compute mean average precision (mAP) for a detector
#'
#' Evaluates a trained detector on a dataset by computing mAP@50 and
#' mAP@50:95 using the COCO protocol.
#'
#' @param model A trained Faster R-CNN model (eval mode).
#' @param dataset A `geo_detection_dataset`.
#' @param iou_thresholds Numeric vector of IoU thresholds. Default is
#'   `seq(0.5, 0.95, by = 0.05)` (COCO standard).
#' @param score_thresh Numeric. Minimum score for a prediction to count.
#'   Default 0.5.
#' @param device Character. `"cpu"` or `"cuda"`.
#'
#' @return A list with `map50` (scalar) and `map50_95` (scalar).
#'
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
    coro::loop(for (batch in dl) {
      img_id  <- img_id + 1L
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
    })
  })

  # Compute mAP per IoU threshold then average
  aps <- vapply(iou_thresholds, function(iou_thresh) {
    .compute_ap_at_iou(all_preds, all_gt, iou_thresh, score_thresh)
  }, numeric(1))

  list(
    map50    = aps[1],
    map50_95 = mean(aps)
  )
}


# Internal: compute AP at a single IoU threshold using standard 11-point interpolation
.compute_ap_at_iou <- function(preds_list, gt_list, iou_thresh, score_thresh) {

  all_scores  <- c()
  all_tp      <- c()
  all_fp      <- c()
  n_gt_total  <- 0L

  for (i in seq_along(preds_list)) {
    pred <- preds_list[[i]]
    gt   <- gt_list[[i]]

    pred_boxes  <- as.matrix(pred$boxes)
    pred_scores <- as.numeric(pred$scores)
    pred_labels <- as.integer(pred$labels)

    gt_boxes  <- as.matrix(gt$boxes)
    n_gt       <- nrow(gt_boxes)
    n_gt_total <- n_gt_total + n_gt

    keep <- pred_scores >= score_thresh
    pred_boxes  <- pred_boxes[keep, , drop = FALSE]
    pred_scores <- pred_scores[keep]

    if (nrow(pred_boxes) == 0) next

    # Sort by descending score
    ord <- order(pred_scores, decreasing = TRUE)
    pred_boxes  <- pred_boxes[ord, , drop = FALSE]
    pred_scores <- pred_scores[ord]

    matched <- rep(FALSE, n_gt)

    for (j in seq_len(nrow(pred_boxes))) {
      if (n_gt == 0) {
        all_tp <- c(all_tp, 0); all_fp <- c(all_fp, 1)
        all_scores <- c(all_scores, pred_scores[j])
        next
      }
      ious <- .box_iou_vec(pred_boxes[j, ], gt_boxes)
      best_iou_idx <- which.max(ious)
      best_iou     <- ious[best_iou_idx]

      if (best_iou >= iou_thresh && !matched[best_iou_idx]) {
        matched[best_iou_idx] <- TRUE
        all_tp <- c(all_tp, 1); all_fp <- c(all_fp, 0)
      } else {
        all_tp <- c(all_tp, 0); all_fp <- c(all_fp, 1)
      }
      all_scores <- c(all_scores, pred_scores[j])
    }
  }

  if (length(all_scores) == 0 || n_gt_total == 0) return(0.0)

  ord  <- order(all_scores, decreasing = TRUE)
  tp_c <- cumsum(all_tp[ord])
  fp_c <- cumsum(all_fp[ord])

  recall    <- tp_c / n_gt_total
  precision <- tp_c / (tp_c + fp_c)

  # 11-point interpolation
  r_levels <- seq(0, 1, by = 0.1)
  ap <- mean(vapply(r_levels, function(r) {
    max(c(0, precision[recall >= r]))
  }, numeric(1)))

  ap
}

# Internal: compute IoU between one box and a matrix of boxes
.box_iou_vec <- function(box, boxes) {
  # box: [xmin, ymin, xmax, ymax]
  # boxes: matrix [N, 4]
  inter_xmin <- pmax(boxes[, 1], box[1])
  inter_ymin <- pmax(boxes[, 2], box[2])
  inter_xmax <- pmin(boxes[, 3], box[3])
  inter_ymax <- pmin(boxes[, 4], box[4])

  inter_w <- pmax(0, inter_xmax - inter_xmin)
  inter_h <- pmax(0, inter_ymax - inter_ymin)
  inter   <- inter_w * inter_h

  area_box  <- (box[3]     - box[1])     * (box[4]     - box[2])
  area_boxes <- (boxes[, 3] - boxes[, 1]) * (boxes[, 4] - boxes[, 2])

  inter / (area_box + area_boxes - inter + 1e-6)
}
