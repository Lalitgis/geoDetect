# geodetect

**Geospatial Object Detection in R using Deep Learning**

`geodetect` provides a complete workflow for training and deploying Faster R-CNN object detectors on satellite imagery, aerial photographs, and drone data. Every detected object is returned as a georeferenced bounding box (`sf` polygon) in the raster's CRS.

---

## Why geodetect?

| Need | Current R options | geodetect |
|---|---|---|
| Semantic segmentation | `geodl` ✓ | ✓ |
| **Object detection (bounding boxes)** | ❌ nothing active | ✓ |
| Instance segmentation | ❌ | planned v0.2 |
| No Python required | mixed | ✓ pure R |
| CRS-aware results as `sf` | ❌ | ✓ |
| Handles arbitrarily large rasters | ❌ | ✓ sliding window |

---

## Installation

```r
# Requires torch and torchvision (R native, no Python)
install.packages(c("torch", "torchvision", "terra", "sf", "luz", "cli", "fs"))

# Install geodetect
remotes::install_github("Lalitgis/geodetect")
```

---

## Quickstart

```r
library(geodetect)
library(terra)
library(sf)

# 1. Tile your raster into chips
chips <- make_chips(rast("scene.tif"), chip_size = 512, overlap = 0.25)

# 2. Assign vector annotations to chips
labels <- assign_labels_to_chips(chips, st_read("labels.gpkg"),
                                 class_map = c(building = 1L, vehicle = 2L))

# 3. Build and train detector
model  <- build_detector(num_classes = 2)
result <- train_detector(model,
                         geo_detection_dataset(chips, labels),
                         epochs = 20, save_path = "detector.pt")

# 4. Detect objects in a new scene
ckpt <- load_detector("detector.pt")
dets <- predict_raster(ckpt$model, rast("new_scene.tif"),
                       class_map = ckpt$class_map)

# 5. Export to GeoPackage
export_detections(dets, "results.gpkg")
```

---

## Architecture

```
Input scene (SpatRaster, any size)
  └─▶ make_chips()                   Tile → overlapping 512×512 chips
        └─▶ geo_detection_dataset()  torch Dataset (image + bbox labels)
              └─▶ build_detector()   Faster R-CNN
                    ResNet-50 backbone (ImageNet pretrained)
                    Feature Pyramid Network (multi-scale features)
                    Region Proposal Network
                    RoI Align → Classification + Box regression heads
              └─▶ train_detector()   SGD + step LR + early stopping
        └─▶ predict_raster()        Sliding-window inference
              └─▶ global NMS         Merge detections across chip boundaries
                    └─▶ sf POLYGON   Georeferenced bounding boxes
```

---

## Real problems solved

- **Disaster response:** Count damaged buildings in post-earthquake aerial imagery
- **Urban planning:** Map vehicle density in parking lots from drone surveys
- **Agriculture:** Detect individual trees or field objects from Sentinel-2
- **Ecology:** Count animals or nests in wildlife camera trap overviews
- **Infrastructure:** Identify road damage or flooded areas from satellite imagery

---

## File structure

```
geodetect/
├── R/
│   ├── chips.R       make_chips(), assign_labels_to_chips()
│   ├── dataset.R     geo_detection_dataset(), detection_collate_fn()
│   ├── model.R       build_detector(), load_detector()
│   ├── train.R       train_detector(), evaluate_detector()
│   ├── predict.R     predict_raster(), plot_detections()
│   └── geodetect.R   package docs, has_gpu(), split_chips(), export_detections()
├── tests/testthat/
│   └── test-geodetect.R
├── vignettes/
│   └── geodetect-intro.Rmd
├── DESCRIPTION
└── LICENSE
```

---

## License

MIT
