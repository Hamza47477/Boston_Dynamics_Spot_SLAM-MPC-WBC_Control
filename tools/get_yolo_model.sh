#!/bin/bash
# Download the YOLOv4-tiny model (people, cars, trucks, ...) used by `detector:=yolo` / `detector:=auto`.
#   cfg      ~3 KB    https://github.com/AlexeyAB/darknet (network definition)
#   weights  ~23 MB   https://github.com/AlexeyAB/darknet/releases (COCO-trained weights)
#   names    ~1 KB    class names
# Files go to ~/.spot_models/. Run once:  bash tools/get_yolo_model.sh
set -e
DIR="$HOME/.spot_models"
mkdir -p "$DIR"
curl -L -o "$DIR/yolov4-tiny.cfg" https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
curl -L -o "$DIR/coco.names" https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
curl -L -o "$DIR/yolov4-tiny.weights" https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
ls -la "$DIR"
