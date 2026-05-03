#!/usr/bin/env python3
"""Defines the Yolo class"""

import tensorflow.keras as K
import numpy as np


class Yolo:
    """Uses the YOLO v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize Yolo"""
        self.model = K.models.load_model(model_path)
        self.class_names = self._load_classes(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def _load_classes(self, classes_path):
        """Load class names"""
        with open(classes_path, 'r') as f:
            return f.read().splitlines()

    def sigmoid(self, x):
        """Sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """Process outputs"""
        boxes = []
        box_confidences = []
        box_class_probs = []

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        image_h, image_w = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            # Extract parts
            t_xy = output[..., :2]
            t_wh = output[..., 2:4]
            confidence = self.sigmoid(output[..., 4:5])
            class_probs = self.sigmoid(output[..., 5:])

            # Grid
            cx = np.arange(grid_w)
            cy = np.arange(grid_h)
            cx, cy = np.meshgrid(cx, cy)

            cx = np.expand_dims(cx, axis=-1)
            cy = np.expand_dims(cy, axis=-1)

            # Calculate bx, by
            bx = (self.sigmoid(t_xy[..., 0]) + cx) / grid_w
            by = (self.sigmoid(t_xy[..., 1]) + cy) / grid_h

            # Anchors
            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]

            anchor_w = anchor_w.reshape((1, 1, anchor_boxes))
            anchor_h = anchor_h.reshape((1, 1, anchor_boxes))

            bw = (anchor_w * np.exp(t_wh[..., 0])) / input_w
            bh = (anchor_h * np.exp(t_wh[..., 1])) / input_h

            # Convert to corners
            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            box = np.stack([x1, y1, x2, y2], axis=-1)

            boxes.append(box)
            box_confidences.append(confidence)
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs
