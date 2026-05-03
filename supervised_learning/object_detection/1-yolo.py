#!/usr/bin/env python3
"""Defines the Yolo class."""

import numpy as np
import tensorflow.keras as K


class Yolo:
    """Uses the YOLO v3 algorithm to perform object detection."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize Yolo."""
        self.model = K.models.load_model(model_path)
        self.class_names = self._load_classes(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def _load_classes(self, classes_path):
        """Load class names."""
        with open(classes_path, 'r') as f:
            return f.read().splitlines()

    def sigmoid(self, x):
        """Sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """Process outputs."""
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_h, image_w = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            confidence = self.sigmoid(output[..., 4:5])
            class_probs = self.sigmoid(output[..., 5:])

            # Grid
            cx = np.arange(grid_w)
            cy = np.arange(grid_h)
            cx, cy = np.meshgrid(cx, cy)

            cx = np.expand_dims(cx, axis=2)
            cy = np.expand_dims(cy, axis=2)

            # bx, by (НЕ делим!)
            bx = self.sigmoid(tx) + cx
            by = self.sigmoid(ty) + cy

            # Anchors
            anchor_w = self.anchors[i, :, 0].reshape((1, 1, anchor_boxes))
            anchor_h = self.anchors[i, :, 1].reshape((1, 1, anchor_boxes))

            # ❗ БЕЗ деления на input
            bw = anchor_w * np.exp(tw)
            bh = anchor_h * np.exp(th)

            # Перевод в координаты изображения
            x1 = (bx - bw / 2) * (image_w / grid_w)
            y1 = (by - bh / 2) * (image_h / grid_h)
            x2 = (bx + bw / 2) * (image_w / grid_w)
            y2 = (by + bh / 2) * (image_h / grid_h)

            box = np.zeros((grid_h, grid_w, anchor_boxes, 4))
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            boxes.append(box)
            box_confidences.append(confidence)
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs
