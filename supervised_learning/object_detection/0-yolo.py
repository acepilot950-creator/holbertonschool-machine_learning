#!/usr/bin/env python3
"""Defines the Yolo class for object detection."""

import tensorflow.keras as K


class Yolo:
    """Uses the YOLO v3 algorithm to perform object detection."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize a Yolo object."""
        self.model = K.models.load_model(model_path)
        self.class_names = self._load_class_names(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def _load_class_names(self, classes_path):
        """Load class names from file."""
        with open(classes_path, 'r', encoding='utf-8') as f:
            return f.read().splitlines()
