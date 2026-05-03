#!/usr/bin/env python3
"""Defines the Yolo class."""

import numpy as np
import tensorflow.keras as K


class Yolo:
    """Uses the YOLO v3 algorithm to perform object detection."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize a Yolo instance."""
        self.model = K.models.load_model(model_path)
        self.class_names = self._load_classes(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def _load_classes(self, classes_path):
        """Load class names from a file."""
        with open(classes_path, 'r') as f:
            return f.read().splitlines()

    def sigmoid(self, x):
        """Apply the sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """Process Darknet model outputs."""
        boxes = []
        box_confidences = []
        box_class_probs = []

        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        image_h, image_w = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            box_confidence = self.sigmoid(output[..., 4:5])
            box_class_prob = self.sigmoid(output[..., 5:])

            col = np.arange(grid_w)
            row = np.arange(grid_h)
            cx, cy = np.meshgrid(col, row)

            cx = np.expand_dims(cx, axis=2)
            cy = np.expand_dims(cy, axis=2)

            bx = (self.sigmoid(tx) + cx) / grid_w
            by = (self.sigmoid(ty) + cy) / grid_h

            anchor_w = self.anchors[i, :, 0].reshape((1, 1, anchor_boxes))
            anchor_h = self.anchors[i, :, 1].reshape((1, 1, anchor_boxes))

            bw = (anchor_w * np.exp(tw)) / input_w
            bh = (anchor_h * np.exp(th)) / input_h

            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            box = np.zeros((grid_h, grid_w, anchor_boxes, 4))
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes based on their object scores."""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]

            classes = np.argmax(scores, axis=-1)
            max_scores = np.max(scores, axis=-1)

            mask = max_scores >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(classes[mask])
            box_scores.append(max_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Apply non-max suppression to filtered boxes."""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in np.unique(box_classes):
            idx = np.where(box_classes == cls)[0]

            cls_boxes = filtered_boxes[idx]
            cls_classes = box_classes[idx]
            cls_scores = box_scores[idx]

            sorted_idx = np.argsort(cls_scores)[::-1]

            while sorted_idx.size > 0:
                current = sorted_idx[0]

                box_predictions.append(cls_boxes[current])
                predicted_box_classes.append(cls_classes[current])
                predicted_box_scores.append(cls_scores[current])

                if sorted_idx.size == 1:
                    break

                current_box = cls_boxes[current]
                other_boxes = cls_boxes[sorted_idx[1:]]

                x1 = np.maximum(current_box[0], other_boxes[:, 0])
                y1 = np.maximum(current_box[1], other_boxes[:, 1])
                x2 = np.minimum(current_box[2], other_boxes[:, 2])
                y2 = np.minimum(current_box[3], other_boxes[:, 3])

                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                intersection = inter_w * inter_h

                current_area = ((current_box[2] - current_box[0]) *
                                (current_box[3] - current_box[1]))

                other_areas = ((other_boxes[:, 2] - other_boxes[:, 0]) *
                               (other_boxes[:, 3] - other_boxes[:, 1]))

                union = current_area + other_areas - intersection
                iou = intersection / union

                keep = np.where(iou <= self.nms_t)[0]
                sorted_idx = sorted_idx[keep + 1]

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores
