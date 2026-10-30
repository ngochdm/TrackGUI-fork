"""Associate rectangle annotations across frames using the existing SORT tracker."""

import copy
import numbers

import numpy as np

from labelme.track_algo.sort import KalmanBoxTracker
from labelme.track_algo.sort import associate_detections_to_trackers


def _rectangle(shape, frame_index, shape_index):
    location = "Frame {}, object {}".format(frame_index + 1, shape_index + 1)
    if shape.get("shape_type") != "rectangle":
        raise ValueError(
            "{}: tracking requires 2D rectangle annotations.".format(location)
        )
    if not isinstance(shape.get("label"), str):
        raise ValueError("{}: each rectangle requires a class label.".format(location))
    try:
        points = np.asarray(shape["points"], dtype=float)
    except (KeyError, TypeError, ValueError):
        raise ValueError("{}: invalid rectangle coordinates.".format(location))
    if points.shape != (2, 2) or not np.isfinite(points).all():
        raise ValueError(
            "{}: rectangles require two finite 2D points.".format(location)
        )
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    if np.any(upper <= lower):
        raise ValueError(
            "{}: rectangles must have positive width and height.".format(location)
        )
    return np.concatenate((lower, upper))


def _initial_track_ids(shapes):
    ids = []
    for index, shape in enumerate(shapes):
        value = shape.get("track_id")
        if isinstance(value, str):
            value = value.strip()
            valid = value.isdecimal()
        else:
            valid = isinstance(value, numbers.Integral) and not isinstance(value, bool)
        if not valid or int(value) < 0:
            raise ValueError(
                "First frame, object {}: assign a nonnegative integer track ID "
                "before tracking from existing IDs.".format(index + 1)
            )
        track_id = int(value)
        if track_id in ids:
            raise ValueError(
                "First frame: track IDs must be unique across all classes."
            )
        ids.append(track_id)
    return ids


def assign_track_ids(frames, use_existing_ids=False):
    """Return copied shapes with string track IDs, preserving detection geometry.

    ``frames`` contains one list of native rectangle shape dictionaries per frame,
    including empty frames. Matching uses SORT's Kalman model and IoU threshold
    0.3 independently for each class. Tracks survive ten missed frames; every
    detection receives an ID immediately, including newly appearing objects.
    IDs are unique across classes and start at zero for each call. When
    ``use_existing_ids`` is true, the first frame's IDs seed the tracks and new
    IDs start above their maximum. IDs in subsequent input frames are replaced.

    All input is validated before tracking. Unsupported shapes, invalid boxes,
    and invalid or duplicate initial IDs raise ``ValueError``. Inputs and SORT's
    global ID counter are never modified; all other shape metadata is retained.
    """
    result = copy.deepcopy(frames)
    rectangles = [
        [_rectangle(shape, frame_index, index) for index, shape in enumerate(shapes)]
        for frame_index, shapes in enumerate(result)
    ]
    initial_ids = _initial_track_ids(result[0]) if use_existing_ids and result else []
    next_id = max(initial_ids) + 1 if initial_ids else 0
    trackers_by_label = {}

    for frame_index, shapes in enumerate(result):
        boxes = rectangles[frame_index]
        if frame_index == 0 and use_existing_ids:
            for shape, box, track_id in zip(shapes, boxes, initial_ids):
                shape["track_id"] = str(track_id)
                trackers_by_label.setdefault(shape["label"], []).append(
                    KalmanBoxTracker(box, id=track_id)
                )
            continue

        detections_by_label = {}
        for index, shape in enumerate(shapes):
            label = shape["label"]
            detections_by_label.setdefault(label, []).append(index)
            trackers_by_label.setdefault(label, [])

        unmatched_indices = []
        for label, trackers in trackers_by_label.items():
            # Predict on every frame, even when that class has no detections.
            active_trackers = []
            predictions = []
            for tracker in trackers:
                prediction = tracker.predict()[0]
                if np.isfinite(prediction).all():
                    active_trackers.append(tracker)
                    predictions.append(prediction)
            trackers_by_label[label] = active_trackers
            indices = detections_by_label.get(label, [])
            detections = np.asarray([boxes[index] for index in indices]).reshape(
                (-1, 4)
            )
            predicted_boxes = np.asarray(predictions).reshape((-1, 4))
            matches, unmatched, _ = associate_detections_to_trackers(
                detections, predicted_boxes, iou_threshold=0.3
            )
            for detection_index, tracker_index in matches:
                tracker = active_trackers[tracker_index]
                tracker.update(detections[detection_index])
                shapes[indices[detection_index]]["track_id"] = str(tracker.id)
            unmatched_indices.extend(indices[index] for index in unmatched)

        # Allocate IDs in original detection order, rather than class order.
        for index in sorted(unmatched_indices):
            shape = shapes[index]
            shape["track_id"] = str(next_id)
            trackers_by_label[shape["label"]].append(
                KalmanBoxTracker(boxes[index], id=next_id)
            )
            next_id += 1

        for label, trackers in trackers_by_label.items():
            trackers_by_label[label] = [
                tracker for tracker in trackers if tracker.time_since_update <= 10
            ]

    return result
