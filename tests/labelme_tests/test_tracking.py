import copy
import unittest
from unittest import mock

from labelme.track_algo.sort import KalmanBoxTracker
from labelme.tracking import assign_track_ids


def rectangle(x=0, y=0, label="car", track_id=None):
    return {
        "label": label,
        "points": [[x, y], [x + 20, y + 20]],
        "shape_type": "rectangle",
        "track_id": track_id,
        "group_id": None,
        "flags": {},
        "other_data": {"score": 0.9, "attributes": {"color": "blue"}},
    }


def ids(frames):
    return [[shape["track_id"] for shape in frame] for frame in frames]


class TrackingTest(unittest.TestCase):
    def test_moving_boxes_keep_ids_when_detection_order_changes(self):
        frames = [
            [rectangle(0), rectangle(100)],
            [rectangle(102), rectangle(2)],
            [rectangle(4), rectangle(104)],
        ]
        self.assertEqual(
            ids(assign_track_ids(frames)), [["0", "1"], ["1", "0"], ["0", "1"]]
        )

    def test_overlapping_boxes_use_one_to_one_detection_associations(self):
        frames = [[rectangle(0), rectangle(2)], [rectangle(3), rectangle(1)]]
        self.assertEqual(ids(assign_track_ids(frames)), [["0", "1"], ["1", "0"]])

    def test_empty_first_middle_and_last_frames(self):
        frames = [[], [rectangle()], [], [rectangle(2)], []]
        self.assertEqual(ids(assign_track_ids(frames)), [[], ["0"], [], ["0"], []])

    def test_empty_sequence_and_all_empty_frames(self):
        self.assertEqual(assign_track_ids([]), [])
        for preserve_ids in (False, True):
            with self.subTest(preserve_ids=preserve_ids):
                frames = [[], [], []]
                result = assign_track_ids(frames, use_existing_ids=preserve_ids)
                self.assertEqual(result, frames)
                self.assertIsNot(result[0], frames[0])

    def test_birth_and_death_assign_new_ids(self):
        frames = [[rectangle()], [rectangle(2), rectangle(100)]]
        frames.extend([[] for _ in range(11)])
        frames.append([rectangle(2), rectangle(100)])
        result = ids(assign_track_ids(frames))
        self.assertEqual(result[:2], [["0"], ["0", "1"]])
        self.assertEqual(result[-1], ["2", "3"])

    def test_track_survives_ten_missed_frames(self):
        frames = [[rectangle()]] + [[] for _ in range(10)] + [[rectangle()]]
        self.assertEqual(ids(assign_track_ids(frames))[-1], ["0"])

    def test_classes_do_not_exchange_tracks_at_identical_boxes(self):
        frames = [
            [rectangle(label="car"), rectangle(label="person")],
            [rectangle(label="person"), rectangle(label="car")],
            [rectangle(label="bicycle"), rectangle(label="person")],
        ]
        self.assertEqual(
            ids(assign_track_ids(frames)), [["0", "1"], ["1", "0"], ["2", "1"]]
        )

    def test_ids_are_allocated_in_detection_order_across_classes(self):
        frames = [[rectangle(), rectangle(100, label="person"), rectangle(200)]]
        self.assertEqual(ids(assign_track_ids(frames)), [["0", "1", "2"]])

    def test_existing_first_frame_ids_and_new_ids(self):
        frames = [
            [rectangle(track_id="12"), rectangle(100, label="person", track_id=4)],
            [
                rectangle(102, label="person", track_id="900"),
                rectangle(2),
                rectangle(200),
            ],
        ]
        self.assertEqual(
            ids(assign_track_ids(frames, use_existing_ids=True)),
            [["12", "4"], ["4", "12", "13"]],
        )

    def test_existing_ids_remain_on_identical_first_frame_boxes(self):
        frames = [[rectangle(track_id="8"), rectangle(track_id="2")]]
        self.assertEqual(
            ids(assign_track_ids(frames, use_existing_ids=True)), [["8", "2"]]
        )

    def test_existing_id_mode_with_empty_first_frame(self):
        frames = [[], [rectangle(track_id="99")]]
        self.assertEqual(
            ids(assign_track_ids(frames, use_existing_ids=True)), [[], ["0"]]
        )

    def test_invalid_existing_ids_are_rejected(self):
        for track_id in (None, "None", "-1", -1, "1.5", 1.5, "", True):
            with self.subTest(track_id=track_id):
                with self.assertRaisesRegex(ValueError, "nonnegative integer track ID"):
                    assign_track_ids(
                        [[rectangle(track_id=track_id)]], use_existing_ids=True
                    )

    def test_duplicate_existing_ids_are_rejected_across_classes(self):
        with self.assertRaisesRegex(ValueError, "unique across all classes"):
            assign_track_ids(
                [[rectangle(track_id="0"), rectangle(label="person", track_id=0)]],
                use_existing_ids=True,
            )

    def test_metadata_and_original_inputs_are_preserved(self):
        frames = [[rectangle(track_id="99")], [rectangle(2, track_id="42")]]
        original = copy.deepcopy(frames)
        result = assign_track_ids(frames)
        self.assertEqual(frames, original)
        self.assertEqual(ids(result), [["0"], ["0"]])
        for frame, original_frame in zip(result, original):
            shape = dict(frame[0])
            shape["track_id"] = original_frame[0]["track_id"]
            self.assertEqual(shape, original_frame[0])
        result[0][0]["other_data"]["attributes"]["color"] = "red"
        self.assertEqual(frames, original)

    def test_reversed_diagonal_points_are_tracked_without_changing_geometry(self):
        shape = rectangle(2)
        shape["points"].reverse()
        result = assign_track_ids([[rectangle()], [shape]])
        self.assertEqual(ids(result), [["0"], ["0"]])
        self.assertEqual(result[1][0]["points"], shape["points"])

    def test_all_shapes_are_validated_before_tracking(self):
        for shape_type in ("polygon", "rotation", "point", "rectangle3D"):
            with self.subTest(shape_type=shape_type):
                unsupported = rectangle()
                unsupported["shape_type"] = shape_type
                with mock.patch("labelme.tracking.KalmanBoxTracker") as tracker:
                    with self.assertRaisesRegex(
                        ValueError, "Frame 2, object 1.*2D rectangle"
                    ):
                        assign_track_ids([[rectangle()], [unsupported]])
                    tracker.assert_not_called()

    def test_invalid_boxes_are_rejected(self):
        for points in (
            [[0, 0], [0, 10]],
            [[0, 0], [10, 0]],
            [[0, 0], [float("nan"), 10]],
            [[0, 0], [float("inf"), 10]],
            [[0, 0]],
            [[0, 0], [10, 0], [10, 10], [0, 10]],
        ):
            with self.subTest(points=points):
                shape = rectangle()
                shape["points"] = points
                with self.assertRaises(ValueError):
                    assign_track_ids([[shape]])

    def test_separate_runs_are_deterministic_without_modifying_sort_counter(self):
        with mock.patch.object(KalmanBoxTracker, "count", 500):
            for _ in range(2):
                self.assertEqual(ids(assign_track_ids([[rectangle()]])), [["0"]])
                self.assertEqual(KalmanBoxTracker.count, 500)


if __name__ == "__main__":
    unittest.main()
