import tempfile
import unittest
from pathlib import Path

from labelme.yolo import YoloImportError
from labelme.yolo import load_class_names
from labelme.yolo import load_yolo_shapes


class YoloImportTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.path = self.directory / "frame.txt"

    def write(self, content, name="frame.txt"):
        path = self.directory / name
        path.write_text(content, encoding="utf-8")
        return path

    def load(self, content):
        return load_yolo_shapes(self.write(content), ["car", "pedestrian"], 640, 480)

    def test_pixel_conversion_and_native_shape_fields(self):
        self.assertEqual(
            self.load("0 0.5 0.5 0.5 0.25\n"),
            [dict(
                label="car", points=[[160.0, 180.0], [480.0, 300.0]],
                group_id=None, track_id=None, shape_type="rectangle", flags={},
                description="", mask=None,
            )],
        )

    def test_confidence_comments_and_multiple_classes(self):
        shapes = self.load(
            "\ufeff# predictions\n\n1 0.5 0.5 0.5 0.25 0.8 # person\n0 0.5 0.5 1 1\n"
        )
        self.assertEqual(len(shapes), 2)
        self.assertEqual(shapes[0]["label"], "pedestrian")
        self.assertEqual(shapes[0]["score"], 0.8)
        self.assertNotIn("score", shapes[1])

    def test_empty_files_are_valid_empty_annotations(self):
        for content in ("", "\n # no detections\n"):
            with self.subTest(content=content):
                self.assertEqual(self.load(content), [])

    def test_boxes_are_clipped_at_all_image_edges(self):
        shapes = self.load("0 0 0 0.5 0.5\n1 1 1 0.5 0.5\n")
        self.assertEqual(shapes[0]["points"], [[0.0, 0.0], [160.0, 120.0]])
        self.assertEqual(shapes[1]["points"], [[480.0, 360.0], [640.0, 480.0]])

    def test_invalid_rows_include_file_and_line(self):
        rows = (
            "0 0.5 0.5 0.5", "0 0.5 0.5 0.5 0.5 0.9 7",
            "cat 0.5 0.5 0.5 0.5", "1.0 0.5 0.5 0.5 0.5",
            "-1 0.5 0.5 0.5 0.5", "2 0.5 0.5 0.5 0.5",
            "0 text 0.5 0.5 0.5", "0 NaN 0.5 0.5 0.5",
            "0 0.5 inf 0.5 0.5", "0 0.5 0.5 -inf 0.5",
            "0 -0.1 0.5 0.5 0.5", "0 0.5 1.1 0.5 0.5",
            "0 0.5 0.5 0 0.5", "0 0.5 0.5 0.5 -0.1",
            "0 0.5 0.5 1.1 0.5", "0 0.5 0.5 0.5 1.1",
            "0 0.5 0.5 0.5 0.5 -0.1", "0 0.5 0.5 0.5 0.5 1.1",
            "0 0.5 0.5 0.5 0.5 nan", "0 1 1 1e-100 1e-100",
        )
        for row in rows:
            with self.subTest(row=row):
                with self.assertRaises(YoloImportError) as caught:
                    self.load("# header\n" + row)
                self.assertIn("frame.txt:2:", str(caught.exception))

    def test_bad_later_row_does_not_return_partial_shapes(self):
        with self.assertRaisesRegex(YoloImportError, "frame.txt:2:"):
            self.load("0 0.5 0.5 0.5 0.5\n3 0.5 0.5 0.5 0.5\n")

    def test_positive_finite_image_dimensions_required(self):
        self.write("0 0.5 0.5 0.5 0.5")
        for dimensions in ((0, 480), (640, -1), (float("nan"), 480),
                           (640, float("inf")), (None, 480), (True, 480)):
            with self.subTest(dimensions=dimensions):
                with self.assertRaisesRegex(YoloImportError, "Image dimensions"):
                    load_yolo_shapes(self.path, ["car"], *dimensions)

    def test_plain_class_names_preserve_spaces_and_unicode(self):
        path = self.write("\ufeffcar\n traffic light \n보행자\n", "classes.txt")
        self.assertEqual(load_class_names(path), ["car", "traffic light", "보행자"])

    def test_yaml_names_list_and_ordered_integer_mapping(self):
        for content in (
            "names: [car, traffic light]\n", "names:\n  1: traffic light\n  0: car\n"
        ):
            with self.subTest(content=content):
                path = self.write(content, "data.yaml")
                self.assertEqual(load_class_names(path), ["car", "traffic light"])

    def test_invalid_class_files(self):
        files = (
            ("classes.txt", ""), ("classes.txt", "car\n\nbus"),
            ("classes.txt", "car\n car\n"), ("data.yaml", "names: []"),
            ("data.yaml", "names: [car, car]"), ("data.yaml", "names: [car, '']"),
            ("data.yaml", "names: [car, 7]"), ("data.yaml", "names: car"),
            ("data.yaml", "train: images"), ("data.yaml", "names: {1: car}"),
            ("data.yaml", "names: {0: car, 2: bus}"),
            ("data.yaml", "names: {'0': car}"), ("data.yaml", "names: {false: car}"),
            ("data.yaml", "names: {0: car, 0: bus}"),
            ("data.yaml", "names: [car"), ("data.yaml", "- car"),
            ("data.yaml", "names: {[]: car}"),
        )
        for name, content in files:
            with self.subTest(content=content):
                with self.assertRaises(YoloImportError) as caught:
                    load_class_names(self.write(content, name))
                self.assertIn(name, str(caught.exception))

    def test_unreadable_inputs_report_source(self):
        missing = self.directory / "missing.txt"
        for loader in (
            load_class_names,
            lambda path: load_yolo_shapes(path, ["car"], 640, 480),
        ):
            with self.subTest(loader=loader):
                with self.assertRaisesRegex(YoloImportError, "missing.txt"):
                    loader(missing)
        self.path.write_bytes(b"\xff")
        with self.assertRaisesRegex(YoloImportError, "frame.txt"):
            load_yolo_shapes(self.path, ["car"], 640, 480)


if __name__ == "__main__":
    unittest.main()
