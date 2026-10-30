import base64
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml
from PIL import Image
from qtpy import QtCore
from qtpy import QtWidgets

from labelme import app
from labelme import config as labelme_config
from labelme.label_file import LabelFile
from labelme.label_file import LabelFileError


class XAnyLabelingTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.image = self.directory / "frame.png"
        Image.new("RGB", (40, 30), "white").save(str(self.image))
        self.path = self.directory / "frame.json"
        self.shape = {
            "label": "car",
            "points": [[5, 6], [20, 6], [20, 25], [5, 25]],
            "shape_type": "rectangle",
            "group_id": 319,
            "description": None,
            "difficult": False,
            "direction": 0,
            "flags": None,
            "attributes": {"color": "blue"},
            "score": 0.95,
            "kie_linking": [],
        }
        self.data = {
            "version": "2.5.0",
            "shapes": [self.shape],
            "flags": None,
            "imagePath": "frame.png",
            "imageData": None,
            "imageHeight": 30,
            "imageWidth": 40,
            "description": "Traffic annotations",
            "checked": True,
        }

    def load(self):
        self.path.write_text(json.dumps(self.data), encoding="utf-8")
        return LabelFile(str(self.path))

    def test_four_corner_rectangle_and_group_id(self):
        original = copy.deepcopy(self.data)
        loaded = self.load()
        self.assertEqual(loaded.shapes[0]["points"], [[5, 6], [20, 25]])
        self.assertEqual(loaded.shapes[0]["track_id"], "319")
        self.assertEqual(loaded.shapes[0]["group_id"], 319)
        self.assertEqual(loaded.shapes[0]["flags"], {})
        self.assertEqual(loaded.flags, {})
        self.assertEqual(loaded.shapes[0]["other_data"]["score"], 0.95)
        self.assertEqual(self.data, original)

    def test_corner_order_does_not_change_rectangle(self):
        self.shape["points"] = [[20, 25], [20, 6], [5, 6], [5, 25]]
        self.assertEqual(self.load().shapes[0]["points"], [[5, 6], [20, 25]])

    def test_explicit_track_id_wins_including_none(self):
        for track_id in (None, "17", 17):
            with self.subTest(track_id=track_id):
                self.shape["track_id"] = track_id
                self.assertEqual(self.load().shapes[0]["track_id"], track_id)

    def test_zero_and_missing_group_ids(self):
        for group_id, expected in ((0, "0"), (None, None)):
            with self.subTest(group_id=group_id):
                self.shape["group_id"] = group_id
                self.assertEqual(self.load().shapes[0]["track_id"], expected)
        del self.shape["group_id"]
        self.assertIsNone(self.load().shapes[0]["track_id"])

    def test_plain_labelme_keeps_grouping_and_two_corner_boxes(self):
        self.data["shapes"] = [
            {
                "label": "car",
                "points": [[5, 6], [20, 25]],
                "shape_type": "rectangle",
                "group_id": 17,
            }
        ]
        loaded = self.load().shapes[0]
        self.assertEqual(loaded["points"], [[5, 6], [20, 25]])
        self.assertEqual(loaded["group_id"], 17)
        self.assertIsNone(loaded["track_id"])

    def test_rotation_and_quadrilateral_preserve_outline_as_polygon(self):
        points = [[10, 3], [25, 13], [15, 28], [0, 18]]
        for shape_type in ("rotation", "quadrilateral"):
            with self.subTest(shape_type=shape_type):
                self.shape.update(shape_type=shape_type, points=points, direction=0.5)
                loaded = self.load().shapes[0]
                self.assertEqual(loaded["shape_type"], "polygon")
                self.assertEqual(loaded["points"], points)
                self.assertEqual(loaded["other_data"]["direction"], 0.5)

    def test_common_shape_types_and_optional_fields(self):
        for shape_type, points in (
            ("polygon", [[1, 2], [3, 2], [3, 4]]),
            ("point", [[1, 2]]),
            ("line", [[1, 2], [3, 4]]),
            ("circle", [[5, 5], [8, 5]]),
            ("linestrip", [[1, 2], [3, 2], [3, 4]]),
        ):
            with self.subTest(shape_type=shape_type):
                self.shape.update(shape_type=shape_type, points=points)
                loaded = self.load().shapes[0]
                self.assertEqual(loaded["shape_type"], shape_type)
                self.assertEqual(loaded["points"], points)
        self.data["shapes"] = [{"label": "car", "points": [[1, 2], [3, 4]]}]
        del self.data["imageData"]
        loaded = self.load().shapes[0]
        self.assertEqual(loaded["shape_type"], "polygon")
        self.assertEqual(loaded["flags"], {})
        self.assertIsNone(loaded["track_id"])

    def test_relative_and_absolute_image_paths(self):
        nested = self.directory / "images"
        nested.mkdir()
        image = nested / "scene.png"
        Image.new("RGB", (40, 30), "blue").save(str(image))
        for image_path in ("images/scene.png", str(image)):
            with self.subTest(image_path=image_path):
                self.data["imagePath"] = image_path
                loaded = self.load()
                self.assertEqual(loaded.imagePath, image_path)
                self.assertIsNotNone(loaded.imageData)

    def test_embedded_image_without_external_file(self):
        self.data["imageData"] = base64.b64encode(self.image.read_bytes()).decode()
        self.data["imagePath"] = "unavailable.png"
        self.assertIsNotNone(self.load().imageData)

    def test_legacy_jpeg_fallback_updates_path(self):
        Image.new("RGB", (40, 30), "white").save(str(self.path.with_suffix(".jpg")))
        self.data["imagePath"] = "/unavailable/dataset/frame.jpg"
        self.assertEqual(self.load().imagePath, "frame.jpg")

    def test_missing_image_reports_label_file_error(self):
        self.data["imagePath"] = "unavailable.png"
        with self.assertRaisesRegex(LabelFileError, "Failed opening image file"):
            self.load()

    def test_unsupported_shape_reports_label_file_error(self):
        self.shape["shape_type"] = "cuboid"
        with self.assertRaisesRegex(LabelFileError, "Unsupported shape_type: cuboid"):
            self.load()

    def test_batch_save_preserves_metadata_without_nesting(self):
        loaded = self.load()
        output = self.directory / "saved.json"
        loaded.save(
            filename=str(output),
            shapes=loaded.shapes,
            imagePath=loaded.imagePath,
            imageHeight=30,
            imageWidth=40,
            otherData=loaded.otherData,
            flags=loaded.flags,
        )
        saved = json.loads(output.read_text())
        self.assertEqual(saved["checked"], True)
        self.assertEqual(saved["description"], "Traffic annotations")
        self.assertEqual(saved["shapes"][0]["attributes"], {"color": "blue"})
        self.assertEqual(saved["shapes"][0]["score"], 0.95)
        self.assertNotIn("other_data", saved["shapes"][0])
        reloaded = LabelFile(str(output))
        self.assertEqual(reloaded.shapes, loaded.shapes)

    def test_gui_open_render_save_and_invalid_file(self):
        application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        for shape_type in ("rotation", "quadrilateral"):
            shape = copy.deepcopy(self.shape)
            shape["shape_type"] = shape_type
            self.data["shapes"].append(shape)
        self.load()
        config_path = Path(labelme_config.__file__).with_name("default_config.yaml")
        config = yaml.safe_load(config_path.read_text())
        settings = QtCore.QSettings(
            str(self.directory / "settings.ini"), QtCore.QSettings.IniFormat
        )
        with mock.patch.object(QtCore, "QSettings", return_value=settings):
            window = app.MainWindow(config=config)
            try:
                self.assertTrue(window.loadFile(str(self.path)))
                window.show()
                application.processEvents()
                self.assertEqual(
                    [shape.shape_type for shape in window.canvas.shapes],
                    ["rectangle", "polygon", "polygon"],
                )
                shape = window.canvas.shapes[0]
                self.assertEqual(len(shape.points), 2)
                self.assertEqual(shape.track_id, "319")
                self.assertEqual(shape.fill_color.getRgb()[:3], (0, 255, 0))
                self.assertEqual(shape.other_data["attributes"], {"color": "blue"})
                shape.points[1] = QtCore.QPointF(22, 26)
                output = self.directory / "edited.json"
                self.assertTrue(window.saveLabels(str(output)))
                reloaded = LabelFile(str(output))
                self.assertEqual(reloaded.shapes[0]["points"], [[5, 6], [22, 26]])
                self.assertEqual(reloaded.shapes[0]["track_id"], "319")
                self.assertEqual(reloaded.shapes[0]["other_data"], shape.other_data)
                self.assertTrue(reloaded.otherData["checked"])

                self.shape["shape_type"] = "cuboid"
                self.path.write_text(json.dumps(self.data), encoding="utf-8")
                with mock.patch.object(window, "errorMessage") as error_message:
                    self.assertFalse(window.loadFile(str(self.path)))
                    error_message.assert_called_once()
                    self.assertIn("cuboid", error_message.call_args[0][1])
            finally:
                window.close()
                window.deleteLater()
                application.processEvents()
                settings.sync()


if __name__ == "__main__":
    unittest.main()
