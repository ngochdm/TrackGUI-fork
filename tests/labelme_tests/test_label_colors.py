import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml
from qtpy import QtCore
from qtpy import QtWidgets

from labelme import app
from labelme import config as labelme_config


def default_config():
    # Read the packaged defaults without creating or loading ~/.labelmerc.
    path = Path(labelme_config.__file__).with_name("default_config.yaml")
    with path.open() as stream:
        return yaml.safe_load(stream)


class LabelColorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = default_config()

    def setUp(self):
        self.window = SimpleNamespace(_config=copy.deepcopy(self.config))

    def color(self, label):
        return app.MainWindow._get_rgb_by_label(self.window, label)

    def test_plain_names_and_unavailable_ids_use_default_color(self):
        labels = [
            "unidentified",
            "pedestrian",
            "small truck",
            "small_truck",
            "vehicle_invalid",
            "vehicle_",
            "vehicle_-1",
            "vehicle_-257",
            "vehicle_255",
            "vehicle_256",
            "vehicle_1000000",
        ]
        for default, expected in [([12, 34, 56], [12, 34, 56]), (None, (0, 255, 0))]:
            self.window._config["default_shape_color"] = default
            for label in labels:
                with self.subTest(default=default, label=label):
                    self.assertEqual(self.color(label), expected)

    def test_missing_palette_key_uses_default_color(self):
        self.window._config["label_colors"] = [{1: [128, 0, 0]}]
        self.window._config["default_shape_color"] = [12, 34, 56]
        self.assertEqual(self.color("car_0"), [12, 34, 56])

    def test_valid_track_colors_are_preserved(self):
        expected_colors = {
            0: [0, 0, 0],
            1: [128, 0, 0],
            254: [96, 224, 192],
        }
        for track_id, expected in expected_colors.items():
            with self.subTest(track_id=track_id):
                self.assertEqual(
                    self.color("small_truck_{}".format(track_id)), expected
                )

    def test_none_suffix_keeps_yellow_color(self):
        self.assertEqual(self.color("unidentified_None"), [224, 224, 0])

    def test_automatic_color_keeps_index_shift_and_wrapping(self):
        self.window._config["shape_color"] = "auto"
        self.window._config["shift_auto_shape_color"] = len(app.LABEL_COLORMAP) + 4
        item = object()
        label_list = mock.Mock()
        label_list.findItemByLabel.return_value = item
        label_list.indexFromItem.return_value.row.return_value = 2
        self.window.uniqLabelList = label_list

        self.assertEqual(
            self.color("unidentified").tolist(), app.LABEL_COLORMAP[7].tolist()
        )
        label_list.findItemByLabel.assert_called_once_with("unidentified")
        label_list.addItem.assert_not_called()


class PlainClassStartupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = (
            QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        )

    def test_startup_with_plain_class_names(self):
        labels = [
            "unidentified",
            "others",
            "pedestrian",
            "micromobility",
            "car",
            "bus",
            "small truck",
            "truck",
        ]
        config = default_config()
        config["labels"] = labels
        config["default_shape_color"] = [12, 34, 56]
        with tempfile.TemporaryDirectory() as directory:
            settings = QtCore.QSettings(
                str(Path(directory) / "labelme.ini"), QtCore.QSettings.IniFormat
            )
            with mock.patch.object(QtCore, "QSettings", return_value=settings):
                window = app.MainWindow(config=config)
                try:
                    window.show()
                    self.application.processEvents()
                    self.assertEqual(window.uniqLabelList.count(), len(labels))
                    for label in labels:
                        with self.subTest(label=label):
                            self.assertIsNotNone(
                                window.uniqLabelList.findItemByLabel(label)
                            )
                            self.assertEqual(
                                window._get_rgb_by_label(label), [12, 34, 56]
                            )
                finally:
                    window.close()
                    window.deleteLater()
                    self.application.processEvents()
                    settings.sync()


if __name__ == "__main__":
    unittest.main()
