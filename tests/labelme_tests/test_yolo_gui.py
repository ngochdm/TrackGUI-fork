import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import yaml
from PIL import Image
from qtpy import QtCore
from qtpy import QtWidgets

from labelme import app
from labelme import config as labelme_config
from labelme.widgets.yolo_import_dialog import YoloImportDialog


class YoloDialogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = (
            QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        )

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.txt = self.directory / "frame.txt"
        self.txt.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    def dialog(self, **kwargs):
        dialog = YoloImportDialog(**kwargs)
        self.addCleanup(dialog.deleteLater)
        return dialog

    def test_sequence_defaults_and_scope_changes(self):
        dialog = self.dialog(class_names=["car"], frame_count=3)
        self.assertEqual(dialog.scope_combo.currentData(), "sequence")
        self.assertEqual(dialog.frame_count_label.text(), "3")
        self.assertTrue(dialog.run_tracking.isChecked())
        self.assertFalse(dialog.overwrite.isChecked())
        dialog.source_edit.setText(str(self.directory))
        dialog.scope_combo.setCurrentIndex(0)
        self.assertEqual(dialog.source_edit.text(), "")
        self.assertEqual(dialog.frame_count_label.text(), "1")
        self.assertFalse(dialog.run_tracking.isEnabled())
        self.assertFalse(dialog.run_tracking.isChecked())
        dialog.scope_combo.setCurrentIndex(1)
        self.assertTrue(dialog.run_tracking.isEnabled())
        self.assertTrue(dialog.run_tracking.isChecked())

    def test_current_image_accepts_valid_txt_with_loaded_classes(self):
        dialog = self.dialog(class_names=["car"], frame_count=1)
        self.assertEqual(dialog.scope_combo.currentData(), "current")
        self.assertFalse(dialog.run_tracking.isEnabled())
        dialog.source_edit.setText(str(self.txt))
        dialog.button_box.button(QtWidgets.QDialogButtonBox.Ok).click()
        self.assertEqual(dialog.result(), QtWidgets.QDialog.Accepted)
        self.assertEqual(dialog.options(), dict(
            sequence=False, source=str(self.txt), classes_file="",
            track=False, overwrite=False,
        ))

    def test_invalid_source_and_missing_classes_are_inline_errors(self):
        dialog = self.dialog(frame_count=1)
        for path in ("", str(self.directory), str(self.directory / "missing.txt")):
            with self.subTest(source=path):
                dialog.source_edit.setText(path)
                dialog.accept()
                self.assertEqual(dialog.result(), QtWidgets.QDialog.Rejected)
                self.assertIn("YOLO TXT", dialog.error_label.text())
        dialog.source_edit.setText(str(self.txt))
        dialog.accept()
        self.assertEqual(dialog.result(), QtWidgets.QDialog.Rejected)
        self.assertIn("class names", dialog.error_label.text())
        dialog.classes_edit.setText(str(self.directory / "missing.yaml"))
        dialog.accept()
        self.assertIn("readable", dialog.error_label.text())
        classes = self.directory / "classes.txt"
        classes.write_text("car\n", encoding="utf-8")
        dialog.classes_edit.setText(str(classes))
        dialog.accept()
        self.assertEqual(dialog.result(), QtWidgets.QDialog.Accepted)
        self.assertEqual(dialog.error_label.text(), "")

    def test_sequence_requires_directory_and_accepts_yaml_path(self):
        dialog = self.dialog(frame_count=2)
        classes = self.directory / "data.yaml"
        classes.write_text("names: [car]\n", encoding="utf-8")
        dialog.classes_edit.setText(str(classes))
        dialog.source_edit.setText(str(self.txt))
        dialog.accept()
        self.assertEqual(dialog.result(), QtWidgets.QDialog.Rejected)
        self.assertIn("folder", dialog.error_label.text())
        dialog.source_edit.setText(str(self.directory))
        dialog.overwrite.setChecked(True)
        dialog.accept()
        self.assertEqual(dialog.result(), QtWidgets.QDialog.Accepted)
        self.assertEqual(dialog.options(), dict(
            sequence=True, source=str(self.directory), classes_file=str(classes),
            track=True, overwrite=True,
        ))


class YoloGuiImportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = (
            QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        )

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.images_dir = self.directory / "images"
        self.labels_dir = self.directory / "labels"
        self.images_dir.mkdir()
        self.labels_dir.mkdir()
        self.images = []
        for index in range(3):
            path = self.images_dir / "frame{:03d}.png".format(index)
            Image.new("RGB", (100, 80), "white").save(str(path))
            self.images.append(path)
            (self.labels_dir / path.with_suffix(".txt").name).write_text(
                "0 {} 0.5 0.2 0.25 0.9\n".format(0.4 + 0.01 * index),
                encoding="utf-8",
            )
        defaults = Path(labelme_config.__file__).with_name("default_config.yaml")
        config = yaml.safe_load(defaults.read_text())
        config["labels"] = ["car", "pedestrian"]
        self.settings = QtCore.QSettings(
            str(self.directory / "settings.ini"), QtCore.QSettings.IniFormat
        )
        settings_patch = mock.patch.object(
            QtCore, "QSettings", return_value=self.settings
        )
        settings_patch.start()
        self.addCleanup(settings_patch.stop)
        self.window = app.MainWindow(config=config)
        self.addCleanup(self.close_window)
        error_patch = mock.patch.object(self.window, "errorMessage")
        info_patch = mock.patch.object(self.window, "informationMessage")
        self.error_message = error_patch.start()
        self.information_message = info_patch.start()
        self.addCleanup(error_patch.stop)
        self.addCleanup(info_patch.stop)
        self.window.importDirImages(str(self.images_dir))
        self.assertEqual(self.window.imageList, [str(path) for path in self.images])
        self.assertEqual(self.window.imagePath, str(self.images[0]))

    def close_window(self):
        self.window.setClean()
        self.window.close()
        self.window.deleteLater()
        self.application.processEvents()
        self.settings.sync()

    def import_yolo(self, sequence=True, track=False, source=None, **kwargs):
        options = dict(
            sequence=sequence,
            source=str(source or self.labels_dir),
            classes_file="", track=track, overwrite=False,
        )
        options.update(kwargs)
        dialog = mock.Mock()
        dialog.exec_.return_value = QtWidgets.QDialog.Accepted
        dialog.options.return_value = options
        with mock.patch.object(app, "YoloImportDialog", return_value=dialog):
            self.window.importYOLO()

    def read_annotation(self, index, output_dir=None):
        path = self.images[index].with_suffix(".json")
        if output_dir:
            path = output_dir / path.name
        return json.loads(path.read_text(encoding="utf-8"))

    def run_sort(self, option):
        dialog = mock.Mock()
        dialog.exec_.return_value = QtWidgets.QDialog.Accepted
        dialog.option_value = option
        dialog.end_frame = QtWidgets.QLineEdit()
        with mock.patch.object(app, "TrackDialog", return_value=dialog):
            self.window.SORT()
        dialog.end_frame.deleteLater()

    def test_single_image_import_reloads_canvas_without_other_json_files(self):
        self.import_yolo(sequence=False, source=self.labels_dir / "frame000.txt")
        self.error_message.assert_not_called()
        shape = self.window.canvas.shapes[0]
        self.assertEqual(shape.label, "car")
        self.assertEqual(shape.shape_type, "rectangle")
        self.assertIsNone(shape.track_id)
        for point, (x, y) in zip(shape.points, [(30.0, 30.0), (50.0, 50.0)]):
            self.assertAlmostEqual(point.x(), x)
            self.assertAlmostEqual(point.y(), y)
        self.assertEqual(self.window.fileListWidget.item(0).checkState(), QtCore.Qt.Checked)
        self.assertFalse(self.images[1].with_suffix(".json").exists())
        self.assertFalse(self.images[2].with_suffix(".json").exists())
        self.information_message.assert_called_once()

    def test_sequence_import_tracks_to_output_dir_and_missing_txt_is_empty(self):
        output_dir = self.directory / "output"
        self.window.output_dir = str(output_dir)
        (self.labels_dir / "frame002.txt").unlink()
        self.import_yolo(track=True)
        self.error_message.assert_not_called()
        first = self.read_annotation(0, output_dir)
        second = self.read_annotation(1, output_dir)
        track_id = first["shapes"][0]["track_id"]
        self.assertIsNotNone(track_id)
        self.assertEqual(second["shapes"][0]["track_id"], track_id)
        self.assertEqual(self.read_annotation(2, output_dir)["shapes"], [])
        self.assertEqual(self.window.canvas.shapes[0].track_id, track_id)
        self.assertFalse(list(self.images_dir.glob("*.json")))
        for index in range(3):
            self.assertEqual(self.window.fileListWidget.item(index).checkState(),
                             QtCore.Qt.Checked)
        message = self.information_message.call_args[0][1]
        self.assertIn("1 frames had no TXT", message)
        self.assertIn("SORT tracking IDs", message)

    def test_class_file_updates_class_names_in_original_order(self):
        classes = self.directory / "classes.txt"
        classes.write_text("zebra\ncar\n", encoding="utf-8")
        self.import_yolo(sequence=False, source=self.labels_dir / "frame000.txt",
                         classes_file=str(classes))
        self.error_message.assert_not_called()
        self.assertEqual(self.window._config["labels"], ["zebra", "car"])
        self.assertEqual(self.window.canvas.shapes[0].label, "zebra")
        self.assertIsNotNone(self.window.uniqLabelList.findItemByLabel("zebra"))

    def test_keep_previous_does_not_fill_explicitly_empty_imported_frame(self):
        (self.labels_dir / "frame001.txt").unlink()
        self.window._config["keep_prev"] = True
        self.import_yolo()
        self.assertTrue(self.window._config["keep_prev"])
        self.assertEqual(len(self.window.canvas.shapes), 1)
        empty_path = self.images[1].with_suffix(".json")
        empty_contents = empty_path.read_bytes()
        self.assertEqual(self.read_annotation(1)["shapes"], [])

        self.window.fileListWidget.setCurrentRow(1)
        self.error_message.assert_not_called()
        self.assertEqual(self.window.imagePath, str(self.images[1]))
        self.assertEqual(self.window.canvas.shapes, [])
        self.assertEqual(empty_path.read_bytes(), empty_contents)
        self.assertFalse(self.window.dirty)

    def test_keep_previous_still_carries_to_frame_without_annotations(self):
        self.import_yolo(sequence=False, source=self.labels_dir / "frame000.txt")
        self.window._config["keep_prev"] = True
        self.assertFalse(self.images[1].with_suffix(".json").exists())
        self.window.fileListWidget.setCurrentRow(1)
        self.error_message.assert_not_called()
        self.assertEqual(self.window.imagePath, str(self.images[1]))
        self.assertEqual(len(self.window.canvas.shapes), 1)
        self.assertEqual(self.window.canvas.shapes[0].label, "car")
        self.assertEqual(self.read_annotation(1)["shapes"][0]["label"], "car")

    def test_import_then_sort_from_scratch_starts_at_first_frame(self):
        self.import_yolo()
        for index in range(3):
            self.assertIsNone(self.read_annotation(index)["shapes"][0]["track_id"])
        self.window.fileListWidget.setCurrentRow(1)
        self.assertEqual(self.window.imagePath, str(self.images[1]))
        self.run_sort(option=1)
        self.error_message.assert_not_called()
        ids = [self.read_annotation(index)["shapes"][0]["track_id"]
               for index in range(3)]
        self.assertIsNotNone(ids[0])
        self.assertEqual(ids, [ids[0]] * 3)
        self.assertEqual(self.window.imagePath, str(self.images[1]))
        self.assertEqual(self.window.canvas.shapes[0].track_id, ids[1])

    def test_sort_with_current_annotation_preserves_seed_and_earlier_frame(self):
        self.import_yolo()
        first_contents = self.images[0].with_suffix(".json").read_bytes()
        second = self.read_annotation(1)
        second["shapes"][0]["track_id"] = "42"
        self.images[1].with_suffix(".json").write_text(
            json.dumps(second), encoding="utf-8"
        )
        self.window.fileListWidget.setCurrentRow(1)
        self.assertEqual(self.window.canvas.shapes[0].track_id, "42")
        self.run_sort(option=2)
        self.error_message.assert_not_called()
        self.assertEqual(self.images[0].with_suffix(".json").read_bytes(), first_contents)
        self.assertEqual(self.read_annotation(1)["shapes"][0]["track_id"], "42")
        self.assertEqual(self.read_annotation(2)["shapes"][0]["track_id"], "42")
        self.assertEqual(self.window.canvas.shapes[0].track_id, "42")

    def test_cancelled_import_does_not_create_annotations(self):
        dialog = mock.Mock()
        dialog.exec_.return_value = QtWidgets.QDialog.Rejected
        with mock.patch.object(app, "YoloImportDialog", return_value=dialog):
            self.window.importYOLO()
        dialog.options.assert_not_called()
        self.assertFalse(list(self.directory.rglob("*.json")))
        self.error_message.assert_not_called()
        self.information_message.assert_not_called()

    def test_invalid_sequence_reports_error_and_writes_no_annotations(self):
        (self.labels_dir / "frame001.txt").write_text(
            "0 invalid 0.5 0.2 0.2\n", encoding="utf-8"
        )
        self.import_yolo(track=True)
        self.error_message.assert_called_once()
        self.assertIn("frame001.txt", self.error_message.call_args[0][1])
        self.assertFalse(list(self.directory.rglob("*.json")))
        self.assertEqual(self.window.canvas.shapes, [])
        self.information_message.assert_not_called()
        self.assertIsNone(QtWidgets.QApplication.overrideCursor())


if __name__ == "__main__":
    unittest.main()
