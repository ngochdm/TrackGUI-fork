import os

from qtpy import QtWidgets


class YoloImportDialog(QtWidgets.QDialog):
    """Choose YOLO detections and their ordered class names for import."""

    def __init__(self, parent=None, class_names=None, frame_count=1):
        super(YoloImportDialog, self).__init__(parent)
        self.setWindowTitle("Import YOLO Detections")
        self.setModal(True)
        self._class_names = list(class_names or [])
        self._frame_count = max(1, frame_count)

        layout = QtWidgets.QVBoxLayout(self)
        format_label = QtWidgets.QLabel(
            "YOLO detection rows: class cx cy width height [confidence].\n"
            "Box coordinates must be normalized to the image size."
        )
        format_label.setWordWrap(True)
        layout.addWidget(format_label)

        form = QtWidgets.QFormLayout()
        self.scope_combo = QtWidgets.QComboBox()
        self.scope_combo.addItem("Current image", "current")
        self.scope_combo.addItem("Loaded image sequence", "sequence")
        self.scope_combo.setCurrentIndex(1 if self._frame_count > 1 else 0)
        form.addRow("Import into:", self.scope_combo)

        self.frame_count_label = QtWidgets.QLabel()
        form.addRow("Frames:", self.frame_count_label)

        self.source_edit = QtWidgets.QLineEdit()
        self.source_browse = QtWidgets.QPushButton("Browse…")
        self.source_browse.clicked.connect(self._browse_source)
        source_row = QtWidgets.QHBoxLayout()
        source_row.addWidget(self.source_edit)
        source_row.addWidget(self.source_browse)
        self.source_label = QtWidgets.QLabel()
        form.addRow(self.source_label, source_row)

        self.classes_edit = QtWidgets.QLineEdit()
        self.classes_edit.setPlaceholderText(
            "Optional classes.txt or data.yaml"
            if self._class_names else "Choose classes.txt or data.yaml"
        )
        self.classes_browse = QtWidgets.QPushButton("Browse…")
        self.classes_browse.clicked.connect(self._browse_classes)
        classes_row = QtWidgets.QHBoxLayout()
        classes_row.addWidget(self.classes_edit)
        classes_row.addWidget(self.classes_browse)
        form.addRow("Class names:", classes_row)
        layout.addLayout(form)

        if self._class_names:
            class_hint = (
                "Leave Class names blank to use the {} loaded class names "
                "in their original order (class IDs start at 0)."
            ).format(len(self._class_names))
        else:
            class_hint = (
                "Choose ordered class names; the first class has ID 0."
            )
        class_label = QtWidgets.QLabel(class_hint)
        class_label.setWordWrap(True)
        layout.addWidget(class_label)

        self.sequence_hint = QtWidgets.QLabel(
            "Each TXT must match its image filename. Missing TXT files in "
            "the sequence are treated as empty detections."
        )
        self.sequence_hint.setWordWrap(True)
        layout.addWidget(self.sequence_hint)

        self.run_tracking = QtWidgets.QCheckBox(
            "Run SORT tracking after import"
        )
        layout.addWidget(self.run_tracking)
        self.overwrite = QtWidgets.QCheckBox(
            "Replace existing JSON annotations"
        )
        self.overwrite.setChecked(False)
        layout.addWidget(self.overwrite)

        output_label = QtWidgets.QLabel(
            "Imported JSON annotations are saved in the active output "
            "directory, or beside the images when no output directory is set."
        )
        output_label.setWordWrap(True)
        layout.addWidget(output_label)

        self.error_label = QtWidgets.QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setText("Import")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.scope_combo.currentIndexChanged.connect(self._scope_changed)
        self._scope_changed()

    def _scope_changed(self, index=None):
        sequence = self.scope_combo.currentData() == "sequence"
        self.source_edit.clear()
        self.source_label.setText("TXT folder:" if sequence else "TXT file:")
        self.source_edit.setPlaceholderText(
            "Folder containing YOLO TXT files" if sequence
            else "YOLO TXT file for the current image"
        )
        self.frame_count_label.setText(
            str(self._frame_count if sequence else 1)
        )
        self.sequence_hint.setVisible(sequence)
        self.run_tracking.setEnabled(sequence)
        self.run_tracking.setChecked(sequence and self._frame_count > 1)
        self.error_label.clear()
        self.error_label.hide()

    def _browse_source(self):
        if self.scope_combo.currentData() == "sequence":
            path = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Choose YOLO TXT folder", self.source_edit.text()
            )
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Choose YOLO TXT file", self.source_edit.text(),
                "YOLO detections (*.txt);;All files (*)"
            )
        if path:
            self.source_edit.setText(path)

    def _browse_classes(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose class names", self.classes_edit.text(),
            "Class names (*.txt *.yaml *.yml);;All files (*)"
        )
        if path:
            self.classes_edit.setText(path)

    def options(self):
        sequence = self.scope_combo.currentData() == "sequence"
        return {
            "sequence": sequence,
            "source": os.path.expanduser(self.source_edit.text().strip()),
            "classes_file": os.path.expanduser(
                self.classes_edit.text().strip()
            ),
            "track": sequence and self.run_tracking.isChecked(),
            "overwrite": self.overwrite.isChecked(),
        }

    def accept(self):
        options = self.options()
        source = options["source"]
        classes_file = options["classes_file"]
        error = None
        if options["sequence"]:
            if not os.path.isdir(source) or not os.access(source, os.R_OK):
                error = "Choose a readable folder containing YOLO TXT files."
        elif (not os.path.isfile(source)
              or not source.lower().endswith(".txt")
              or not os.access(source, os.R_OK)):
            error = "Choose a readable YOLO TXT file for the current image."
        if error is None and classes_file:
            if (not os.path.isfile(classes_file)
                    or not os.access(classes_file, os.R_OK)):
                error = "Choose a readable classes.txt or data.yaml file."
        elif error is None and not self._class_names:
            error = "Choose a classes.txt or data.yaml file with class names."

        if error:
            self.error_label.setText(error)
            self.error_label.show()
            return
        self.error_label.clear()
        self.error_label.hide()
        super(YoloImportDialog, self).accept()
