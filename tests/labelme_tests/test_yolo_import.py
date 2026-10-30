import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from labelme.annotation_import import import_yolo_annotations
from labelme.annotation_import import track_annotations
from labelme.yolo import YoloImportError


class YoloSequenceImportTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.images = self.directory / "images"
        self.labels = self.directory / "labels"
        self.images.mkdir()
        self.labels.mkdir()
        self.names = ["car", "traffic light"]

    def image(self, name, size=(640, 480)):
        path = self.images / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, "white").save(str(path))
        return path

    def label(self, name, content="0 0.5 0.5 0.5 0.25\n"):
        path = self.labels / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def document(self, image):
        return json.loads(image.with_suffix(".json").read_text(encoding="utf-8"))

    def test_sequence_uses_each_image_size_and_keeps_empty_frames(self):
        images = [
            self.image("001.png"), self.image("002.png", (320, 240)),
            self.image("003.png"), self.image("004.png"),
        ]
        self.label("001.txt")
        confidence_file = self.label("002.txt", "1 0.5 0.5 0.5 0.25 0.9\n")
        original_txt = confidence_file.read_bytes()
        self.label("003.txt", "")
        result = import_yolo_annotations(images, self.labels, self.names)
        self.assertEqual(
            result, dict(frames=4, detections=2, missing=1, tracked=False)
        )
        first, second, empty, missing = map(self.document, images)
        self.assertEqual(first["shapes"][0]["points"], [[160, 180], [480, 300]])
        self.assertEqual(second["shapes"][0]["points"], [[80, 90], [240, 150]])
        self.assertEqual(second["shapes"][0]["label"], "traffic light")
        self.assertEqual(second["shapes"][0]["score"], 0.9)
        self.assertIsNone(second["shapes"][0]["track_id"])
        self.assertEqual((second["imageWidth"], second["imageHeight"]), (320, 240))
        self.assertIsNone(second["imageData"])
        self.assertEqual(empty["shapes"], [])
        self.assertEqual(missing["shapes"], [])
        self.assertEqual(confidence_file.read_bytes(), original_txt)

    def test_import_and_track_preserves_ids_across_empty_frame_and_classes(self):
        images = [self.image("{:03d}.png".format(index)) for index in range(4)]
        self.label("000.txt", "0 0.3 0.5 0.2 0.2\n1 0.3 0.5 0.2 0.2\n")
        self.label("001.txt", "0 0.31 0.5 0.2 0.2\n1 0.31 0.5 0.2 0.2\n")
        self.label("002.txt", "")
        self.label(
            "003.txt",
            "1 0.33 0.5 0.2 0.2\n0 0.33 0.5 0.2 0.2\n0 0.8 0.5 0.2 0.2\n",
        )
        result = import_yolo_annotations(images, self.labels, self.names, track=True)
        frames = [self.document(image)["shapes"] for image in images]
        self.assertTrue(result["tracked"])
        self.assertEqual([shape["track_id"] for shape in frames[0]], ["0", "1"])
        self.assertEqual([shape["track_id"] for shape in frames[1]], ["0", "1"])
        self.assertEqual(frames[2], [])
        self.assertEqual([shape["track_id"] for shape in frames[3]], ["1", "0", "2"])
        self.assertEqual([shape["label"] for shape in frames[3]],
                         ["traffic light", "car", "car"])

    def test_single_image_accepts_a_differently_named_txt(self):
        image = self.image("scene.png")
        source = self.label("custom_predictions.txt")
        import_yolo_annotations([image], source, self.names)
        self.assertEqual(self.document(image)["shapes"][0]["label"], "car")
        self.assertFalse(source.with_suffix(".json").exists())

    def test_output_directory_relative_paths_and_subsequent_tracking(self):
        images = [self.image("001.png"), self.image("002.png")]
        for image in images:
            self.label(image.with_suffix(".txt").name)
        output_dir = self.directory / "output"
        import_yolo_annotations(images, self.labels, self.names, output_dir=output_dir)
        result = track_annotations(images, output_dir=output_dir)
        self.assertEqual(result, dict(frames=2, detections=2))
        for image in images:
            output = output_dir / image.with_suffix(".json").name
            data = json.loads(output.read_text())
            self.assertEqual((output.parent / data["imagePath"]).resolve(), image)
            self.assertEqual(data["shapes"][0]["track_id"], "0")
            self.assertFalse(image.with_suffix(".json").exists())

    def test_malformed_late_row_writes_no_files(self):
        images = [self.image("001.png"), self.image("002.png")]
        self.label("001.txt")
        self.label("002.txt", "0 0.5 0.5 0.5 0.25\n0 bad 0.5 0.5 0.25\n")
        with self.assertRaisesRegex(YoloImportError, "002.txt:2:"):
            import_yolo_annotations(images, self.labels, self.names)
        self.assertEqual(list(self.images.glob("*.json")), [])

    def test_existing_conflict_protects_all_outputs_until_explicit_overwrite(self):
        images = [self.image("001.png"), self.image("002.png")]
        for image in images:
            self.label(image.with_suffix(".txt").name)
        existing = images[1].with_suffix(".json")
        original = '{"manually_edited": true}\n'
        existing.write_text(original)
        with self.assertRaisesRegex(YoloImportError, "already exist"):
            import_yolo_annotations(images, self.labels, self.names)
        self.assertEqual(existing.read_text(), original)
        self.assertFalse(images[0].with_suffix(".json").exists())
        import_yolo_annotations(images, self.labels, self.names, overwrite=True)
        self.assertNotIn("manually_edited", self.document(images[1]))
        self.assertEqual(len(self.document(images[0])["shapes"]), 1)

    def test_no_matching_txt_files_fails_without_writing(self):
        image = self.image("scene.png")
        self.label("different.txt")
        with self.assertRaisesRegex(YoloImportError, "No TXT filenames match"):
            import_yolo_annotations([image], self.labels, self.names)
        self.assertFalse(image.with_suffix(".json").exists())

    def test_duplicate_output_names_fail_before_writing(self):
        images = [self.image("a/frame.png"), self.image("b/frame.png")]
        self.label("a/frame.txt")
        self.label("b/frame.txt")
        output_dir = self.directory / "output"
        with self.assertRaisesRegex(YoloImportError, "same JSON filename"):
            import_yolo_annotations(
                images, self.labels, self.names, output_dir=output_dir,
                image_root=self.images,
            )
        self.assertFalse(output_dir.exists())

    def test_mirrored_nested_labels_work_and_ambiguous_flat_names_fail(self):
        images = [self.image("a/frame.png"), self.image("b/frame.png")]
        first = self.label("a/frame.txt")
        second = self.label("b/frame.txt", "1 0.5 0.5 0.5 0.25\n")
        import_yolo_annotations(images, self.labels, self.names, image_root=self.images)
        self.assertEqual(self.document(images[0])["shapes"][0]["label"], "car")
        self.assertEqual(self.document(images[1])["shapes"][0]["label"], "traffic light")
        originals = [image.with_suffix(".json").read_bytes() for image in images]
        first.unlink()
        second.unlink()
        self.label("frame.txt")
        with self.assertRaisesRegex(YoloImportError, "Ambiguous TXT"):
            import_yolo_annotations(
                images, self.labels, self.names, image_root=self.images, overwrite=True
            )
        self.assertEqual(
            [image.with_suffix(".json").read_bytes() for image in images], originals
        )

    def test_duplicate_flat_names_outside_stale_image_root_do_not_overwrite(self):
        images = [self.image("a/frame.png"), self.image("b/frame.png")]
        self.label("frame.txt")
        for index, image in enumerate(images):
            image.with_suffix(".json").write_text(
                json.dumps(dict(shapes=[], description="Keep {}".format(index)))
            )
        originals = [image.with_suffix(".json").read_bytes() for image in images]
        with self.assertRaisesRegex(YoloImportError, "Ambiguous TXT"):
            import_yolo_annotations(
                images, self.labels, self.names,
                image_root=self.directory / "old-image-directory", overwrite=True,
            )
        self.assertEqual(
            [image.with_suffix(".json").read_bytes() for image in images], originals
        )

    def test_tracking_keeps_existing_metadata_and_seed_ids(self):
        images = [self.image("001.png"), self.image("002.png")]
        for image in images:
            self.label(image.with_suffix(".txt").name, "0 0.5 0.5 0.5 0.25 0.9\n")
        import_yolo_annotations(images, self.labels, self.names)
        for image in images:
            data = self.document(image)
            data.update(checked=True, description="Keep image notes")
            shape = data["shapes"][0]
            shape.update(
                track_id="43", group_id=8, description="Keep object notes",
                attributes={"color": "blue"}, flags={"occluded": True},
            )
            image.with_suffix(".json").write_text(json.dumps(data))
        track_annotations(images, use_existing_ids=True)
        for image in images:
            data = self.document(image)
            shape = data["shapes"][0]
            self.assertTrue(data["checked"])
            self.assertEqual(data["description"], "Keep image notes")
            self.assertEqual(shape["description"], "Keep object notes")
            self.assertEqual(shape["attributes"], {"color": "blue"})
            self.assertEqual(shape["flags"], {"occluded": True})
            self.assertEqual(shape["group_id"], 8)
            self.assertEqual(shape["score"], 0.9)
            self.assertEqual(shape["track_id"], "43")
            self.assertEqual(shape["points"], [[160, 180], [480, 300]])
            self.assertNotIn("other_data", shape)

    def test_tracking_missing_or_invalid_later_annotation_never_partially_writes(self):
        images = [self.image("001.png"), self.image("002.png")]
        for image in images:
            self.label(image.with_suffix(".txt").name)
        import_yolo_annotations(images, self.labels, self.names)
        first, second = [image.with_suffix(".json") for image in images]
        first_original = first.read_bytes()
        valid_second = self.document(images[1])
        for failure in ("missing", "unsupported_shape", "malformed_json"):
            with self.subTest(failure=failure):
                if failure == "missing":
                    second.unlink()
                elif failure == "unsupported_shape":
                    data = json.loads(json.dumps(valid_second))
                    data["shapes"][0]["shape_type"] = "polygon"
                    second.write_text(json.dumps(data))
                else:
                    second.write_text("not json")
                second_original = second.read_bytes() if second.exists() else None
                with self.assertRaises(ValueError):
                    track_annotations(images)
                self.assertEqual(first.read_bytes(), first_original)
                self.assertEqual(
                    second.read_bytes() if second.exists() else None, second_original
                )

    def test_malformed_annotation_structure_reports_source_without_overwriting(self):
        images = [self.image("001.png"), self.image("002.png")]
        for image in images:
            self.label(image.with_suffix(".txt").name)
        import_yolo_annotations(images, self.labels, self.names)
        first, second = [image.with_suffix(".json") for image in images]
        first_original = first.read_bytes()
        for malformed in (
            {"shapes": "rectangle"}, {"shapes": ""},
            {"shapes": ["rectangle"]}, {"shapes": {}}, {"shapes": None},
            "document", [], None,
        ):
            with self.subTest(document=malformed):
                first.write_bytes(first_original)
                second.write_text(json.dumps(malformed))
                second_original = second.read_bytes()
                with self.assertRaisesRegex(YoloImportError, "002.json"):
                    track_annotations(images)
                self.assertEqual(first.read_bytes(), first_original)
                self.assertEqual(second.read_bytes(), second_original)


if __name__ == "__main__":
    unittest.main()
