"""Import detection sequences and save tracked annotations."""

import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image

from labelme import __version__
from labelme.label_file import LabelFile
from labelme.logger import logger
from labelme.tracking import assign_track_ids
from labelme.yolo import YoloImportError
from labelme.yolo import load_yolo_shapes


def _annotation_paths(image_paths, output_dir):
    images = [Path(path).resolve() for path in image_paths]
    if not images:
        raise YoloImportError("Open an image or image directory first.")
    outputs = [
        Path(output_dir).resolve() / image.with_suffix(".json").name
        if output_dir else image.with_suffix(".json")
        for image in images
    ]
    if len(set(outputs)) != len(outputs):
        raise YoloImportError(
            "Multiple images would write the same JSON filename. "
            "Use a sequence with unique image names or separate output directories."
        )
    return images, outputs


def _write_annotations(outputs, documents):
    # Validate serialization and stage every file before replacing any annotation.
    contents = [json.dumps(data, ensure_ascii=False, indent=2) for data in documents]
    staged = []
    written = 0
    try:
        for output, content in zip(outputs, contents):
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=str(output.parent),
                prefix=".trackgui-", suffix=".tmp", delete=False,
            ) as stream:
                staged.append(Path(stream.name))
                stream.write(content)
        for temporary, output in zip(staged, outputs):
            os.replace(str(temporary), str(output))
            written += 1
    except OSError as error:
        raise YoloImportError(
            "Could not save annotations ({} of {} files written): {}".format(
                written, len(outputs), error
            )
        ) from error
    finally:
        for temporary in staged:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning("Could not remove temporary annotation: %s", temporary)


def import_yolo_annotations(
    image_paths, source, class_names, output_dir=None, image_root=None,
    overwrite=False, track=False,
):
    """Import one TXT file or matching sequence files, optionally assigning IDs."""
    images, outputs = _annotation_paths(image_paths, output_dir)
    source = Path(source).resolve()
    if not source.exists():
        raise YoloImportError("YOLO source does not exist: {}".format(source))
    if source.is_file() and len(images) != 1:
        raise YoloImportError("Choose a TXT directory to import an image sequence.")
    conflicts = [path for path in outputs if path.exists()]
    if conflicts and not overwrite:
        raise YoloImportError(
            "{} JSON file(s) already exist, including {}. Select 'Replace existing "
            "JSON annotations' or choose a different Output Dir.".format(
                len(conflicts), conflicts[0]
            )
        )
    root = Path(image_root).resolve() if image_root else Path(
        os.path.commonpath([str(image.parent) for image in images])
    )
    names = Counter(image.stem for image in images)
    matched = 0
    documents = []
    for image, output in zip(images, outputs):
        if source.is_file():
            annotation = source
        else:
            try:
                annotation = source / image.relative_to(root).with_suffix(".txt")
            except ValueError:
                annotation = None
            if annotation is None or not annotation.is_file():
                flat_annotation = source / image.with_suffix(".txt").name
                if flat_annotation.is_file() and names[image.stem] > 1:
                    raise YoloImportError(
                        "Ambiguous TXT file {}: multiple images share this name. "
                        "Mirror the image subdirectories in the labels directory."
                        .format(flat_annotation)
                    )
                annotation = flat_annotation
        try:
            with Image.open(str(image)) as opened:
                width, height = opened.size
                if opened.getexif().get(274) in (5, 6, 7, 8):
                    width, height = height, width
        except (OSError, ValueError) as error:
            raise YoloImportError(
                "Cannot read image {}: {}".format(image, error)
            ) from error
        shapes = []
        if annotation.is_file():
            shapes = load_yolo_shapes(annotation, class_names, width, height)
            matched += 1
        documents.append(dict(
            version=__version__, flags={}, shapes=shapes,
            imagePath=os.path.relpath(str(image), str(output.parent)),
            imageData=None, imageHeight=height, imageWidth=width,
        ))
    if not matched:
        raise YoloImportError(
            "No TXT filenames match the loaded images in {}. "
            "Each TXT file must have the same stem as its image.".format(source)
        )
    if track:
        frames = assign_track_ids([data["shapes"] for data in documents])
        for data, shapes in zip(documents, frames):
            data["shapes"] = shapes
    _write_annotations(outputs, documents)
    return dict(
        frames=len(images), detections=sum(len(data["shapes"]) for data in documents),
        missing=len(images) - matched, tracked=track,
    )


def track_annotations(image_paths, output_dir=None, use_existing_ids=False):
    """Assign tracking IDs to existing JSON files in the given frame order."""
    _, outputs = _annotation_paths(image_paths, output_dir)
    documents = []
    frames = []
    for path in outputs:
        try:
            with path.open(encoding="utf-8") as stream:
                document = json.load(stream)
            if (
                not isinstance(document, dict)
                or not isinstance(document.get("shapes"), list)
                or any(not isinstance(shape, dict) for shape in document["shapes"])
            ):
                raise ValueError("Expected a JSON object with a list of shape objects.")
            frames.append(LabelFile.normalize_shapes(document["shapes"]))
            documents.append(document)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise YoloImportError(
                "Cannot read annotations {}: {}".format(path, error)
            ) from error
    tracked = assign_track_ids(frames, use_existing_ids=use_existing_ids)
    for document, shapes in zip(documents, tracked):
        serialized = []
        for shape in shapes:
            data = dict(shape.get("other_data") or {})
            data.update(
                (key, value) for key, value in shape.items() if key != "other_data"
            )
            serialized.append(data)
        document["shapes"] = serialized
        document["version"] = __version__
    _write_annotations(outputs, documents)
    return dict(frames=len(outputs), detections=sum(map(len, tracked)))
