"""Read YOLO detection annotations without modifying their source files."""

import math
from pathlib import Path

import yaml


class YoloImportError(ValueError):
    pass


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError:
                raise yaml.constructor.ConstructorError(
                    None, None, "Invalid mapping key", key_node.start_mark
                )
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    None, None, "Duplicate mapping key", key_node.start_mark
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise YoloImportError("{}: {}".format(path, error)) from error


def _validate_class_names(names, path):
    if not isinstance(names, (list, tuple)) or not names:
        raise YoloImportError(
            "{}: Expected a nonempty list of class names".format(path)
        )
    cleaned = []
    seen = set()
    for index, name in enumerate(names):
        if not isinstance(name, str) or not name.strip():
            raise YoloImportError(
                "{}: Empty or invalid class name for class ID {}".format(path, index)
            )
        name = name.strip()
        if name in seen:
            raise YoloImportError("{}: Duplicate class name {!r}".format(path, name))
        seen.add(name)
        cleaned.append(name)
    return cleaned


def load_class_names(path):
    """Load ordered names from classes.txt or a YOLO data.yaml file."""
    content = _read_text(path)
    if Path(path).suffix.lower() in (".yaml", ".yml"):
        try:
            data = yaml.load(content, Loader=_UniqueKeyLoader)
        except yaml.YAMLError as error:
            mark = getattr(error, "problem_mark", None)
            location = "{}:{}".format(path, mark.line + 1) if mark else str(path)
            raise YoloImportError(
                "{}: Invalid class YAML: {}".format(location, error)
            ) from error
        if not isinstance(data, dict) or "names" not in data:
            raise YoloImportError(
                "{}: YAML must contain a 'names' list or mapping".format(path)
            )
        names = data["names"]
        if isinstance(names, dict):
            if any(type(key) is not int for key in names) or set(names) != set(
                range(len(names))
            ):
                raise YoloImportError(
                    "{}: Class IDs must be consecutive integers starting at 0".format(
                        path
                    )
                )
            names = [names[index] for index in range(len(names))]
    else:
        names = content.splitlines()
    return _validate_class_names(names, path)


def load_yolo_shapes(path, class_names, image_width, image_height):
    """Read normalized YOLO boxes as native, untracked rectangle dictionaries.

    Each row contains ``class_id cx cy width height [confidence]``. Boxes are
    clipped to the image boundaries; blank lines and ``#`` comments are ignored.
    """
    class_names = _validate_class_names(class_names, path)
    try:
        dimensions = [float(image_width), float(image_height)]
    except (TypeError, ValueError, OverflowError) as error:
        raise YoloImportError(
            "{}: Image dimensions must be positive finite numbers".format(path)
        ) from error
    if any(isinstance(value, bool) for value in (image_width, image_height)) or any(
        not math.isfinite(value) or value <= 0 for value in dimensions
    ):
        raise YoloImportError(
            "{}: Image dimensions must be positive finite numbers".format(path)
        )
    image_width, image_height = dimensions
    shapes = []
    for line_number, line in enumerate(_read_text(path).splitlines(), start=1):
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        location = "{}:{}".format(path, line_number)
        if len(fields) not in (5, 6):
            raise YoloImportError(
                "{}: Expected class_id cx cy width height [confidence]".format(location)
            )
        try:
            class_id = int(fields[0])
        except ValueError as error:
            raise YoloImportError(
                "{}: Class ID must be an integer".format(location)
            ) from error
        if class_id < 0 or class_id >= len(class_names):
            raise YoloImportError("{}: Unknown class ID {}".format(location, class_id))
        try:
            values = [float(field) for field in fields[1:]]
        except ValueError as error:
            raise YoloImportError(
                "{}: Box coordinates and confidence must be numeric".format(location)
            ) from error
        if not all(math.isfinite(value) for value in values):
            raise YoloImportError(
                "{}: Box coordinates and confidence must be finite".format(location)
            )
        center_x, center_y, width, height = values[:4]
        if not (
            0 <= center_x <= 1
            and 0 <= center_y <= 1
            and 0 < width <= 1
            and 0 < height <= 1
        ):
            raise YoloImportError(
                "{}: Centers must be in [0, 1] and box sizes in (0, 1]".format(location)
            )
        if len(values) == 5 and not 0 <= values[4] <= 1:
            raise YoloImportError("{}: Confidence must be in [0, 1]".format(location))
        x_min = max(0.0, (center_x - width / 2) * image_width)
        y_min = max(0.0, (center_y - height / 2) * image_height)
        x_max = min(image_width, (center_x + width / 2) * image_width)
        y_max = min(image_height, (center_y + height / 2) * image_height)
        if x_max <= x_min or y_max <= y_min:
            raise YoloImportError("{}: Box has no area after clipping".format(location))
        shape = dict(
            label=class_names[class_id],
            points=[[x_min, y_min], [x_max, y_max]],
            group_id=None,
            track_id=None,
            shape_type="rectangle",
            flags={},
            description="",
            mask=None,
        )
        if len(values) == 5:
            shape["score"] = values[4]
        shapes.append(shape)
    return shapes
