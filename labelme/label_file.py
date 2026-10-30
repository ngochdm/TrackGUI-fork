import base64
import contextlib
import io
import json
import os.path as osp

import PIL.Image

from labelme import PY2
from labelme import QT4
from labelme import __version__
from labelme import utils
from labelme.logger import logger

PIL.Image.MAX_IMAGE_PIXELS = None


@contextlib.contextmanager
def open(name, mode):
    assert mode in ["r", "w"]
    if PY2:
        mode += "b"
        encoding = None
    else:
        encoding = "utf-8"
    with io.open(name, mode, encoding=encoding) as stream:
        yield stream


class LabelFileError(Exception):
    pass


class LabelFile(object):
    suffix = ".json"

    def __init__(self, filename=None):
        self.shapes = []
        self.imagePath = None
        self.imageData = None
        if filename is not None:
            self.load(filename)
        self.filename = filename

    @staticmethod
    def load_image_file(filename):
        try:
            image_pil = PIL.Image.open(filename)
        except IOError:
            logger.error("Failed opening image file: {}".format(filename))
            return

        # apply orientation to image according to exif
        image_pil = utils.apply_exif_orientation(image_pil)

        with io.BytesIO() as f:
            ext = osp.splitext(filename)[1].lower()
            if PY2 and QT4:
                format = "PNG"
            elif ext in [".jpg", ".jpeg"]:
                format = "JPEG"
            else:
                format = "PNG"
            image_pil.save(f, format=format)
            f.seek(0)
            return f.read()

    @staticmethod
    def normalize_shapes(shapes):
        """Read Labelme and X-AnyLabeling shapes into TrackGUI's representation."""
        shape_keys = {
            "label",
            "points",
            "group_id",
            "track_id",
            "shape_type",
            "flags",
            "description",
            "mask",
            "other_data",
        }
        xany_fields = {"score", "difficult", "direction", "attributes", "kie_linking"}
        is_xanylabeling = any(
            xany_fields.intersection(shape)
            or shape.get("shape_type") in ("rotation", "quadrilateral")
            or (shape.get("shape_type") == "rectangle" and len(shape["points"]) == 4)
            for shape in shapes
        )
        normalized = []
        for shape in shapes:
            points = shape["points"]
            shape_type = shape.get("shape_type") or "polygon"
            if shape_type == "rectangle" and len(points) == 4:
                xs, ys = zip(*points)
                points = [[min(xs), min(ys)], [max(xs), max(ys)]]
            elif shape_type in ("rotation", "quadrilateral"):
                # Preserve the outline using the existing polygon editor.
                shape_type = "polygon"
            if shape_type not in (
                "polygon",
                "rectangle",
                "point",
                "line",
                "circle",
                "linestrip",
                "points",
                "mask",
            ):
                raise ValueError("Unsupported shape_type: {}".format(shape_type))
            track_id = shape.get("track_id")
            if "track_id" not in shape and is_xanylabeling:
                group_id = shape.get("group_id")
                track_id = str(group_id) if group_id is not None else None
            other_data = dict(shape.get("other_data") or {})
            other_data.update(
                (key, value) for key, value in shape.items() if key not in shape_keys
            )
            normalized.append(
                dict(
                    label=shape["label"],
                    points=points,
                    shape_type=shape_type,
                    flags=shape.get("flags") or {},
                    description=shape.get("description"),
                    group_id=shape.get("group_id"),
                    track_id=track_id,
                    mask=(
                        utils.img_b64_to_arr(shape["mask"])
                        if shape.get("mask")
                        else None
                    ),
                    other_data=other_data,
                )
            )
        return normalized

    def load(self, filename):
        keys = [
            "version",
            "imageData",
            "imagePath",
            "shapes",  # polygonal annotations
            "flags",  # image level flags
            "imageHeight",
            "imageWidth",
        ]
        try:
            with open(filename, "r") as f:
                data = json.load(f)

            imagePath = data["imagePath"]
            if data.get("imageData") is not None:
                imageData = base64.b64decode(data["imageData"])
                if PY2 and QT4:
                    imageData = utils.img_data_to_png_data(imageData)
            else:
                resolved_path = osp.join(osp.dirname(filename), imagePath)
                if not osp.isfile(resolved_path):
                    # Retain support for moved datasets with same-stem JPEGs.
                    fallback_path = osp.splitext(filename)[0] + ".jpg"
                    if osp.isfile(fallback_path):
                        resolved_path = fallback_path
                        imagePath = osp.basename(fallback_path)
                imageData = self.load_image_file(resolved_path)
                if imageData is None:
                    raise ValueError("Failed opening image file: {}".format(resolved_path))
            flags = data.get("flags") or {}
            self._check_image_height_and_width(
                base64.b64encode(imageData).decode("utf-8"),
                data.get("imageHeight"),
                data.get("imageWidth"),
            )
            shapes = self.normalize_shapes(data["shapes"])
        except Exception as e:
            raise LabelFileError(e)

        otherData = {}
        for key, value in data.items():
            if key not in keys:
                otherData[key] = value

        # Only replace data after everything is loaded.
        self.flags = flags
        self.shapes = shapes
        self.imagePath = imagePath
        self.imageData = imageData
        self.filename = filename
        self.otherData = otherData

    @staticmethod
    def _check_image_height_and_width(imageData, imageHeight, imageWidth):
        img_arr = utils.img_b64_to_arr(imageData)
        if imageHeight is not None and img_arr.shape[0] != imageHeight:
            logger.error(
                "imageHeight does not match with imageData or imagePath, "
                "so getting imageHeight from actual image."
            )
            imageHeight = img_arr.shape[0]
        if imageWidth is not None and img_arr.shape[1] != imageWidth:
            logger.error(
                "imageWidth does not match with imageData or imagePath, "
                "so getting imageWidth from actual image."
            )
            imageWidth = img_arr.shape[1]
        return imageHeight, imageWidth

    def save(
        self,
        filename,
        shapes,
        imagePath,
        imageHeight,
        imageWidth,
        imageData=None,
        otherData=None,
        flags=None,
    ):
        if imageData is not None:
            imageData = base64.b64encode(imageData).decode("utf-8")
            imageHeight, imageWidth = self._check_image_height_and_width(
                imageData, imageHeight, imageWidth
            )
        if otherData is None:
            otherData = {}
        if flags is None:
            flags = {}
        serialized_shapes = []
        for shape in shapes:
            serialized_shape = dict(shape.get("other_data") or {})
            serialized_shape.update(
                (key, value) for key, value in shape.items() if key != "other_data"
            )
            serialized_shapes.append(serialized_shape)
        data = dict(
            version=__version__,
            flags=flags,
            shapes=serialized_shapes,
            imagePath=imagePath,
            imageData=imageData,
            imageHeight=imageHeight,
            imageWidth=imageWidth,
        )
        for key, value in otherData.items():
            assert key not in data
            data[key] = value
        try:
            with open(filename, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.filename = filename
        except Exception as e:
            raise LabelFileError(e)

    @staticmethod
    def is_label_file(filename):
        return osp.splitext(filename)[1].lower() == LabelFile.suffix
