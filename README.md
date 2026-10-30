<h4 align="center">
  TrackMe
</h4>

<h4 align="center">
  Simple Tracking Annotation Tool based on LabelMe
</h4>

## Description
This tool is built to integrate tracking visualization and annotation capabilities into LabelMe. <br>
Please feel free to use if your project/work includes visualization of multi-object tracking and editting of object information. <br>
TrackMe annotation format is compatible with LabelMe annotation format (.json) without conversion.

<img src="examples/trackgui/TrackMe_overview.png" width="100%"/> <br>
<i>TrackMe saves and displays the tracking information of multiple objects on the right. It generates unique colors for different combinations of object label and ID  </i>

## Features
- Add/remove tracking ID.
- Associate boxes (assign IDs) for existing non-ID detection boxes in the video folder (SORT).
- Interpolate boxes in long video range in case no pre-defined detection boxes.
- Modify/Delete boxes of same info throughout a list of continuous frames.
- Display homegeneous color for same object info (for the sake of multi-view object tracking).
- Open native X-AnyLabeling JSON annotations alongside Labelme files.
- Import YOLO detection TXT files and assign tracking IDs with SORT.

## Installation
Please install Anaconda environment on your computer beforehand.

conda create --name=trackGUI python=3.8 <br>
conda activate trackGUI <br>
pip install -e . <br>

## Usage
conda activate trackGUI <br>
labelme

### X-AnyLabeling annotations

Open an X-AnyLabeling `.json` file or its image directory with the usual Open/Open
Dir actions or `labelme /path/to/annotations`. No conversion command is needed.
Images referenced by `imagePath` are resolved relative to the JSON file; embedded
image data is also supported.

Four-corner rectangles are imported as TrackGUI's two-corner boxes. For files
identified by X-AnyLabeling shape fields or geometry, `group_id` also supplies
the tracking ID when `track_id` is absent. Explicit tracking IDs, including null,
take precedence; ordinary Labelme group IDs retain their grouping behavior.
IDs without a configured color use the default shape color.

Rotated boxes and quadrilaterals are imported as polygons with the same outline.
Saving writes TrackGUI/Labelme JSON and preserves image and shape metadata such
as descriptions, review status, scores, difficulty, and attributes. Specialized
X-AnyLabeling editing tools and 3D cuboids are not supported.

### YOLO detection import and tracking

1. Open your image sequence with **Open Dir**. Images are processed in the order
   shown in the file list.
2. Choose **File > Import YOLO Detections** (also available under **Track**).
3. Select **Loaded image sequence** and its TXT folder, or **Current image** and
   one TXT file. For sequences, TXT filenames must match image names, such as
   `000001.txt` for `000001.jpg`. Mirrored subdirectories are supported.
4. Choose `classes.txt` (one class per line) or a YOLO `data.yaml` with a `names`
   list or mapping. Leave it blank to use the original order of `--labels`.
5. Leave **Run SORT tracking after import** checked to assign IDs immediately,
   or import boxes first and run **Track > ID Association** later.

Each detection row uses normalized coordinates and an optional confidence:

```text
class_id center_x center_y width height [confidence]
0 0.5 0.5 0.2 0.3
1 0.25 0.4 0.1 0.2 0.95
```

Class IDs start at zero. Empty or missing TXT files in a sequence become empty
frames; at least one TXT filename must match. Invalid rows report their filename
and line, and all inputs are validated before annotations are written. Existing
JSON files are protected unless **Replace existing JSON annotations** is checked.
Imported JSON files go to the active **Output Dir**, or beside the images.
The source TXT files are left intact.

Tracking associates boxes within the same class, handles empty frames and newly
appearing objects, and saves string `track_id` values in JSON. The original box
coordinates and confidence scores are retained. This imports existing detection
results; it does not run a YOLO model. Segmentation, pose, and oriented-box TXT
formats are not supported.

## Note
Each frame needs a JSON annotation for tracking; it may contain no boxes. YOLO
sequence import creates these files automatically.

+ **Track from Scratch** starts at the first loaded frame and assigns fresh IDs.
+ **Track w/ Current Annotation** starts at the current frame, preserving its
  existing IDs. Every box in that frame must have a unique nonnegative ID.
+ **End Frame** is the inclusive frame number in the loaded image list.

## Citation
If you find our work helpful, please consider citing our paper:
```
@article{phan2024trackme,
  title={TrackMe: A Simple and Effective Multiple Object Tracking Annotation Tool},
  author={Phan, Thinh and Phillips, Isaac and Lockett, Andrew and Kidd, Michael T and Le, Ngan},
  journal={arXiv preprint arXiv:2410.15518},
  year={2024}
}
```
