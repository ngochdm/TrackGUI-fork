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

## Note
All frames must have labeled boxes
+ Track from scratch: track from the first frame to the end frame with automatic ID assignment
+ Track from Current Frame w/ Annotation: track from the current (being opened) frame with the modified ID or manual ID assignment
+ Track from Current Frame w/0 Annotation: track from the current (being opened) frame with automatic ID assignment

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
