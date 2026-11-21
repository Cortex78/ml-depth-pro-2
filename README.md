<h1 align="center">Depth Pro 2: Sharp and Temporally Consistent Metric Monocular Depth</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2611.26603"><img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-b31b1b"></a>
  <a href="https://apple-aiml-research.github.io/ml-depth-pro-2/"><img alt="Project website" src="https://img.shields.io/badge/Project-Website-1f6feb"></a>
  <a href="https://huggingface.co/apple/DepthPro2"><img alt="Models" src="https://img.shields.io/badge/Models-Hugging%20Face-ffd21e"></a>
</p>

![Depth Pro 2 on images: inputs on top, predicted depth below](assets/images.jpg)

Depth Pro 2 estimates the depth of every pixel, in meters, from images taken with an ordinary camera. One network handles a single photo, a whole video, or a live camera feed, at the input's full resolution. Its depth is sharp at object edges and steady from frame to frame.

This repository contains the code to run the models and links to their weights. It does not include training or evaluation code.

## Getting started

On Linux, Windows or an Apple silicon Mac, install [uv](https://docs.astral.sh/uv/), clone this repository, and run (on Windows, in PowerShell; see [Windows](#windows) for the GPU):

```bash
uv run depth-pro-2 predict -i assets/example.jpg -o out/   # a photo
uv run depth-pro-2 predict -i assets/example.mp4 -o out/   # a video
```

The first run downloads the weights of the default model, H+ (3.5 GB). On a laptop, add `-m depthpro2-s` to use S (97 MB) instead. The command runs on the first available device: an NVIDIA GPU (CUDA), an Apple silicon GPU (MPS), or the CPU.

To process many images in one run, pass a folder or repeat `-i`. The model loads once for the whole run. Each image is predicted independently, with its own focal length. For the frames of a video, pass a video file instead, or pass the frames to `infer_video` in Python. Both process the frames jointly, with one focal length for the whole video.

### Windows

On Windows, `uv sync` (or the first `uv run`) installs PyTorch with CUDA 13.0 from the [PyTorch index](https://download.pytorch.org/whl/cu130), because PyPI only has CPU builds of PyTorch for Windows. CUDA 13.0 needs an NVIDIA GPU from the GeForce RTX 20 series (Turing) or newer and NVIDIA driver 580 or later. Install uv and set up the repository in PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
git clone https://github.com/Cortex78/ml-depth-pro-2.git
cd ml-depth-pro-2
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The last line should print a version ending in `+cu130` and `True`, and `predict` should report `Loading ... onto cuda`. If it prints `False`, update the NVIDIA driver. For an older GPU, such as the GeForce GTX 10 series, replace every `cu130` in `pyproject.toml` with `cu126` and run `uv sync` again.

Install with uv: `pip install .` ignores the CUDA index in `pyproject.toml` and installs the CPU build of PyTorch. When a video does not fit in GPU memory, the Windows driver moves the overflow to system memory, and the run slows down a lot instead of stopping with an out-of-memory error. If a video runs much slower than its first frames suggest, lower `--window` and `--overlap` as described in [Choosing settings](#choosing-settings), or set the driver to stop instead: in the NVIDIA Control Panel, under Manage 3D settings, set CUDA - Sysmem Fallback Policy to Prefer No Sysmem Fallback.

### Google Colab

[`notebooks/depth_pro_2_colab_a100.ipynb`](notebooks/depth_pro_2_colab_a100.ipynb) installs the repository on a Colab GPU runtime, runs the examples, times the four model sizes and compares the precisions, then runs on your own photos and videos. [Open it in Colab](https://colab.research.google.com/github/Cortex78/ml-depth-pro-2/blob/claude/festive-cerf-65vitz/notebooks/depth_pro_2_colab_a100.ipynb), choose Runtime → Change runtime type → A100 GPU, and run all cells.

For one picture only, [`notebooks/depth_pro_2_picture_normals.ipynb`](notebooks/depth_pro_2_picture_normals.ipynb) takes an uploaded picture and returns its depth, normal map and point cloud from the largest model, H+. [Open it in Colab](https://colab.research.google.com/github/Cortex78/ml-depth-pro-2/blob/claude/festive-cerf-65vitz/notebooks/depth_pro_2_picture_normals.ipynb).

## Outputs

The command writes its results to the folder given by `-o`:

| Option | Photo | Video |
|---|---|---|
| Always | `<name>-depth.png` | `<name>-depth.mp4` |
| `--save-png16` | `<name>-depth16.png` | `<name>-depth16/`, a PNG per frame |
| `--save-npz` | `<name>-depth.npz` | `<name>-depth/`, a file per frame |
| `--save-normals` | `<name>-normals.png` | `<name>-normals.mp4` |
| `--save-ply` | `<name>-points.ply` | `<name>-points/`, a file per frame |
| `--save-all` | all of the above | all of the above |

The network predicts two things: depth, at every pixel, and the field of view, which gives the focal length. Everything else here is computed from them: the normals and the 3D points follow from the depth and the focal length. For a 3D model of a whole scene from a video, see [3D reconstruction](#3d-reconstruction).

- The PNG or MP4 shows depth as a color image, red near and blue far.
- `--save-png16` adds inverse depth as 16-bit grayscale PNGs, white near and black far, for editing and compositing tools. Read them as raw data (Raw in Nuke, Non-Color in Blender). The values are scaled to the depth range of the photo, or of a video's first frame, so points that come nearer later in a video saturate at white. The command prints how to convert the values to meters.
- `--save-npz` adds the depth itself. Each file holds `depth`, in meters, `focallength_px`, the focal length used, in pixels, and `intrinsics`, the 3x3 camera matrix. With `--save-normals`, it also holds `normals`, `(H, W, 3)` unit vectors.
- `--save-normals` adds surface normals as a normal map: red points right, green up and blue toward the camera, as in OpenGL and Blender, so a wall facing the camera is light blue and the ground is green. Pixels without a normal, such as sky, are black. Each normal spans a few pixels (6 at 1920 wide), since depth varies too finely from pixel to pixel to give a steady normal on distant surfaces, and stays on its own side of an occlusion boundary.
- `--save-ply` adds a colored point cloud with normals, in meters, for MeshLab, CloudCompare or Blender. It leaves out sky (`--max-depth`, 1000 m by default) and points on occlusion boundaries, which would float between the two surfaces. Each point takes 27 bytes: at 1920x1440, a file is about 40 MB, and up to 75 MB without sky.
- The arrays and point clouds are in the camera frame of OpenCV and COLMAP: x right, y down and z forward. `intrinsics` follows OpenCV: the top-left pixel's center is at (0, 0), so the principal point is at ((width - 1) / 2, (height - 1) / 2).

### Interpreting the depth

- `depth` is the distance along the camera's viewing direction (the z coordinate of each point), not the straight-line distance from the camera.
- `focallength_px` is in pixels of the input and applies both horizontally and vertically. To turn depth into 3D points, use it with the image center as the principal point, or with your camera's principal point if the camera is calibrated.
- Sky pixels have infinite depth or, especially with the smaller models, a finite depth of several kilometers. Clip the depth range before making a point cloud or a color image.
- Depth scales with the focal length: a focal length 10% too long places everything 10% too far.
- In Python, `inverse_depth` is normalized by the focal length: it equals `focallength_px / (width * depth)`, where `width` is the image width in pixels.

## 3D reconstruction

Depth Pro 2's depth is metric and consistent from frame to frame, so a video of a static scene is a sequence of RGB-D images, and `reconstruct` turns it into a 3D model:

```bash
uv sync --extra reconstruct
uv run depth-pro-2 reconstruct -i assets/example.mp4 -o out/ -m depthpro2-s --colmap
```

1. It predicts depth for every frame, as `predict` does in buffered mode, with one focal length for the video.
2. It finds the camera's motion between each keyframe and the few before it: SIFT matches, lifted to 3D by the metric depth, give each pair's motion by PnP with RANSAC; where features are too few, dense RGB-D odometry stands in. A pose graph reconciles the pairs. As the depth is metric, the camera path is in meters.
3. It fuses the keyframes' depth in a truncated signed distance volume (TSDF), on the GPU if Open3D has CUDA, keeping surfaces seen by at least two keyframes.

| Output | |
|---|---|
| `<name>-mesh.ply` | Colored triangle mesh, in meters |
| `<name>-cloud.ply` | Its vertices as a point cloud with normals |
| `<name>-cameras.json` | The intrinsics, and each keyframe's camera-to-world pose |
| `<name>-trajectory.txt` | The same poses in the TUM format |
| `<name>-colmap/` | With `--colmap`: a COLMAP text model (`sparse/0`) and the keyframes (`images/`) |

The world frame is the first keyframe's camera frame: x right, y down and z forward. The COLMAP model can be refined with COLMAP, or used directly to train Gaussian splatting; its 3D points come from the fused cloud, without 2D observations.

Settings: `--stride N` uses every N-th frame as a keyframe (depth is still predicted for every frame); by default, every frame up to 300 keyframes, and enough of a stride to keep 300 for longer videos, which bounds memory. A folder of photos is taken in the order of the file names, numbers compared as numbers. `--resolution` is the size of the frames used for poses and fusion (960 px on the longer side by default). `--max-depth` leaves out farther depth, by default 3 times the median depth, and `--voxel` sets the size of the fused volume's voxels, by default the median depth divided by 300.

It works best on static scenes filmed with steady motion and some texture. Moving objects are mostly left out of the mesh but can mislead the poses when they fill the frame; mirrors, glass and featureless walls are hard; and a long video that never revisits a place drifts slowly, as there is no loop closure beyond the last few keyframes. On Linux without a display, Open3D also needs the EGL and OpenGL libraries: `sudo apt-get install libegl1 libgl1 libgomp1`.

## Models

Depth Pro 2 comes in four sizes. Larger models give sharper, steadier depth; smaller ones run faster. H+ is the default. On a laptop, start with S (`-m depthpro2-s`).

| Model | `-m` | Parameters | Download |
|:---:|:---:|:---:|:---:|
| S | `depthpro2-s` | 24M | [97 MB](https://ml-site.cdn-apple.com/models/depth-pro-2/depthpro2-s-bgcc6f8bg3.pt) |
| B | `depthpro2-b` | 97M | [386 MB](https://ml-site.cdn-apple.com/models/depth-pro-2/depthpro2-b-xb7m9egktw.pt) |
| L | `depthpro2-l` | 334M | [1.3 GB](https://ml-site.cdn-apple.com/models/depth-pro-2/depthpro2-l-tegstu7fim.pt) |
| H+ | `depthpro2-h` | 872M | [3.5 GB](https://ml-site.cdn-apple.com/models/depth-pro-2/depthpro2-h-eivtm5w7m5.pt) |

All sizes share a field-of-view estimator ([94 MB](https://ml-site.cdn-apple.com/models/depth-pro-2/depthpro2-fov-ew55zg4kq7.pt)),
which predicts the focal length when it is unknown. The paper compares the sizes, including their speed and memory. We also provide exported [Core ML](https://huggingface.co/apple/DepthPro2-coreml/) Depth Pro 2 models for Apple hardware.

## Choosing settings

The defaults follow the recommendations of the paper, which gives the evidence for them.

- **Video files.** By default (`--mode buffered`), the model processes clips of 16 frames that
  overlap by 4 frames, which keeps the depth steady where clips meet. `--window` sets the clip
  length and `--overlap` the overlap ($N$ and $K$ in the paper). Longer clips and larger overlaps
  give slightly steadier depth but run slower.
- **Live video.** With `--mode unbuffered`, the model predicts each frame as soon as it arrives,
  from the frame itself and the 15 frames before it (`--window`, $C$ in the paper). Use this mode
  when depth is needed immediately, as with a camera. To keep up with a camera, reduce the model
  size, the input size, and the window. The first frames of a stream have fewer frames before them
  to attend to, so their depth is less steady.
- **Resolution and memory.** Keep the input at full resolution for the sharpest edges. Memory
  grows with the frame size and the window, not with the length of the video. If a video does not
  fit in memory, first shorten `--window` (and `--overlap` with it), then switch to a smaller
  model. Resize the input only as a last resort: a much smaller input loses detail and can shift
  the scale of the depth.
- **Focal length.** Depth in meters requires the camera's focal length. Photos usually record it
  in their EXIF data, as a 35mm-equivalent focal length. For other photos and for videos, the
  field-of-view estimator predicts it. A video gets one focal length for all its frames: the median
  of the estimates over its first window. A live stream in Python sees no frames ahead, so it uses
  the estimate so far until its first window is full, and keeps that value from then on.
- **A known focal length.** Pass it in Python as `f_px`, in pixels of the frames you pass; if you
  resize the frames, scale `f_px` by the same factor. With a calibrated camera, undistort the
  frames and pass the horizontal focal length, fx. When `f_px` is given, the model skips the
  field-of-view estimator.
- **Precision.** On Apple GPUs and on NVIDIA GPUs from Ampere (GeForce RTX 30) on, the model
  runs under bfloat16 autocast by default (`--precision bf16`), as in the paper; on the CPU, it
  runs in float32. Older NVIDIA GPUs, such as the GeForce RTX 20 and GTX 16 series, only emulate
  bfloat16, so they default to float16, and GPUs before Volta to float32. `--precision fp16` uses
  float16 autocast instead, and `--precision fp32` runs in float32 without autocast. In Python,
  pass `precision=torch.float16` or `precision=torch.float32` to `create_model_and_transforms`.

## Python

```python
import av
from depth_pro_2 import create_model_and_transforms, load_rgb, upright

model, transform = create_model_and_transforms("depthpro2-h")  # on the GPU if there is one

# A photo. f_px is the focal length recorded in the photo, or None.
image, f_px = load_rgb("assets/example.jpg")
depth = model.infer(transform(image), f_px=f_px)["depth"]  # meters, shape (height, width)

# A video file, 16 frames at a time. upright turns phone videos the right way up.
video = av.open("assets/example.mp4")
frames = (transform(upright(f)) for f in video.decode(video=0))
for prediction in model.infer_video(frames):
    depth = prediction["depth"]

# A live feed, one frame at a time.
stream = model.stream(mode="unbuffered")
for f in av.open("assets/example.mp4").decode(video=0):  # stands in for a camera
    for prediction in stream.step(transform(upright(f))):
        depth = prediction["depth"]
```

Results are tensors on the model's device. `transform` takes an RGB uint8 array `(H, W, 3)`, as
`load_rgb` and `upright` return; convert BGR frames, such as those from OpenCV, first. `infer`
treats a stack of frames `(T, 3, H, W)` as one clip: the frames attend to each other and share one
focal length. Pass unrelated photos one at a time, each at its own size: padding or resizing them
to a shared size changes the depth.

A stream belongs to one video. At its end, `flush` returns the predictions still held. It does not
reset the stream, which keeps its recent frames and, once set, the focal length: open a new stream
for another video, camera or frame size.

## Mac demo

`demo/` holds a minimal Mac app that shows live depth from the camera, running Depth Pro 2 with Core
ML. It needs Apple silicon, macOS 15 or later, and Xcode 16 or later, or its command line tools.

```bash
uv run --extra coreml python -m demo.export stream -m depthpro2-s -w 4
./demo/run.sh depthpro2-s_480x854_w4.mlpackage
```

The [demo's README](demo/README.md) details the options, and shows how to export the model to Core ML and use it in your own app. It also has the runtime on an M4 Max MacBook Pro. For convenience, we also provide exported [Core ML models](https://huggingface.co/apple/DepthPro2-coreml/).

## Citation

If you find our work useful, please cite the following paper:

```bibtex
@article{DepthPro2:2026:arxiv,
  title      = {Depth Pro 2: Sharp and Temporally Consistent Metric Monocular Depth},
  author     = {Peiyun Hu and Marko Mihajlovic and Lyndon Duong and Lahav Lipson and Wei Dong and Ama\"{e}l Delaunoy and Stephan R. Richter and Vladlen Koltun},
  journal    = {arXiv preprint arXiv:2611.26603},
  year       = {2026},
  url        = {https://arxiv.org/abs/2611.26603},
}
```

## License

The code is under [LICENSE](LICENSE). The model weights are under [LICENSE_MODEL](LICENSE_MODEL),
which permits use "exclusively for Research Purposes", defined as "non-commercial scientific
research and academic development activities".

This codebase builds on third-party code and
models, listed with their licenses in [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS). We thank the authors of [TIMM](https://github.com/huggingface/pytorch-image-models) and [DINOv3](https://github.com/facebookresearch/dinov3/), on which the Depth Pro 2 code and models build.
