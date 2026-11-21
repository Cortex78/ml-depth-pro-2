#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import logging
import math
import re
import sys
from collections import Counter, deque
from contextlib import ExitStack
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Mapping, Sequence

import av
import click
import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image

from .geometry import intrinsics, point_cloud, write_ply
from .model import (
    DEFAULT_MODEL,
    DEFAULT_OVERLAP,
    DEFAULT_WINDOW,
    PRESETS,
    DepthPro2,
    create_model_and_transforms,
)
from .utils import default_device, get_supported_image_extensions, load_rgb, upright

VIDEO_EXTENSIONS = (".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".mts")

COLOR_LUT = (colormaps["turbo"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
#: Room above the nearest point of the frame that fixes the color scale, for nearer ones later.
COLOR_HEADROOM = 0.2

PRECISIONS = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

FRAGMENTED_MP4 = "frag_keyframe+empty_moov+default_base_moof"

#: Log-spaced depth bins from 1 cm to 10 km, and the pixel stride, of the summary's depth range.
DEPTH_BINS = np.logspace(-2, 4, 1001)
DEPTH_STRIDE = 8


@click.group()
def main_cli():
    """Run Depth Pro 2 inference."""


@main_cli.command("predict")
@click.option(
    "-i",
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    multiple=True,
    required=True,
    help="An image, a video, or a folder of them. Repeat -i for more; the model loads once.",
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Folder to write the depth to.",
)
@click.option(
    "-m",
    "--model",
    type=click.Choice(sorted(PRESETS)),
    default=DEFAULT_MODEL,
    show_default=True,
    help="Which Depth Pro 2 variant to run.",
)
@click.option(
    "--mode",
    type=click.Choice(["buffered", "unbuffered"]),
    default="buffered",
    show_default=True,
    help="Video input only: buffered processes overlapping clips of --window frames, so a frame "
    "also sees the frames after it. unbuffered predicts each frame as it arrives, from itself "
    "and the frames before it, for a fraction of the latency.",
)
@click.option(
    "--window",
    type=click.IntRange(min=1),
    default=DEFAULT_WINDOW,
    show_default=True,
    help="Video input only: frames attended together: the clip length (buffered) or the cache "
    "size, the current frame included (unbuffered).",
)
@click.option(
    "--overlap",
    type=click.IntRange(min=0),
    default=DEFAULT_OVERLAP,
    show_default=True,
    help="Video input only: frames shared between consecutive clips, fewer than --window. "
    "Buffered mode only; unbuffered has no clips to overlap.",
)
@click.option(
    "--precision",
    type=click.Choice(sorted(PRECISIONS)),
    default=None,
    help="Autocast dtype of the backbones and the decoder, the heads staying float32. By default "
    "bf16, as trained and evaluated, on Apple GPUs and NVIDIA GPUs from Ampere (RTX 30) on; fp16 "
    "on older NVIDIA GPUs, which only emulate bf16; and fp32 on the CPU, where bf16 is much "
    "slower. fp16 is closer to fp32 than bf16 is; fp32 runs without autocast.",
)
@click.option(
    "--save-npz",
    is_flag=True,
    help="Also write depth in meters and the focal length as .npz. Large for a video, which "
    "gets one file per frame: a 4K frame is 33 MB, so a minute of footage is about 60 GB.",
)
@click.option(
    "--save-png16",
    is_flag=True,
    help="Also write inverse depth as 16-bit grayscale PNG, for editing and compositing tools: "
    "white is near and black is infinitely far, on the color preview's scale.",
)
@click.option(
    "--save-ply",
    is_flag=True,
    help="Also write a colored 3D point cloud as .ply, in meters in the camera frame. "
    "A video gets one file per frame: 15 bytes per point, about 25 MB at 1920x1440.",
)
@click.option(
    "--max-depth",
    type=click.FloatRange(min=0, min_open=True),
    default=1000.0,
    show_default=True,
    help="--save-ply only: leave out points farther than this many meters, such as sky.",
)
@click.option(
    "--save-all",
    is_flag=True,
    help="Write every output: --save-npz, --save-png16 and --save-ply.",
)
def predict_cli(
    input_path: tuple[Path, ...],
    output_path: Path,
    model: str,
    mode: str,
    window: int,
    overlap: int,
    precision: str | None,
    save_npz: bool,
    save_png16: bool,
    save_ply: bool,
    max_depth: float,
    save_all: bool,
):
    """Predict metric depth for images and videos.

    \b
    Writes into the output folder, for each input NAME.ext:
      NAME-depth.png                        inverse depth in color, for an image
      NAME-depth.mp4                        the same for a video
      NAME-depth.npz                        with --save-npz: depth in meters, focallength_px
                                            and the 3x3 intrinsics
      NAME-depth/000000.npz                 the same for a video, one file per frame
      NAME-depth16.png                      with --save-png16: inverse depth in 16-bit gray
      NAME-depth16/NAME-depth16.000000.png  the same for a video, one file per frame
      NAME-points.ply                       with --save-ply: colored point cloud
      NAME-points/000000.ply                the same for a video, one file per frame
    Colors follow the Turbo map: dark blue is infinitely far, where depth is inf. Arrays and
    point clouds are in the camera frame of OpenCV: x right, y down and z forward.
    """  # noqa: D301 - click keeps a paragraph after \b as written.
    # A redirected stdout on Windows uses the ANSI code page, which cannot encode every file name.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    logging.basicConfig(format="%(message)s")
    logging.getLogger("depth_pro_2").setLevel(logging.INFO)

    if mode == "buffered" and overlap >= window:
        raise click.BadParameter(
            f"{overlap} must be less than --window ({window}).", param_hint="'--overlap'"
        )
    outputs = Outputs(
        npz=save_npz or save_all,
        png16=save_png16 or save_all,
        ply=save_ply or save_all,
        max_depth=max_depth,
    )
    inputs = collect_inputs(input_path, output_path)
    device = default_device()
    net, transform = create_model_and_transforms(
        model, device, None if precision is None else PRECISIONS[precision]
    )
    click.echo(f"Loading {model} onto {device} in {str(net.precision).removeprefix('torch.')}.")
    output_path.mkdir(parents=True, exist_ok=True)
    for path in inputs:
        try:
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                predict_video(net, transform, path, output_path, mode, window, overlap, outputs)
            else:
                predict_image(net, transform, path, output_path, outputs)
        except torch.OutOfMemoryError as error:
            raise click.ClickException(
                f"Out of {device.type.upper()} memory on {path}. For a video, lower --window and "
                f"--overlap first; then use a smaller model, such as -m depthpro2-s."
            ) from error


@main_cli.command("reconstruct")
@click.option(
    "-i",
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="A video, or a folder of photos of the same size of a static scene, taken in the order "
    "of their file names (frame2 before frame10).",
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Folder to write the reconstruction to.",
)
@click.option(
    "-m",
    "--model",
    type=click.Choice(sorted(PRESETS)),
    default=DEFAULT_MODEL,
    show_default=True,
    help="Which Depth Pro 2 variant to run.",
)
@click.option("--window", type=click.IntRange(min=1), default=DEFAULT_WINDOW, show_default=True)
@click.option("--overlap", type=click.IntRange(min=0), default=DEFAULT_OVERLAP, show_default=True)
@click.option("--precision", type=click.Choice(sorted(PRECISIONS)), default=None)
@click.option(
    "--stride",
    type=click.IntRange(min=1),
    default=None,
    help="Use every N-th frame as a keyframe; depth is still predicted for every frame, so it "
    "stays consistent. Default: 1, or enough for at most 300 keyframes, which bounds memory.",
)
@click.option(
    "--resolution",
    type=click.IntRange(min=64),
    default=960,
    show_default=True,
    help="Longer side, in pixels, of the frames used for poses and fusion.",
)
@click.option(
    "--max-depth",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    help="Leave out depth beyond this many meters, such as sky. Default: 3 times the median depth.",
)
@click.option(
    "--voxel",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    help="Voxel size of the fused volume in meters. Default: the median depth divided by 300.",
)
@click.option(
    "--neighbors",
    type=click.IntRange(min=1),
    default=3,
    show_default=True,
    help="Odometry pairs from each keyframe to the ones before it.",
)
@click.option(
    "--min-views",
    type=click.IntRange(min=1),
    default=2,
    show_default=True,
    help="Keep only surfaces seen by at least this many keyframes, against moving objects.",
)
@click.option(
    "--colmap",
    is_flag=True,
    help="Also write a COLMAP model with the keyframes, to refine in COLMAP or to train "
    "Gaussian splatting from.",
)
def reconstruct_cli(
    input_path: Path,
    output_path: Path,
    model: str,
    window: int,
    overlap: int,
    precision: str | None,
    stride: int | None,
    resolution: int,
    max_depth: float | None,
    voxel: float | None,
    neighbors: int,
    min_views: int,
    colmap: bool,
):
    """Reconstruct a static scene in 3D from a video: camera poses, a mesh and a point cloud.

    \b
    Writes into the output folder, for an input NAME:
      NAME-mesh.ply             colored triangle mesh, in meters
      NAME-cloud.ply            its vertices as a colored point cloud
      NAME-cameras.json         intrinsics, and each keyframe's camera-to-world pose
      NAME-trajectory.txt       the same poses in the TUM format
      NAME-colmap/              with --colmap: sparse/0/*.txt and the keyframes in images/
    The world frame is the first frame's camera: x right, y down and z forward.
    Moving objects, mirrors and long videos without revisits make the poses drift.
    """  # noqa: D301 - click keeps a paragraph after \b as written.
    from . import reconstruct as rec

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    logging.basicConfig(format="%(message)s")
    logging.getLogger("depth_pro_2").setLevel(logging.INFO)
    if overlap >= window:
        raise click.BadParameter(
            f"{overlap} must be less than --window ({window}).", param_hint="'--overlap'"
        )
    try:
        rec.import_open3d()
    except ImportError as error:
        raise click.ClickException(str(error)) from error

    device = default_device()
    net, transform = create_model_and_transforms(
        model, device, None if precision is None else PRECISIONS[precision]
    )
    click.echo(f"Loading {model} onto {device} in {str(net.precision).removeprefix('torch.')}.")
    output_path.mkdir(parents=True, exist_ok=True)
    name = input_path.stem if input_path.is_file() else input_path.resolve().name or "scene"
    colmap_images = output_path / f"{name}-colmap" / "images" if colmap else None
    if colmap_images is not None:
        colmap_images.mkdir(parents=True, exist_ok=True)

    def save_keyframe(index: int, image: np.ndarray, prediction: Mapping) -> None:
        if colmap_images is not None:
            Image.fromarray(image).save(colmap_images / rec.colmap_image_name(index), quality=95)

    options = rec.ReconstructionOptions(1, resolution, max_depth, voxel, neighbors, min_views)
    times: list[float] | None = None
    pending: deque[np.ndarray] = deque()
    fps = None
    with ExitStack() as stack:
        if input_path.is_dir():
            paths = sorted(
                (
                    p
                    for p in input_path.iterdir()
                    if p.is_file()
                    and p.suffix.lower() in get_supported_image_extensions()
                    and not p.name.startswith(".")
                ),
                key=natural_key,
            )
            if len(paths) < 2:
                raise click.UsageError(f"Fewer than two images in {input_path}.")
            first, f_px = load_rgb(paths[0])
            expected = len(paths)

            def frames():
                for path in paths:
                    image = first if path == paths[0] else load_rgb(path)[0]
                    if image.shape != first.shape:
                        raise click.UsageError(f"{path} differs in size from {paths[0]}.")
                    pending.append(image)
                    yield transform(image)

        else:
            container = stack.enter_context(av.open(str(input_path)))
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            fps = float(stream.average_rate or 30.0)
            expected = stream.frames or round((container.duration or 0) / av.time_base * fps)
            f_px = None
            times = []

            def frames():
                for frame in container.decode(stream):
                    image = upright(frame)
                    pending.append(image)
                    times.append(float(frame.time) if frame.time is not None else len(times) / fps)
                    yield transform(image)

        if stride is None:
            stride = max(1, math.ceil(expected / MAX_KEYFRAMES)) if expected else 1
            if stride > 1:
                click.echo(f"Keyframes: one every {stride} frames (--stride).")
        options.stride = stride
        predictions = net.infer_video(frames(), window, overlap, f_px, mode="buffered")
        bar = stack.enter_context(
            click.progressbar(predictions, expected or None, label=f"Depth of {input_path.name}")
        )
        images = (pending.popleft() for _ in iter(int, 1))
        try:
            result = rec.reconstruct(bar, images, options, save_keyframe)
        except torch.OutOfMemoryError as error:
            raise click.ClickException(
                f"Out of {device.type.upper()} memory. Lower --window and --overlap first; then "
                f"use a smaller model, such as -m depthpro2-s."
            ) from error

    if not len(result.mesh.vertices):
        raise click.ClickException(
            f"Fusion kept no surface: nothing within {result.max_depth:.1f} m was seen by "
            f"--min-views ({min_views}) keyframes. Lower --min-views or raise --max-depth."
        )
    written = rec.write_outputs(result, output_path, name, times, colmap_images)
    path_length = sum(
        float(np.linalg.norm(b[:3, 3] - a[:3, 3])) for a, b in zip(result.poses, result.poses[1:])
    )
    click.echo(f"Wrote {', '.join(str(p) for p in written)}.")
    click.echo(
        f"{len(result.keyframes)} keyframes, camera path {path_length:.2f} m, "
        f"{len(result.mesh.triangles)} triangles, {len(result.cloud.points)} points; "
        f"depth up to {result.max_depth:.1f} m, {100 * result.voxel:.1f} cm voxels."
    )
    if result.failed_edges:
        click.echo(
            f"Warning: odometry failed between {result.failed_edges} consecutive keyframes, "
            f"where the poses guess constant motion. Lower --stride, or film more slowly."
        )


#: The default --stride keeps at most this many keyframes, which bounds reconstruct's memory.
MAX_KEYFRAMES = 300


def natural_key(path: Path) -> list:
    """Sort key of a file name with its numbers compared as numbers: frame2 before frame10."""
    return [
        int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name)
    ]


def collect_inputs(paths: Sequence[Path], output_path: Path) -> list[Path]:
    """The images and videos given, folders expanded, checked to have distinct output names."""
    supported = VIDEO_EXTENSIONS + get_supported_image_extensions()
    inputs = []
    for path in paths:
        if path.is_dir():
            found = sorted(
                p
                for p in path.iterdir()
                if p.is_file() and p.suffix.lower() in supported and not p.name.startswith(".")
            )
            if path.resolve() == output_path.resolve():
                tags = ("depth", "depth16")
                written = {f"{p.stem}-{tag}" for p in found for tag in tags}
                found = [p for p in found if p.stem not in written]
            if not found:
                raise click.UsageError(f"No supported image or video in {path}.")
            inputs += found
        elif path.suffix.lower() in supported:
            inputs.append(path)
        else:
            raise click.UsageError(
                f"No supported image or video found at {path}. Images are any format Pillow "
                f"reads, such as .jpg, .png, .heic and .avif. Videos are "
                f"{', '.join(VIDEO_EXTENSIONS)}."
            )
    # A photo and a video may share a name (a Live Photo): one writes a .png, the other an .mp4.
    # Names compare without case, as on macOS's default file system.
    names = Counter((p.stem.casefold(), p.suffix.lower() in VIDEO_EXTENSIONS) for p in inputs)
    shared = sorted({stem for (stem, _), count in names.items() if count > 1})
    if shared:
        raise click.UsageError(
            f"Inputs share a name, so their outputs would overwrite each other: "
            f"{', '.join(shared)}."
        )
    return inputs


@dataclass(frozen=True)
class Outputs:
    """The outputs to write besides the color preview."""

    npz: bool = False
    png16: bool = False
    ply: bool = False
    max_depth: float | None = None


def predict_image(
    net: DepthPro2,
    transform: Callable[[np.ndarray], torch.Tensor],
    input_path: Path,
    output_path: Path,
    outputs: Outputs,
) -> None:
    """Predict one image, write its preview and arrays, and print a summary."""
    image, f_px = load_rgb(input_path)
    prediction = net.infer(transform(image), f_px=f_px)
    inverse_depth = inverse_metric_depth(prediction)

    png = output_path / f"{input_path.stem}-depth.png"
    scale = color_scale(inverse_depth)
    Image.fromarray(colorize(inverse_depth, scale)).save(png, compress_level=1)
    written = [str(png)]
    if outputs.npz:
        write_npz(png.with_suffix(".npz"), prediction)
        written.append(str(png.with_suffix(".npz")))
    if outputs.png16:
        png16 = output_path / f"{input_path.stem}-depth16.png"
        grayscale16(inverse_depth, scale).save(png16, compress_level=1)
        written.append(str(png16))
    if outputs.ply:
        ply = output_path / f"{input_path.stem}-points.ply"
        write_frame_ply(ply, prediction, image, outputs.max_depth)
        written.append(str(ply))

    counts = np.zeros(len(DEPTH_BINS), dtype=np.int64)
    count_depth(counts, inverse_depth)
    source = "from EXIF" if f_px is not None else "estimated"
    print_summary(written, prediction, source, counts, scale if outputs.png16 else None)


def predict_video(
    net: DepthPro2,
    transform: Callable[[np.ndarray], torch.Tensor],
    input_path: Path,
    output_path: Path,
    mode: str,
    window: int,
    overlap: int,
    outputs: Outputs,
) -> None:
    """Predict a video frame by frame, write its preview and arrays, and print a summary."""
    video_path = output_path / f"{input_path.stem}-depth.mp4"
    npz_dir = output_path / f"{input_path.stem}-depth"
    png16_dir = output_path / f"{input_path.stem}-depth16"
    ply_dir = output_path / f"{input_path.stem}-points"
    candidates = ((npz_dir, outputs.npz), (png16_dir, outputs.png16), (ply_dir, outputs.ply))
    folders = [folder for folder, wanted in candidates if wanted]
    for folder in folders:
        folder.mkdir(exist_ok=True)
    counts = np.zeros(len(DEPTH_BINS), dtype=np.int64)
    scale = prediction = None
    count = 0
    # The RGB frames waiting for their prediction, to color the point clouds.
    pending: deque[np.ndarray] = deque()

    with av.open(str(input_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate or 30.0)
        expected = stream.frames or round((container.duration or 0) / av.time_base * fps)

        def frames():
            for frame in container.decode(stream):
                image = upright(frame)
                if outputs.ply:
                    pending.append(image)
                yield transform(image)

        predictions = net.infer_video(frames(), window, overlap, mode=mode)
        with ExitStack() as stack:
            writer = stack.enter_context(VideoWriter(video_path, fps))
            bar = stack.enter_context(
                click.progressbar(predictions, expected or None, label=input_path.name)
            )
            for prediction in bar:
                inverse_depth = inverse_metric_depth(prediction)
                if scale is None:
                    scale = color_scale(inverse_depth)
                writer.write(colorize(inverse_depth, scale))
                if outputs.npz:
                    write_npz(npz_dir / f"{count:06d}.npz", prediction)
                if outputs.png16:
                    grayscale16(inverse_depth, scale).save(
                        png16_dir / f"{png16_dir.name}.{count:06d}.png", compress_level=1
                    )
                if outputs.ply:
                    write_frame_ply(
                        ply_dir / f"{count:06d}.ply",
                        prediction,
                        pending.popleft(),
                        outputs.max_depth,
                    )
                count_depth(counts, inverse_depth)
                count += 1

    if prediction is None:
        raise click.ClickException(f"No frames decoded from {input_path}.")
    written = [f"{count} frames at {fps:.5g} fps to {video_path}"]
    written += [f"{folder}/" for folder in folders]
    source = f"estimated from the first {min(window, count)} frames"
    print_summary(written, prediction, source, counts, scale if outputs.png16 else None)


def write_frame_ply(
    path: Path,
    prediction: Mapping[str, torch.Tensor],
    image: np.ndarray,
    max_depth: float | None,
) -> None:
    """Write a frame's colored point cloud, without sky and depth edges."""
    points, colors = point_cloud(
        prediction["depth"], prediction["focallength_px"], image, max_depth
    )
    write_ply(path, points, colors)


def inverse_metric_depth(prediction: Mapping[str, torch.Tensor]) -> np.ndarray:
    """``1 / depth`` in 1/m as a float32 array on the host, zero where infinitely far."""
    inverse_depth = prediction["inverse_depth"].float().cpu().numpy()
    return inverse_depth * (inverse_depth.shape[-1] / float(prediction["focallength_px"]))


def write_npz(path: Path, prediction: Mapping[str, torch.Tensor]) -> None:
    """Write depth in meters, the focal length in pixels and the intrinsics.

    The npz is uncompressed. ``intrinsics`` is the 3x3 camera matrix, with the principal point at
    the image center.
    """
    depth = prediction["depth"].float().cpu().numpy()
    f_px = float(prediction["focallength_px"])
    np.savez(
        path,
        depth=depth,
        focallength_px=np.float32(f_px),
        intrinsics=intrinsics(f_px, depth.shape[1], depth.shape[0]),
    )


def color_scale(inverse_depth: np.ndarray) -> float:
    """Inverse depth mapped to the top of the color map, anchored on one frame of a clip."""
    high = float(inverse_depth.max())
    return high * (1.0 + COLOR_HEADROOM) if high > 0 else 1.0


def colorize(inverse_depth: np.ndarray, scale: float) -> np.ndarray:
    """Color-map inverse depth from 0, at infinite depth, to ``scale`` as RGB uint8."""
    return COLOR_LUT[np.clip(inverse_depth * (256 / scale), 0, 255).astype(np.int32)]


def grayscale16(inverse_depth: np.ndarray, scale: float) -> Image.Image:
    """Inverse depth from 0, at infinite depth, to ``scale`` as a 16-bit grayscale image."""
    return Image.fromarray((np.clip(inverse_depth / scale, 0, 1) * 65535).round().astype(np.uint16))


class VideoWriter:
    """Encode frames into an H.264 MP4 that stays playable while it is being written."""

    def __init__(self, path: Path, fps: float):
        """Defer opening until the first frame fixes the resolution."""
        self.path = path
        self.fps = fps
        self.container: av.container.OutputContainer | None = None

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *_) -> None:
        """Flush the encoder and finish the file."""
        if self.container is not None:
            self.container.mux(self.stream.encode())
            self.container.close()

    def write(self, image: np.ndarray) -> None:
        """Append one ``(H, W, 3)`` uint8 frame, edge-padded to even sides if it has an odd one."""
        height, width = image.shape[:2]
        if height % 2 or width % 2:
            image = np.pad(image, ((0, height % 2), (0, width % 2), (0, 0)), mode="edge")
        if self.container is None:
            self.container = av.open(
                str(self.path), "w", options={"movflags": FRAGMENTED_MP4, "flush_packets": "1"}
            )
            self.stream = self.container.add_stream(
                "libx264", rate=Fraction(self.fps).limit_denominator(1001)
            )
            self.stream.height, self.stream.width = image.shape[:2]
            self.stream.pix_fmt = "yuv420p"
            # Tag what the RGB to YUV conversion produces, or QuickTime, Finder and Photos assume
            # BT.709 and shift the colors: BT.601 limited range, from sRGB.
            self.stream.colorspace = 6  # BT.601 (SMPTE 170M)
            self.stream.color_range = 1  # limited (MPEG)
            self.stream.color_primaries = 1  # BT.709, the primaries of sRGB
            self.stream.color_trc = 13  # sRGB
            self.stream.gop_size = max(1, round(self.fps))
            # B-frames would delay the first frame: a fragmented MP4 has no edit list to undo it.
            self.stream.max_b_frames = 0
        self.container.mux(self.stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")))


def count_depth(counts: np.ndarray, inverse_depth: np.ndarray) -> None:
    """Add a strided sample of one frame's depth to ``counts``, the last bin infinite depth."""
    sample = inverse_depth[::DEPTH_STRIDE, ::DEPTH_STRIDE]
    near = sample[sample > 0]
    counts[:-1] += np.histogram(1 / near, DEPTH_BINS)[0]
    counts[-1] += sample.size - near.size


def print_summary(
    written: list[str],
    prediction: Mapping[str, torch.Tensor],
    source: str,
    counts: np.ndarray,
    png16_scale: float | None,
) -> None:
    """Print what was written, the focal length of the last frame, and the depth range.

    Given ``png16_scale``, also how to turn the 16-bit PNG's values into meters.
    """
    click.echo(f"Wrote {', '.join(written)}.")
    click.echo(f"Focal length: {float(prediction['focallength_px']):.1f} px, {source}.")
    finite = counts[:-1]
    depth = ""
    if finite.sum():
        centers = np.sqrt(DEPTH_BINS[:-1] * DEPTH_BINS[1:])
        low, high = centers[np.searchsorted(np.cumsum(finite) / finite.sum(), (0.05, 0.95))]
        depth = f"{low:.2f} m to {high:.2f} m (5th to 95th percentile), "
    click.echo(f"Depth: {depth}{100 * counts[-1] / max(1, counts.sum()):.1f}% infinitely far.")
    if png16_scale is not None:
        k = 65535 / png16_scale
        click.echo(f"16-bit PNG: depth in meters = {k:.2f} / value, 0 is infinitely far.")
