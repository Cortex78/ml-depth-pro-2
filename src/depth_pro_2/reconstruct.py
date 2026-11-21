#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""3D reconstruction of a static scene from a video, with Depth Pro 2's metric depth.

Depth Pro 2 gives each frame metric depth that is consistent from frame to frame, and one focal
length for the video. That makes each frame an RGB-D image, and the video an RGB-D sequence:

1. Camera poses: SIFT matches between each keyframe and the few before it, lifted to 3D by
   the metric depth, give each pair's motion by PnP with RANSAC; dense RGB-D odometry stands in
   where features are too few. A pose graph reconciles the pairs. The depth is metric, so the
   trajectory is in meters.
2. Fusion: each keyframe's depth goes into a truncated signed distance volume (TSDF) at its
   pose, from which come a colored mesh and its vertices as a point cloud.

It needs Open3D and OpenCV: ``uv sync --extra reconstruct``. On Linux without a display, Open3D
also needs the EGL and OpenGL libraries (``apt-get install libegl1 libgl1 libgomp1`` on Debian or
Ubuntu).

The camera frame is OpenCV's (x right, y down, z forward), as in :mod:`depth_pro_2.geometry`,
and the world frame is the first keyframe's camera frame. Poses are camera-to-world.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from PIL import Image

LOGGER = logging.getLogger(__name__)


@dataclass
class ReconstructionOptions:
    """How to reconstruct; ``None`` picks a value from the scene."""

    #: Use every ``stride``-th frame as a keyframe. Depth is predicted for every frame.
    stride: int = 1
    #: Longer side, in pixels, of the frames used for odometry and fusion.
    resolution: int = 960
    #: Depth beyond this many meters is left out. Default: 3 times the median depth.
    max_depth: float | None = None
    #: TSDF voxel size in meters. Default: the median depth divided by 300.
    voxel: float | None = None
    #: Odometry edges from each keyframe to the ones before it.
    neighbors: int = 3
    #: Keep only surfaces seen by at least this many keyframes.
    min_views: int = 2


@dataclass
class Keyframe:
    """A keyframe at the reconstruction's resolution."""

    index: int
    rgb: np.ndarray
    depth: np.ndarray  # float32 meters, 0 where infinite


@dataclass
class Reconstruction:
    """Poses and geometry of a video."""

    keyframes: list[Keyframe]
    f_px: float  # At the reconstruction's resolution.
    full_size: tuple[int, int]  # (width, height) of the video.
    poses: list[np.ndarray]  # 4x4 camera-to-world per keyframe.
    max_depth: float
    voxel: float
    failed_edges: int = 0
    mesh: object = None
    cloud: object = None
    info: dict = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of the keyframes."""
        return self.keyframes[0].depth.shape[1], self.keyframes[0].depth.shape[0]


def import_cv2():
    """Import OpenCV, or say how to install it."""
    try:
        import cv2
    except ImportError as error:
        raise ImportError(
            "3D reconstruction needs OpenCV: run `uv sync --extra reconstruct`."
        ) from error
    return cv2


def import_open3d():
    """Import Open3D, or say how to install it."""
    try:
        import open3d
    except ImportError as error:
        raise ImportError(
            "3D reconstruction needs Open3D: run `uv sync --extra reconstruct`. On Linux without "
            "a display, also install the EGL and OpenGL libraries, e.g. "
            "`apt-get install libegl1 libgl1 libgomp1`."
        ) from error
    return open3d


def collect_keyframes(
    predictions: Iterable,
    images: Iterable[np.ndarray],
    options: ReconstructionOptions,
    on_frame: Callable[[int, np.ndarray, dict], None] | None = None,
) -> tuple[list[Keyframe], float, tuple[int, int]]:
    """Keep every ``stride``-th prediction, downsized, with its RGB frame.

    ``predictions`` and ``images`` run in step, one per frame. ``on_frame(index, image,
    prediction)`` is called for each keyframe at full resolution, before it is downsized.

    Returns:
        The keyframes, the focal length at their size, and the video's (width, height).
    """
    keyframes = []
    f_px = full_size = None
    for index, (prediction, image) in enumerate(zip(predictions, images)):
        if index % options.stride:
            continue
        depth = prediction["depth"]
        height, width = depth.shape
        scale = min(1.0, options.resolution / max(height, width))
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        if f_px is None:
            f_px = float(prediction["focallength_px"]) * size[0] / width
            full_size = (width, height)
        if on_frame is not None:
            on_frame(index, image, prediction)
        # Nearest neighbor: averaging depth across an occlusion boundary would create points
        # between the two surfaces.
        small = torch.nn.functional.interpolate(
            depth[None, None].float(), size=(size[1], size[0]), mode="nearest-exact"
        )[0, 0]
        small = torch.where(torch.isfinite(small), small, torch.zeros_like(small))
        rgb = np.asarray(Image.fromarray(image).resize(size, Image.Resampling.BILINEAR))
        keyframes.append(Keyframe(index, np.ascontiguousarray(rgb), small.cpu().numpy()))
    if not keyframes:
        raise ValueError("No frames to reconstruct.")
    return keyframes, f_px, full_size


def scene_scale(keyframes: list[Keyframe], options: ReconstructionOptions) -> tuple[float, float]:
    """The depth cutoff and voxel size in meters, from the options or the median depth."""
    sample = np.concatenate([k.depth[::4, ::4].ravel() for k in keyframes])
    median = float(np.median(sample[sample > 0])) if (sample > 0).any() else 1.0
    max_depth = options.max_depth or 3.0 * median
    voxel = options.voxel or median / 300.0
    return max_depth, voxel


def estimate_poses(
    keyframes: list[Keyframe], f_px: float, max_depth: float, neighbors: int
) -> tuple[list[np.ndarray], int]:
    """Camera-to-world poses of the keyframes, from feature matches lifted by depth.

    For each keyframe and the ``neighbors`` before it, SIFT matches are lifted to 3D in the
    earlier frame with its metric depth, and PnP with RANSAC finds the motion that projects them
    onto the later frame. Where too few matches agree, as on bare walls, dense RGB-D odometry
    takes over. A pose graph then reconciles all the pairs.

    Returns:
        The poses, and the number of consecutive pairs for which both methods failed.
    """
    o3d = import_open3d()
    reg = o3d.pipelines.registration
    match = FeatureMatcher(keyframes, f_px, max_depth)
    dense = DenseOdometry(keyframes, f_px, max_depth)

    graph = reg.PoseGraph()
    graph.nodes.append(reg.PoseGraphNode(np.eye(4)))
    poses = [np.eye(4)]
    motion = np.eye(4)  # The last relative motion, for a constant velocity guess.
    failed = 0
    for t in range(1, len(keyframes)):
        # Only the last few keyframes are compared: free what is older.
        match.forget_before(t - neighbors)
        dense.forget_before(t - neighbors)
        # The consecutive pair first: it sets frame t's pose, the guess for the other pairs.
        for s in [t - 1, *range(max(0, t - neighbors), t - 1)]:
            consecutive = s == t - 1
            # transform maps frame s's camera coordinates to frame t's.
            transform, weight = match(s, t)
            if transform is None:
                guess = motion if consecutive else np.linalg.inv(poses[t]) @ poses[s]
                transform, weight = dense(s, t, guess)
            if consecutive:
                if transform is None:
                    failed += 1
                    transform, weight = motion, 1.0
                motion = transform
                poses.append(poses[s] @ np.linalg.inv(transform))
                graph.nodes.append(reg.PoseGraphNode(poses[t]))
            if transform is not None:
                information = np.eye(6) * weight
                graph.edges.append(
                    reg.PoseGraphEdge(s, t, transform, information, uncertain=not consecutive)
                )

    if len(keyframes) > 2:
        reg.global_optimization(
            graph,
            reg.GlobalOptimizationLevenbergMarquardt(),
            reg.GlobalOptimizationConvergenceCriteria(),
            reg.GlobalOptimizationOption(
                max_correspondence_distance=dense.depth_diff_max,
                edge_prune_threshold=0.25,
                preference_loop_closure=0.1,
                reference_node=0,
            ),
        )
        poses = [np.asarray(node.pose) for node in graph.nodes]
    return poses, failed


class FeatureMatcher:
    """Relative poses of keyframe pairs from SIFT matches, metric depth and PnP with RANSAC."""

    #: Fewer RANSAC inliers than this and a pair is left to dense odometry.
    MIN_INLIERS = 30

    def __init__(self, keyframes: list[Keyframe], f_px: float, max_depth: float):
        """Detect the keyframes' features."""
        cv2 = import_cv2()
        self.cv2, self.keyframes, self.f_px, self.max_depth = cv2, keyframes, f_px, max_depth
        height, width = keyframes[0].depth.shape
        self.camera = np.array(
            [[f_px, 0.0, (width - 1) / 2.0], [0.0, f_px, (height - 1) / 2.0], [0.0, 0.0, 1.0]]
        )
        # 2 pixels at 640 wide, in proportion above.
        self.max_error = 2.0 * max(1.0, width / 640)
        self.sift = cv2.SIFT_create(nfeatures=4000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.features: dict[int, tuple] = {}

    def feature(self, i: int) -> tuple:
        """Keyframe i's SIFT keypoints and descriptors, detected on first use."""
        if i not in self.features:
            gray = self.cv2.cvtColor(self.keyframes[i].rgb, self.cv2.COLOR_RGB2GRAY)
            self.features[i] = self.sift.detectAndCompute(gray, None)
        return self.features[i]

    def forget_before(self, i: int) -> None:
        """Free the features of keyframes before i."""
        for j in [j for j in self.features if j < i]:
            del self.features[j]

    def __call__(self, s: int, t: int) -> tuple[np.ndarray | None, float]:
        """The motion from frame s to frame t and its number of inliers, or ``(None, 0)``."""
        cv2 = self.cv2
        (points_s, desc_s), (points_t, desc_t) = self.feature(s), self.feature(t)
        if desc_s is None or desc_t is None or len(desc_s) < 2 or len(desc_t) < 2:
            return None, 0.0
        matches = [
            pair[0]
            for pair in self.matcher.knnMatch(desc_s, desc_t, k=2)
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
        ]
        if len(matches) < self.MIN_INLIERS:
            return None, 0.0
        uv_s = np.array([points_s[m.queryIdx].pt for m in matches])
        uv_t = np.array([points_t[m.trainIdx].pt for m in matches])
        depth = self.keyframes[s].depth
        height, width = depth.shape
        z = depth[
            uv_s[:, 1].round().astype(int).clip(0, height - 1),
            uv_s[:, 0].round().astype(int).clip(0, width - 1),
        ]
        keep = (z > 0) & (z < self.max_depth)
        uv_s, uv_t, z = uv_s[keep], uv_t[keep], z[keep]
        if len(z) < self.MIN_INLIERS:
            return None, 0.0
        k = self.camera
        points = np.stack(
            [(uv_s[:, 0] - k[0, 2]) * z / self.f_px, (uv_s[:, 1] - k[1, 2]) * z / self.f_px, z], 1
        )
        found, rvec, tvec, inliers = cv2.solvePnPRansac(
            points,
            uv_t,
            k,
            None,
            reprojectionError=self.max_error,
            iterationsCount=2000,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not found or inliers is None or len(inliers) < self.MIN_INLIERS:
            return None, 0.0
        inliers = inliers[:, 0]
        rvec, tvec = cv2.solvePnPRefineLM(points[inliers], uv_t[inliers], k, None, rvec, tvec)
        transform = np.eye(4)
        transform[:3, :3] = cv2.Rodrigues(rvec)[0]
        transform[:3, 3] = tvec[:, 0]
        return transform, float(len(inliers))


class DenseOdometry:
    """Relative poses of keyframe pairs from dense RGB-D odometry, for pairs without features."""

    def __init__(self, keyframes: list[Keyframe], f_px: float, max_depth: float):
        """Prepare the grayscale RGB-D images."""
        o3d = import_open3d()
        self.o3d, self.keyframes, self.max_depth = o3d, keyframes, max_depth
        height, width = keyframes[0].depth.shape
        self.pixels = height * width
        self.intrinsic = pinhole(o3d, f_px, width, height)
        sample = np.concatenate([k.depth[::8, ::8].ravel() for k in keyframes[:8]])
        median = float(np.median(sample[sample > 0])) if (sample > 0).any() else 1.0
        self.option = o3d.pipelines.odometry.OdometryOption()
        # Correspondences may differ in depth by about 1% of the scene's depth.
        self.depth_diff_max = 0.01 * median
        self.option.depth_diff_max = self.depth_diff_max
        self.option.depth_min = 0.0
        self.option.depth_max = max_depth
        self.images: dict[int, object] = {}

    def image(self, i: int):
        """Keyframe i as a grayscale Open3D RGB-D image, made on first use."""
        if i not in self.images:
            self.images[i] = rgbd_image(self.o3d, self.keyframes[i], self.max_depth, True)
        return self.images[i]

    def forget_before(self, i: int) -> None:
        """Free the images of keyframes before i."""
        for j in [j for j in self.images if j < i]:
            del self.images[j]

    def __call__(self, s: int, t: int, guess: np.ndarray) -> tuple[np.ndarray | None, float]:
        """The motion from frame s to frame t, from ``guess``, and a weight, or ``(None, 0)``."""
        odometry = self.o3d.pipelines.odometry
        success, transform, information = odometry.compute_rgbd_odometry(
            self.image(s),
            self.image(t),
            self.intrinsic,
            guess,
            odometry.RGBDOdometryJacobianFromHybridTerm(),
            self.option,
        )
        # Open3D reports success even without correspondences; information[5, 5] counts them.
        if not success or information[5, 5] < 0.05 * self.pixels:
            return None, 0.0
        # A weight below a feature match's: dense odometry is the fallback.
        return transform, float(FeatureMatcher.MIN_INLIERS)


def fuse(
    keyframes: list[Keyframe],
    poses: list[np.ndarray],
    f_px: float,
    max_depth: float,
    voxel: float,
    min_views: int = 2,
):
    """A colored mesh, and its vertices as a point cloud, fused in a TSDF.

    The volume is a sparse grid of voxel blocks, on the GPU if Open3D was built with CUDA. Only
    surfaces seen by at least ``min_views`` keyframes are kept (or by all, if fewer), which
    leaves out most of what moves and most one-off errors.
    """
    o3d = import_open3d()
    o3c = o3d.core
    device = o3c.Device("CUDA:0" if o3c.cuda.is_available() else "CPU:0")
    width, height = keyframes[0].depth.shape[1], keyframes[0].depth.shape[0]
    intrinsic = o3c.Tensor(
        [[f_px, 0.0, (width - 1) / 2.0], [0.0, f_px, (height - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=o3c.float64,
    )
    grid = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=voxel,
        block_resolution=8,
        block_count=10_000,  # Grows as needed.
        device=device,
    )
    integrated = 0
    for keyframe, pose in zip(keyframes, poses):
        # Open3D aborts on a frame that touches no voxel block: all sky, or all beyond max_depth.
        if not ((keyframe.depth > 0) & (keyframe.depth < max_depth)).any():
            LOGGER.warning(
                "Frame %d has no depth within %.1f m: not fused.", keyframe.index, max_depth
            )
            continue
        integrated += 1
        depth = o3d.t.geometry.Image(keyframe.depth).to(device)
        color = o3d.t.geometry.Image((keyframe.rgb / np.float32(255)).astype(np.float32))
        color = color.to(device)
        extrinsic = o3c.Tensor(np.linalg.inv(pose), dtype=o3c.float64)
        blocks = grid.compute_unique_block_coordinates(depth, intrinsic, extrinsic, 1.0, max_depth)
        grid.integrate(blocks, depth, color, intrinsic, extrinsic, 1.0, max_depth)
    if not integrated:
        return o3d.geometry.TriangleMesh(), o3d.geometry.PointCloud()
    # Each keyframe adds a weight of 1 to the voxels it sees; the threshold is exclusive.
    views = max(1, min(min_views, integrated))
    mesh = grid.extract_triangle_mesh(weight_threshold=views - 0.5).to_legacy()
    # Averaged colors can overshoot 1 by a rounding error.
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(np.asarray(mesh.vertex_colors), 0, 1))
    cloud = o3d.geometry.PointCloud(mesh.vertices)
    cloud.colors = mesh.vertex_colors
    return mesh, cloud


def reconstruct(
    predictions: Iterable,
    images: Iterable[np.ndarray],
    options: ReconstructionOptions,
    on_frame: Callable[[int, np.ndarray, dict], None] | None = None,
) -> Reconstruction:
    """Reconstruct a static scene from a video's predictions and RGB frames, in step."""
    keyframes, f_px, full_size = collect_keyframes(predictions, images, options, on_frame)
    max_depth, voxel = scene_scale(keyframes, options)
    LOGGER.info(
        "Reconstructing %d keyframes at %dx%d: depth up to %.1f m, %.1f cm voxels.",
        len(keyframes),
        keyframes[0].depth.shape[1],
        keyframes[0].depth.shape[0],
        max_depth,
        100 * voxel,
    )
    poses, failed = estimate_poses(keyframes, f_px, max_depth, options.neighbors)
    mesh, cloud = fuse(keyframes, poses, f_px, max_depth, voxel, options.min_views)
    return Reconstruction(keyframes, f_px, full_size, poses, max_depth, voxel, failed, mesh, cloud)


def pinhole(o3d, f_px: float, width: int, height: int):
    """Open3D's intrinsics, with pixel centers at whole coordinates, as in OpenCV."""
    return o3d.camera.PinholeCameraIntrinsic(
        width, height, f_px, f_px, (width - 1) / 2.0, (height - 1) / 2.0
    )


def rgbd_image(o3d, keyframe: Keyframe, max_depth: float, intensity: bool):
    """An Open3D RGB-D image of a keyframe, in grayscale for odometry or in color for fusion."""
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(keyframe.rgb),
        o3d.geometry.Image(keyframe.depth),
        depth_scale=1.0,
        depth_trunc=max_depth,
        convert_rgb_to_intensity=intensity,
    )


def rotation_to_quaternion(r: np.ndarray) -> np.ndarray:
    """The unit quaternion ``(w, x, y, z)``, with ``w >= 0``, of a 3x3 rotation matrix."""
    trace = np.trace(r)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    q = np.array(q, dtype=np.float64)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def write_outputs(
    result: Reconstruction,
    output_path: Path,
    name: str,
    times: Sequence[float] | None,
    colmap_images: Path | None,
) -> list[Path]:
    """Write the mesh, point cloud, cameras, trajectory and, if images are given, a COLMAP model.

    ``times`` are the frames' timestamps in seconds, indexed by frame; without them, the
    trajectory's times are the frame numbers.

    Returns:
        The paths written.
    """
    o3d = import_open3d()
    written = []
    mesh_path = output_path / f"{name}-mesh.ply"
    if o3d.io.write_triangle_mesh(str(mesh_path), result.mesh):
        written.append(mesh_path)
    cloud_path = output_path / f"{name}-cloud.ply"
    if o3d.io.write_point_cloud(str(cloud_path), result.cloud):
        written.append(cloud_path)

    # Cameras at the video's full resolution.
    full_w, full_h = result.full_size
    f_full = result.f_px * full_w / result.size[0]
    cameras = {
        "convention": "OpenCV camera (x right, y down, z forward); camera_to_world 4x4; meters; "
        "pixel centers at whole coordinates",
        "width": full_w,
        "height": full_h,
        "fx": f_full,
        "fy": f_full,
        "cx": (full_w - 1) / 2.0,
        "cy": (full_h - 1) / 2.0,
        "max_depth": result.max_depth,
        "voxel": result.voxel,
        "frames": [
            {
                "frame": k.index,
                "time": None if times is None else times[k.index],
                "camera_to_world": pose.tolist(),
            }
            for k, pose in zip(result.keyframes, result.poses)
        ],
    }
    cameras_path = output_path / f"{name}-cameras.json"
    cameras_path.write_text(json.dumps(cameras, indent=1))
    written.append(cameras_path)

    # TUM format: time tx ty tz qx qy qz qw, camera-to-world.
    trajectory_path = output_path / f"{name}-trajectory.txt"
    with open(trajectory_path, "w") as f:
        f.write("# time tx ty tz qx qy qz qw (camera-to-world, meters)\n")
        for k, pose in zip(result.keyframes, result.poses):
            w, x, y, z = rotation_to_quaternion(pose[:3, :3])
            time = float(k.index) if times is None else times[k.index]
            tx, ty, tz = pose[:3, 3]
            f.write(f"{time:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {x:.8f} {y:.8f} {z:.8f} {w:.8f}\n")
    written.append(trajectory_path)

    if colmap_images is not None:
        written.append(write_colmap(result, colmap_images.parent, f_full))
    return written


def write_colmap(
    result: Reconstruction, root: Path, f_full: float, max_points: int = 200_000
) -> Path:
    """Write a COLMAP text model to ``root/sparse/0``, for keyframe images in ``root/images``.

    Camera 1 is a PINHOLE camera at the video's full resolution. The 3D points come from the
    fused point cloud, without 2D observations: enough to start Gaussian splatting or to
    triangulate with COLMAP from known poses.
    """
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    full_w, full_h = result.full_size
    # COLMAP puts pixel centers at half coordinates: the image center is (w / 2, h / 2).
    (sparse / "cameras.txt").write_text(
        "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n"
        f"1 PINHOLE {full_w} {full_h} {f_full:.6f} {f_full:.6f} {full_w / 2:.6f} {full_h / 2:.6f}\n"
    )
    with open(sparse / "images.txt", "w") as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME, then a line of 2D points\n")
        for image_id, (k, pose) in enumerate(zip(result.keyframes, result.poses), start=1):
            world_to_camera = np.linalg.inv(pose)
            qw, qx, qy, qz = rotation_to_quaternion(world_to_camera[:3, :3])
            tx, ty, tz = world_to_camera[:3, 3]
            f.write(
                f"{image_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {tx:.9f} {ty:.9f} {tz:.9f} 1 "
                f"{colmap_image_name(k.index)}\n\n"
            )
    cloud = result.cloud
    points = np.asarray(cloud.points)
    colors = (np.asarray(cloud.colors) * 255).round().astype(np.uint8)
    if len(points) > max_points:
        keep = np.random.default_rng(0).choice(len(points), max_points, replace=False)
        points, colors = points[keep], colors[keep]
    with open(sparse / "points3D.txt", "w") as f:
        f.write("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
        for i, ((x, y, z), (r, g, b)) in enumerate(zip(points, colors), start=1):
            f.write(f"{i} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0\n")
    return root


def colmap_image_name(index: int) -> str:
    """The file name of frame ``index`` in the COLMAP model's images folder."""
    return f"{index:06d}.jpg"
