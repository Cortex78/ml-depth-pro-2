#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""What follows from metric depth and the focal length: intrinsics, 3D points and normals.

All of it is in the camera frame of OpenCV and COLMAP: +x right, +y down and +z forward, along
the viewing direction, with the principal point at the image center.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

#: A neighbor farther than this, relative to a pixel's depth, is across a depth edge.
EDGE_THRESHOLD = 0.05


def intrinsics(f_px: float, width: int, height: int) -> np.ndarray:
    """The 3x3 camera matrix of a frame, with the principal point at its center.

    As in OpenCV, pixel ``(u, v)`` has its center at ``(u, v)``, so the center of the image is
    ``((width - 1) / 2, (height - 1) / 2)``. COLMAP puts pixel centers at ``(u + 0.5, v + 0.5)``.
    """
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    return np.array([[f_px, 0.0, cx], [0.0, f_px, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def unproject(depth: torch.Tensor, f_px: float | torch.Tensor) -> torch.Tensor:
    """3D points ``(H, W, 3)`` in meters of depth ``(H, W)``, at the pixel centers.

    Points where the depth is infinite are infinite or NaN; mask them with ``torch.isfinite``.
    """
    height, width = depth.shape
    f_px = torch.as_tensor(f_px, dtype=depth.dtype, device=depth.device)
    u = torch.arange(width, dtype=depth.dtype, device=depth.device) + 0.5 - width / 2.0
    v = torch.arange(height, dtype=depth.dtype, device=depth.device) + 0.5 - height / 2.0
    x = depth * (u / f_px)[None, :]
    y = depth * (v / f_px)[:, None]
    return torch.stack([x, y, depth], dim=-1)


def depth_edges(depth: torch.Tensor, threshold: float = EDGE_THRESHOLD) -> torch.Tensor:
    """``(H, W)`` mask of pixels whose 4-neighbors are all across a depth edge on one axis.

    Such pixels lie on an occlusion boundary, where depth jumps between two surfaces and a 3D
    point between them belongs to neither ("flying pixels").
    """
    jump = _relative_jumps(depth, threshold)
    left, right, up, down = jump
    return (left & right) | (up & down)


def normals(
    depth: torch.Tensor, f_px: float | torch.Tensor, step: int | None = None
) -> torch.Tensor:
    """Unit surface normals ``(H, W, 3)`` of depth ``(H, W)``, facing the camera.

    The tangents span ``step`` pixels on each side, by default 1 per 320 pixels of the longer
    side (6 at 1920): neighboring pixels' depth carries fine noise that would dominate the
    normals of distant surfaces. On each axis, the side whose depth is closest to the pixel's
    own is used, so a normal stays on its own surface at an occlusion boundary. Where both sides
    of an axis are across a depth edge, as on a pole thinner than the step, shorter steps take
    over, down to one pixel. The normal is zero where it is undefined: infinite depth, or a depth
    edge on both sides of an axis even one pixel away.
    """
    height, width = depth.shape
    k = max(1, round(max(height, width) / 320)) if step is None else step
    points = unproject(depth.float(), f_px)
    finite = torch.isfinite(depth)
    points = torch.where(finite[..., None], points, torch.zeros_like(points))

    dx, dy, valid_x, valid_y = _tangents(points, depth, k)
    while k > 1:
        k = max(1, k // 2)
        dx_k, dy_k, valid_x_k, valid_y_k = _tangents(points, depth, k)
        dx = torch.where((~valid_x & valid_x_k)[..., None], dx_k, dx)
        dy = torch.where((~valid_y & valid_y_k)[..., None], dy_k, dy)
        valid_x, valid_y = valid_x | valid_x_k, valid_y | valid_y_k

    # x right and y down make dx x dy point away from the camera: flip it to face the camera.
    n = torch.linalg.cross(dy, dx, dim=-1)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    valid = finite & valid_x & valid_y
    return torch.where(valid[..., None] & torch.isfinite(n), n, torch.zeros_like(n))


def _tangents(
    points: torch.Tensor, depth: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-sided tangents ``k`` pixels long along x and y, and where each is usable.

    On each axis, the tangent goes to the neighbor whose depth is closest to the pixel's own, as
    a step right or down. It is unusable where both neighbors are across a depth edge.
    """
    neighbors = _neighbors(points.permute(2, 0, 1), k)
    left, right, up, down = (n.permute(1, 2, 0) - points for n in neighbors)
    # Over k pixels, a smooth surface changes depth by up to about k times as much as over one.
    jump_left, jump_right, jump_up, jump_down = _relative_jumps(
        depth, min(0.5, EDGE_THRESHOLD * k), k, borders=True
    )
    use_right = (right[..., 2].abs() <= left[..., 2].abs()) & ~jump_right | jump_left
    dx = torch.where(use_right[..., None], right, -left)
    use_down = (down[..., 2].abs() <= up[..., 2].abs()) & ~jump_down | jump_up
    dy = torch.where(use_down[..., None], down, -up)
    return dx, dy, ~(jump_left & jump_right), ~(jump_up & jump_down)


def _neighbors(x: torch.Tensor, k: int) -> tuple[torch.Tensor, ...]:
    """The left, right, up and down neighbors ``k`` pixels away of ``(C, H, W)``, edge-padded."""
    height, width = x.shape[-2:]
    padded = torch.nn.functional.pad(x[None], (k, k, k, k), mode="replicate")[0]
    return (
        padded[:, k : k + height, :width],
        padded[:, k : k + height, 2 * k :],
        padded[:, :height, k : k + width],
        padded[:, 2 * k :, k : k + width],
    )


def _relative_jumps(
    depth: torch.Tensor, threshold: float, k: int = 1, borders: bool = False
) -> tuple[torch.Tensor, ...]:
    """Masks of a jump to the left, right, up and down neighbor ``k`` pixels away.

    A jump is a depth difference above ``threshold`` times the nearer depth. With ``borders``,
    a neighbor outside the image counts as a jump, as it cannot be used.
    """
    d = torch.where(torch.isfinite(depth), depth.float(), torch.full_like(depth, float("inf")))
    jumps = []
    for neighbor in _neighbors(d[None], k):
        neighbor = neighbor[0]
        # Infinite against finite is a jump; infinite against infinite is not.
        both_inf = torch.isinf(d) & torch.isinf(neighbor)
        diff = (neighbor - d).abs()
        jumps.append(~both_inf & ~(diff <= threshold * torch.minimum(d, neighbor)))
    if borders:
        jumps[0][:, :k] = True
        jumps[1][:, -k:] = True
        jumps[2][:k, :] = True
        jumps[3][-k:, :] = True
    return tuple(jumps)


def normals_to_rgb(n: np.ndarray) -> np.ndarray:
    """Normals ``(H, W, 3)`` in the camera frame as a normal map, RGB uint8.

    The colors follow the common normal map convention (OpenGL, Blender): +x right, +y up and +z
    toward the camera map to red, green and blue, so a surface facing the camera is light blue.
    Pixels without a normal are black.
    """
    gl = n * np.array([1.0, -1.0, -1.0], dtype=n.dtype)
    rgb = np.clip((gl * 0.5 + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rgb[~n.any(axis=-1)] = 0
    return rgb


def write_ply(
    path: Path | str,
    points: np.ndarray,
    colors: np.ndarray | None = None,
    point_normals: np.ndarray | None = None,
) -> None:
    """Write a binary point cloud: ``(N, 3)`` points, and uint8 colors and normals if given."""
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if point_normals is not None:
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    if colors is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    data = np.empty(len(points), dtype=fields)
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    if point_normals is not None:
        data["nx"], data["ny"], data["nz"] = point_normals.T
    if colors is not None:
        data["red"], data["green"], data["blue"] = colors.T
    types = {"<f4": "float", "u1": "uchar"}
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(points)}"]
    header += [f"property {types[t]} {name}" for name, t in fields]
    header.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(data.tobytes())


def point_cloud(
    depth: torch.Tensor,
    f_px: float | torch.Tensor,
    image: np.ndarray | None = None,
    max_depth: float | None = None,
    stride: int = 1,
    frame_normals: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """The points, colors and normals of a frame, as ``(N, 3)`` arrays on the host.

    Leaves out infinite depth, depth beyond ``max_depth`` meters, points on depth edges, which
    float between two surfaces, and points without a normal. ``stride`` keeps every
    ``stride``-th pixel on each axis. ``frame_normals`` are the frame's :func:`normals`, if
    already computed.
    """
    points = unproject(depth.float(), f_px)
    n = normals(depth, f_px) if frame_normals is None else frame_normals
    keep = torch.isfinite(depth) & n.any(dim=-1) & ~depth_edges(depth)
    if max_depth is not None:
        keep &= depth <= max_depth
    keep = keep[::stride, ::stride]
    points, n = points[::stride, ::stride][keep], n[::stride, ::stride][keep]
    colors = None
    if image is not None:
        mask = keep.cpu().numpy()
        colors = image[::stride, ::stride][mask]
    return points.cpu().numpy(), colors, n.cpu().numpy()
