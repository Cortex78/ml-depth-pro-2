#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""What follows from metric depth and the focal length: intrinsics and 3D points.

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


def write_ply(path: Path | str, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Write a binary point cloud: ``(N, 3)`` points, and uint8 colors if given."""
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if colors is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    data = np.empty(len(points), dtype=fields)
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
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
) -> tuple[np.ndarray, np.ndarray | None]:
    """The points and colors of a frame, as ``(N, 3)`` arrays on the host.

    Leaves out infinite depth, depth beyond ``max_depth`` meters, and points on depth edges,
    which float between two surfaces. ``stride`` keeps every ``stride``-th pixel on each axis.
    """
    points = unproject(depth.float(), f_px)
    keep = torch.isfinite(depth) & ~depth_edges(depth)
    if max_depth is not None:
        keep &= depth <= max_depth
    keep = keep[::stride, ::stride]
    points = points[::stride, ::stride][keep]
    colors = None
    if image is not None:
        colors = image[::stride, ::stride][keep.cpu().numpy()]
    return points.cpu().numpy(), colors
