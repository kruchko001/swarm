from __future__ import annotations

from typing import Optional, Iterable

from pathlib import Path

import numpy as np
import onnxruntime as ort

DEFAULT_WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float32)
DEPTH_MIN_M = 0.5
DEPTH_MAX_M = 20.0
MAX_YAW_RATE = 3.141
SIM_DT = 1 / 50
CAMERA_FOV_DEG = 90.0
CAMERA_OFFSET_M = 0.13
CAMERA_UP_OFFSET_M = 0.05


def _prep_depth(depth_map: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    elif depth.ndim != 2:
        raise ValueError(f"expected depth map with shape (H,W) or (H,W,1), got {depth.shape}")

    if depth.shape[0] < 2 or depth.shape[1] < 2:
        raise ValueError(f"depth map is too small: {depth.shape}")

    return np.clip(depth, 0.0, 1.0)


def _norm_depth_m(depth: np.ndarray) -> np.ndarray:
    return DEPTH_MIN_M + depth * (DEPTH_MAX_M - DEPTH_MIN_M)


def _cam_vec_world(
    camera_vector: np.ndarray,
    camera_forward: np.ndarray,
    camera_right: np.ndarray,
    camera_up: np.ndarray,
) -> np.ndarray:
    world = (
        camera_right * float(camera_vector[0])
        + camera_up * float(camera_vector[1])
        + camera_forward * float(camera_vector[2])
    ).astype(np.float32)
    return _norm_vec(world)


def _cam_basis(
    camera_target: np.ndarray,
    *,
    camera_position: np.ndarray,
    camera_target_is_point: bool,
    camera_up: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_position = np.asarray(camera_position, dtype=np.float32).reshape(-1)
    if camera_position.shape != (3,):
        raise ValueError(f"expected camera_position shape (3,), got {camera_position.shape}")

    target = np.asarray(camera_target, dtype=np.float32).reshape(-1)
    if target.shape != (3,):
        raise ValueError(f"expected camera_target shape (3,), got {target.shape}")

    if camera_target_is_point:
        target = target - camera_position

    forward = _norm_vec(target)
    up_guess = DEFAULT_WORLD_UP if camera_up is None else _norm_vec(camera_up)

    right = np.cross(forward, up_guess).astype(np.float32)
    if float(np.linalg.norm(right)) <= 1e-8:
        fallback_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, fallback_up).astype(np.float32)

    right = _norm_vec(right)
    up = _norm_vec(np.cross(right, forward))
    return forward, right, up


def _parse_fov(fov_deg: float | Iterable[float]) -> tuple[float, float]:
    if np.isscalar(fov_deg):
        fov_x = float(fov_deg)
        fov_y = float(fov_deg)
    else:
        values = tuple(float(v) for v in fov_deg)
        if len(values) != 2:
            raise ValueError(f"expected scalar FOV or (horizontal, vertical), got {values}")
        fov_x, fov_y = values

    if not (0.0 < fov_x < 180.0 and 0.0 < fov_y < 180.0):
        raise ValueError(f"invalid FOV values: {(fov_x, fov_y)}")
    return fov_x, fov_y


def _min_pool(depth_m: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    row_idx = np.linspace(0, depth_m.shape[0], out_h + 1, dtype=np.int32)
    col_idx = np.linspace(0, depth_m.shape[1], out_w + 1, dtype=np.int32)
    temp = np.minimum.reduceat(depth_m, row_idx[:-1], axis=0)
    return np.minimum.reduceat(temp, col_idx[:-1], axis=1).astype(np.float32)


def _work_shape(height: int, width: int, working_resolution: int) -> tuple[int, int]:
    if working_resolution < 3:
        raise ValueError(f"working_resolution must be >= 3, got {working_resolution}")

    scale = min(1.0, float(working_resolution) / float(max(height, width)))
    out_h = max(3, int(round(height * scale)))
    out_w = max(3, int(round(width * scale)))

    if out_h % 2 == 0:
        out_h = max(3, out_h - 1)
    if out_w % 2 == 0:
        out_w = max(3, out_w - 1)

    return min(out_h, height), min(out_w, width)


def _cand_rays(
    height: int,
    width: int,
    fov_x_deg: float,
    fov_y_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(1.0, -1.0, height, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x, y)

    tan_half_x = float(np.tan(np.deg2rad(np.float32(fov_x_deg)) * 0.5))
    tan_half_y = float(np.tan(np.deg2rad(np.float32(fov_y_deg)) * 0.5))

    ray_points = np.stack(
        [
            x_grid * tan_half_x,
            y_grid * tan_half_y,
            np.ones_like(x_grid, dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.float32)

    ray_norms = np.linalg.norm(ray_points, axis=-1, keepdims=True)
    ray_dirs = (ray_points / np.maximum(ray_norms, 1e-8)).astype(np.float32)
    return ray_points, ray_dirs


def _cam_vecs_world(
    camera_vectors: np.ndarray,
    camera_forward: np.ndarray,
    camera_right: np.ndarray,
    camera_up: np.ndarray,
) -> np.ndarray:
    vectors = np.asarray(camera_vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(f"expected camera_vectors shape (N, 3), got {vectors.shape}")

    world = (
        vectors[:, 0:1] * camera_right[None, :]
        + vectors[:, 1:2] * camera_up[None, :]
        + vectors[:, 2:3] * camera_forward[None, :]
    ).astype(np.float32)
    norms = np.linalg.norm(world, axis=1, keepdims=True)
    return (world / np.maximum(norms, 1e-8)).astype(np.float32)


def _safe_ctx(
    depth_map: np.ndarray,
    camera_position: np.ndarray,
    camera_target: np.ndarray,
    fov_deg: float | Iterable[float],
    *,
    current_direction: np.ndarray | None = None,
    camera_target_is_point: bool = False,
    camera_up: np.ndarray | None = None,
    working_resolution: int = 49,
) -> dict:
    depth = _prep_depth(depth_map)
    depth_m = _norm_depth_m(depth)
    fov_x_deg, fov_y_deg = _parse_fov(fov_deg)
    camera_forward, camera_right, camera_up_vector = _cam_basis(
        camera_target,
        camera_position=np.asarray(camera_position, dtype=np.float32),
        camera_target_is_point=camera_target_is_point,
        camera_up=camera_up,
    )

    pooled_h, pooled_w = _work_shape(depth_m.shape[0], depth_m.shape[1], working_resolution)
    pooled_depth_m = _min_pool(depth_m, pooled_h, pooled_w)
    ray_points_cam, ray_dirs_cam = _cand_rays(pooled_h, pooled_w, fov_x_deg, fov_y_deg)

    surface_points_cam = (ray_points_cam * pooled_depth_m[..., None]).astype(np.float32)
    surface_ranges = np.linalg.norm(surface_points_cam, axis=-1)

    candidate_dirs = ray_dirs_cam.reshape(-1, 3)
    candidate_world_dirs = _cam_vecs_world(
        candidate_dirs,
        camera_forward=camera_forward,
        camera_right=camera_right,
        camera_up=camera_up_vector,
    )

    if current_direction is None:
        preferred_direction = camera_forward
    else:
        preferred_candidate = np.asarray(current_direction, dtype=np.float32).reshape(-1)
        if preferred_candidate.shape != (3,):
            raise ValueError(f"expected current_direction shape (3,), got {preferred_candidate.shape}")
        if float(np.linalg.norm(preferred_candidate)) <= 1e-8:
            preferred_direction = camera_forward
        else:
            preferred_direction = _norm_vec(preferred_candidate)

    preference_angles = np.arccos(
        np.clip(candidate_world_dirs @ preferred_direction, -1.0, 1.0)
    ).astype(np.float32)

    obstacle_ranges_sq = np.sum(
        surface_points_cam.reshape(-1, 3) ** 2, axis=1, dtype=np.float32
    )
    projections = candidate_dirs @ surface_points_cam.reshape(-1, 3).T

    return {
        "candidate_dirs": candidate_dirs,
        "camera_forward": camera_forward,
        "camera_right": camera_right,
        "camera_up_vector": camera_up_vector,
        "surface_ranges": surface_ranges,
        "obstacle_ranges_sq": obstacle_ranges_sq,
        "projections": projections,
        "preference_angles": preference_angles,
    }


def _pick_ctx(
    ctx: dict,
    *,
    drone_radius_m: float = 0.06,
    safety_margin_m: float = 0.03,
    preferred_clearance_m: float = 3.0,
    max_lookahead_m: float = 8.0,
) -> np.ndarray:
    candidate_dirs = ctx["candidate_dirs"]
    surface_ranges = ctx["surface_ranges"]
    obstacle_ranges_sq = ctx["obstacle_ranges_sq"]
    projections = ctx["projections"]
    preference_angles = ctx["preference_angles"]

    effective_radius = float(drone_radius_m + safety_margin_m)
    relevant_points = surface_ranges.ravel() <= float(max_lookahead_m + effective_radius)

    if np.any(relevant_points):
        rel_idx = np.flatnonzero(relevant_points)
        proj_rel = projections[:, rel_idx]
        orsq_rel = obstacle_ranges_sq[rel_idx]

        lateral_sq = np.clip(orsq_rel[None, :] - proj_rel * proj_rel, 0.0, None)

        collision_mask = (proj_rel > 0.0) & (lateral_sq < (effective_radius * effective_radius))
        collision_distances = np.full_like(proj_rel, np.inf, dtype=np.float32)

        if np.any(collision_mask):
            penetration = np.sqrt(
                np.maximum((effective_radius * effective_radius) - lateral_sq[collision_mask], 0.0)
            ).astype(np.float32)
            collision_distances[collision_mask] = np.maximum(
                proj_rel[collision_mask] - penetration,
                0.0,
            )

        min_collision_distance = np.min(collision_distances, axis=1)
        min_collision_distance[~np.isfinite(min_collision_distance)] = float(max_lookahead_m)
        clearance_distance = np.minimum(min_collision_distance, float(max_lookahead_m)).astype(np.float32)
    else:
        clearance_distance = np.full(candidate_dirs.shape[0], float(max_lookahead_m), dtype=np.float32)

    safe_candidates = np.flatnonzero(clearance_distance >= float(preferred_clearance_m))
    if safe_candidates.size > 0:
        best_order = np.lexsort(
            (
                -clearance_distance[safe_candidates],
                preference_angles[safe_candidates],
            )
        )
        best_idx = int(safe_candidates[best_order[0]])
    else:
        best_order = np.lexsort((preference_angles, -clearance_distance))
        best_idx = int(best_order[0])

    best_camera_direction = candidate_dirs[best_idx]
    return _cam_vec_world(
        best_camera_direction,
        camera_forward=ctx["camera_forward"],
        camera_right=ctx["camera_right"],
        camera_up=ctx["camera_up_vector"],
    )


def find_safe_dir(
    depth_map: np.ndarray,
    camera_position: np.ndarray,
    camera_target: np.ndarray,
    fov_deg: float | Iterable[float] = (90.0, 90.0),
    *,
    current_direction: np.ndarray | None = None,
    camera_target_is_point: bool = False,
    camera_up: np.ndarray | None = None,
    drone_radius_m: float = 0.06,
    safety_margin_m: float = 0.03,
    preferred_clearance_m: float = 3.0,
    max_lookahead_m: float = 8.0,
    working_resolution: int = 49,
) -> np.ndarray:
    ctx = _safe_ctx(
        depth_map, camera_position, camera_target, fov_deg,
        current_direction=current_direction,
        camera_target_is_point=camera_target_is_point,
        camera_up=camera_up,
        working_resolution=working_resolution,
    )
    return _pick_ctx(
        ctx,
        drone_radius_m=drone_radius_m,
        safety_margin_m=safety_margin_m,
        preferred_clearance_m=preferred_clearance_m,
        max_lookahead_m=max_lookahead_m,
    )


def _norm_vec(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"expected a 3D vector, got shape {arr.shape}")

    norm = float(np.linalg.norm(arr))
    if norm <= eps:
        raise ValueError("cannot normalize a near-zero vector")
    return (arr / norm).astype(np.float32)

def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Right-handed rotation: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    vector_world = R @ vector_body
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rotation_x = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr],
    ], dtype=np.float32)

    rotation_y = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp],
    ], dtype=np.float32)

    rotation_z = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1],
    ], dtype=np.float32)

    return (rotation_z @ rotation_y @ rotation_x).astype(np.float32)

def _cam_geom(
    drone_position: np.ndarray,
    drone_rpy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotation_matrix = rpy_to_rot(drone_rpy[0], drone_rpy[1], drone_rpy[2])
    forward_direction = _norm_vec(rotation_matrix @ np.array([1.0, 0.0, 0.0], dtype=np.float32))
    up_guess = _norm_vec(rotation_matrix @ np.array([0.0, 0.0, 1.0], dtype=np.float32))
    right_direction = np.cross(forward_direction, up_guess).astype(np.float32)

    if float(np.linalg.norm(right_direction)) <= 1e-8:
        right_direction = np.cross(forward_direction, np.array([0.0, 0.0, 1.0], dtype=np.float32)).astype(np.float32)

    right_direction = _norm_vec(right_direction)
    up_direction = _norm_vec(np.cross(right_direction, forward_direction))

    camera_position = (
        np.asarray(drone_position, dtype=np.float32)
        + forward_direction * CAMERA_OFFSET_M
        + up_guess * CAMERA_UP_OFFSET_M
    ).astype(np.float32)
    camera_target = (camera_position + forward_direction * 20.0).astype(np.float32)

    return camera_position, camera_target, right_direction, up_direction, forward_direction


def slerp_dir(a, b, t, eps=1e-12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        return b
    a /= na
    b /= nb

    if t <= 0.0: return a
    if t >= 1.0: return b
 
    cross = np.cross(a, b)
    cross_norm = np.linalg.norm(cross)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    angle = np.arctan2(cross_norm, dot)  # in [0, π]

    if angle < 1e-8: 
        v = b
    else:
        if np.pi - angle < 1e-8:  
            x = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(a, x)) > 0.9:
                x = np.array([0.0, 1.0, 0.0])
            k = np.cross(a, x)
            k /= np.linalg.norm(k)
        else:
            k = cross / cross_norm

        c, s = np.cos(t * angle), np.sin(t * angle)
        v = a * c + np.cross(k, a) * s + k * (np.dot(k, a)) * (1 - c)

    return v / np.linalg.norm(v)


class DroneFlightController:
    def __init__(
        self,
        *,
        goal_detector_model_path: Optional[Path] = None,
    ):
        script_dir = Path(__file__).resolve().parent

        if goal_detector_model_path is None:
            goal_detector_model_path = script_dir / "goal_detector.onnx"

        self._load_model(goal_detector_model_path)

        self._mode = "takeoff"
        self._last_action = None
        self._landing_platform_position = None

    def act(self, observation):
        state = np.asarray(observation.get("state", None), dtype=np.float32).squeeze()

        drone_position = np.array([state[0], state[1], state[2]], dtype=float)
        drone_rpy = np.array([state[3], state[4], state[5]], dtype=float)
        drone_velocity = np.array([state[6], state[7], state[8]], dtype=float)
        drone_speed = float(np.linalg.norm(drone_velocity))
        drone_altitude = state[-4] * 20.0

        search_area_vector = np.array([state[-3], state[-2], state[-1]], dtype=float)
        search_area_position = search_area_vector + drone_position

        search_area_position[2] += 2.5
        search_area_vector = search_area_position - drone_position
        distance_to_search_area = float(np.linalg.norm(search_area_vector))

        depth = np.asarray(observation["depth"], dtype=np.float32)

        is_goal_visible = False
        visible_goal_position = None

        if distance_to_search_area <= DEPTH_MAX_M or self._mode in ("navigation", "landing"):
            goal_visibility_prob, predicted_goal_position = self._predict_goal(
                depth,
                state,
                drone_position,
                drone_rpy,
            )
            is_goal_visible = bool(goal_visibility_prob >= self._goal_visibility_threshold)

            if is_goal_visible:
                visible_goal_position = predicted_goal_position.copy()

        slerp_steps = 0.06

        if self._mode == "takeoff":
            yaw_to_search_area = np.arctan2(search_area_vector[1], search_area_vector[0])
            yaw_to_search_area_diff = (yaw_to_search_area - drone_rpy[2] + np.pi) % (2.0 * np.pi) - np.pi
            yaw_command = yaw_to_search_area / np.pi
            z_command = 0.0
            speed_command = 0.3
            min_altitude = 1.5

            if drone_altitude < min_altitude:
                z_command = 1.0

            if drone_altitude >= min_altitude and is_goal_visible:
                self._mode = "navigation"
            elif drone_altitude >= min_altitude and abs(yaw_to_search_area_diff) < (np.pi / 36):
                self._mode = "search"

            action = np.array([0.0, 0.0, z_command, speed_command, yaw_command], dtype=np.float32)
        elif self._mode == "search":
            acceleration_rate = 0.05
            brake_rate = 0.025
            drone_speed_normalized = min(drone_speed, 3.0) / 3.0

            if self._landing_platform_position is None:
                if distance_to_search_area > 3.0:
                    yaw_command = np.arctan2(search_area_vector[1], search_area_vector[0]) / np.pi
                    speed_command = min(drone_speed_normalized + acceleration_rate, 1.0)
                else:
                    yaw_command = self._rot_left(drone_rpy[2])
                    speed_command = distance_to_search_area / 3.0

                    if speed_command < self._last_action[3] - brake_rate:
                        speed_command = self._last_action[3] - brake_rate

                    speed_command = max(speed_command, 0.0)

                direction = search_area_vector

                if float(np.linalg.norm(direction)) > 1e-6:
                    direction_norm = direction / np.linalg.norm(direction)
                else:
                    direction_norm = np.array([0.0, 0.0, 0.0], dtype=np.float32)

                action = np.concatenate([direction_norm, [speed_command, yaw_command]], dtype=np.float32)
            else:
                direction = self._last_action[0:3]
                speed_command = self._last_action[3] - brake_rate
                speed_command = max(speed_command, 0.0)
                yaw_command = self._rot_left(drone_rpy[2])

                action = np.concatenate([direction, [speed_command, yaw_command]], dtype=np.float32)

            if is_goal_visible:
                self._mode = "navigation"
        elif self._mode == "navigation":
            acceleration_rate = 0.05
            drone_speed_normalized = min(drone_speed, 3.0) / 3.0
            speed_command = min(drone_speed_normalized + acceleration_rate, 1.0)

            if visible_goal_position is not None:
                goal_position = visible_goal_position.copy()
                goal_position[2] += 0.5

                direction = goal_position - drone_position
                distance_to_goal = float(np.linalg.norm(direction))
                yaw_command = np.arctan2(direction[1], direction[0]) / np.pi

                if distance_to_goal > 1e-6:
                    direction[2] *= 5.0
                    direction_norm = direction / np.linalg.norm(direction)
                else:
                    direction_norm = np.array([0.0, 0.0, 0.0], dtype=np.float32)

                if drone_altitude < 0.5:
                    direction_norm = np.array([direction_norm[0] * 0.1, direction_norm[1] * 0.1, 1.0], dtype=np.float32)

                if distance_to_goal < 4.0:
                    self._mode = "landing"
                    self._landing_platform_position = None

                action = np.concatenate([direction_norm, [speed_command, yaw_command]], dtype=np.float32)
            else:
                action = self._last_action.copy()

            if not is_goal_visible or visible_goal_position is None:
                self._mode = "search"
        elif self._mode == "landing":
            if visible_goal_position is not None:
                goal_position = visible_goal_position.copy()
                goal_position[2] += 0.3

                if self._landing_platform_position is None:
                    self._landing_platform_position = goal_position.copy()
                else:
                    dist_drone_to_anchor = float(np.linalg.norm(drone_position - self._landing_platform_position))
                    dxy = float(np.linalg.norm(goal_position[0:2] - self._landing_platform_position[0:2]))
                    if dist_drone_to_anchor < 1.5 and dxy < 0.5:
                        self._landing_platform_position[0:2] = (
                            0.8 * self._landing_platform_position[0:2] + 0.2 * goal_position[0:2]
                        )

                direction_to_landing_point = self._landing_platform_position - drone_position
                direction_to_current_goal_position = goal_position - drone_position
                distance_to_landing_point = float(np.linalg.norm(direction_to_landing_point))
                horizontal_distance_to_current_goal_position = float(
                    np.linalg.norm(goal_position[0:2] - drone_position[0:2]))

                brake_rate = 0.01
                speed_command = self._last_action[3] - brake_rate
                speed_command = max(speed_command, 0.1)

                yaw_command = np.arctan2(direction_to_current_goal_position[1],
                                         direction_to_current_goal_position[0]) / np.pi

                if distance_to_landing_point > 0.2:
                    direction_to_landing_point[2] *= 2.0
                    direction_norm = direction_to_landing_point / np.linalg.norm(direction_to_landing_point)
                else:
                    direction_norm = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    speed_command = 0.0

                if horizontal_distance_to_current_goal_position < 0.6:
                    direction_norm = np.array([0.0, 0.0, -1.0], dtype=np.float32)
                    speed_command = 0.3

                action = np.concatenate([direction_norm, [speed_command, yaw_command]], dtype=np.float32)
            else:
                action = self._last_action.copy()

            if not is_goal_visible or visible_goal_position is None:
                self._mode = "search"
        else:
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        if (self._mode == "search" and distance_to_search_area > 3.0) or self._mode == "navigation":
            camera_position, camera_target, _, up_direction, _ = _cam_geom(
                drone_position,
                drone_rpy,
            )

            ctx = _safe_ctx(
                depth_map=depth,
                camera_position=camera_position,
                camera_target=camera_target,
                fov_deg=CAMERA_FOV_DEG,
                current_direction=action[0:3],
                camera_target_is_point=True,
                camera_up=up_direction,
                working_resolution=32,
            )

            waypoint_direction = _pick_ctx(
                ctx,
                safety_margin_m=0.35,
                preferred_clearance_m=10.0,
                max_lookahead_m=15.0,
            )

            waypoint_direction_close = _pick_ctx(
                ctx,
                safety_margin_m=1.2,
                preferred_clearance_m=1.5,
                max_lookahead_m=3.0,
            )

            if np.linalg.norm(action[0:3]) > 0 and np.linalg.norm(waypoint_direction_close) > 0:
                dir_1 = action[0:3] / np.linalg.norm(action[0:3])
                dir_2 = waypoint_direction_close / np.linalg.norm(waypoint_direction_close)
                dir_similarity = np.dot(dir_1, dir_2)

                if dir_similarity < 0.99:
                    waypoint_direction = waypoint_direction_close

            action = np.concatenate([waypoint_direction, [action[3], action[4]]]).astype(np.float32)

        if self._last_action is not None:
            slerp_dir_vector = slerp_dir(
                a=self._last_action[0:3],
                b=action[0:3],
                t=slerp_steps
            )

            action = np.concatenate([slerp_dir_vector, [action[3], action[4]]]).astype(np.float32)

        action = np.clip(action, -1.0, 1.0)

        self._last_action = action

        return action[None, :]

    def reset(self):
        self._mode = "takeoff"
        self._last_action = None
        self._landing_platform_position = None

    def _rot_left(self, drone_yaw):
        max_yaw_change = MAX_YAW_RATE * SIM_DT
        new_drone_yaw_angle = drone_yaw + max_yaw_change - 1e-4
        new_drone_yaw_angle_normalized = new_drone_yaw_angle / np.pi

        if new_drone_yaw_angle_normalized > 1.0:
            new_drone_yaw_angle_normalized = (new_drone_yaw_angle_normalized - 1.0) - 1.0

        if new_drone_yaw_angle_normalized < -1.0:
            new_drone_yaw_angle_normalized = (new_drone_yaw_angle_normalized + 1.0) + 1.0

        return np.clip(new_drone_yaw_angle_normalized, -1.0, 1.0)

    def _load_model(self, goal_detector_model_path: Path):
        goal_detector_model_path = Path(goal_detector_model_path)
        if not goal_detector_model_path.exists():
            raise FileNotFoundError(f"Model not found: {goal_detector_model_path}")

        try:
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opts.intra_op_num_threads = 2
            session = ort.InferenceSession(
                str(goal_detector_model_path),
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load ONNX goal detector. If the model was exported with external data, "
                f"keep the referenced .onnx.data file next to {goal_detector_model_path}."
            ) from exc

        inputs = {input_info.name: input_info for input_info in session.get_inputs()}
        required_inputs = {"depth", "camera_pos", "camera_target", "fov"}
        missing_inputs = sorted(required_inputs - set(inputs))
        if missing_inputs:
            raise KeyError(f"ONNX goal detector missing required inputs: {missing_inputs}")

        outputs = session.get_outputs()
        if len(outputs) != 1:
            raise KeyError(f"ONNX goal detector expected one output, got {len(outputs)}")

        state_shape = inputs["state"].shape if "state" in inputs else None
        state_dim = None
        if state_shape is not None and len(state_shape) == 2 and isinstance(state_shape[1], int):
            state_dim = int(state_shape[1])

        self._goal_detector_session = session
        self._goal_detector_providers = session.get_providers()
        self._goal_detector_inputs = inputs
        self._goal_detector_output_name = outputs[0].name
        self._goal_detector_state_dim = state_dim
        self._goal_visibility_threshold = 0.5

    def _predict_goal(
        self,
        depth: np.ndarray,
        state: np.ndarray,
        drone_position: np.ndarray,
        drone_rpy: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        depth_input = np.asarray(depth, dtype=np.float32)
        if depth_input.ndim != 3 or depth_input.shape[-1] != 1:
            raise ValueError(f"expected depth shape (H,W,1), got {depth_input.shape}")

        camera_position, camera_target, _, _, _ = _cam_geom(
            drone_position,
            drone_rpy,
        )

        model_inputs = {
            "depth": depth_input[None, ...],
            "camera_pos": camera_position[None, :].astype(np.float32, copy=False),
            "camera_target": camera_target[None, :].astype(np.float32, copy=False),
            "fov": np.asarray([CAMERA_FOV_DEG], dtype=np.float32),
        }
        if "state" in self._goal_detector_inputs:
            state_input = np.asarray(state, dtype=np.float32).reshape(-1)
            if self._goal_detector_state_dim is not None and state_input.shape[0] != self._goal_detector_state_dim:
                raise ValueError(
                    "ONNX goal detector state shape mismatch: "
                    f"expected {self._goal_detector_state_dim}, got {state_input.shape[0]}"
                )
            model_inputs["state"] = state_input[None, :]

        prediction = self._goal_detector_session.run(
            [self._goal_detector_output_name],
            model_inputs,
        )[0]
        prediction = np.asarray(prediction, dtype=np.float32)
        if prediction.shape != (1, 4):
            raise ValueError(f"expected ONNX goal detector output shape (1,4), got {prediction.shape}")

        visibility_prob = float(prediction[0, 0])
        pred_world = prediction[0, 1:4].astype(np.float32, copy=True)
        return visibility_prob, pred_world
