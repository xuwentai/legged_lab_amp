from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Literal
import sys

import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.camera import CameraData

from legged_lab.utils.warp.raycast import raycast_mesh_grouped

from .grouped_ray_caster import GroupedRayCaster

if TYPE_CHECKING:
    from .grouped_ray_caster_camera_cfg import GroupedRayCasterCameraCfg

sys.modules.setdefault("isaaclab.sensors.ray_caster.grouped_ray_caster_camera", sys.modules[__name__])


def _isaacsim_camera_modules():
    import isaacsim.core.utils.stage as stage_utils
    import omni.physics.tensors.impl.api as physx
    from isaacsim.core.prims import XFormPrim

    return stage_utils, physx, XFormPrim


class GroupedRayCasterCamera(GroupedRayCaster):
    """Grouped ray-cast camera sensor."""

    # Keep IsaacLab's factory registration happy while loading the class from
    # legged_lab.
    __module__ = "isaaclab.sensors.ray_caster.grouped_ray_caster_camera"

    cfg: GroupedRayCasterCameraCfg

    """The configuration parameters."""
    UNSUPPORTED_TYPES: ClassVar[set[str]] = {
        "rgb",
        "instance_id_segmentation",
        "instance_id_segmentation_fast",
        "instance_segmentation",
        "instance_segmentation_fast",
        "semantic_segmentation",
        "skeleton_data",
        "motion_vectors",
        "bounding_box_2d_tight",
        "bounding_box_2d_tight_fast",
        "bounding_box_2d_loose",
        "bounding_box_2d_loose_fast",
        "bounding_box_3d",
        "bounding_box_3d_fast",
    }
    """A set of sensor types that are not supported by the ray-caster camera."""

    def __init__(self, cfg: GroupedRayCasterCameraCfg):
        """Initializes the camera object.

        Args:
            cfg: The configuration parameters.

        Raises:
            ValueError: If the provided data types are not supported by the grouped-ray-caster camera.
        """
        # perform check on supported data types
        self._check_supported_data_types(cfg)
        # initialize base class
        super().__init__(cfg)
        # create empty variables for storing output data
        self._data = CameraData()
        self._history_write_index = {}
        self._history_buffers = {}

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Grouped-Ray-Caster-Camera @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {len(self.meshes)}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}\n"
            f"\timage shape          : {self.image_shape}"
        )

    """
    Properties
    """

    @property
    def data(self) -> CameraData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    @property
    def image_shape(self) -> tuple[int, int]:
        """A tuple containing (height, width) of the camera sensor."""
        return (self.cfg.pattern_cfg.height, self.cfg.pattern_cfg.width)

    @property
    def frame(self) -> torch.tensor:
        """Frame number when the measurement took place."""
        return self._frame

    """
    Operations.
    NOTE: Since RayCasterCamera is a direct subclass of RayCaster, GroupedRayCasterCamera has to copy some of the code
    from RayCasterCamera. (Code duplication is not ideal, shall be optimized in the future.)
    """

    def set_intrinsic_matrices(
        self, matrices: torch.Tensor, focal_length: float = 1.0, env_ids: Sequence[int] | None = None
    ):
        """Set the intrinsic matrix of the camera.

        Args:
            matrices: The intrinsic matrices for the camera. Shape is (N, 3, 3).
            focal_length: Focal length to use when computing aperture values (in cm). Defaults to 1.0.
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.
        """
        # resolve env_ids
        if env_ids is None:
            env_ids = slice(None)
        # save new intrinsic matrices and focal length
        self._data.intrinsic_matrices[env_ids] = matrices.to(self._device)
        self._focal_length = focal_length
        # recompute ray directions
        self.ray_starts[env_ids], self.ray_directions[env_ids] = self.cfg.pattern_cfg.func(
            self.cfg.pattern_cfg, self._data.intrinsic_matrices[env_ids], self._device
        )

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timestamps
        # Safety: create drift buffer if _initialize_impl hasn't been called yet
        if not hasattr(self, "drift"):
            self.drift = torch.zeros(self._view.count, 3, device=self._device)
            # Also initialise remaining buffers that should come from _initialize_rays_impl
            if not hasattr(self, "_frame"):
                self._frame = torch.zeros(self._view.count, device=self._device, dtype=torch.long)
            if not hasattr(self, "_ALL_INDICES"):
                self._ALL_INDICES = torch.arange(self._view.count, device=self._device, dtype=torch.long)
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
        # reset the data
        # note: this recomputation is useful if one performs events such as randomizations on the camera poses.
        pos_w, quat_w = self._compute_camera_world_poses(env_ids)
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w_world[env_ids] = quat_w
        # Reset the frame count
        self._frame[env_ids] = 0
        for data_type in self._history_write_index:
            self._history_write_index[data_type][env_ids] = 0
            self._history_buffers[data_type][env_ids] = 0.0
            self._data.output[f"{data_type}_history"][env_ids] = 0.0

    def set_world_poses(
        self,
        positions: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        env_ids: Sequence[int] | None = None,
        convention: Literal["opengl", "ros", "world"] = "ros",
    ):
        """Set the pose of the camera w.r.t. the world frame using specified convention.

        Since different fields use different conventions for camera orientations, the method allows users to
        set the camera poses in the specified convention. Possible conventions are:

        - :obj:`"opengl"` - forward axis: -Z - up axis +Y - Offset is applied in the OpenGL (Usd.Camera) convention
        - :obj:`"ros"`    - forward axis: +Z - up axis -Y - Offset is applied in the ROS convention
        - :obj:`"world"`  - forward axis: +X - up axis +Z - Offset is applied in the World Frame convention

        See :meth:`isaaclab.utils.maths.convert_camera_frame_orientation_convention` for more details
        on the conventions.

        Args:
            positions: The cartesian coordinates (in meters). Shape is (N, 3).
                Defaults to None, in which case the camera position in not changed.
            orientations: The quaternion orientation in (w, x, y, z). Shape is (N, 4).
                Defaults to None, in which case the camera orientation in not changed.
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.
            convention: The convention in which the poses are fed. Defaults to "ros".

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` method first.
        """
        # resolve env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES

        # get current positions
        pos_w, quat_w = self._compute_view_world_poses(env_ids)
        if positions is not None:
            # transform to camera frame
            pos_offset_world_frame = positions - pos_w
            self._offset_pos[env_ids] = math_utils.quat_apply(math_utils.quat_inv(quat_w), pos_offset_world_frame)
        if orientations is not None:
            # convert rotation matrix from input convention to world
            quat_w_set = math_utils.convert_camera_frame_orientation_convention(
                orientations, origin=convention, target="world"
            )
            self._offset_quat[env_ids] = math_utils.quat_mul(math_utils.quat_inv(quat_w), quat_w_set)

        # update the data
        pos_w, quat_w = self._compute_camera_world_poses(env_ids)
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w_world[env_ids] = quat_w

    def randomize_offsets(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        pos_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
        rpy_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    ) -> None:
        """Randomize per-environment camera mounting offsets from the configured nominal pose."""
        if env_ids is None:
            env_ids = self._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self._device, dtype=torch.long)

        pos_ranges = pos_ranges or ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
        rpy_ranges = rpy_ranges or ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
        num_envs = len(env_ids)

        base_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device).repeat(num_envs, 1)
        pos_noise = torch.empty((num_envs, 3), device=self._device)
        for axis, axis_range in enumerate(pos_ranges):
            pos_noise[:, axis].uniform_(axis_range[0], axis_range[1])
        self._offset_pos[env_ids] = base_pos + pos_noise

        base_quat = math_utils.convert_camera_frame_orientation_convention(
            torch.tensor([self.cfg.offset.rot], device=self._device),
            origin=self.cfg.offset.convention,
            target="world",
        ).repeat(num_envs, 1)
        base_roll, base_pitch, base_yaw = math_utils.euler_xyz_from_quat(base_quat)
        rpy_noise = torch.empty((num_envs, 3), device=self._device)
        for axis, axis_range in enumerate(rpy_ranges):
            rpy_noise[:, axis].uniform_(axis_range[0], axis_range[1])
        self._offset_quat[env_ids] = math_utils.quat_from_euler_xyz(
            base_roll + rpy_noise[:, 0],
            base_pitch + rpy_noise[:, 1],
            base_yaw + rpy_noise[:, 2],
        )

        pos_w, quat_w = self._compute_camera_world_poses(env_ids)
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w_world[env_ids] = quat_w

    def set_world_poses_from_view(
        self, eyes: torch.Tensor, targets: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        """Set the poses of the camera from the eye position and look-at target position.

        Args:
            eyes: The positions of the camera's eye. Shape is N, 3).
            targets: The target locations to look at. Shape is (N, 3).
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` method first.
            NotImplementedError: If the stage up-axis is not "Y" or "Z".
        """
        stage_utils, _, _ = _isaacsim_camera_modules()
        # get up axis of current stage
        up_axis = stage_utils.get_stage_up_axis()
        # camera position and rotation in opengl convention
        orientations = math_utils.quat_from_matrix(
            math_utils.create_rotation_matrix_from_view(eyes, targets, up_axis=up_axis, device=self._device)
        )
        self.set_world_poses(eyes, orientations, env_ids, convention="opengl")

    """
    Implementation.
    """

    def _initialize_rays_impl(self):
        # Create all indices buffer
        self._ALL_INDICES = torch.arange(self._view.count, device=self._device, dtype=torch.long)
        # Create frame count buffer
        self._frame = torch.zeros(self._view.count, device=self._device, dtype=torch.long)
        # create buffers
        self._create_buffers()
        # compute intrinsic matrices
        self._compute_intrinsic_matrices()
        # compute ray stars and directions
        self.ray_starts, self.ray_directions = self.cfg.pattern_cfg.func(
            self.cfg.pattern_cfg, self._data.intrinsic_matrices, self._device
        )
        self.num_rays = self.ray_directions.shape[1]
        # create buffer to store ray hits
        self.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)
        # set offsets
        quat_w = math_utils.convert_camera_frame_orientation_convention(
            torch.tensor([self.cfg.offset.rot], device=self._device), origin=self.cfg.offset.convention, target="world"
        )
        self._offset_quat = quat_w.repeat(self._view.count, 1)
        self._offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device).repeat(self._view.count, 1)
        self._create_ray_collision_groups()

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        self._update_mesh_transforms(env_ids)

        # increment frame count
        self._frame[env_ids] += 1

        # compute poses from current view
        pos_w, quat_w = self._compute_camera_world_poses(env_ids)
        # update the data
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w_world[env_ids] = quat_w

        # note: full orientation is considered
        ray_starts_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
        ray_starts_w += pos_w.unsqueeze(1)
        ray_directions_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])

        # ray cast and store the hits
        # note: we set max distance to 1e6 during the ray-casting. THis is because we clip the distance
        # to the image plane and distance to the camera to the maximum distance afterwards in-order to
        # match the USD camera behavior.

        # TODO: Make ray-casting work for multiple meshes?
        # necessary for regular dictionaries.

        ray_group_ids = self._ray_collision_groups[env_ids]

        self.ray_hits_w, ray_depth, ray_normal, _ = raycast_mesh_grouped(
            mesh_prototypes=self.meshes,
            mesh_prototype_ids=self.mesh_prototype_ids_pyt,
            mesh_transforms=self.mesh_transforms_pyt,
            mesh_inv_transforms=self.mesh_inv_transforms_pyt,
            ray_group_ids=ray_group_ids,
            mesh_ids_for_group=self._mesh_ids_for_group,
            mesh_ids_slice_for_group=self._mesh_ids_slice_for_group,
            ray_starts=ray_starts_w,
            ray_directions=ray_directions_w,
            max_dist=self.cfg.max_distance,
            min_dist=self.cfg.min_distance,
            return_distance=any(
                [name in self.cfg.data_types for name in ["distance_to_image_plane", "distance_to_camera"]]
            ),
            return_normal="normals" in self.cfg.data_types,
        )

        # update output buffers
        if "distance_to_image_plane" in self.cfg.data_types:
            # note: data is in camera frame so we only take the first component (z-axis of camera frame)
            distance_to_image_plane = (
                math_utils.quat_apply(
                    math_utils.quat_inv(quat_w).repeat(1, self.num_rays),
                    (ray_depth[:, :, None] * ray_directions_w),
                )
            )[:, :, 0]
            # apply the maximum distance after the transformation
            if self.cfg.depth_clipping_behavior == "max":
                distance_to_image_plane = torch.clip(distance_to_image_plane, max=self.cfg.max_distance)
                distance_to_image_plane[torch.isnan(distance_to_image_plane)] = self.cfg.max_distance
            elif self.cfg.depth_clipping_behavior == "zero":
                distance_to_image_plane[distance_to_image_plane > self.cfg.max_distance] = 0.0
                distance_to_image_plane[torch.isnan(distance_to_image_plane)] = 0.0
            self._data.output["distance_to_image_plane"][env_ids] = distance_to_image_plane.view(
                -1, *self.image_shape, 1
            )

        if "distance_to_camera" in self.cfg.data_types:
            if self.cfg.depth_clipping_behavior == "max":
                ray_depth = torch.clip(ray_depth, max=self.cfg.max_distance)
            elif self.cfg.depth_clipping_behavior == "zero":
                ray_depth[ray_depth > self.cfg.max_distance] = 0.0
            self._data.output["distance_to_camera"][env_ids] = ray_depth.view(-1, *self.image_shape, 1)

        if "normals" in self.cfg.data_types:
            self._data.output["normals"][env_ids] = ray_normal.view(-1, *self.image_shape, 3)

        self._update_noised_outputs(env_ids)
        self._update_history_outputs(env_ids)

    def _debug_vis_callback(self, event):
        if not hasattr(self, "ray_hits_w"):
            return
        viz_points = self.ray_hits_w.reshape(-1, 3)
        viz_points = viz_points[~torch.any(torch.isinf(viz_points), dim=1)]
        translations = torch.cat([viz_points, self._data.pos_w], dim=0)
        orientations = torch.cat(
            [torch.zeros((viz_points.shape[0], 4), device=self._device), self._data.quat_w_world], dim=0
        )
        marker_indices = torch.cat(
            [torch.zeros(viz_points.shape[0], dtype=torch.int), torch.ones(self._num_envs, dtype=torch.int)]
        )
        self.ray_visualizer.visualize(translations, orientations, marker_indices=marker_indices)

    """
    Private Helpers
    """

    def _check_supported_data_types(self, cfg: GroupedRayCasterCameraCfg):
        """Checks if the data types are supported by the grouped-ray-caster camera."""
        # check if there is any intersection in unsupported types
        # reason: we cannot obtain this data from simplified warp-based ray caster
        common_elements = set(cfg.data_types) & GroupedRayCasterCamera.UNSUPPORTED_TYPES
        if common_elements:
            raise ValueError(
                f"GroupedRayCasterCamera class does not support the following sensor types: {common_elements}."
                "\n\tThis is because these sensor types cannot be obtained in a fast way using ''warp''."
                "\n\tHint: If you need to work with these sensor types, we recommend using the USD camera"
                " interface from the isaaclab.sensors.camera module."
            )

    def _create_buffers(self):
        """Create buffers for storing data."""
        # prepare drift
        self.drift = torch.zeros(self._view.count, 3, device=self._device)
        # create the data object
        # -- pose of the cameras
        self._data.pos_w = torch.zeros((self._view.count, 3), device=self._device)
        self._data.quat_w_world = torch.zeros((self._view.count, 4), device=self._device)
        # -- intrinsic matrix
        self._data.intrinsic_matrices = torch.zeros((self._view.count, 3, 3), device=self._device)
        self._data.intrinsic_matrices[:, 2, 2] = 1.0
        self._data.image_shape = self.image_shape
        # -- output data
        # create the buffers to store the annotator data.
        self._data.output = {}
        self._data.info = [{name: None for name in self.cfg.data_types}] * self._view.count
        for name in self.cfg.data_types:
            if name in ["distance_to_image_plane", "distance_to_camera"]:
                shape = (self.cfg.pattern_cfg.height, self.cfg.pattern_cfg.width, 1)
            elif name in ["normals"]:
                shape = (self.cfg.pattern_cfg.height, self.cfg.pattern_cfg.width, 3)
            else:
                raise ValueError(f"Received unknown data type: {name}. Please check the configuration.")
            # allocate tensor to store the data
            self._data.output[name] = torch.zeros((self._view.count, *shape), device=self._device)
            if name in ["distance_to_image_plane", "distance_to_camera"]:
                noised_name = f"{name}_noised"
                noised_shape = self._processed_image_shape(shape)
                self._data.output[noised_name] = torch.zeros((self._view.count, *noised_shape), device=self._device)

        self._history_write_index = {}
        self._history_buffers = {}
        for data_type, history_length in self.cfg.data_histories.items():
            if data_type not in self._data.output:
                if data_type.endswith("_noised"):
                    base_data_type = data_type.removesuffix("_noised")
                    if base_data_type in self._data.output:
                        shape = self._processed_image_shape(self._data.output[base_data_type].shape[1:])
                        self._data.output[data_type] = torch.zeros((self._view.count, *shape), device=self._device)
                    else:
                        raise ValueError(f"Cannot create history for unknown camera data type: {data_type!r}")
                else:
                    raise ValueError(f"Cannot create history for unknown camera data type: {data_type!r}")
            data_shape = self._data.output[data_type].shape[1:]
            self._data.output[f"{data_type}_history"] = torch.zeros(
                (self._view.count, history_length, *data_shape), device=self._device
            )
            self._history_buffers[data_type] = torch.zeros(
                (self._view.count, history_length, *data_shape), device=self._device
            )
            self._history_write_index[data_type] = torch.zeros(self._view.count, dtype=torch.long, device=self._device)

    def _processed_image_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        shape = input_shape
        pipeline = self.cfg.noise_pipeline
        if pipeline is None:
            return shape
        for stage in pipeline:
            if stage["name"] == "crop_and_resize":
                top, bottom, left, right = stage.get("crop_region", (0, 0, 0, 0))
                h = shape[0] - top - bottom
                w = shape[1] - left - right
                if stage.get("resize_shape"):
                    h, w = stage["resize_shape"]
                shape = (h, w, shape[2])
        return shape

    def _apply_noise_pipeline(self, image: torch.Tensor) -> torch.Tensor:
        pipeline = self.cfg.noise_pipeline
        if pipeline is None:
            return image

        from legged_lab.utils import depth_noise

        for stage in pipeline:
            name = stage["name"]
            if name == "crop_and_resize":
                image = depth_noise.crop_and_resize(
                    image,
                    crop_region=stage.get("crop_region", (0, 0, 0, 0)),
                    resize_shape=stage.get("resize_shape", None),
                )
            elif name == "gaussian_blur":
                image = depth_noise.gaussian_blur(
                    image,
                    kernel_size=stage.get("kernel_size", 3),
                    sigma=stage.get("sigma", 1.0),
                )
            elif name == "scale_randomization":
                image = depth_noise.scale_randomization(
                    image,
                    apply_probability=stage.get("apply_probability", 1.0),
                    scale_min=stage.get("scale_min", 0.90),
                    scale_max=stage.get("scale_max", 1.10),
                )
            elif name == "stereo_fusion":
                image = depth_noise.stereo_fusion(
                    image,
                    apply_probability=stage.get("apply_probability", 0.5),
                    disparity_grad_threshold=stage.get("disparity_grad_threshold", 0.08),
                    texture_var_threshold=stage.get("texture_var_threshold", 0.0005),
                    hole_probability=stage.get("hole_probability", 0.08),
                    hole_kernel_size=stage.get("hole_kernel_size", 3),
                    hole_value=stage.get("hole_value", 0.0),
                )
            elif name == "random_conv":
                image = depth_noise.random_conv(
                    image,
                    apply_probability=stage.get("apply_probability", 0.5),
                    kernel_std=stage.get("kernel_std", 0.12),
                    center_weight=stage.get("center_weight", 1.0),
                )
            elif name == "perlin_noise":
                image = depth_noise.perlin_noise(
                    image,
                    apply_probability=stage.get("apply_probability", 0.5),
                    octaves=stage.get("octaves", 4),
                    base_frequency=stage.get("base_frequency", 8.0),
                    lacunarity=stage.get("lacunarity", 2.0),
                    persistence=stage.get("persistence", 0.5),
                    amplitude=stage.get("amplitude", 1.0),
                    noise_std=stage.get("noise_std", 0.02),
                )
            elif name == "pixel_failures":
                image = depth_noise.pixel_failure(
                    image,
                    apply_probability=stage.get("apply_probability", 0.7),
                    dead_pixel_prob=stage.get("dead_pixel_prob", 0.001),
                    saturated_pixel_prob=stage.get("saturated_pixel_prob", 0.001),
                    dead_value=stage.get("dead_value", 0.0),
                    saturated_value=stage.get("saturated_value", 1.0),
                )
            elif name == "depth_normalization":
                image = depth_noise.depth_normalization(
                    image,
                    depth_range=stage.get("depth_range", (0.0, self.cfg.max_distance)),
                    normalize=stage.get("normalize", True),
                    output_range=stage.get("output_range", (0.0, 1.0)),
                )
            else:
                raise ValueError(f"Unknown depth camera noise pipeline stage: {name!r}")
        return image

    def _update_noised_outputs(self, env_ids: Sequence[int]):
        for name in self.cfg.data_types:
            noised_name = f"{name}_noised"
            if noised_name not in self._data.output:
                continue
            self._data.output[noised_name][env_ids] = self._apply_noise_pipeline(self._data.output[name][env_ids])

    def _update_history_outputs(self, env_ids: Sequence[int]):
        env_indices = self._ALL_INDICES if isinstance(env_ids, slice) else env_ids
        for data_type, history_length in self.cfg.data_histories.items():
            history_name = f"{data_type}_history"
            if history_name not in self._data.output:
                continue
            write_index = self._history_write_index[data_type][env_indices] % history_length
            self._history_buffers[data_type][env_indices, write_index] = self._data.output[data_type][env_indices]
            self._history_write_index[data_type][env_indices] += 1
            order = (
                torch.arange(history_length, device=self._device).unsqueeze(0)
                + self._history_write_index[data_type][env_indices].unsqueeze(1)
            ) % history_length
            self._data.output[history_name][env_indices] = self._history_buffers[data_type][env_indices].gather(
                1,
                order.reshape(-1, history_length, 1, 1, 1).expand_as(self._history_buffers[data_type][env_indices]),
            )

    def _compute_intrinsic_matrices(self):
        """Computes the intrinsic matrices for the camera based on the config provided."""
        # get the sensor properties
        pattern_cfg = self.cfg.pattern_cfg

        # check if vertical aperture is provided
        # if not then it is auto-computed based on the aspect ratio to preserve squared pixels
        if pattern_cfg.vertical_aperture is None:
            pattern_cfg.vertical_aperture = pattern_cfg.horizontal_aperture * pattern_cfg.height / pattern_cfg.width

        # compute the intrinsic matrix
        f_x = pattern_cfg.width * pattern_cfg.focal_length / pattern_cfg.horizontal_aperture
        f_y = pattern_cfg.height * pattern_cfg.focal_length / pattern_cfg.vertical_aperture
        c_x = pattern_cfg.horizontal_aperture_offset * f_x + pattern_cfg.width / 2
        c_y = pattern_cfg.vertical_aperture_offset * f_y + pattern_cfg.height / 2
        # allocate the intrinsic matrices
        self._data.intrinsic_matrices[:, 0, 0] = f_x
        self._data.intrinsic_matrices[:, 0, 2] = c_x
        self._data.intrinsic_matrices[:, 1, 1] = f_y
        self._data.intrinsic_matrices[:, 1, 2] = c_y

        # save focal length
        self._focal_length = pattern_cfg.focal_length

    def _compute_view_world_poses(self, env_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Obtains the pose of the view the camera is attached to in the world frame.

        Returns:
            A tuple of the position (in meters) and quaternion (w, x, y, z).
        """
        _, physx, XFormPrim = _isaacsim_camera_modules()
        # obtain the poses of the sensors
        # note: clone arg doesn't exist for xform prim view so we need to do this manually
        if isinstance(self._view, XFormPrim):
            pos_w, quat_w = self._view.get_world_poses(env_ids)
        elif isinstance(self._view, physx.ArticulationView):
            pos_w, quat_w = self._view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = math_utils.convert_quat(quat_w, to="wxyz")
        elif isinstance(self._view, physx.RigidBodyView):
            pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = math_utils.convert_quat(quat_w, to="wxyz")
        else:
            raise RuntimeError(f"Unsupported view type: {type(self._view)}")
        # return the pose
        return pos_w.clone(), quat_w.clone()

    def _compute_camera_world_poses(self, env_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the camera in the world frame.

        This function applies the offset pose to the pose of the view the camera is attached to.

        Returns:
            A tuple of the position (in meters) and quaternion (w, x, y, z) in "world" convention.
        """
        # get the pose of the view the camera is attached to
        pos_w, quat_w = self._compute_view_world_poses(env_ids)
        # apply offsets
        # need to apply quat because offset relative to parent frame
        pos_w += math_utils.quat_apply(quat_w, self._offset_pos[env_ids])
        quat_w = math_utils.quat_mul(quat_w, self._offset_quat[env_ids])

        return pos_w, quat_w
