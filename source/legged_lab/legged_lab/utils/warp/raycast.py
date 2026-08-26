import torch
import warp as wp

wp.config.quiet = True
wp.init()

from . import kernels


def raycast_mesh_grouped(
    mesh_prototypes: dict[str, list[wp.Mesh]],
    mesh_prototype_ids: torch.Tensor,
    mesh_transforms: torch.Tensor,
    mesh_inv_transforms: torch.Tensor,
    ray_group_ids: torch.Tensor,
    mesh_ids_for_group: torch.Tensor,
    mesh_ids_slice_for_group: torch.Tensor,
    ray_starts: torch.Tensor,
    ray_directions: torch.Tensor,
    max_dist: float = 1e6,
    min_dist: float = 0.0,
    return_distance: bool = False,
    return_normal: bool = False,
    return_face_id: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Ray-cast against meshes filtered by collision group.

    Meshes with group -1 are global and visible to every ray. Meshes with group N
    are visible only to rays from environment N.
    """
    shape = ray_starts.shape
    device = ray_starts.device
    first_mesh_list = next(iter(mesh_prototypes.values()))
    torch_device = wp.device_to_torch(first_mesh_list[0].device)

    ray_starts = ray_starts.to(torch_device).view(-1, 3).contiguous()
    ray_directions = ray_directions.to(torch_device).view(-1, 3).contiguous()
    ray_group_ids = ray_group_ids.to(torch_device).view(-1).contiguous()
    mesh_ids_for_group = mesh_ids_for_group.to(torch_device).contiguous()
    mesh_ids_slice_for_group = mesh_ids_slice_for_group.to(torch_device).view(-1).contiguous()
    num_rays = ray_starts.shape[0]

    ray_hits = torch.full((num_rays, 3), max_dist, device=torch_device).contiguous()

    mesh_prototype_ids_wp = wp.from_torch(mesh_prototype_ids.to(torch_device), dtype=wp.uint64)
    mesh_transforms_wp = wp.from_torch(mesh_transforms.to(torch_device), dtype=wp.transform)
    mesh_inv_transforms_wp = wp.from_torch(mesh_inv_transforms.to(torch_device), dtype=wp.transform)
    ray_group_ids_wp = wp.from_torch(ray_group_ids, dtype=wp.int32)
    mesh_ids_for_group_wp = wp.from_torch(mesh_ids_for_group, dtype=wp.int32)
    mesh_ids_slice_for_group_wp = wp.from_torch(mesh_ids_slice_for_group, dtype=wp.int32)
    ray_starts_wp = wp.from_torch(ray_starts, dtype=wp.vec3)
    ray_directions_wp = wp.from_torch(ray_directions, dtype=wp.vec3)
    ray_hits_wp = wp.from_torch(ray_hits, dtype=wp.vec3)

    if return_distance:
        ray_distance = torch.full((num_rays,), float("inf"), device=torch_device).contiguous()
        ray_distance_wp = wp.from_torch(ray_distance, dtype=wp.float32)
    else:
        ray_distance = None
        ray_distance_wp = wp.empty((1,), dtype=wp.float32, device=torch_device)

    if return_normal:
        ray_normal = torch.full((num_rays, 3), float("inf"), device=torch_device).contiguous()
        ray_normal_wp = wp.from_torch(ray_normal, dtype=wp.vec3)
    else:
        ray_normal = None
        ray_normal_wp = wp.empty((1,), dtype=wp.vec3, device=torch_device)

    if return_face_id:
        ray_face_id = torch.full((num_rays,), -1, dtype=torch.int32, device=torch_device).contiguous()
        ray_face_id_wp = wp.from_torch(ray_face_id, dtype=wp.int32)
    else:
        ray_face_id = None
        ray_face_id_wp = wp.empty((1,), dtype=wp.int32, device=torch_device)

    wp.launch(
        kernel=kernels.raycast_mesh_kernel_grouped_transformed,
        dim=num_rays,
        inputs=[
            mesh_prototype_ids_wp,
            mesh_transforms_wp,
            mesh_inv_transforms_wp,
            ray_group_ids_wp,
            mesh_ids_for_group_wp,
            mesh_ids_slice_for_group_wp,
            ray_starts_wp,
            ray_directions_wp,
            ray_hits_wp,
            ray_distance_wp,
            ray_normal_wp,
            ray_face_id_wp,
            max_dist,
            min_dist,
            int(return_distance),
            int(return_normal),
            int(return_face_id),
        ],
        device=first_mesh_list[0].device,
    )
    wp.synchronize()

    if return_distance:
        ray_distance = ray_distance.to(device).view(shape[0], shape[1])
    if return_normal:
        ray_normal = ray_normal.to(device).view(shape)
    if return_face_id:
        ray_face_id = ray_face_id.to(device).view(shape[0], shape[1])

    return ray_hits.to(device).view(shape), ray_distance, ray_normal, ray_face_id
