import warp as wp


@wp.kernel(enable_backward=False)
def points_penetrate_cylinder_kernel(
    points: wp.array(dtype=wp.vec3),
    cylinder_start: wp.array(dtype=wp.vec3),
    cylinder_end: wp.array(dtype=wp.vec3),
    cylinder_radius: wp.array(dtype=wp.float32),
    cell_offsets: wp.array(dtype=wp.int32),
    cell_indices: wp.array(dtype=wp.int32),
    grid_res: wp.vec3i,
    bbox_min: wp.vec3,
    cell_size: wp.vec3,
    penetrate_offset: wp.array(dtype=wp.vec3),
):
    """Compute the deepest penetration offset from points into nearby cylinders."""
    tid = wp.tid()
    p = points[tid]

    rel = p - bbox_min
    ix = int(rel[0] / cell_size[0])
    iy = int(rel[1] / cell_size[1])
    iz = int(rel[2] / cell_size[2])

    depth = float(0.0)
    offset = wp.vec3(0.0, 0.0, 0.0)

    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                x = ix + dx
                y = iy + dy
                z = iz + dz

                if x < 0 or x >= grid_res.x or y < 0 or y >= grid_res.y or z < 0 or z >= grid_res.z:
                    continue

                flat = x * grid_res.y * grid_res.z + y * grid_res.z + z
                start = cell_offsets[flat]
                end = cell_offsets[flat + 1]

                for i in range(start, end):
                    cid = cell_indices[i]
                    a = cylinder_start[cid]
                    b = cylinder_end[cid]
                    radius = cylinder_radius[cid]

                    ab = b - a
                    ab_len = wp.length(ab)
                    if ab_len <= 1.0e-6:
                        continue

                    ab_dir = ab / ab_len
                    t = wp.dot(p - a, ab_dir)
                    if t < 0.0 or t > ab_len:
                        continue

                    proj = a + t * ab_dir
                    dist = wp.length(p - proj)
                    if dist < radius:
                        d = radius - dist
                        if d > depth:
                            depth = d
                            if dist > 1.0e-6:
                                offset = (proj - p) * (d / dist)
                            else:
                                offset = wp.vec3(0.0, 0.0, d)

    if depth > 0.0:
        penetrate_offset[tid] = offset

