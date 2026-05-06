#include <metal_stdlib>
using namespace metal;

#ifndef FOAM_TILE_WIDTH
#define FOAM_TILE_WIDTH 8u
#endif

#ifndef FOAM_TILE_THREADS
#define FOAM_TILE_THREADS 64u
#endif

// Compile-time draft of the v2 DynamicPowerFoam kernel family. This file is not wired
// into the current Python extension yet; it defines the intended buffer ABI and
// the exact reverse-replay math for the new tiled path.

struct FoamTileMetaI32 {
  int batch_size;
  int height;
  int width;
  int cell_count;
  int feature_dim;
  int tiles_y;
  int tiles_x;
  int tile_count_per_image;
  int total_batch_tiles;
  int camera_stride;
};

struct FoamTileMetaF32 {
  float near_plane;
  float alpha_threshold;
  float transmittance_threshold;
  float max_alpha;
  float eps;
  float projection_radius_scale;
};

struct FoamCamera {
  float3 eye;
  float3 right;
  float3 up;
  float3 forward;
  float fx;
  float fy;
  float cx;
  float cy;
};

struct FoamInterval {
  bool hit;
  float t_near;
  float t_far;
  int near_id;
  int far_id;
};

struct FoamEndpointGrad {
  float3 center;
  float radius;
  float3 adj_center;
  float adj_radius;
};

inline float3 foam_load3(const device float* values, uint idx) {
  uint base = idx * 3u;
  return float3(values[base + 0u], values[base + 1u], values[base + 2u]);
}

inline FoamCamera foam_load_camera(const device float* cameras, uint batch, constant FoamTileMetaI32& mi) {
  uint base = batch * uint(mi.camera_stride);
  FoamCamera cam;
  cam.eye = float3(cameras[base + 0u], cameras[base + 1u], cameras[base + 2u]);
  cam.right = float3(cameras[base + 3u], cameras[base + 4u], cameras[base + 5u]);
  cam.up = float3(cameras[base + 6u], cameras[base + 7u], cameras[base + 8u]);
  cam.forward = float3(cameras[base + 9u], cameras[base + 10u], cameras[base + 11u]);
  cam.fx = cameras[base + 12u];
  cam.fy = cameras[base + 13u];
  cam.cx = cameras[base + 14u];
  cam.cy = cameras[base + 15u];
  return cam;
}

inline uint foam_float_sort_key(float value) {
  uint bits = as_type<uint>(value);
  uint mask = ((bits & 0x80000000u) != 0u) ? 0xffffffffu : 0x80000000u;
  return bits ^ mask;
}

inline void foam_atomic_add3(device atomic_float* values, uint base, float3 v) {
  atomic_fetch_add_explicit(&values[base + 0u], v.x, memory_order_relaxed);
  atomic_fetch_add_explicit(&values[base + 1u], v.y, memory_order_relaxed);
  atomic_fetch_add_explicit(&values[base + 2u], v.z, memory_order_relaxed);
}

inline bool foam_project_cell_bounds(
    const device float* points,
    const device float* radii,
    uint cell,
    FoamCamera cam,
    constant FoamTileMetaI32& mi,
    constant FoamTileMetaF32& mf,
    thread int& tile_x0,
    thread int& tile_y0,
    thread int& tile_x1,
    thread int& tile_y1,
    thread float& sort_power) {
  float3 p = foam_load3(points, cell);
  float radius = radii[cell];
  if (!(radius > 0.0f) || !isfinite(radius)) {
    return false;
  }

  float3 v = p - cam.eye;
  float x_cam = dot(v, cam.right);
  float y_cam = dot(v, cam.up);
  float z_cam = dot(v, cam.forward);
  if (z_cam <= max(mf.near_plane + radius, mf.eps)) {
    return false;
  }

  float inv_z = 1.0f / max(z_cam, mf.eps);
  float u = cam.fx * x_cam * inv_z + cam.cx;
  float vv = cam.fy * y_cam * inv_z + cam.cy;
  float denom = max(z_cam - radius, mf.eps);
  float pixel_radius = max(abs(cam.fx), abs(cam.fy)) * radius / denom;
  pixel_radius *= max(mf.projection_radius_scale, 1.0f);

  int x0 = max(0, int(floor(u - pixel_radius)));
  int y0 = max(0, int(floor(vv - pixel_radius)));
  int x1 = min(mi.width - 1, int(ceil(u + pixel_radius)));
  int y1 = min(mi.height - 1, int(ceil(vv + pixel_radius)));
  if (x1 < x0 || y1 < y0) {
    return false;
  }

  tile_x0 = x0 / int(FOAM_TILE_WIDTH);
  tile_y0 = y0 / int(FOAM_TILE_WIDTH);
  tile_x1 = x1 / int(FOAM_TILE_WIDTH);
  tile_y1 = y1 / int(FOAM_TILE_WIDTH);
  sort_power = dot(v, v) - radius * radius;
  return true;
}

inline FoamInterval foam_ray_sphere_interval(
    float3 origin,
    float3 direction,
    float3 center,
    float radius,
    constant FoamTileMetaF32& mf) {
  FoamInterval out;
  out.hit = false;
  out.t_near = 0.0f;
  out.t_far = 0.0f;
  out.near_id = -2;
  out.far_id = -2;

  float3 oc = origin - center;
  float qa = dot(direction, direction);
  float qb = 2.0f * dot(oc, direction);
  float qc = dot(oc, oc) - radius * radius;
  float disc = qb * qb - 4.0f * qa * qc;
  if (disc < 0.0f || qa <= mf.eps) {
    return out;
  }

  float root = sqrt(max(disc, 0.0f));
  float inv = 0.5f / qa;
  float t0 = (-qb - root) * inv;
  float t1 = (-qb + root) * inv;
  if (t1 <= mf.near_plane) {
    return out;
  }

  out.hit = true;
  out.t_near = max(t0, mf.near_plane);
  out.t_far = t1;
  out.near_id = (t0 >= mf.near_plane) ? -1 : -2;
  out.far_id = -1;
  return out;
}

inline bool foam_clip_power_face(
    float3 origin,
    float3 direction,
    float3 pi,
    float ri,
    float3 pj,
    float rj,
    int face_id,
    constant FoamTileMetaF32& mf,
    thread FoamInterval& interval) {
  float3 n = pj - pi;
  float h = 0.5f * (dot(pj, pj) - dot(pi, pi) + ri * ri - rj * rj);
  float dp = dot(direction, n);
  float num = h - dot(origin, n);

  if (abs(dp) <= mf.eps) {
    return num >= -mf.eps;
  }

  float t_face = num / dp;
  if (dp > 0.0f) {
    if (t_face < interval.t_far) {
      interval.t_far = t_face;
      interval.far_id = face_id;
    }
  } else {
    if (t_face > interval.t_near) {
      interval.t_near = t_face;
      interval.near_id = face_id;
    }
  }

  return interval.t_far > interval.t_near;
}

inline FoamInterval foam_clipped_cell_interval(
    const device float* points,
    const device float* radii,
    const device int* adjacency,
    const device int* adjacency_offsets,
    uint cell,
    float3 origin,
    float3 direction,
    constant FoamTileMetaI32& mi,
    constant FoamTileMetaF32& mf) {
  float3 center = foam_load3(points, cell);
  float radius = radii[cell];
  FoamInterval hit = foam_ray_sphere_interval(origin, direction, center, radius, mf);
  if (!hit.hit) {
    return hit;
  }

  int edge_begin = adjacency_offsets[cell];
  int edge_end = adjacency_offsets[cell + 1u];
  if (edge_begin < 0 || edge_end < edge_begin) {
    hit.hit = false;
    return hit;
  }

  for (int edge = edge_begin; edge < edge_end; ++edge) {
    int neighbor_i = adjacency[edge];
    if (neighbor_i < 0 || neighbor_i >= mi.cell_count || uint(neighbor_i) == cell) {
      continue;
    }
    float3 pj = foam_load3(points, uint(neighbor_i));
    if (!foam_clip_power_face(
            origin,
            direction,
            center,
            radius,
            pj,
            radii[uint(neighbor_i)],
            edge - edge_begin,
            mf,
            hit)) {
      hit.hit = false;
      return hit;
    }
  }

  hit.hit = hit.t_far > hit.t_near;
  return hit;
}

inline FoamEndpointGrad foam_ray_sphere_endpoint_bwd(
    float3 origin,
    float3 direction,
    float3 center,
    float radius,
    bool use_far,
    constant FoamTileMetaF32& mf) {
  FoamEndpointGrad out;
  out.center = float3(0.0f);
  out.radius = 0.0f;
  out.adj_center = float3(0.0f);
  out.adj_radius = 0.0f;

  float3 oc = origin - center;
  float qa = dot(direction, direction);
  float qb = 2.0f * dot(oc, direction);
  float qc = dot(oc, oc) - radius * radius;
  float disc = qb * qb - 4.0f * qa * qc;
  if (disc <= mf.eps || qa <= mf.eps) {
    return out;
  }

  float root = sqrt(disc);
  float3 common = (2.0f * qa * oc - qb * direction) / max(qa * root, mf.eps);
  if (use_far) {
    out.center = direction / qa + common;
    out.radius = 2.0f * radius / root;
  } else {
    out.center = direction / qa - common;
    out.radius = -2.0f * radius / root;
  }
  return out;
}

inline FoamEndpointGrad foam_power_face_endpoint_bwd(
    float3 origin,
    float3 direction,
    float3 center,
    float radius,
    float3 adj_center,
    float adj_radius,
    constant FoamTileMetaF32& mf) {
  FoamEndpointGrad out;
  out.center = float3(0.0f);
  out.radius = 0.0f;
  out.adj_center = float3(0.0f);
  out.adj_radius = 0.0f;

  float3 face_n = adj_center - center;
  float h = 0.5f * (dot(adj_center, adj_center) - dot(center, center) + radius * radius - adj_radius * adj_radius);
  float num = h - dot(origin, face_n);
  float dp = dot(direction, face_n);
  if (abs(dp) <= mf.eps) {
    return out;
  }

  float inv_dp = 1.0f / dp;
  float inv_dp2 = inv_dp * inv_dp;
  out.center = (-center + origin) * inv_dp + num * direction * inv_dp2;
  out.adj_center = (adj_center - origin) * inv_dp - num * direction * inv_dp2;
  out.radius = radius * inv_dp;
  out.adj_radius = -adj_radius * inv_dp;
  return out;
}

inline void foam_route_endpoint_grad(
    int endpoint_id,
    float dldt,
    bool use_far,
    uint cell,
    const device float* points,
    const device float* radii,
    const device int* adjacency,
    const device int* adjacency_offsets,
    float3 origin,
    float3 direction,
    constant FoamTileMetaI32& mi,
    constant FoamTileMetaF32& mf,
    device atomic_float* grad_points,
    device atomic_float* grad_radii) {
  if (endpoint_id == -2 || dldt == 0.0f) {
    return;
  }

  float3 center = foam_load3(points, cell);
  float radius = radii[cell];
  if (endpoint_id == -1) {
    FoamEndpointGrad g = foam_ray_sphere_endpoint_bwd(origin, direction, center, radius, use_far, mf);
    foam_atomic_add3(grad_points, cell * 3u, dldt * g.center);
    atomic_fetch_add_explicit(&grad_radii[cell], dldt * g.radius, memory_order_relaxed);
    return;
  }

  int edge_begin = adjacency_offsets[cell];
  int edge = edge_begin + endpoint_id;
  if (edge < edge_begin || edge >= adjacency_offsets[cell + 1u]) {
    return;
  }
  int neighbor_i = adjacency[edge];
  if (neighbor_i < 0 || neighbor_i >= mi.cell_count) {
    return;
  }

  uint neighbor = uint(neighbor_i);
  float3 adj_center = foam_load3(points, neighbor);
  float adj_radius = radii[neighbor];
  FoamEndpointGrad g = foam_power_face_endpoint_bwd(origin, direction, center, radius, adj_center, adj_radius, mf);
  foam_atomic_add3(grad_points, cell * 3u, dldt * g.center);
  atomic_fetch_add_explicit(&grad_radii[cell], dldt * g.radius, memory_order_relaxed);
  foam_atomic_add3(grad_points, neighbor * 3u, dldt * g.adj_center);
  atomic_fetch_add_explicit(&grad_radii[neighbor], dldt * g.adj_radius, memory_order_relaxed);
}

kernel void powerfoam_v2_count_visible_pinhole(
    const device float* points [[buffer(0)]],
    const device float* radii [[buffer(1)]],
    const device float* cameras [[buffer(2)]],
    constant FoamTileMetaI32& mi [[buffer(3)]],
    constant FoamTileMetaF32& mf [[buffer(4)]],
    device atomic_uint* tile_counts [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  uint total = uint(mi.batch_size * mi.cell_count);
  if (gid >= total) {
    return;
  }

  uint batch = gid / uint(mi.cell_count);
  uint cell = gid - batch * uint(mi.cell_count);
  FoamCamera cam = foam_load_camera(cameras, batch, mi);

  int tx0 = 0;
  int ty0 = 0;
  int tx1 = 0;
  int ty1 = 0;
  float sort_power = 0.0f;
  if (!foam_project_cell_bounds(points, radii, cell, cam, mi, mf, tx0, ty0, tx1, ty1, sort_power)) {
    return;
  }

  uint tile_base = batch * uint(mi.tile_count_per_image);
  for (int ty = ty0; ty <= ty1; ++ty) {
    for (int tx = tx0; tx <= tx1; ++tx) {
      uint tile = tile_base + uint(ty * mi.tiles_x + tx);
      atomic_fetch_add_explicit(&tile_counts[tile + 1u], 1u, memory_order_relaxed);
    }
  }
}

kernel void powerfoam_v2_write_visible_pinhole(
    const device float* points [[buffer(0)]],
    const device float* radii [[buffer(1)]],
    const device float* cameras [[buffer(2)]],
    const device ulong* tile_offsets [[buffer(3)]],
    device atomic_uint* tile_cursors [[buffer(4)]],
    constant FoamTileMetaI32& mi [[buffer(5)]],
    constant FoamTileMetaF32& mf [[buffer(6)]],
    device int* tile_cell_ids [[buffer(7)]],
    device ulong* sort_keys [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
  uint total = uint(mi.batch_size * mi.cell_count);
  if (gid >= total) {
    return;
  }

  uint batch = gid / uint(mi.cell_count);
  uint cell = gid - batch * uint(mi.cell_count);
  FoamCamera cam = foam_load_camera(cameras, batch, mi);

  int tx0 = 0;
  int ty0 = 0;
  int tx1 = 0;
  int ty1 = 0;
  float sort_power = 0.0f;
  if (!foam_project_cell_bounds(points, radii, cell, cam, mi, mf, tx0, ty0, tx1, ty1, sort_power)) {
    return;
  }

  uint tile_base = batch * uint(mi.tile_count_per_image);
  uint sortable_power = foam_float_sort_key(sort_power);
  for (int ty = ty0; ty <= ty1; ++ty) {
    for (int tx = tx0; tx <= tx1; ++tx) {
      uint tile = tile_base + uint(ty * mi.tiles_x + tx);
      uint local = atomic_fetch_add_explicit(&tile_cursors[tile], 1u, memory_order_relaxed);
      ulong dst = tile_offsets[tile] + ulong(local);
      if (dst < tile_offsets[tile + 1u]) {
        tile_cell_ids[dst] = int(cell);
        sort_keys[dst] = (ulong(tile) << 32u) | ulong(sortable_power);
      }
    }
  }
}

kernel void powerfoam_v2_forward_tiles(
    const device float* points [[buffer(0)]],
    const device float* radii [[buffer(1)]],
    const device float* densities [[buffer(2)]],
    const device float* features [[buffer(3)]],
    const device int* adjacency [[buffer(4)]],
    const device int* adjacency_offsets [[buffer(5)]],
    const device ulong* tile_offsets [[buffer(6)]],
    const device int* tile_cell_ids [[buffer(7)]],
    const device float* rays [[buffer(8)]],
    constant FoamTileMetaI32& mi [[buffer(9)]],
    constant FoamTileMetaF32& mf [[buffer(10)]],
    device int* tile_stop_counts [[buffer(11)]],
    device float* out_features [[buffer(12)]],
    device float* out_alpha [[buffer(13)]],
    device float* out_log_t [[buffer(14)]],
    uint tile_group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float tg_trans[FOAM_TILE_THREADS];

  uint batch = tile_group / uint(mi.tile_count_per_image);
  uint tile = tile_group - batch * uint(mi.tile_count_per_image);
  uint tile_y = tile / uint(mi.tiles_x);
  uint tile_x = tile - tile_y * uint(mi.tiles_x);
  uint local_x = tid % FOAM_TILE_WIDTH;
  uint local_y = tid / FOAM_TILE_WIDTH;
  uint x = tile_x * FOAM_TILE_WIDTH + local_x;
  uint y = tile_y * FOAM_TILE_WIDTH + local_y;
  bool valid = batch < uint(mi.batch_size) && y < uint(mi.height) && x < uint(mi.width);

  uint pixel = (batch * uint(mi.height) + y) * uint(mi.width) + x;
  uint out_base = pixel * uint(mi.feature_dim);
  if (valid) {
    for (uint f = 0u; f < uint(mi.feature_dim); ++f) {
      out_features[out_base + f] = 0.0f;
    }
    out_alpha[pixel] = 0.0f;
    out_log_t[pixel] = 0.0f;
  }

  float3 origin = float3(0.0f);
  float3 direction = float3(0.0f, 0.0f, 1.0f);
  if (valid) {
    uint ray_base = pixel * 6u;
    origin = float3(rays[ray_base + 0u], rays[ray_base + 1u], rays[ray_base + 2u]);
    direction = float3(rays[ray_base + 3u], rays[ray_base + 4u], rays[ray_base + 5u]);
    direction = direction / max(length(direction), mf.eps);
  }

  float log_t = 0.0f;
  ulong start = tile_offsets[tile_group];
  ulong end = tile_offsets[tile_group + 1u];
  int stop_count = int(end - start);

  for (ulong k = start; k < end; ++k) {
    tg_trans[tid] = valid ? exp(log_t) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = FOAM_TILE_THREADS / 2u; stride > 0u; stride >>= 1u) {
      if (tid < stride) {
        tg_trans[tid] = max(tg_trans[tid], tg_trans[tid + stride]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tg_trans[0] < mf.transmittance_threshold) {
      stop_count = int(k - start);
      break;
    }

    int cell_i = tile_cell_ids[k];
    if (valid && cell_i >= 0 && cell_i < mi.cell_count) {
      uint cell = uint(cell_i);
      float sigma = max(densities[cell], 0.0f);
      if (sigma > 0.0f) {
        FoamInterval interval = foam_clipped_cell_interval(
            points, radii, adjacency, adjacency_offsets, cell, origin, direction, mi, mf);
        float dt = interval.t_far - interval.t_near;
        if (interval.hit && dt > 0.0f) {
          float delta = -sigma * dt;
          float alpha = clamp(1.0f - exp(delta), 0.0f, mf.max_alpha);
          if (alpha >= mf.alpha_threshold) {
            float weight = exp(log_t) * alpha;
            uint feat_base = cell * uint(mi.feature_dim);
            for (uint f = 0u; f < uint(mi.feature_dim); ++f) {
              out_features[out_base + f] += weight * features[feat_base + f];
            }
            log_t += delta;
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0u) {
    tile_stop_counts[tile_group] = stop_count;
  }
  if (valid) {
    out_log_t[pixel] = log_t;
    out_alpha[pixel] = 1.0f - exp(log_t);
  }
}

kernel void powerfoam_v2_backward_tiles_global_atomic(
    const device float* points [[buffer(0)]],
    const device float* radii [[buffer(1)]],
    const device float* densities [[buffer(2)]],
    const device float* features [[buffer(3)]],
    const device int* adjacency [[buffer(4)]],
    const device int* adjacency_offsets [[buffer(5)]],
    const device ulong* tile_offsets [[buffer(6)]],
    const device int* tile_cell_ids [[buffer(7)]],
    const device int* tile_stop_counts [[buffer(8)]],
    const device float* rays [[buffer(9)]],
    const device float* out_log_t [[buffer(10)]],
    const device float* grad_out_features [[buffer(11)]],
    const device float* grad_out_alpha [[buffer(12)]],
    constant FoamTileMetaI32& mi [[buffer(13)]],
    constant FoamTileMetaF32& mf [[buffer(14)]],
    device atomic_float* grad_points [[buffer(15)]],
    device atomic_float* grad_radii [[buffer(16)]],
    device atomic_float* grad_densities [[buffer(17)]],
    device atomic_float* grad_features [[buffer(18)]],
    uint tile_group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  uint batch = tile_group / uint(mi.tile_count_per_image);
  uint tile = tile_group - batch * uint(mi.tile_count_per_image);
  uint tile_y = tile / uint(mi.tiles_x);
  uint tile_x = tile - tile_y * uint(mi.tiles_x);
  uint local_x = tid % FOAM_TILE_WIDTH;
  uint local_y = tid / FOAM_TILE_WIDTH;
  uint x = tile_x * FOAM_TILE_WIDTH + local_x;
  uint y = tile_y * FOAM_TILE_WIDTH + local_y;
  bool valid = batch < uint(mi.batch_size) && y < uint(mi.height) && x < uint(mi.width);
  if (!valid) {
    return;
  }

  uint pixel = (batch * uint(mi.height) + y) * uint(mi.width) + x;
  uint ray_base = pixel * 6u;
  float3 origin = float3(rays[ray_base + 0u], rays[ray_base + 1u], rays[ray_base + 2u]);
  float3 direction = float3(rays[ray_base + 3u], rays[ray_base + 4u], rays[ray_base + 5u]);
  direction = direction / max(length(direction), mf.eps);

  uint out_base = pixel * uint(mi.feature_dim);
  float log_t_after = out_log_t[pixel];
  float g_log_t_after = -grad_out_alpha[pixel] * exp(log_t_after);

  ulong start = tile_offsets[tile_group];
  int stop_count = tile_stop_counts[tile_group];
  for (int local_k = stop_count - 1; local_k >= 0; --local_k) {
    ulong k = start + ulong(local_k);
    int cell_i = tile_cell_ids[k];
    if (cell_i < 0 || cell_i >= mi.cell_count) {
      continue;
    }

    uint cell = uint(cell_i);
    float sigma = max(densities[cell], 0.0f);
    if (sigma <= 0.0f) {
      continue;
    }

    FoamInterval interval = foam_clipped_cell_interval(
        points, radii, adjacency, adjacency_offsets, cell, origin, direction, mi, mf);
    float dt = interval.t_far - interval.t_near;
    if (!interval.hit || dt <= 0.0f) {
      continue;
    }

    float delta = -sigma * dt;
    float log_t_before = log_t_after - delta;
    float trans_before = exp(log_t_before);
    float exp_delta = exp(delta);
    float alpha = clamp(1.0f - exp_delta, 0.0f, mf.max_alpha);
    if (alpha < mf.alpha_threshold) {
      log_t_after = log_t_before;
      continue;
    }

    uint feat_base = cell * uint(mi.feature_dim);
    float feature_dot_grad = 0.0f;
    for (uint f = 0u; f < uint(mi.feature_dim); ++f) {
      float go = grad_out_features[out_base + f];
      feature_dot_grad += go * features[feat_base + f];
      atomic_fetch_add_explicit(&grad_features[feat_base + f], go * trans_before * alpha, memory_order_relaxed);
    }

    float g_delta = g_log_t_after - feature_dot_grad * trans_before * exp_delta;
    atomic_fetch_add_explicit(&grad_densities[cell], g_delta * -dt, memory_order_relaxed);

    float g_t_near = g_delta * sigma;
    float g_t_far = g_delta * -sigma;
    foam_route_endpoint_grad(
        interval.near_id,
        g_t_near,
        false,
        cell,
        points,
        radii,
        adjacency,
        adjacency_offsets,
        origin,
        direction,
        mi,
        mf,
        grad_points,
        grad_radii);
    foam_route_endpoint_grad(
        interval.far_id,
        g_t_far,
        true,
        cell,
        points,
        radii,
        adjacency,
        adjacency_offsets,
        origin,
        direction,
        mi,
        mf,
        grad_points,
        grad_radii);

    float g_log_t_before = g_log_t_after + feature_dot_grad * trans_before * alpha;
    g_log_t_after = g_log_t_before;
    log_t_after = log_t_before;
  }
}
