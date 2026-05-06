#include <metal_stdlib>
using namespace metal;

#ifndef FOAM_STREAM_TILE_WIDTH
#define FOAM_STREAM_TILE_WIDTH 16u
#endif

#ifndef FOAM_STREAM_TILE_THREADS
#define FOAM_STREAM_TILE_THREADS 256u
#endif

#ifndef FOAM_RAYTRACE_MAX_EVENTS
#define FOAM_RAYTRACE_MAX_EVENTS 64
#endif

inline uint stream_tiles_x(constant FoamStreamMetaI32& mi) {
  return (uint(mi.width) + FOAM_STREAM_TILE_WIDTH - 1u) / FOAM_STREAM_TILE_WIDTH;
}

inline uint stream_tiles_y(constant FoamStreamMetaI32& mi) {
  return (uint(mi.height) + FOAM_STREAM_TILE_WIDTH - 1u) / FOAM_STREAM_TILE_WIDTH;
}

inline uint stream_tile_count_per_batch(constant FoamStreamMetaI32& mi) {
  return stream_tiles_x(mi) * stream_tiles_y(mi);
}

inline bool stream_bounds_overlap_tile(int4 bounds, uint tile_x, uint tile_y, constant FoamStreamMetaI32& mi) {
  if (bounds.z < bounds.x || bounds.w < bounds.y) {
    return false;
  }
  int x0 = int(tile_x * FOAM_STREAM_TILE_WIDTH);
  int y0 = int(tile_y * FOAM_STREAM_TILE_WIDTH);
  int x1 = min(mi.width - 1, x0 + int(FOAM_STREAM_TILE_WIDTH) - 1);
  int y1 = min(mi.height - 1, y0 + int(FOAM_STREAM_TILE_WIDTH) - 1);
  return !(bounds.z < x0 || bounds.x > x1 || bounds.w < y0 || bounds.y > y1);
}

inline void stream_local_feature_add(thread float* values, uint idx, float value) {
  if (idx < 128u) {
    values[idx] += value;
  }
}

inline void stream_local_feature_add3(thread float* values, uint base, float3 value) {
  stream_local_feature_add(values, base + 0u, value.x);
  stream_local_feature_add(values, base + 1u, value.y);
  stream_local_feature_add(values, base + 2u, value.z);
}

inline void stream_route_sv_color_grad_local(
    const device float* features,
    uint cell_feature_base,
    uint sbase,
    float3 view_dir,
    float3 g_color,
    constant FoamStreamMetaI32& mi,
    constant FoamStreamMetaF32& mf,
    thread float* local_grad_features) {
  float weights[16];
  float values[3];
  values[0] = 0.0f;
  values[1] = 0.0f;
  values[2] = 0.0f;
  float denom = 0.0f;
  uint dof = uint(mi.sv_dof);
  for (uint d = 0u; d < dof; ++d) {
    uint abase = stream_sv_axis_base(sbase, d);
    float3 raw_axis = float3(features[abase + 0u], features[abase + 1u], features[abase + 2u]);
    float temp = max(length(raw_axis), mf.eps);
    float3 axis = raw_axis / temp;
    float dist = max(length(view_dir - axis), mf.eps);
    float w = exp(-temp * dist);
    if (d < 16u) {
      weights[d] = w;
    }
    denom += w;
    uint rbase = stream_sv_rgb_base(sbase, d, mi);
    values[0] += w * features[rbase + 0u];
    values[1] += w * features[rbase + 1u];
    values[2] += w * features[rbase + 2u];
  }
  denom = max(denom, mf.eps);
  values[0] /= denom;
  values[1] /= denom;
  values[2] /= denom;

  float active0 = (values[0] + 0.5f > 0.0f) ? 1.0f : 0.0f;
  float active1 = (values[1] + 0.5f > 0.0f) ? 1.0f : 0.0f;
  float active2 = (values[2] + 0.5f > 0.0f) ? 1.0f : 0.0f;
  g_color *= float3(active0, active1, active2);

  for (uint d = 0u; d < dof; ++d) {
    uint abase = stream_sv_axis_base(sbase, d);
    uint rbase = stream_sv_rgb_base(sbase, d, mi);
    float3 raw_axis = float3(features[abase + 0u], features[abase + 1u], features[abase + 2u]);
    float raw_norm = length(raw_axis);
    float temp = max(raw_norm, mf.eps);
    float3 axis = raw_axis / temp;
    float3 diff = view_dir - axis;
    float dist = max(length(diff), mf.eps);
    float w = (d < 16u) ? weights[d] : exp(-temp * dist);

    stream_local_feature_add(local_grad_features, rbase + 0u - cell_feature_base, g_color.x * w / denom);
    stream_local_feature_add(local_grad_features, rbase + 1u - cell_feature_base, g_color.y * w / denom);
    stream_local_feature_add(local_grad_features, rbase + 2u - cell_feature_base, g_color.z * w / denom);

    float g_w = dot(g_color, float3(
        features[rbase + 0u] - values[0],
        features[rbase + 1u] - values[1],
        features[rbase + 2u] - values[2])) / denom;
    float g_e = g_w * w;
    float g_temp = -dist * g_e;
    float3 g_axis = temp * g_e * diff / dist;
    float3 g_raw = (g_axis - axis * dot(g_axis, axis)) / temp;
    if (raw_norm > mf.eps) {
      g_raw += g_temp * axis;
    }
    stream_local_feature_add3(local_grad_features, abase - cell_feature_base, g_raw);
  }
}

kernel void powerfoam_tiled_count_bounds_sorted(
    const device int* screen_bounds [[buffer(0)]],
    const device int* sorted_ids [[buffer(1)]],
    constant FoamStreamMetaI32& mi [[buffer(2)]],
    device int* tile_counts [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  uint tiles_per_batch = stream_tile_count_per_batch(mi);
  uint total_tiles = uint(mi.batch_size) * tiles_per_batch;
  if (gid >= total_tiles) {
    return;
  }
  if (gid == 0u) {
    tile_counts[0] = 0;
  }

  uint batch = gid / tiles_per_batch;
  uint tile = gid - batch * tiles_per_batch;
  uint tiles_x = stream_tiles_x(mi);
  uint tile_y = tile / tiles_x;
  uint tile_x = tile - tile_y * tiles_x;
  uint sorted_base = batch * uint(mi.cell_count);

  int count = 0;
  for (int order = 0; order < mi.cell_count; ++order) {
    int cell_i = sorted_ids[sorted_base + uint(order)];
    if (cell_i < 0 || cell_i >= mi.cell_count) {
      continue;
    }
    int4 bounds = stream_load_bounds(screen_bounds, batch, uint(cell_i), mi);
    if (stream_bounds_overlap_tile(bounds, tile_x, tile_y, mi)) {
      ++count;
    }
  }
  tile_counts[gid + 1u] = count;
}

kernel void powerfoam_tiled_write_bounds_sorted(
    const device int* screen_bounds [[buffer(0)]],
    const device int* sorted_ids [[buffer(1)]],
    const device int* tile_offsets [[buffer(2)]],
    constant FoamStreamMetaI32& mi [[buffer(3)]],
    device int* tile_cell_ids [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  uint tiles_per_batch = stream_tile_count_per_batch(mi);
  uint total_tiles = uint(mi.batch_size) * tiles_per_batch;
  if (gid >= total_tiles) {
    return;
  }

  uint batch = gid / tiles_per_batch;
  uint tile = gid - batch * tiles_per_batch;
  uint tiles_x = stream_tiles_x(mi);
  uint tile_y = tile / tiles_x;
  uint tile_x = tile - tile_y * tiles_x;
  uint sorted_base = batch * uint(mi.cell_count);
  int write = tile_offsets[gid];
  int end = tile_offsets[gid + 1u];

  for (int order = 0; order < mi.cell_count && write < end; ++order) {
    int cell_i = sorted_ids[sorted_base + uint(order)];
    if (cell_i < 0 || cell_i >= mi.cell_count) {
      continue;
    }
    int4 bounds = stream_load_bounds(screen_bounds, batch, uint(cell_i), mi);
    if (stream_bounds_overlap_tile(bounds, tile_x, tile_y, mi)) {
      tile_cell_ids[write] = cell_i;
      ++write;
    }
  }
}

kernel void powerfoam_tiled_emit_count_bounds(
    const device int* screen_bounds [[buffer(0)]],
    const device int* sorted_ids [[buffer(1)]],
    constant FoamStreamMetaI32& mi [[buffer(2)]],
    device atomic_int* tile_counts [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  uint total = uint(mi.batch_size * mi.cell_count);
  if (gid >= total) {
    return;
  }

  uint batch = gid / uint(mi.cell_count);
  uint order = gid - batch * uint(mi.cell_count);
  int cell_i = sorted_ids[batch * uint(mi.cell_count) + order];
  if (cell_i < 0 || cell_i >= mi.cell_count) {
    return;
  }

  int4 bounds = stream_load_bounds(screen_bounds, batch, uint(cell_i), mi);
  if (bounds.z < bounds.x || bounds.w < bounds.y) {
    return;
  }
  int bx0 = clamp(bounds.x, 0, mi.width - 1);
  int by0 = clamp(bounds.y, 0, mi.height - 1);
  int bx1 = clamp(bounds.z, 0, mi.width - 1);
  int by1 = clamp(bounds.w, 0, mi.height - 1);
  uint tx0 = uint(bx0) / FOAM_STREAM_TILE_WIDTH;
  uint ty0 = uint(by0) / FOAM_STREAM_TILE_WIDTH;
  uint tx1 = uint(bx1) / FOAM_STREAM_TILE_WIDTH;
  uint ty1 = uint(by1) / FOAM_STREAM_TILE_WIDTH;
  uint tiles_x = stream_tiles_x(mi);
  uint tile_base = batch * stream_tile_count_per_batch(mi);
  for (uint ty = ty0; ty <= ty1; ++ty) {
    for (uint tx = tx0; tx <= tx1; ++tx) {
      uint tile_group = tile_base + ty * tiles_x + tx;
      atomic_fetch_add_explicit(&tile_counts[tile_group + 1u], 1, memory_order_relaxed);
    }
  }
}

kernel void powerfoam_tiled_emit_write_bounds(
    const device int* screen_bounds [[buffer(0)]],
    const device int* sorted_ids [[buffer(1)]],
    const device int* tile_offsets [[buffer(2)]],
    device atomic_int* tile_cursors [[buffer(3)]],
    constant FoamStreamMetaI32& mi [[buffer(4)]],
    device long* sort_keys [[buffer(5)]],
    device int* tile_cell_ids [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  uint total = uint(mi.batch_size * mi.cell_count);
  if (gid >= total) {
    return;
  }

  uint batch = gid / uint(mi.cell_count);
  uint order = gid - batch * uint(mi.cell_count);
  int cell_i = sorted_ids[batch * uint(mi.cell_count) + order];
  if (cell_i < 0 || cell_i >= mi.cell_count) {
    return;
  }

  int4 bounds = stream_load_bounds(screen_bounds, batch, uint(cell_i), mi);
  if (bounds.z < bounds.x || bounds.w < bounds.y) {
    return;
  }
  int bx0 = clamp(bounds.x, 0, mi.width - 1);
  int by0 = clamp(bounds.y, 0, mi.height - 1);
  int bx1 = clamp(bounds.z, 0, mi.width - 1);
  int by1 = clamp(bounds.w, 0, mi.height - 1);
  uint tx0 = uint(bx0) / FOAM_STREAM_TILE_WIDTH;
  uint ty0 = uint(by0) / FOAM_STREAM_TILE_WIDTH;
  uint tx1 = uint(bx1) / FOAM_STREAM_TILE_WIDTH;
  uint ty1 = uint(by1) / FOAM_STREAM_TILE_WIDTH;
  uint tiles_x = stream_tiles_x(mi);
  uint tile_base = batch * stream_tile_count_per_batch(mi);
  for (uint ty = ty0; ty <= ty1; ++ty) {
    for (uint tx = tx0; tx <= tx1; ++tx) {
      uint tile_group = tile_base + ty * tiles_x + tx;
      int dst = atomic_fetch_add_explicit(&tile_cursors[tile_group], 1, memory_order_relaxed);
      if (dst >= tile_offsets[tile_group] && dst < tile_offsets[tile_group + 1u]) {
        sort_keys[dst] = long((ulong(tile_group) << 32u) | ulong(order));
        tile_cell_ids[dst] = cell_i;
      }
    }
  }
}

kernel void powerfoam_tiled_forward(
    const device float* points [[buffer(0)]],
    const device float* radii [[buffer(1)]],
    const device float* densities [[buffer(2)]],
    const device float* features [[buffer(3)]],
    const device int* adjacency [[buffer(4)]],
    const device int* adjacency_offsets [[buffer(5)]],
    const device float* adjacency_diff [[buffer(6)]],
    const device int* screen_bounds [[buffer(7)]],
    const device int* tile_offsets [[buffer(8)]],
    const device int* tile_cell_ids [[buffer(9)]],
    const device float* rays [[buffer(10)]],
    constant FoamStreamMetaI32& mi [[buffer(11)]],
    constant FoamStreamMetaF32& mf [[buffer(12)]],
    device float* out_features [[buffer(13)]],
    device float* out_alpha [[buffer(14)]],
    device float* out_normal_distance [[buffer(15)]],
    device float* out_log_t [[buffer(16)]],
    device int* out_tile_stop [[buffer(17)]],
    uint tile_group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float tg_trans[FOAM_STREAM_TILE_THREADS];

  uint tiles_per_batch = stream_tile_count_per_batch(mi);
  uint total_tiles = uint(mi.batch_size) * tiles_per_batch;
  if (tile_group >= total_tiles) {
    return;
  }

  uint batch = tile_group / tiles_per_batch;
  uint tile = tile_group - batch * tiles_per_batch;
  uint tiles_x = stream_tiles_x(mi);
  uint tile_y = tile / tiles_x;
  uint tile_x = tile - tile_y * tiles_x;
  uint local_x = tid % FOAM_STREAM_TILE_WIDTH;
  uint local_y = tid / FOAM_STREAM_TILE_WIDTH;
  uint x = tile_x * FOAM_STREAM_TILE_WIDTH + local_x;
  uint y = tile_y * FOAM_STREAM_TILE_WIDTH + local_y;
  bool valid = batch < uint(mi.batch_size) && y < uint(mi.height) && x < uint(mi.width);
  uint pixel = (batch * uint(mi.height) + y) * uint(mi.width) + x;
  uint out_base = pixel * uint(mi.output_dim);

  if (valid) {
    for (uint f = 0u; f < uint(mi.output_dim); ++f) {
      out_features[out_base + f] = 0.0f;
    }
    out_alpha[pixel] = 0.0f;
    out_normal_distance[pixel] = 0.0f;
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
  int start = tile_offsets[tile_group];
  int end = tile_offsets[tile_group + 1u];
  int stop_count = end - start;

  for (int local_order = 0; local_order < end - start; ++local_order) {
    tg_trans[tid] = valid ? exp(log_t) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = FOAM_STREAM_TILE_THREADS / 2u; stride > 0u; stride >>= 1u) {
      if (tid < stride) {
        tg_trans[tid] = max(tg_trans[tid], tg_trans[tid + stride]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tg_trans[0] < mf.transmittance_threshold) {
      stop_count = local_order;
      break;
    }

    int cell_i = tile_cell_ids[start + local_order];
    if (valid && cell_i >= 0 && cell_i < mi.cell_count) {
      uint cell = uint(cell_i);
      int4 bounds = stream_load_bounds(screen_bounds, batch, cell, mi);
      if (!(int(x) < bounds.x || int(y) < bounds.y || int(x) > bounds.z || int(y) > bounds.w)) {
        float3 center = stream_load3(points, cell);
        float radius = radii[cell];
        float sigma = max(densities[cell], 0.0f);
        if (sigma > 0.0f) {
          FoamStreamInterval interval = stream_clipped_cell_interval_diff(
              points, radii, adjacency, adjacency_offsets, adjacency_diff, cell, origin, direction, mi, mf);
          if (interval.hit &&
              (mi.feature_mode == 2 || mi.feature_mode == 3 || mi.feature_mode == 4 || stream_texel_has_height(mi))) {
            float3 surface_normal = stream_surface_normal(features, cell, mi);
            if (stream_texel_has_height(mi)) {
              interval.hit =
                  stream_clip_height_surface(features, cell, origin, direction, center, radius, surface_normal, mi, mf, interval);
            } else {
              interval.hit = stream_clip_surface(origin, direction, center, surface_normal, mf, interval);
            }
          }
          float dt = interval.t_far - interval.t_near;
          if (interval.hit && dt > 0.0f) {
            float delta = -sigma * dt;
            float alpha = clamp(1.0f - exp(delta), 0.0f, mf.max_alpha);
            if (alpha >= mf.alpha_threshold) {
              float weight = exp(log_t) * alpha;
              uint feat_base = cell * uint(mi.feature_dim);
              float3 local_coord = float3(0.0f);
              if (mi.feature_mode == 1 || mi.feature_mode == 2 || mi.feature_mode == 3 || mi.feature_mode == 4 ||
                  stream_texel_has_height(mi)) {
                float t_sample = 0.5f * (interval.t_near + interval.t_far);
                if (mi.feature_mode == 2 || mi.feature_mode == 3 || mi.feature_mode == 4) {
                  float3 n = stream_surface_normal(features, cell, mi);
                  t_sample = (dot(center, n) - dot(origin, n)) / dot(direction, n);
                } else if (stream_texel_has_height(mi)) {
                  t_sample = stream_height_texel_sample_t(interval);
                }
                local_coord = (origin + direction * t_sample - center) / max(radius, mf.eps);
              }
              if (mi.feature_mode == 4 || stream_texel_has_height(mi)) {
                uint c = uint(mi.output_dim);
                uint stride = stream_texel_stride(mi);
                uint color_offset = stream_texel_color_offset(mi);
                uint texel_count = stream_texel_count(mi);
                float3 tangent = stream_surface_tangent(features, cell, mi);
                float3 bitangent = stream_surface_bitangent(features, cell, mi);
                float2 texel_coord = float2(dot(local_coord, tangent), dot(local_coord, bitangent));
                float texel_weights[16];
                float denom = 0.0f;
                for (uint s = 0u; s < texel_count; ++s) {
                  uint sbase = feat_base + s * stride;
                  float2 diff = texel_coord - float2(features[sbase + 0u], features[sbase + 1u]);
                  float texel_weight = exp(-mf.texel_temperature * dot(diff, diff));
                  if (s < 16u) {
                    texel_weights[s] = texel_weight;
                  }
                  denom += texel_weight;
                }
                denom = max(denom, mf.eps);
                if (mi.feature_mode == 6) {
                  float3 numer3 = float3(0.0f);
                  for (uint s = 0u; s < texel_count; ++s) {
                    uint sbase = feat_base + s * stride;
                    float2 diff = texel_coord - float2(features[sbase + 0u], features[sbase + 1u]);
                    float texel_weight = (s < 16u) ? texel_weights[s] : exp(-mf.texel_temperature * dot(diff, diff));
                    float2 site = float2(features[sbase + 0u], features[sbase + 1u]);
                    float3 texel_world = center + radius * (site.x * tangent + site.y * bitangent);
                    float3 view_vec = texel_world - origin;
                    float3 view_dir = view_vec / max(length(view_vec), mf.eps);
                    numer3 += texel_weight * stream_sv_texel_color(features, sbase, view_dir, mi, mf);
                  }
                  float3 value3 = numer3 / denom;
                  out_features[out_base + 0u] += weight * value3.x;
                  out_features[out_base + 1u] += weight * value3.y;
                  out_features[out_base + 2u] += weight * value3.z;
                } else {
                  for (uint f = 0u; f < c; ++f) {
                    float numer = 0.0f;
                    for (uint s = 0u; s < texel_count; ++s) {
                      uint sbase = feat_base + s * stride;
                      float2 diff = texel_coord - float2(features[sbase + 0u], features[sbase + 1u]);
                      float texel_weight = exp(-mf.texel_temperature * dot(diff, diff));
                      numer += texel_weight * features[sbase + color_offset + f];
                    }
                    out_features[out_base + f] += weight * (numer / denom);
                  }
                }
              } else {
                for (uint f = 0u; f < uint(mi.output_dim); ++f) {
                  float value = features[feat_base + f];
                  if (mi.feature_mode == 1 || mi.feature_mode == 2 || mi.feature_mode == 3) {
                    uint c = uint(mi.output_dim);
                    value += local_coord.x * features[feat_base + c + f];
                    value += local_coord.y * features[feat_base + 2u * c + f];
                    value += local_coord.z * features[feat_base + 3u * c + f];
                  }
                  out_features[out_base + f] += weight * value;
                }
              }
              float3 n_aux = stream_surface_normal(features, cell, mi);
              float ndv = dot(n_aux, direction);
              if (ndv > 0.0f) {
                out_normal_distance[pixel] += ndv * ndv * weight;
              }
              log_t += delta;
            }
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0u) {
    out_tile_stop[tile_group] = stop_count;
  }
  if (valid) {
    out_log_t[pixel] = log_t;
    out_alpha[pixel] = 1.0f - exp(log_t);
  }
}

kernel void powerfoam_tiled_aux_forward(
    const device float* points [[buffer(0)]],
    const device float* radii [[buffer(1)]],
    const device float* densities [[buffer(2)]],
    const device float* features [[buffer(3)]],
    const device int* adjacency [[buffer(4)]],
    const device int* adjacency_offsets [[buffer(5)]],
    const device float* adjacency_diff [[buffer(6)]],
    const device int* screen_bounds [[buffer(7)]],
    const device int* tile_offsets [[buffer(8)]],
    const device int* tile_cell_ids [[buffer(9)]],
    const device float* rays [[buffer(10)]],
    constant FoamStreamMetaI32& mi [[buffer(11)]],
    constant FoamStreamMetaF32& mf [[buffer(12)]],
    const device float* target_features [[buffer(13)]],
    const device float* depth_quantiles [[buffer(14)]],
    device float* out_normal_distance [[buffer(15)]],
    device float* out_normal [[buffer(16)]],
    device float* out_depth_quantiles [[buffer(17)]],
    device atomic_float* out_contrib [[buffer(18)]],
    device atomic_float* out_point_error [[buffer(19)]],
    device atomic_int* out_visible [[buffer(20)]],
    uint tile_group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float tg_trans[FOAM_STREAM_TILE_THREADS];

  uint tiles_per_batch = stream_tile_count_per_batch(mi);
  uint total_tiles = uint(mi.batch_size) * tiles_per_batch;
  if (tile_group >= total_tiles) {
    return;
  }

  uint batch = tile_group / tiles_per_batch;
  uint tile = tile_group - batch * tiles_per_batch;
  uint tiles_x = stream_tiles_x(mi);
  uint tile_y = tile / tiles_x;
  uint tile_x = tile - tile_y * tiles_x;
  uint local_x = tid % FOAM_STREAM_TILE_WIDTH;
  uint local_y = tid / FOAM_STREAM_TILE_WIDTH;
  uint x = tile_x * FOAM_STREAM_TILE_WIDTH + local_x;
  uint y = tile_y * FOAM_STREAM_TILE_WIDTH + local_y;
  bool valid = batch < uint(mi.batch_size) && y < uint(mi.height) && x < uint(mi.width);
  uint pixel = (batch * uint(mi.height) + y) * uint(mi.width) + x;
  uint target_base = pixel * uint(mi.output_dim);

  if (valid) {
    out_normal_distance[pixel] = 0.0f;
    uint normal_base = pixel * 3u;
    out_normal[normal_base + 0u] = 0.0f;
    out_normal[normal_base + 1u] = 0.0f;
    out_normal[normal_base + 2u] = 0.0f;
    uint q_base = pixel * uint(mi.depth_quantile_count);
    for (uint q = 0u; q < uint(mi.depth_quantile_count); ++q) {
      out_depth_quantiles[q_base + q] = -1.0f;
    }
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
  int start = tile_offsets[tile_group];
  int end = tile_offsets[tile_group + 1u];
  float pixel_norm = 1.0f / float(max(mi.height * mi.width, 1));

  for (int local_order = 0; local_order < end - start; ++local_order) {
    tg_trans[tid] = valid ? exp(log_t) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = FOAM_STREAM_TILE_THREADS / 2u; stride > 0u; stride >>= 1u) {
      if (tid < stride) {
        tg_trans[tid] = max(tg_trans[tid], tg_trans[tid + stride]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tg_trans[0] < mf.transmittance_threshold) {
      break;
    }

    int cell_i = tile_cell_ids[start + local_order];
    if (valid && cell_i >= 0 && cell_i < mi.cell_count) {
      uint cell = uint(cell_i);
      int4 bounds = stream_load_bounds(screen_bounds, batch, cell, mi);
      if (!(int(x) < bounds.x || int(y) < bounds.y || int(x) > bounds.z || int(y) > bounds.w)) {
        float3 center = stream_load3(points, cell);
        float radius = radii[cell];
        float sigma = max(densities[cell], 0.0f);
        if (sigma > 0.0f) {
          FoamStreamInterval interval = stream_clipped_cell_interval_diff(
              points, radii, adjacency, adjacency_offsets, adjacency_diff, cell, origin, direction, mi, mf);
          if (interval.hit &&
              (mi.feature_mode == 2 || mi.feature_mode == 3 || mi.feature_mode == 4 || stream_texel_has_height(mi))) {
            float3 surface_normal = stream_surface_normal(features, cell, mi);
            if (stream_texel_has_height(mi)) {
              interval.hit =
                  stream_clip_height_surface(features, cell, origin, direction, center, radius, surface_normal, mi, mf, interval);
            } else {
              interval.hit = stream_clip_surface(origin, direction, center, surface_normal, mf, interval);
            }
          }
          float dt = interval.t_far - interval.t_near;
          if (interval.hit && dt > 0.0f) {
            float delta = -sigma * dt;
            float alpha = clamp(1.0f - exp(delta), 0.0f, mf.max_alpha);
            if (alpha >= mf.alpha_threshold) {
              float weight = exp(log_t) * alpha;
              float next_log_t = log_t + delta;
              uint q_base = pixel * uint(mi.depth_quantile_count);
              float trans_before = exp(log_t);
              float trans_after = exp(next_log_t);
              for (uint q = 0u; q < uint(mi.depth_quantile_count); ++q) {
                float quantile = clamp(depth_quantiles[q], 0.0f, 1.0f);
                float target_trans = max(1.0f - quantile, mf.eps);
                if (out_depth_quantiles[q_base + q] < 0.0f && trans_after < target_trans) {
                  out_depth_quantiles[q_base + q] =
                      interval.t_near + log(trans_before / target_trans) / max(sigma, mf.eps);
                }
              }
              uint feat_base = cell * uint(mi.feature_dim);
              float3 local_coord = float3(0.0f);
              if (mi.feature_mode == 1 || mi.feature_mode == 2 || mi.feature_mode == 3 || mi.feature_mode == 4 ||
                  stream_texel_has_height(mi)) {
                float t_sample = 0.5f * (interval.t_near + interval.t_far);
                if (mi.feature_mode == 2 || mi.feature_mode == 3 || mi.feature_mode == 4) {
                  float3 n = stream_surface_normal(features, cell, mi);
                  t_sample = (dot(center, n) - dot(origin, n)) / dot(direction, n);
                } else if (stream_texel_has_height(mi)) {
                  t_sample = stream_height_texel_sample_t(interval);
                }
                local_coord = (origin + direction * t_sample - center) / max(radius, mf.eps);
              }

              float point_error = 0.0f;
              if (mi.feature_mode == 4 || stream_texel_has_height(mi)) {
                uint c = uint(mi.output_dim);
                uint stride = stream_texel_stride(mi);
                uint color_offset = stream_texel_color_offset(mi);
                uint texel_count = stream_texel_count(mi);
                float3 tangent = stream_surface_tangent(features, cell, mi);
                float3 bitangent = stream_surface_bitangent(features, cell, mi);
                float2 texel_coord = float2(dot(local_coord, tangent), dot(local_coord, bitangent));
                float texel_weights[16];
                float denom = 0.0f;
                for (uint s = 0u; s < texel_count; ++s) {
                  uint sbase = feat_base + s * stride;
                  float2 diff = texel_coord - float2(features[sbase + 0u], features[sbase + 1u]);
                  float texel_weight = exp(-mf.texel_temperature * dot(diff, diff));
                  if (s < 16u) {
                    texel_weights[s] = texel_weight;
                  }
                  denom += texel_weight;
                }
                denom = max(denom, mf.eps);
                if (mi.feature_mode == 6) {
                  float3 numer3 = float3(0.0f);
                  for (uint s = 0u; s < texel_count; ++s) {
                    uint sbase = feat_base + s * stride;
                    float2 diff = texel_coord - float2(features[sbase + 0u], features[sbase + 1u]);
                    float texel_weight = (s < 16u) ? texel_weights[s] : exp(-mf.texel_temperature * dot(diff, diff));
                    float2 site = float2(features[sbase + 0u], features[sbase + 1u]);
                    float3 texel_world = center + radius * (site.x * tangent + site.y * bitangent);
                    float3 view_vec = texel_world - origin;
                    float3 view_dir = view_vec / max(length(view_vec), mf.eps);
                    numer3 += texel_weight * stream_sv_texel_color(features, sbase, view_dir, mi, mf);
                  }
                  float3 value3 = numer3 / denom;
                  point_error += abs(value3.x - target_features[target_base + 0u]);
                  point_error += abs(value3.y - target_features[target_base + 1u]);
                  point_error += abs(value3.z - target_features[target_base + 2u]);
                } else {
                  for (uint f = 0u; f < c; ++f) {
                    float numer = 0.0f;
                    for (uint s = 0u; s < texel_count; ++s) {
                      uint sbase = feat_base + s * stride;
                      float2 diff = texel_coord - float2(features[sbase + 0u], features[sbase + 1u]);
                      float texel_weight = exp(-mf.texel_temperature * dot(diff, diff));
                      numer += texel_weight * features[sbase + color_offset + f];
                    }
                    float value = numer / denom;
                    point_error += abs(value - target_features[target_base + f]);
                  }
                }
              } else {
                for (uint f = 0u; f < uint(mi.output_dim); ++f) {
                  float value = features[feat_base + f];
                  if (mi.feature_mode == 1 || mi.feature_mode == 2 || mi.feature_mode == 3) {
                    uint c = uint(mi.output_dim);
                    value += local_coord.x * features[feat_base + c + f];
                    value += local_coord.y * features[feat_base + 2u * c + f];
                    value += local_coord.z * features[feat_base + 3u * c + f];
                  }
                  point_error += abs(value - target_features[target_base + f]);
                }
              }

              float3 n_aux = stream_surface_normal(features, cell, mi);
              float ndv = dot(n_aux, direction);
              if (ndv > 0.0f) {
                out_normal_distance[pixel] += ndv * ndv * weight;
              }
              uint normal_base = pixel * 3u;
              out_normal[normal_base + 0u] += weight * n_aux.x;
              out_normal[normal_base + 1u] += weight * n_aux.y;
              out_normal[normal_base + 2u] += weight * n_aux.z;
              uint cell_slot = batch * uint(mi.cell_count) + cell;
              atomic_fetch_add_explicit(&out_contrib[cell_slot], weight * pixel_norm, memory_order_relaxed);
              atomic_fetch_add_explicit(&out_point_error[cell_slot], weight * point_error * pixel_norm, memory_order_relaxed);
              atomic_store_explicit(&out_visible[cell_slot], 1, memory_order_relaxed);
              log_t = next_log_t;
            }
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

kernel void powerfoam_tiled_backward_global_atomic(
    const device float* points [[buffer(0)]],
    const device float* radii [[buffer(1)]],
    const device float* densities [[buffer(2)]],
    const device float* features [[buffer(3)]],
    const device int* adjacency [[buffer(4)]],
    const device int* adjacency_offsets [[buffer(5)]],
    const device float* adjacency_diff [[buffer(6)]],
    const device int* screen_bounds [[buffer(7)]],
    const device int* tile_offsets [[buffer(8)]],
    const device int* tile_cell_ids [[buffer(9)]],
    const device int* tile_stop [[buffer(10)]],
    const device float* rays [[buffer(11)]],
    const device float* out_log_t [[buffer(12)]],
    const device float* grad_out_features [[buffer(13)]],
    const device float* grad_out_alpha [[buffer(14)]],
    const device float* grad_out_normal_distance [[buffer(15)]],
    constant FoamStreamMetaI32& mi [[buffer(16)]],
    constant FoamStreamMetaF32& mf [[buffer(17)]],
    device atomic_float* grad_points [[buffer(18)]],
    device atomic_float* grad_radii [[buffer(19)]],
    device atomic_float* grad_densities [[buffer(20)]],
    device atomic_float* grad_features [[buffer(21)]],
    uint tile_group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  uint tiles_per_batch = stream_tile_count_per_batch(mi);
  uint total_tiles = uint(mi.batch_size) * tiles_per_batch;
  if (tile_group >= total_tiles) {
    return;
  }

  uint batch = tile_group / tiles_per_batch;
  uint tile = tile_group - batch * tiles_per_batch;
  uint tiles_x = stream_tiles_x(mi);
  uint tile_y = tile / tiles_x;
  uint tile_x = tile - tile_y * tiles_x;
  uint local_x = tid % FOAM_STREAM_TILE_WIDTH;
  uint local_y = tid / FOAM_STREAM_TILE_WIDTH;
  uint x = tile_x * FOAM_STREAM_TILE_WIDTH + local_x;
  uint y = tile_y * FOAM_STREAM_TILE_WIDTH + local_y;
  bool valid = batch < uint(mi.batch_size) && y < uint(mi.height) && x < uint(mi.width);
  if (!valid) {
    return;
  }

  uint pixel = (batch * uint(mi.height) + y) * uint(mi.width) + x;
  uint out_base = pixel * uint(mi.output_dim);
  uint ray_base = pixel * 6u;
  float3 origin = float3(rays[ray_base + 0u], rays[ray_base + 1u], rays[ray_base + 2u]);
  float3 direction = float3(rays[ray_base + 3u], rays[ray_base + 4u], rays[ray_base + 5u]);
  direction = direction / max(length(direction), mf.eps);

  float log_t_after = out_log_t[pixel];
  float g_log_t_after = -grad_out_alpha[pixel] * exp(log_t_after);
  int start = tile_offsets[tile_group];
  int stop = clamp(tile_stop[tile_group], 0, tile_offsets[tile_group + 1u] - start);

  for (int local_order = stop - 1; local_order >= 0; --local_order) {
    int cell_i = tile_cell_ids[start + local_order];
    if (cell_i < 0 || cell_i >= mi.cell_count) {
      continue;
    }
    uint cell = uint(cell_i);
    int4 bounds = stream_load_bounds(screen_bounds, batch, cell, mi);
    if (int(x) < bounds.x || int(y) < bounds.y || int(x) > bounds.z || int(y) > bounds.w) {
      continue;
    }

    float3 center = stream_load3(points, cell);
    float radius = radii[cell];
    float sigma = max(densities[cell], 0.0f);
    if (sigma <= 0.0f) {
      continue;
    }

    FoamStreamInterval interval = stream_clipped_cell_interval_diff(
        points, radii, adjacency, adjacency_offsets, adjacency_diff, cell, origin, direction, mi, mf);
    FoamStreamInterval base_interval = interval;
    if (interval.hit &&
        (mi.feature_mode == 2 || mi.feature_mode == 3 || mi.feature_mode == 4 || stream_texel_has_height(mi))) {
      float3 surface_normal = stream_surface_normal(features, cell, mi);
      if (stream_texel_has_height(mi)) {
        interval.hit =
            stream_clip_height_surface(features, cell, origin, direction, center, radius, surface_normal, mi, mf, interval);
      } else {
        interval.hit = stream_clip_surface(origin, direction, center, surface_normal, mf, interval);
      }
    }
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
    float3 local_coord = float3(0.0f);
    float3 hit_offset = float3(0.0f);
    if (mi.feature_mode == 1 || mi.feature_mode == 2 || mi.feature_mode == 3 || mi.feature_mode == 4 ||
        stream_texel_has_height(mi)) {
      float t_sample = 0.5f * (interval.t_near + interval.t_far);
      if (mi.feature_mode == 2 || mi.feature_mode == 3 || mi.feature_mode == 4) {
        float3 n = stream_surface_normal(features, cell, mi);
        t_sample = (dot(center, n) - dot(origin, n)) / dot(direction, n);
      } else if (stream_texel_has_height(mi)) {
        t_sample = stream_height_texel_sample_t(interval);
      }
      hit_offset = origin + direction * t_sample - center;
      local_coord = hit_offset / max(radius, mf.eps);
    }
    float3 g_local = float3(0.0f);
    float weight = trans_before * alpha;
    if (mi.feature_mode == 4 || stream_texel_has_height(mi)) {
      uint c = uint(mi.output_dim);
      uint stride = stream_texel_stride(mi);
      uint color_offset = stream_texel_color_offset(mi);
      uint texel_count = stream_texel_count(mi);
      uint frame_base = stream_texel_frame_base(cell, mi);
      float3 tangent = stream_surface_tangent(features, cell, mi);
      float3 bitangent = stream_surface_bitangent(features, cell, mi);
      float2 texel_coord = float2(dot(local_coord, tangent), dot(local_coord, bitangent));
      float texel_weights[16];
      float denom = 0.0f;
      for (uint s = 0u; s < texel_count; ++s) {
        uint sbase = feat_base + s * stride;
        float2 diff = texel_coord - float2(features[sbase + 0u], features[sbase + 1u]);
        float texel_weight = exp(-mf.texel_temperature * dot(diff, diff));
        if (s < 16u) {
          texel_weights[s] = texel_weight;
        }
        denom += texel_weight;
      }
      denom = max(denom, mf.eps);

      if (mi.feature_mode == 6) {
        float3 sv_colors[16];
        float3 sv_raw_colors[16];
        float sv_denoms[16];
        float3 value3 = float3(0.0f);
        for (uint s = 0u; s < texel_count; ++s) {
          uint sbase = feat_base + s * stride;
          float2 diff = texel_coord - float2(features[sbase + 0u], features[sbase + 1u]);
          float texel_weight = (s < 16u) ? texel_weights[s] : exp(-mf.texel_temperature * dot(diff, diff));
          float2 site = float2(features[sbase + 0u], features[sbase + 1u]);
          float3 texel_world = center + radius * (site.x * tangent + site.y * bitangent);
          float3 view_vec = texel_world - origin;
          float3 view_dir = view_vec / max(length(view_vec), mf.eps);
          float sv_denom = 0.0f;
          float3 sv_raw = stream_sv_texel_color_raw(features, sbase, view_dir, mi, mf, sv_denom);
          float3 sv_color = max(sv_raw + 0.5f, float3(0.0f));
          if (s < 16u) {
            texel_weights[s] = texel_weight;
            sv_colors[s] = sv_color;
            sv_raw_colors[s] = sv_raw;
            sv_denoms[s] = sv_denom;
          }
          value3 += texel_weight * sv_color;
        }
        value3 /= denom;
        float3 go3 = float3(
            grad_out_features[out_base + 0u],
            grad_out_features[out_base + 1u],
            grad_out_features[out_base + 2u]);
        feature_dot_grad += dot(go3, value3);

        for (uint s = 0u; s < texel_count; ++s) {
          uint sbase = feat_base + s * stride;
          float2 site = float2(features[sbase + 0u], features[sbase + 1u]);
          float2 diff = texel_coord - site;
          float texel_weight = (s < 16u) ? texel_weights[s] : exp(-mf.texel_temperature * dot(diff, diff));
          float3 sv_color = (s < 16u) ? sv_colors[s] : float3(0.0f);
          float3 texel_world = center + radius * (site.x * tangent + site.y * bitangent);
          float3 view_vec = texel_world - origin;
          float3 view_dir = view_vec / max(length(view_vec), mf.eps);
          if (s >= 16u) {
            sv_color = stream_sv_texel_color(features, sbase, view_dir, mi, mf);
          }
          float3 sv_color_grad = weight * texel_weight / denom * go3;
          if (s < 16u) {
            stream_route_sv_color_grad_known_value(
                features, sbase, view_dir, sv_color_grad, sv_raw_colors[s], sv_denoms[s], mi, mf, grad_features);
          } else {
            stream_route_sv_color_grad(features, sbase, view_dir, sv_color_grad, mi, mf, grad_features);
          }
          float g_w = dot(go3, weight * (sv_color - value3) / denom);
          float2 g_diff = g_w * (-2.0f * mf.texel_temperature * texel_weight) * diff;
          g_local += g_diff.x * tangent + g_diff.y * bitangent;
          stream_atomic_add3(grad_features, frame_base + 3u, g_diff.x * local_coord);
          stream_atomic_add3(grad_features, frame_base + 6u, g_diff.y * local_coord);
          atomic_fetch_add_explicit(&grad_features[sbase + 0u], -g_diff.x, memory_order_relaxed);
          atomic_fetch_add_explicit(&grad_features[sbase + 1u], -g_diff.y, memory_order_relaxed);
        }
      } else {
        for (uint f = 0u; f < c; ++f) {
          float numer = 0.0f;
          for (uint s = 0u; s < texel_count; ++s) {
            uint sbase = feat_base + s * stride;
            float2 diff = texel_coord - float2(features[sbase + 0u], features[sbase + 1u]);
            float texel_weight = exp(-mf.texel_temperature * dot(diff, diff));
            float2 site = float2(features[sbase + 0u], features[sbase + 1u]);
            numer += texel_weight * features[sbase + color_offset + f];
          }
          float value = numer / denom;
          float go = grad_out_features[out_base + f];
          feature_dot_grad += go * value;
          for (uint s = 0u; s < texel_count; ++s) {
            uint sbase = feat_base + s * stride;
            float2 diff = texel_coord - float2(features[sbase + 0u], features[sbase + 1u]);
            float texel_weight = exp(-mf.texel_temperature * dot(diff, diff));
            atomic_fetch_add_explicit(
                &grad_features[sbase + color_offset + f], go * weight * texel_weight / denom, memory_order_relaxed);
          }
        }
      }
    } else {
      for (uint f = 0u; f < uint(mi.output_dim); ++f) {
        float go = grad_out_features[out_base + f];
        float value = features[feat_base + f];
        if (mi.feature_mode == 1 || mi.feature_mode == 2 || mi.feature_mode == 3) {
          uint c = uint(mi.output_dim);
          float cx = features[feat_base + c + f];
          float cy = features[feat_base + 2u * c + f];
          float cz = features[feat_base + 3u * c + f];
          value += local_coord.x * cx + local_coord.y * cy + local_coord.z * cz;
          atomic_fetch_add_explicit(&grad_features[feat_base + c + f], go * weight * local_coord.x, memory_order_relaxed);
          atomic_fetch_add_explicit(&grad_features[feat_base + 2u * c + f], go * weight * local_coord.y, memory_order_relaxed);
          atomic_fetch_add_explicit(&grad_features[feat_base + 3u * c + f], go * weight * local_coord.z, memory_order_relaxed);
          g_local += go * weight * float3(cx, cy, cz);
        }
        feature_dot_grad += go * value;
        atomic_fetch_add_explicit(&grad_features[feat_base + f], go * weight, memory_order_relaxed);
      }
    }

    float g_delta = g_log_t_after - feature_dot_grad * trans_before * exp_delta;
    float g_normal_distance = grad_out_normal_distance[pixel];
    if (g_normal_distance != 0.0f) {
      float3 n_aux = stream_surface_normal(features, cell, mi);
      float ndv = dot(n_aux, direction);
      if (ndv > 0.0f) {
        feature_dot_grad += g_normal_distance * ndv * ndv;
        g_delta -= g_normal_distance * ndv * ndv * trans_before * exp_delta;
        uint normal_base = stream_surface_normal_base(cell, mi);
        if (normal_base != 0xffffffffu) {
          stream_atomic_add3(grad_features, normal_base, g_normal_distance * weight * 2.0f * ndv * direction);
        }
      }
    }
    atomic_fetch_add_explicit(&grad_densities[cell], g_delta * -dt, memory_order_relaxed);

    float g_t_near = g_delta * sigma;
    float g_t_far = g_delta * -sigma;
    if (mi.feature_mode == 1) {
      float safe_radius = max(radius, mf.eps);
      float3 g_hit = g_local / safe_radius;
      float g_mid = dot(g_hit, direction);
      g_t_near += 0.5f * g_mid;
      g_t_far += 0.5f * g_mid;
      stream_atomic_add3(grad_points, cell * 3u, -g_local / safe_radius);
      atomic_fetch_add_explicit(
          &grad_radii[cell], -dot(g_local, hit_offset) / max(safe_radius * safe_radius, mf.eps), memory_order_relaxed);
    } else if (mi.feature_mode == 2 || mi.feature_mode == 3 || mi.feature_mode == 4) {
      float safe_radius = max(radius, mf.eps);
      float3 g_hit = g_local / safe_radius;
      float3 n = stream_surface_normal(features, cell, mi);
      float dp = dot(direction, n);
      if (abs(dp) > mf.eps) {
        float g_sample_t = dot(g_hit, direction);
        stream_atomic_add3(grad_points, cell * 3u, g_sample_t * n / dp - g_local / safe_radius);
        if (mi.feature_mode == 3 || mi.feature_mode == 4) {
          uint normal_base = cell * uint(mi.feature_dim) + 4u * uint(mi.output_dim);
          if (mi.feature_mode == 4) {
            normal_base = stream_texel_frame_base(cell, mi);
          }
          float num = dot(center - origin, n);
          float inv_dp = 1.0f / dp;
          float3 grad_n = (center - origin) * inv_dp - num * direction * inv_dp * inv_dp;
          stream_atomic_add3(grad_features, normal_base, g_sample_t * grad_n);
        }
      } else {
        stream_atomic_add3(grad_points, cell * 3u, -g_local / safe_radius);
      }
      atomic_fetch_add_explicit(
          &grad_radii[cell], -dot(g_local, hit_offset) / max(safe_radius * safe_radius, mf.eps), memory_order_relaxed);
    } else if (stream_texel_has_height(mi)) {
      float safe_radius = max(radius, mf.eps);
      float3 g_hit = g_local / safe_radius;
      float g_sample_t = dot(g_hit, direction);
      if (stream_height_texel_sample_uses_far(interval)) {
        g_t_far += g_sample_t;
      } else {
        g_t_near += g_sample_t;
      }
      stream_atomic_add3(grad_points, cell * 3u, -g_local / safe_radius);
      atomic_fetch_add_explicit(
          &grad_radii[cell], -dot(g_local, hit_offset) / max(safe_radius * safe_radius, mf.eps), memory_order_relaxed);
    }
    if (interval.near_id == -4) {
      stream_route_height_surface_endpoint_grad(
          base_interval,
          g_t_near,
          cell,
          points,
          radii,
          adjacency,
          adjacency_offsets,
          features,
          origin,
          direction,
          mi,
          mf,
          grad_features,
          grad_points,
          grad_radii);
    } else {
      stream_route_endpoint_grad(
          interval.near_id,
          g_t_near,
          false,
          cell,
          points,
          radii,
          adjacency,
          adjacency_offsets,
          features,
          origin,
          direction,
          mi,
          mf,
          grad_features,
          grad_points,
          grad_radii);
    }
    if (interval.far_id == -4) {
      stream_route_height_surface_endpoint_grad(
          base_interval,
          g_t_far,
          cell,
          points,
          radii,
          adjacency,
          adjacency_offsets,
          features,
          origin,
          direction,
          mi,
          mf,
          grad_features,
          grad_points,
          grad_radii);
    } else {
      stream_route_endpoint_grad(
          interval.far_id,
          g_t_far,
          true,
          cell,
          points,
          radii,
          adjacency,
          adjacency_offsets,
          features,
          origin,
          direction,
          mi,
          mf,
          grad_features,
          grad_points,
          grad_radii);
    }

    float g_log_t_before = g_log_t_after + feature_dot_grad * trans_before * alpha;
    g_log_t_after = g_log_t_before;
    log_t_after = log_t_before;
  }
}

kernel void powerfoam_tiled_backward_constant_reduced(
    const device float* points [[buffer(0)]],
    const device float* radii [[buffer(1)]],
    const device float* densities [[buffer(2)]],
    const device float* features [[buffer(3)]],
    const device int* adjacency [[buffer(4)]],
    const device int* adjacency_offsets [[buffer(5)]],
    const device float* adjacency_diff [[buffer(6)]],
    const device int* screen_bounds [[buffer(7)]],
    const device int* tile_offsets [[buffer(8)]],
    const device int* tile_cell_ids [[buffer(9)]],
    const device int* tile_stop [[buffer(10)]],
    const device float* rays [[buffer(11)]],
    const device float* out_log_t [[buffer(12)]],
    const device float* grad_out_features [[buffer(13)]],
    const device float* grad_out_alpha [[buffer(14)]],
    const device float* grad_out_normal_distance [[buffer(15)]],
    constant FoamStreamMetaI32& mi [[buffer(16)]],
    constant FoamStreamMetaF32& mf [[buffer(17)]],
    device atomic_float* grad_points [[buffer(18)]],
    device atomic_float* grad_radii [[buffer(19)]],
    device atomic_float* grad_densities [[buffer(20)]],
    device atomic_float* grad_features [[buffer(21)]],
    uint tile_group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float tg_reduce[FOAM_STREAM_TILE_THREADS];

  uint tiles_per_batch = stream_tile_count_per_batch(mi);
  uint total_tiles = uint(mi.batch_size) * tiles_per_batch;
  if (tile_group >= total_tiles || mi.feature_mode != 0) {
    return;
  }

  uint batch = tile_group / tiles_per_batch;
  uint tile = tile_group - batch * tiles_per_batch;
  uint tiles_x = stream_tiles_x(mi);
  uint tile_y = tile / tiles_x;
  uint tile_x = tile - tile_y * tiles_x;
  uint local_x = tid % FOAM_STREAM_TILE_WIDTH;
  uint local_y = tid / FOAM_STREAM_TILE_WIDTH;
  uint x = tile_x * FOAM_STREAM_TILE_WIDTH + local_x;
  uint y = tile_y * FOAM_STREAM_TILE_WIDTH + local_y;
  bool valid = batch < uint(mi.batch_size) && y < uint(mi.height) && x < uint(mi.width);

  uint pixel = (batch * uint(mi.height) + y) * uint(mi.width) + x;
  uint out_base = pixel * uint(mi.output_dim);
  uint ray_base = pixel * 6u;
  float3 origin = float3(0.0f);
  float3 direction = float3(0.0f, 0.0f, 1.0f);
  float log_t_after = 0.0f;
  float g_log_t_after = 0.0f;
  if (valid) {
    origin = float3(rays[ray_base + 0u], rays[ray_base + 1u], rays[ray_base + 2u]);
    direction = float3(rays[ray_base + 3u], rays[ray_base + 4u], rays[ray_base + 5u]);
    direction = direction / max(length(direction), mf.eps);
    log_t_after = out_log_t[pixel];
    g_log_t_after = -grad_out_alpha[pixel] * exp(log_t_after);
  }

  int start = tile_offsets[tile_group];
  int stop = clamp(tile_stop[tile_group], 0, tile_offsets[tile_group + 1u] - start);

  for (int local_order = stop - 1; local_order >= 0; --local_order) {
    int cell_i = tile_cell_ids[start + local_order];
    bool active = valid && cell_i >= 0 && cell_i < mi.cell_count;
    uint cell = uint(max(cell_i, 0));
    int4 bounds = int4(1, 1, 0, 0);
    if (active) {
      bounds = stream_load_bounds(screen_bounds, batch, cell, mi);
      active = !(int(x) < bounds.x || int(y) < bounds.y || int(x) > bounds.z || int(y) > bounds.w);
    }

    float sigma = 0.0f;
    float dt = 0.0f;
    float delta = 0.0f;
    float log_t_before = log_t_after;
    float trans_before = 0.0f;
    float exp_delta = 1.0f;
    float alpha = 0.0f;
    float feature_dot_grad = 0.0f;
    FoamStreamInterval interval;
    interval.hit = false;
    interval.t_near = 0.0f;
    interval.t_far = 0.0f;
    interval.near_id = -2;
    interval.far_id = -2;

    if (active) {
      sigma = max(densities[cell], 0.0f);
      if (sigma > 0.0f) {
        interval =
            stream_clipped_cell_interval_diff(points, radii, adjacency, adjacency_offsets, adjacency_diff, cell, origin, direction, mi, mf);
        dt = interval.t_far - interval.t_near;
        active = interval.hit && dt > 0.0f;
      } else {
        active = false;
      }
    }
    if (active) {
      delta = -sigma * dt;
      log_t_before = log_t_after - delta;
      trans_before = exp(log_t_before);
      exp_delta = exp(delta);
      alpha = clamp(1.0f - exp_delta, 0.0f, mf.max_alpha);
      active = alpha >= mf.alpha_threshold;
    }
    if (active) {
      uint feat_base = cell * uint(mi.feature_dim);
      for (uint f = 0u; f < uint(mi.output_dim); ++f) {
        feature_dot_grad += grad_out_features[out_base + f] * features[feat_base + f];
      }
    }

    uint feat_base_reduce = uint(max(cell_i, 0)) * uint(mi.feature_dim);
    float weight = active ? trans_before * alpha : 0.0f;
    for (uint f = 0u; f < uint(mi.output_dim); ++f) {
      tg_reduce[tid] = active ? grad_out_features[out_base + f] * weight : 0.0f;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint stride = FOAM_STREAM_TILE_THREADS / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
          tg_reduce[tid] += tg_reduce[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if (tid == 0u && cell_i >= 0 && cell_i < mi.cell_count && tg_reduce[0] != 0.0f) {
        atomic_fetch_add_explicit(&grad_features[feat_base_reduce + f], tg_reduce[0], memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float g_delta = active ? g_log_t_after - feature_dot_grad * trans_before * exp_delta : 0.0f;
    tg_reduce[tid] = active ? g_delta * -dt : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = FOAM_STREAM_TILE_THREADS / 2u; stride > 0u; stride >>= 1u) {
      if (tid < stride) {
        tg_reduce[tid] += tg_reduce[tid + stride];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u && cell_i >= 0 && cell_i < mi.cell_count && tg_reduce[0] != 0.0f) {
      atomic_fetch_add_explicit(&grad_densities[uint(cell_i)], tg_reduce[0], memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active) {
      float g_t_near = g_delta * sigma;
      float g_t_far = g_delta * -sigma;
      stream_route_endpoint_grad(
          interval.near_id,
          g_t_near,
          false,
          cell,
          points,
          radii,
          adjacency,
          adjacency_offsets,
          features,
          origin,
          direction,
          mi,
          mf,
          grad_features,
          grad_points,
          grad_radii);
      stream_route_endpoint_grad(
          interval.far_id,
          g_t_far,
          true,
          cell,
          points,
          radii,
          adjacency,
          adjacency_offsets,
          features,
          origin,
          direction,
          mi,
          mf,
          grad_features,
          grad_points,
          grad_radii);

      float g_log_t_before = g_log_t_after + feature_dot_grad * trans_before * alpha;
      g_log_t_after = g_log_t_before;
      log_t_after = log_t_before;
    } else if (valid && sigma > 0.0f && interval.hit && dt > 0.0f) {
      log_t_after = log_t_before;
    }
  }
}

kernel void powerfoam_tiled_backward_height_sv_feature_reduced(
    const device float* points [[buffer(0)]],
    const device float* radii [[buffer(1)]],
    const device float* densities [[buffer(2)]],
    const device float* features [[buffer(3)]],
    const device int* adjacency [[buffer(4)]],
    const device int* adjacency_offsets [[buffer(5)]],
    const device float* adjacency_diff [[buffer(6)]],
    const device int* screen_bounds [[buffer(7)]],
    const device int* tile_offsets [[buffer(8)]],
    const device int* tile_cell_ids [[buffer(9)]],
    const device int* tile_stop [[buffer(10)]],
    const device float* rays [[buffer(11)]],
    const device float* out_log_t [[buffer(12)]],
    const device float* grad_out_features [[buffer(13)]],
    const device float* grad_out_alpha [[buffer(14)]],
    constant FoamStreamMetaI32& mi [[buffer(15)]],
    constant FoamStreamMetaF32& mf [[buffer(16)]],
    device atomic_float* grad_points [[buffer(17)]],
    device atomic_float* grad_radii [[buffer(18)]],
    device atomic_float* grad_densities [[buffer(19)]],
    device atomic_float* grad_features [[buffer(20)]],
    uint tile_group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float tg_reduce[FOAM_STREAM_TILE_THREADS];

  uint tiles_per_batch = stream_tile_count_per_batch(mi);
  uint total_tiles = uint(mi.batch_size) * tiles_per_batch;
  if (tile_group >= total_tiles || mi.feature_mode != 6 || mi.feature_dim > 128) {
    return;
  }

  uint batch = tile_group / tiles_per_batch;
  uint tile = tile_group - batch * tiles_per_batch;
  uint tiles_x = stream_tiles_x(mi);
  uint tile_y = tile / tiles_x;
  uint tile_x = tile - tile_y * tiles_x;
  uint local_x = tid % FOAM_STREAM_TILE_WIDTH;
  uint local_y = tid / FOAM_STREAM_TILE_WIDTH;
  uint x = tile_x * FOAM_STREAM_TILE_WIDTH + local_x;
  uint y = tile_y * FOAM_STREAM_TILE_WIDTH + local_y;
  bool valid = batch < uint(mi.batch_size) && y < uint(mi.height) && x < uint(mi.width);

  uint pixel = (batch * uint(mi.height) + y) * uint(mi.width) + x;
  uint out_base = pixel * uint(mi.output_dim);
  uint ray_base = pixel * 6u;
  float3 origin = float3(0.0f);
  float3 direction = float3(0.0f, 0.0f, 1.0f);
  float log_t_after = 0.0f;
  float g_log_t_after = 0.0f;
  if (valid) {
    origin = float3(rays[ray_base + 0u], rays[ray_base + 1u], rays[ray_base + 2u]);
    direction = float3(rays[ray_base + 3u], rays[ray_base + 4u], rays[ray_base + 5u]);
    direction = direction / max(length(direction), mf.eps);
    log_t_after = out_log_t[pixel];
    g_log_t_after = -grad_out_alpha[pixel] * exp(log_t_after);
  }

  int start = tile_offsets[tile_group];
  int stop = clamp(tile_stop[tile_group], 0, tile_offsets[tile_group + 1u] - start);

  for (int local_order = stop - 1; local_order >= 0; --local_order) {
    float local_feature_grad[128];
    for (uint lf = 0u; lf < 128u; ++lf) {
      local_feature_grad[lf] = 0.0f;
    }

    int cell_i = tile_cell_ids[start + local_order];
    bool active = valid && cell_i >= 0 && cell_i < mi.cell_count;
    uint cell = uint(max(cell_i, 0));
    int4 bounds = int4(1, 1, 0, 0);
    if (active) {
      bounds = stream_load_bounds(screen_bounds, batch, cell, mi);
      active = !(int(x) < bounds.x || int(y) < bounds.y || int(x) > bounds.z || int(y) > bounds.w);
    }

    float3 center = float3(0.0f);
    float radius = 0.0f;
    float sigma = 0.0f;
    float dt = 0.0f;
    float delta = 0.0f;
    float log_t_before = log_t_after;
    float trans_before = 0.0f;
    float exp_delta = 1.0f;
    float alpha = 0.0f;
    float feature_dot_grad = 0.0f;
    float3 local_coord = float3(0.0f);
    float3 hit_offset = float3(0.0f);
    float3 g_local = float3(0.0f);
    FoamStreamInterval interval;
    interval.hit = false;
    interval.t_near = 0.0f;
    interval.t_far = 0.0f;
    interval.near_id = -2;
    interval.far_id = -2;

    if (active) {
      center = stream_load3(points, cell);
      radius = radii[cell];
      sigma = max(densities[cell], 0.0f);
      if (sigma > 0.0f) {
        interval =
            stream_clipped_cell_interval_diff(points, radii, adjacency, adjacency_offsets, adjacency_diff, cell, origin, direction, mi, mf);
        if (interval.hit) {
          float3 surface_normal = stream_surface_normal(features, cell, mi);
          interval.hit =
              stream_clip_height_surface(features, cell, origin, direction, center, radius, surface_normal, mi, mf, interval);
        }
        dt = interval.t_far - interval.t_near;
        active = interval.hit && dt > 0.0f;
      } else {
        active = false;
      }
    }
    if (active) {
      delta = -sigma * dt;
      log_t_before = log_t_after - delta;
      trans_before = exp(log_t_before);
      exp_delta = exp(delta);
      alpha = clamp(1.0f - exp_delta, 0.0f, mf.max_alpha);
      active = alpha >= mf.alpha_threshold;
    }

    uint feat_base = uint(max(cell_i, 0)) * uint(mi.feature_dim);
    float weight = active ? trans_before * alpha : 0.0f;
    if (active) {
      hit_offset = origin + direction * stream_height_texel_sample_t(interval) - center;
      local_coord = hit_offset / max(radius, mf.eps);
      uint c = uint(mi.output_dim);
      uint stride = stream_texel_stride(mi);
      uint texel_count = stream_texel_count(mi);
      uint frame_base = stream_texel_frame_base(cell, mi);
      float3 tangent = stream_surface_tangent(features, cell, mi);
      float3 bitangent = stream_surface_bitangent(features, cell, mi);
      float2 texel_coord = float2(dot(local_coord, tangent), dot(local_coord, bitangent));
      float denom = 0.0f;
      for (uint s = 0u; s < texel_count; ++s) {
        uint sbase = feat_base + s * stride;
        float2 diff = texel_coord - float2(features[sbase + 0u], features[sbase + 1u]);
        denom += exp(-mf.texel_temperature * dot(diff, diff));
      }
      denom = max(denom, mf.eps);

      for (uint f = 0u; f < c; ++f) {
        float numer = 0.0f;
        for (uint s = 0u; s < texel_count; ++s) {
          uint sbase = feat_base + s * stride;
          float2 diff = texel_coord - float2(features[sbase + 0u], features[sbase + 1u]);
          float texel_weight = exp(-mf.texel_temperature * dot(diff, diff));
          float2 site = float2(features[sbase + 0u], features[sbase + 1u]);
          float3 texel_world = center + radius * (site.x * tangent + site.y * bitangent);
          float3 view_vec = texel_world - origin;
          float3 view_dir = view_vec / max(length(view_vec), mf.eps);
          numer += texel_weight * stream_sv_texel_color_component(features, sbase, view_dir, f, mi, mf);
        }
        float value = numer / denom;
        float go = grad_out_features[out_base + f];
        feature_dot_grad += go * value;
      }

      for (uint s = 0u; s < texel_count; ++s) {
        uint sbase = feat_base + s * stride;
        float2 site = float2(features[sbase + 0u], features[sbase + 1u]);
        float2 diff = texel_coord - site;
        float texel_weight = exp(-mf.texel_temperature * dot(diff, diff));
        float3 texel_world = center + radius * (site.x * tangent + site.y * bitangent);
        float3 view_vec = texel_world - origin;
        float3 view_dir = view_vec / max(length(view_vec), mf.eps);
        float3 sv_color = stream_sv_texel_color(features, sbase, view_dir, mi, mf);
        float3 sv_color_grad = weight * texel_weight / denom *
            float3(grad_out_features[out_base + 0u], grad_out_features[out_base + 1u], grad_out_features[out_base + 2u]);
        stream_route_sv_color_grad_local(features, feat_base, sbase, view_dir, sv_color_grad, mi, mf, local_feature_grad);

        float g_w = 0.0f;
        for (uint f = 0u; f < c; ++f) {
          float numer = 0.0f;
          for (uint t = 0u; t < texel_count; ++t) {
            uint tbase = feat_base + t * stride;
            float2 tdiff = texel_coord - float2(features[tbase + 0u], features[tbase + 1u]);
            float tw = exp(-mf.texel_temperature * dot(tdiff, tdiff));
            float2 tsite = float2(features[tbase + 0u], features[tbase + 1u]);
            float3 t_texel_world = center + radius * (tsite.x * tangent + tsite.y * bitangent);
            float3 t_view_vec = t_texel_world - origin;
            float3 t_view_dir = t_view_vec / max(length(t_view_vec), mf.eps);
            numer += tw * stream_sv_texel_color_component(features, tbase, t_view_dir, f, mi, mf);
          }
          float value = numer / denom;
          float texel_value = sv_color[f];
          g_w += grad_out_features[out_base + f] * weight * (texel_value - value) / denom;
        }
        float2 g_diff = g_w * (-2.0f * mf.texel_temperature * texel_weight) * diff;
        g_local += g_diff.x * tangent + g_diff.y * bitangent;
        stream_local_feature_add3(local_feature_grad, frame_base + 3u - feat_base, g_diff.x * local_coord);
        stream_local_feature_add3(local_feature_grad, frame_base + 6u - feat_base, g_diff.y * local_coord);
        stream_local_feature_add(local_feature_grad, sbase + 0u - feat_base, -g_diff.x);
        stream_local_feature_add(local_feature_grad, sbase + 1u - feat_base, -g_diff.y);
      }
    }

    float g_delta = active ? g_log_t_after - feature_dot_grad * trans_before * exp_delta : 0.0f;
    tg_reduce[tid] = active ? g_delta * -dt : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = FOAM_STREAM_TILE_THREADS / 2u; stride > 0u; stride >>= 1u) {
      if (tid < stride) {
        tg_reduce[tid] += tg_reduce[tid + stride];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u && cell_i >= 0 && cell_i < mi.cell_count && tg_reduce[0] != 0.0f) {
      atomic_fetch_add_explicit(&grad_densities[uint(cell_i)], tg_reduce[0], memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint f = 0u; f < uint(mi.feature_dim); ++f) {
      tg_reduce[tid] = active ? local_feature_grad[f] : 0.0f;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint stride = FOAM_STREAM_TILE_THREADS / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
          tg_reduce[tid] += tg_reduce[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if (tid == 0u && cell_i >= 0 && cell_i < mi.cell_count && tg_reduce[0] != 0.0f) {
        atomic_fetch_add_explicit(&grad_features[feat_base + f], tg_reduce[0], memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
      float g_t_near = g_delta * sigma;
      float g_t_far = g_delta * -sigma;
      float safe_radius = max(radius, mf.eps);
      float3 g_hit = g_local / safe_radius;
      float g_sample_t = dot(g_hit, direction);
      if (stream_height_texel_sample_uses_far(interval)) {
        g_t_far += g_sample_t;
      } else {
        g_t_near += g_sample_t;
      }
      stream_atomic_add3(grad_points, cell * 3u, -g_local / safe_radius);
      atomic_fetch_add_explicit(
          &grad_radii[cell], -dot(g_local, hit_offset) / max(safe_radius * safe_radius, mf.eps), memory_order_relaxed);
      stream_route_endpoint_grad(
          interval.near_id,
          g_t_near,
          false,
          cell,
          points,
          radii,
          adjacency,
          adjacency_offsets,
          features,
          origin,
          direction,
          mi,
          mf,
          grad_features,
          grad_points,
          grad_radii);
      stream_route_endpoint_grad(
          interval.far_id,
          g_t_far,
          true,
          cell,
          points,
          radii,
          adjacency,
          adjacency_offsets,
          features,
          origin,
          direction,
          mi,
          mf,
          grad_features,
          grad_points,
          grad_radii);

      float g_log_t_before = g_log_t_after + feature_dot_grad * trans_before * alpha;
      g_log_t_after = g_log_t_before;
      log_t_after = log_t_before;
    } else if (valid && sigma > 0.0f && interval.hit && dt > 0.0f) {
      log_t_after = log_t_before;
    }
  }
}

kernel void powerfoam_raytrace_forward(
    const device float* points [[buffer(0)]],
    const device float* radii [[buffer(1)]],
    const device float* densities [[buffer(2)]],
    const device float* features [[buffer(3)]],
    const device int* adjacency [[buffer(4)]],
    const device int* adjacency_offsets [[buffer(5)]],
    const device float* adjacency_diff [[buffer(6)]],
    const device int* start_ids [[buffer(7)]],
    const device float* rays [[buffer(8)]],
    constant FoamStreamMetaI32& mi [[buffer(9)]],
    constant FoamStreamMetaF32& mf [[buffer(10)]],
    device float* out_features [[buffer(11)]],
    device float* out_alpha [[buffer(12)]],
    device float* out_normal_distance [[buffer(13)]],
    device float* out_normal [[buffer(14)]],
    device int* out_steps [[buffer(15)]],
    uint gid [[thread_position_in_grid]]) {
  uint total = uint(mi.batch_size * mi.height * mi.width);
  if (gid >= total) {
    return;
  }

  uint pixels_per_batch = uint(mi.height * mi.width);
  uint batch = gid / pixels_per_batch;
  uint out_base = gid * uint(mi.output_dim);
  for (uint f = 0u; f < uint(mi.output_dim); ++f) {
    out_features[out_base + f] = 0.0f;
  }
  out_alpha[gid] = 0.0f;
  out_normal_distance[gid] = 0.0f;
  uint normal_base = gid * 3u;
  out_normal[normal_base + 0u] = 0.0f;
  out_normal[normal_base + 1u] = 0.0f;
  out_normal[normal_base + 2u] = 0.0f;
  out_steps[gid] = 0;

  int cell_i = start_ids[(mi.start_mode == 1) ? gid : batch];
  if (cell_i < 0 || cell_i >= mi.cell_count) {
    return;
  }

  uint ray_base = gid * 6u;
  float3 origin = float3(rays[ray_base + 0u], rays[ray_base + 1u], rays[ray_base + 2u]);
  float3 direction = float3(rays[ray_base + 3u], rays[ray_base + 4u], rays[ray_base + 5u]);
  direction = direction / max(length(direction), mf.eps);

  float log_t = 0.0f;
  float walk_near = mf.near_plane;
  for (int walk = 0; walk < mi.cell_count; ++walk) {
    if (exp(log_t) < mf.transmittance_threshold || cell_i < 0 || cell_i >= mi.cell_count) {
      break;
    }
    out_steps[gid] = walk + 1;

    uint cell = uint(cell_i);
    float3 center = stream_load3(points, cell);
    float radius = radii[cell];
    float sigma = max(densities[cell], 0.0f);
    FoamStreamInterval interval = stream_ray_sphere_interval(origin, direction, center, radius, mf);
    if (interval.hit && interval.t_near < walk_near) {
      interval.t_near = walk_near;
    }

    int edge_begin = adjacency_offsets[cell];
    int edge_end = adjacency_offsets[cell + 1u];
    int next_cell = 2147483647;
    float next_t = 1.0e20f;
    if (edge_begin < 0 || edge_end < edge_begin) {
      break;
    }
    for (int edge = edge_begin; edge < edge_end; ++edge) {
      int neighbor_i = adjacency[edge];
      if (neighbor_i < 0 || neighbor_i >= mi.cell_count || uint(neighbor_i) == cell) {
        continue;
      }
      uint diff_base = uint(edge) * 4u;
      float3 diff = float3(adjacency_diff[diff_base + 0u], adjacency_diff[diff_base + 1u], adjacency_diff[diff_base + 2u]);
      float pm_diff = adjacency_diff[diff_base + 3u];
      float dp = dot(direction, diff);
      if (abs(dp) <= mf.eps) {
        continue;
      }
      float t_face = (pm_diff - dot(origin, diff)) / dp;
      if (dp >= 0.0f && t_face > walk_near + 1.0e-5f && t_face < next_t) {
        next_t = t_face;
        next_cell = neighbor_i;
      }
      if (interval.hit) {
        if (dp >= 0.0f) {
          interval.t_far = min(interval.t_far, t_face);
        } else {
          interval.t_near = max(interval.t_near, t_face);
        }
      }
    }

    if (interval.hit && interval.t_far > interval.t_near && sigma > 0.0f &&
        (mi.feature_mode == 2 || mi.feature_mode == 3 || mi.feature_mode == 4 || stream_texel_has_height(mi))) {
      float3 surface_normal = stream_surface_normal(features, cell, mi);
      if (stream_texel_has_height(mi)) {
        interval.hit =
            stream_clip_height_surface(features, cell, origin, direction, center, radius, surface_normal, mi, mf, interval);
      } else {
        interval.hit = stream_clip_surface(origin, direction, center, surface_normal, mf, interval);
      }
    }

    float dt = interval.t_far - interval.t_near;
    if (interval.hit && dt > 0.0f && sigma > 0.0f) {
      float delta = -sigma * dt;
      float alpha = clamp(1.0f - exp(delta), 0.0f, mf.max_alpha);
      if (alpha >= mf.alpha_threshold) {
        float weight = exp(log_t) * alpha;
        uint feat_base = cell * uint(mi.feature_dim);
        float3 local_coord = float3(0.0f);
        if (mi.feature_mode == 1 || mi.feature_mode == 2 || mi.feature_mode == 3 || mi.feature_mode == 4 ||
            stream_texel_has_height(mi)) {
          float t_sample = 0.5f * (interval.t_near + interval.t_far);
          if (mi.feature_mode == 2 || mi.feature_mode == 3 || mi.feature_mode == 4) {
            float3 n = stream_surface_normal(features, cell, mi);
            t_sample = (dot(center, n) - dot(origin, n)) / dot(direction, n);
          } else if (stream_texel_has_height(mi)) {
            t_sample = stream_height_texel_sample_t(interval);
          }
          local_coord = (origin + direction * t_sample - center) / max(radius, mf.eps);
        }

        if (mi.feature_mode == 4 || stream_texel_has_height(mi)) {
          uint c = uint(mi.output_dim);
          uint stride = stream_texel_stride(mi);
          uint color_offset = stream_texel_color_offset(mi);
          uint texel_count = stream_texel_count(mi);
          float3 tangent = stream_surface_tangent(features, cell, mi);
          float3 bitangent = stream_surface_bitangent(features, cell, mi);
          float2 texel_coord = float2(dot(local_coord, tangent), dot(local_coord, bitangent));
          float denom = 0.0f;
          for (uint s = 0u; s < texel_count; ++s) {
            uint sbase = feat_base + s * stride;
            float2 diff = texel_coord - float2(features[sbase + 0u], features[sbase + 1u]);
            denom += exp(-mf.texel_temperature * dot(diff, diff));
          }
          denom = max(denom, mf.eps);
          if (mi.feature_mode == 6) {
            float3 numer3 = float3(0.0f);
            for (uint s = 0u; s < texel_count; ++s) {
              uint sbase = feat_base + s * stride;
              float2 diff = texel_coord - float2(features[sbase + 0u], features[sbase + 1u]);
              float texel_weight = exp(-mf.texel_temperature * dot(diff, diff));
              float2 site = float2(features[sbase + 0u], features[sbase + 1u]);
              float3 texel_world = center + radius * (site.x * tangent + site.y * bitangent);
              float3 view_vec = texel_world - origin;
              float3 view_dir = view_vec / max(length(view_vec), mf.eps);
              numer3 += texel_weight * stream_sv_texel_color(features, sbase, view_dir, mi, mf);
            }
            float3 value3 = numer3 / denom;
            out_features[out_base + 0u] += weight * value3.x;
            out_features[out_base + 1u] += weight * value3.y;
            out_features[out_base + 2u] += weight * value3.z;
          } else {
            for (uint f = 0u; f < c; ++f) {
              float numer = 0.0f;
              for (uint s = 0u; s < texel_count; ++s) {
                uint sbase = feat_base + s * stride;
                float2 diff = texel_coord - float2(features[sbase + 0u], features[sbase + 1u]);
                float texel_weight = exp(-mf.texel_temperature * dot(diff, diff));
                numer += texel_weight * features[sbase + color_offset + f];
              }
              out_features[out_base + f] += weight * (numer / denom);
            }
          }
        } else {
          for (uint f = 0u; f < uint(mi.output_dim); ++f) {
            float value = features[feat_base + f];
            if (mi.feature_mode == 1 || mi.feature_mode == 2 || mi.feature_mode == 3) {
              uint c = uint(mi.output_dim);
              value += local_coord.x * features[feat_base + c + f];
              value += local_coord.y * features[feat_base + 2u * c + f];
              value += local_coord.z * features[feat_base + 3u * c + f];
            }
            out_features[out_base + f] += weight * value;
          }
        }
        float3 n_aux = stream_surface_normal(features, cell, mi);
        float ndv = dot(n_aux, direction);
        if (ndv > 0.0f) {
          out_normal_distance[gid] += ndv * ndv * weight;
        }
        uint normal_base = gid * 3u;
        out_normal[normal_base + 0u] += weight * n_aux.x;
        out_normal[normal_base + 1u] += weight * n_aux.y;
        out_normal[normal_base + 2u] += weight * n_aux.z;
        log_t += delta;
      }
    }

    if (next_cell == 2147483647 || next_t >= 1.0e19f) {
      break;
    }
    cell_i = next_cell;
    walk_near = max(walk_near, next_t);
  }

  out_alpha[gid] = 1.0f - exp(log_t);
}

kernel void powerfoam_raytrace_backward_height_sv_global_atomic(
    const device float* points [[buffer(0)]],
    const device float* radii [[buffer(1)]],
    const device float* densities [[buffer(2)]],
    const device float* features [[buffer(3)]],
    const device int* adjacency [[buffer(4)]],
    const device int* adjacency_offsets [[buffer(5)]],
    const device float* adjacency_diff [[buffer(6)]],
    const device int* start_ids [[buffer(7)]],
    const device float* rays [[buffer(8)]],
    const device float* grad_out_features [[buffer(9)]],
    const device float* grad_out_alpha [[buffer(10)]],
    const device float* grad_out_normal_distance [[buffer(11)]],
    const device float* grad_out_normal [[buffer(12)]],
    constant FoamStreamMetaI32& mi [[buffer(13)]],
    constant FoamStreamMetaF32& mf [[buffer(14)]],
    device atomic_float* grad_points [[buffer(15)]],
    device atomic_float* grad_radii [[buffer(16)]],
    device atomic_float* grad_densities [[buffer(17)]],
    device atomic_float* grad_features [[buffer(18)]],
    uint gid [[thread_position_in_grid]]) {
  uint total = uint(mi.batch_size * mi.height * mi.width);
  if (gid >= total || mi.feature_mode != 6 || mi.output_dim != 3 || mi.feature_dim > 384) {
    return;
  }

  uint pixels_per_batch = uint(mi.height * mi.width);
  uint batch = gid / pixels_per_batch;
  int cell_i = start_ids[(mi.start_mode == 1) ? gid : batch];
  if (cell_i < 0 || cell_i >= mi.cell_count) {
    return;
  }

  uint ray_base = gid * 6u;
  float3 origin = float3(rays[ray_base + 0u], rays[ray_base + 1u], rays[ray_base + 2u]);
  float3 direction = float3(rays[ray_base + 3u], rays[ray_base + 4u], rays[ray_base + 5u]);
  direction = direction / max(length(direction), mf.eps);

  int event_cells[FOAM_RAYTRACE_MAX_EVENTS];
  float event_t_near[FOAM_RAYTRACE_MAX_EVENTS];
  float event_t_far[FOAM_RAYTRACE_MAX_EVENTS];
  int event_near_id[FOAM_RAYTRACE_MAX_EVENTS];
  int event_far_id[FOAM_RAYTRACE_MAX_EVENTS];
  float base_t_near[FOAM_RAYTRACE_MAX_EVENTS];
  float base_t_far[FOAM_RAYTRACE_MAX_EVENTS];
  int base_near_id[FOAM_RAYTRACE_MAX_EVENTS];
  int base_far_id[FOAM_RAYTRACE_MAX_EVENTS];

  float log_t = 0.0f;
  float walk_near = mf.near_plane;
  uint event_count = 0u;
  for (int walk = 0; walk < mi.cell_count && event_count < FOAM_RAYTRACE_MAX_EVENTS; ++walk) {
    if (exp(log_t) < mf.transmittance_threshold || cell_i < 0 || cell_i >= mi.cell_count) {
      break;
    }

    uint cell = uint(cell_i);
    float3 center = stream_load3(points, cell);
    float radius = radii[cell];
    float sigma = max(densities[cell], 0.0f);
    FoamStreamInterval interval = stream_ray_sphere_interval(origin, direction, center, radius, mf);

    int edge_begin = adjacency_offsets[cell];
    int edge_end = adjacency_offsets[cell + 1u];
    int next_cell = 2147483647;
    float next_t = 1.0e20f;
    if (edge_begin < 0 || edge_end < edge_begin) {
      break;
    }
    for (int edge = edge_begin; edge < edge_end; ++edge) {
      int neighbor_i = adjacency[edge];
      if (neighbor_i < 0 || neighbor_i >= mi.cell_count || uint(neighbor_i) == cell) {
        continue;
      }
      uint diff_base = uint(edge) * 4u;
      float3 diff = float3(adjacency_diff[diff_base + 0u], adjacency_diff[diff_base + 1u], adjacency_diff[diff_base + 2u]);
      float pm_diff = adjacency_diff[diff_base + 3u];
      float dp = dot(direction, diff);
      if (abs(dp) > mf.eps) {
        float t_face = (pm_diff - dot(origin, diff)) / dp;
        if (dp >= 0.0f && t_face > walk_near + 1.0e-5f && t_face < next_t) {
          next_t = t_face;
          next_cell = neighbor_i;
        }
      }
      if (interval.hit &&
          !stream_clip_power_face_diff(origin, direction, diff, pm_diff, edge - edge_begin, mf, interval)) {
        interval.hit = false;
      }
    }
    if (interval.hit && interval.t_near < walk_near) {
      interval.t_near = walk_near;
    }

    FoamStreamInterval base_interval = interval;
    if (interval.hit && interval.t_far > interval.t_near && sigma > 0.0f) {
      float3 surface_normal = stream_surface_normal(features, cell, mi);
      interval.hit = stream_clip_height_surface(features, cell, origin, direction, center, radius, surface_normal, mi, mf, interval);
    }

    float dt = interval.t_far - interval.t_near;
    if (interval.hit && dt > 0.0f && sigma > 0.0f) {
      float delta = -sigma * dt;
      float alpha = clamp(1.0f - exp(delta), 0.0f, mf.max_alpha);
      if (alpha >= mf.alpha_threshold) {
        uint slot = event_count++;
        event_cells[slot] = cell_i;
        event_t_near[slot] = interval.t_near;
        event_t_far[slot] = interval.t_far;
        event_near_id[slot] = interval.near_id;
        event_far_id[slot] = interval.far_id;
        base_t_near[slot] = base_interval.t_near;
        base_t_far[slot] = base_interval.t_far;
        base_near_id[slot] = base_interval.near_id;
        base_far_id[slot] = base_interval.far_id;
        log_t += delta;
      }
    }

    if (next_cell == 2147483647 || next_t >= 1.0e19f) {
      break;
    }
    cell_i = next_cell;
    walk_near = max(walk_near, next_t);
  }

  float log_t_after = log_t;
  float g_log_t_after = -grad_out_alpha[gid] * exp(log_t_after);
  uint out_base = gid * uint(mi.output_dim);

  for (int event = int(event_count) - 1; event >= 0; --event) {
    uint cell = uint(event_cells[event]);
    float3 center = stream_load3(points, cell);
    float radius = radii[cell];
    float sigma = max(densities[cell], 0.0f);

    FoamStreamInterval interval;
    interval.hit = true;
    interval.t_near = event_t_near[event];
    interval.t_far = event_t_far[event];
    interval.near_id = event_near_id[event];
    interval.far_id = event_far_id[event];

    FoamStreamInterval base_interval;
    base_interval.hit = true;
    base_interval.t_near = base_t_near[event];
    base_interval.t_far = base_t_far[event];
    base_interval.near_id = base_near_id[event];
    base_interval.far_id = base_far_id[event];

    float dt = interval.t_far - interval.t_near;
    float delta = -sigma * dt;
    float log_t_before = log_t_after - delta;
    float trans_before = exp(log_t_before);
    float exp_delta = exp(delta);
    float alpha = clamp(1.0f - exp_delta, 0.0f, mf.max_alpha);
    float weight = trans_before * alpha;

    uint feat_base = cell * uint(mi.feature_dim);
    uint stride = stream_texel_stride(mi);
    uint texel_count = stream_texel_count(mi);
    uint frame_base = stream_texel_frame_base(cell, mi);
    float3 tangent = stream_surface_tangent(features, cell, mi);
    float3 bitangent = stream_surface_bitangent(features, cell, mi);
    float3 hit_offset = origin + direction * stream_height_texel_sample_t(interval) - center;
    float3 local_coord = hit_offset / max(radius, mf.eps);
    float2 texel_coord = float2(dot(local_coord, tangent), dot(local_coord, bitangent));

    float texel_weights[16];
    float denom = 0.0f;
    for (uint s = 0u; s < texel_count; ++s) {
      uint sbase = feat_base + s * stride;
      float2 diff = texel_coord - float2(features[sbase + 0u], features[sbase + 1u]);
      float texel_weight = exp(-mf.texel_temperature * dot(diff, diff));
      if (s < 16u) {
        texel_weights[s] = texel_weight;
      }
      denom += texel_weight;
    }
    denom = max(denom, mf.eps);

    float3 sv_colors[16];
    float3 sv_raw_colors[16];
    float sv_denoms[16];
    float3 value3 = float3(0.0f);
    for (uint s = 0u; s < texel_count; ++s) {
      uint sbase = feat_base + s * stride;
      float texel_weight = (s < 16u) ? texel_weights[s] : exp(-mf.texel_temperature * dot(texel_coord - float2(features[sbase + 0u], features[sbase + 1u]), texel_coord - float2(features[sbase + 0u], features[sbase + 1u])));
      float2 site = float2(features[sbase + 0u], features[sbase + 1u]);
      float3 texel_world = center + radius * (site.x * tangent + site.y * bitangent);
      float3 view_vec = texel_world - origin;
      float3 view_dir = view_vec / max(length(view_vec), mf.eps);
      float sv_denom = 0.0f;
      float3 sv_raw = stream_sv_texel_color_raw(features, sbase, view_dir, mi, mf, sv_denom);
      float3 sv_color = max(sv_raw + 0.5f, float3(0.0f));
      if (s < 16u) {
        sv_colors[s] = sv_color;
        sv_raw_colors[s] = sv_raw;
        sv_denoms[s] = sv_denom;
      }
      value3 += texel_weight * sv_color;
    }
    value3 /= denom;

    float3 go3 = float3(
        grad_out_features[out_base + 0u],
        grad_out_features[out_base + 1u],
        grad_out_features[out_base + 2u]);
    float feature_dot_grad = dot(go3, value3);
    float3 g_local = float3(0.0f);

    for (uint s = 0u; s < texel_count; ++s) {
      uint sbase = feat_base + s * stride;
      float2 site = float2(features[sbase + 0u], features[sbase + 1u]);
      float2 diff = texel_coord - site;
      float texel_weight = (s < 16u) ? texel_weights[s] : exp(-mf.texel_temperature * dot(diff, diff));
      float3 texel_world = center + radius * (site.x * tangent + site.y * bitangent);
      float3 view_vec = texel_world - origin;
      float3 view_dir = view_vec / max(length(view_vec), mf.eps);
      float3 sv_color = (s < 16u) ? sv_colors[s] : stream_sv_texel_color(features, sbase, view_dir, mi, mf);
      float3 sv_color_grad = weight * texel_weight / denom * go3;
      if (s < 16u) {
        stream_route_sv_color_grad_known_value(
            features, sbase, view_dir, sv_color_grad, sv_raw_colors[s], sv_denoms[s], mi, mf, grad_features);
      } else {
        stream_route_sv_color_grad(features, sbase, view_dir, sv_color_grad, mi, mf, grad_features);
      }
      float g_w = dot(go3, weight * (sv_color - value3) / denom);
      float2 g_diff = g_w * (-2.0f * mf.texel_temperature * texel_weight) * diff;
      g_local += g_diff.x * tangent + g_diff.y * bitangent;
      stream_atomic_add3(grad_features, frame_base + 3u, g_diff.x * local_coord);
      stream_atomic_add3(grad_features, frame_base + 6u, g_diff.y * local_coord);
      atomic_fetch_add_explicit(&grad_features[sbase + 0u], -g_diff.x, memory_order_relaxed);
      atomic_fetch_add_explicit(&grad_features[sbase + 1u], -g_diff.y, memory_order_relaxed);
    }

    float g_delta = g_log_t_after - feature_dot_grad * trans_before * exp_delta;
    float g_normal_distance = grad_out_normal_distance[gid];
    if (g_normal_distance != 0.0f) {
      float3 n_aux = stream_surface_normal(features, cell, mi);
      float ndv = dot(n_aux, direction);
      if (ndv > 0.0f) {
        feature_dot_grad += g_normal_distance * ndv * ndv;
        g_delta -= g_normal_distance * ndv * ndv * trans_before * exp_delta;
        uint normal_base = stream_surface_normal_base(cell, mi);
        if (normal_base != 0xffffffffu) {
          stream_atomic_add3(grad_features, normal_base, g_normal_distance * weight * 2.0f * ndv * direction);
        }
      }
    }
    uint normal_grad_base = gid * 3u;
    float3 g_rendered_normal =
        float3(grad_out_normal[normal_grad_base + 0u], grad_out_normal[normal_grad_base + 1u],
               grad_out_normal[normal_grad_base + 2u]);
    if (g_rendered_normal.x != 0.0f || g_rendered_normal.y != 0.0f || g_rendered_normal.z != 0.0f) {
      float3 n_aux = stream_surface_normal(features, cell, mi);
      float normal_dot_grad = dot(g_rendered_normal, n_aux);
      feature_dot_grad += normal_dot_grad;
      g_delta -= normal_dot_grad * trans_before * exp_delta;
      uint normal_base = stream_surface_normal_base(cell, mi);
      if (normal_base != 0xffffffffu) {
        stream_atomic_add3(grad_features, normal_base, weight * g_rendered_normal);
      }
    }
    atomic_fetch_add_explicit(&grad_densities[cell], g_delta * -dt, memory_order_relaxed);

    float g_t_near = g_delta * sigma;
    float g_t_far = g_delta * -sigma;
    float safe_radius = max(radius, mf.eps);
    float3 g_hit = g_local / safe_radius;
    float g_sample_t = dot(g_hit, direction);
    if (stream_height_texel_sample_uses_far(interval)) {
      g_t_far += g_sample_t;
    } else {
      g_t_near += g_sample_t;
    }
    stream_atomic_add3(grad_points, cell * 3u, -g_local / safe_radius);
    atomic_fetch_add_explicit(
        &grad_radii[cell], -dot(g_local, hit_offset) / max(safe_radius * safe_radius, mf.eps), memory_order_relaxed);

    if (interval.near_id == -4) {
      stream_route_height_surface_endpoint_grad(
          base_interval,
          g_t_near,
          cell,
          points,
          radii,
          adjacency,
          adjacency_offsets,
          features,
          origin,
          direction,
          mi,
          mf,
          grad_features,
          grad_points,
          grad_radii);
    } else {
      stream_route_endpoint_grad(
          interval.near_id,
          g_t_near,
          false,
          cell,
          points,
          radii,
          adjacency,
          adjacency_offsets,
          features,
          origin,
          direction,
          mi,
          mf,
          grad_features,
          grad_points,
          grad_radii);
    }
    if (interval.far_id == -4) {
      stream_route_height_surface_endpoint_grad(
          base_interval,
          g_t_far,
          cell,
          points,
          radii,
          adjacency,
          adjacency_offsets,
          features,
          origin,
          direction,
          mi,
          mf,
          grad_features,
          grad_points,
          grad_radii);
    } else {
      stream_route_endpoint_grad(
          interval.far_id,
          g_t_far,
          true,
          cell,
          points,
          radii,
          adjacency,
          adjacency_offsets,
          features,
          origin,
          direction,
          mi,
          mf,
          grad_features,
          grad_points,
          grad_radii);
    }

    float g_log_t_before = g_log_t_after + feature_dot_grad * trans_before * alpha;
    g_log_t_after = g_log_t_before;
    log_t_after = log_t_before;
  }
}
