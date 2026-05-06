#include <metal_stdlib>
using namespace metal;

// Compile-time draft of the non-tiled PowerFoam path. This is the simpler
// memory-safe baseline to benchmark before paying tile count/write/sort costs.

struct FoamStreamMetaI32 {
  int batch_size;
  int height;
  int width;
  int cell_count;
  int feature_dim;
  int output_dim;
  int feature_mode;
  int sv_dof;
  int depth_quantile_count;
  int start_mode;
};

struct FoamStreamMetaF32 {
  float near_plane;
  float alpha_threshold;
  float transmittance_threshold;
  float max_alpha;
  float eps;
  float texel_temperature;
};

struct FoamStreamInterval {
  bool hit;
  float t_near;
  float t_far;
  int near_id;
  int far_id;
};

struct FoamStreamEndpointGrad {
  float3 center;
  float radius;
  float3 adj_center;
  float adj_radius;
};

inline float3 stream_load3(const device float* values, uint idx) {
  uint base = idx * 3u;
  return float3(values[base + 0u], values[base + 1u], values[base + 2u]);
}

inline int4 stream_load_bounds(const device int* screen_bounds, uint batch, uint cell, constant FoamStreamMetaI32& mi) {
  uint base = (batch * uint(mi.cell_count) + cell) * 4u;
  return int4(screen_bounds[base + 0u], screen_bounds[base + 1u], screen_bounds[base + 2u], screen_bounds[base + 3u]);
}

inline void stream_atomic_add3(device atomic_float* values, uint base, float3 v) {
  atomic_fetch_add_explicit(&values[base + 0u], v.x, memory_order_relaxed);
  atomic_fetch_add_explicit(&values[base + 1u], v.y, memory_order_relaxed);
  atomic_fetch_add_explicit(&values[base + 2u], v.z, memory_order_relaxed);
}

inline uint stream_surface_normal_base(uint cell, constant FoamStreamMetaI32& mi) {
  if (mi.feature_mode == 3) {
    return cell * uint(mi.feature_dim) + 4u * uint(mi.output_dim);
  }
  if (mi.feature_mode == 4 || mi.feature_mode == 5 || mi.feature_mode == 6) {
    uint stride = (mi.feature_mode == 6) ? uint(3 + 6 * mi.sv_dof) : uint(mi.output_dim + ((mi.feature_mode == 5) ? 3 : 2));
    uint texel_count = uint(mi.feature_dim - 9) / stride;
    return cell * uint(mi.feature_dim) + texel_count * stride;
  }
  return 0xffffffffu;
}

inline float3 stream_surface_normal(const device float* features, uint cell, constant FoamStreamMetaI32& mi) {
  uint base = stream_surface_normal_base(cell, mi);
  if (base != 0xffffffffu) {
    return float3(features[base + 0u], features[base + 1u], features[base + 2u]);
  }
  return float3(0.0f, 0.0f, -1.0f);
}

inline uint stream_texel_stride(constant FoamStreamMetaI32& mi) {
  if (mi.feature_mode == 6) {
    return uint(3 + 6 * mi.sv_dof);
  }
  return uint(mi.output_dim + ((mi.feature_mode == 5) ? 3 : 2));
}

inline uint stream_texel_color_offset(constant FoamStreamMetaI32& mi) {
  return (mi.feature_mode == 5) ? 3u : 2u;
}

inline bool stream_texel_has_height(constant FoamStreamMetaI32& mi) {
  return mi.feature_mode == 5 || mi.feature_mode == 6;
}

inline uint stream_sv_axis_base(uint sbase, uint d) {
  return sbase + 3u + d * 3u;
}

inline uint stream_sv_rgb_base(uint sbase, uint d, constant FoamStreamMetaI32& mi) {
  return sbase + 3u + uint(mi.sv_dof) * 3u + d * 3u;
}

inline uint stream_texel_count(constant FoamStreamMetaI32& mi) {
  return uint(mi.feature_dim - 9) / stream_texel_stride(mi);
}

inline uint stream_texel_frame_base(uint cell, constant FoamStreamMetaI32& mi) {
  return cell * uint(mi.feature_dim) + stream_texel_count(mi) * stream_texel_stride(mi);
}

inline float3 stream_surface_tangent(const device float* features, uint cell, constant FoamStreamMetaI32& mi) {
  uint base = stream_texel_frame_base(cell, mi) + 3u;
  return float3(features[base + 0u], features[base + 1u], features[base + 2u]);
}

inline float3 stream_surface_bitangent(const device float* features, uint cell, constant FoamStreamMetaI32& mi) {
  uint base = stream_texel_frame_base(cell, mi) + 6u;
  return float3(features[base + 0u], features[base + 1u], features[base + 2u]);
}

inline float stream_sv_texel_color_component(
    const device float* features,
    uint sbase,
    float3 view_dir,
    uint component,
    constant FoamStreamMetaI32& mi,
    constant FoamStreamMetaF32& mf) {
  float denom = 0.0f;
  float numer = 0.0f;
  for (uint d = 0u; d < uint(mi.sv_dof); ++d) {
    uint abase = stream_sv_axis_base(sbase, d);
    float3 raw_axis = float3(features[abase + 0u], features[abase + 1u], features[abase + 2u]);
    float temp = max(length(raw_axis), mf.eps);
    float3 axis = raw_axis / temp;
    float dist = max(length(view_dir - axis), mf.eps);
    float w = exp(-temp * dist);
    uint rbase = stream_sv_rgb_base(sbase, d, mi);
    denom += w;
    numer += w * features[rbase + component];
  }
  return max(numer / max(denom, mf.eps) + 0.5f, 0.0f);
}

inline float3 stream_sv_texel_color_raw(
    const device float* features,
    uint sbase,
    float3 view_dir,
    constant FoamStreamMetaI32& mi,
    constant FoamStreamMetaF32& mf,
    thread float& denom_out) {
  float denom = 0.0f;
  float3 numer = float3(0.0f);
  for (uint d = 0u; d < uint(mi.sv_dof); ++d) {
    uint abase = stream_sv_axis_base(sbase, d);
    float3 raw_axis = float3(features[abase + 0u], features[abase + 1u], features[abase + 2u]);
    float temp = max(length(raw_axis), mf.eps);
    float3 axis = raw_axis / temp;
    float dist = max(length(view_dir - axis), mf.eps);
    float w = exp(-temp * dist);
    uint rbase = stream_sv_rgb_base(sbase, d, mi);
    denom += w;
    numer += w * float3(features[rbase + 0u], features[rbase + 1u], features[rbase + 2u]);
  }
  denom_out = max(denom, mf.eps);
  return numer / denom_out;
}

inline float3 stream_sv_texel_color(
    const device float* features,
    uint sbase,
    float3 view_dir,
    constant FoamStreamMetaI32& mi,
    constant FoamStreamMetaF32& mf) {
  float denom = 0.0f;
  return max(stream_sv_texel_color_raw(features, sbase, view_dir, mi, mf, denom) + 0.5f, float3(0.0f));
}

inline void stream_route_sv_color_grad(
    const device float* features,
    uint sbase,
    float3 view_dir,
    float3 g_color,
    constant FoamStreamMetaI32& mi,
    constant FoamStreamMetaF32& mf,
    device atomic_float* grad_features) {
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

    atomic_fetch_add_explicit(&grad_features[rbase + 0u], g_color.x * w / denom, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_features[rbase + 1u], g_color.y * w / denom, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_features[rbase + 2u], g_color.z * w / denom, memory_order_relaxed);

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
    stream_atomic_add3(grad_features, abase, g_raw);
  }
}

inline void stream_route_sv_color_grad_known_value(
    const device float* features,
    uint sbase,
    float3 view_dir,
    float3 g_color,
    float3 raw_value,
    float denom,
    constant FoamStreamMetaI32& mi,
    constant FoamStreamMetaF32& mf,
    device atomic_float* grad_features) {
  denom = max(denom, mf.eps);
  float active0 = (raw_value.x + 0.5f > 0.0f) ? 1.0f : 0.0f;
  float active1 = (raw_value.y + 0.5f > 0.0f) ? 1.0f : 0.0f;
  float active2 = (raw_value.z + 0.5f > 0.0f) ? 1.0f : 0.0f;
  g_color *= float3(active0, active1, active2);

  for (uint d = 0u; d < uint(mi.sv_dof); ++d) {
    uint abase = stream_sv_axis_base(sbase, d);
    uint rbase = stream_sv_rgb_base(sbase, d, mi);
    float3 raw_axis = float3(features[abase + 0u], features[abase + 1u], features[abase + 2u]);
    float raw_norm = length(raw_axis);
    float temp = max(raw_norm, mf.eps);
    float3 axis = raw_axis / temp;
    float3 diff = view_dir - axis;
    float dist = max(length(diff), mf.eps);
    float w = exp(-temp * dist);

    atomic_fetch_add_explicit(&grad_features[rbase + 0u], g_color.x * w / denom, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_features[rbase + 1u], g_color.y * w / denom, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_features[rbase + 2u], g_color.z * w / denom, memory_order_relaxed);

    float3 rgb = float3(features[rbase + 0u], features[rbase + 1u], features[rbase + 2u]);
    float g_w = dot(g_color, rgb - raw_value) / denom;
    float g_e = g_w * w;
    float g_temp = -dist * g_e;
    float3 g_axis = temp * g_e * diff / dist;
    float3 g_raw = (g_axis - axis * dot(g_axis, axis)) / temp;
    if (raw_norm > mf.eps) {
      g_raw += g_temp * axis;
    }
    stream_atomic_add3(grad_features, abase, g_raw);
  }
}

inline FoamStreamInterval stream_ray_sphere_interval(
    float3 origin,
    float3 direction,
    float3 center,
    float radius,
    constant FoamStreamMetaF32& mf) {
  FoamStreamInterval out;
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

inline bool stream_clip_power_face(
    float3 origin,
    float3 direction,
    float3 pi,
    float ri,
    float3 pj,
    float rj,
    int face_id,
    constant FoamStreamMetaF32& mf,
    thread FoamStreamInterval& interval) {
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

inline bool stream_clip_power_face_diff(
    float3 origin,
    float3 direction,
    float3 diff,
    float pm_diff,
    int face_id,
    constant FoamStreamMetaF32& mf,
    thread FoamStreamInterval& interval) {
  float dp = dot(direction, diff);
  float num = pm_diff - dot(origin, diff);
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

inline bool stream_clip_surface(
    float3 origin,
    float3 direction,
    float3 center,
    float3 normal,
    constant FoamStreamMetaF32& mf,
    thread FoamStreamInterval& interval) {
  float dp = dot(direction, normal);
  if (abs(dp) <= mf.eps) {
    return false;
  }
  float t_surface = (dot(center, normal) - dot(origin, normal)) / dp;
  if (dp >= 0.0f) {
    if (t_surface < interval.t_far) {
      interval.t_far = t_surface;
      interval.far_id = -3;
    }
  } else {
    if (t_surface > interval.t_near) {
      interval.t_near = t_surface;
      interval.near_id = -3;
    }
  }
  return interval.t_far > interval.t_near;
}

inline float stream_height_surface_t(
    const device float* features,
    uint cell,
    float3 origin,
    float3 direction,
    float3 center,
    float radius,
    float3 normal,
    float t_near,
    constant FoamStreamMetaI32& mi,
    constant FoamStreamMetaF32& mf) {
  float dp = dot(direction, normal);
  if (abs(dp) <= mf.eps) {
    return NAN;
  }
  float t_flat = (dot(center, normal) - dot(origin, normal)) / dp;
  float t_query0 = (dp >= 0.0f) ? t_near : max(t_near, t_flat);
  float3 hit0 = origin + direction * t_query0;
  float3 local0 = (hit0 - center) / max(radius, mf.eps);
  uint feat_base = cell * uint(mi.feature_dim);
  uint stride = stream_texel_stride(mi);
  uint texel_count = stream_texel_count(mi);
  float3 tangent = stream_surface_tangent(features, cell, mi);
  float3 bitangent = stream_surface_bitangent(features, cell, mi);
  float2 texel_coord0 = float2(dot(local0, tangent), dot(local0, bitangent));

  float height_num = 0.0f;
  float denom = 0.0f;
  for (uint s = 0u; s < texel_count; ++s) {
    uint sbase = feat_base + s * stride;
    float2 diff = texel_coord0 - float2(features[sbase + 0u], features[sbase + 1u]);
    float w = exp(-mf.texel_temperature * dot(diff, diff));
    denom += w;
    height_num += w * features[sbase + 2u];
  }
  float height = height_num / max(denom, mf.eps);
  return (dot(center, normal) - dot(origin, normal) + height) / dp;
}

inline bool stream_clip_height_surface(
    const device float* features,
    uint cell,
    float3 origin,
    float3 direction,
    float3 center,
    float radius,
    float3 normal,
    constant FoamStreamMetaI32& mi,
    constant FoamStreamMetaF32& mf,
    thread FoamStreamInterval& interval) {
  float dp = dot(direction, normal);
  if (abs(dp) <= mf.eps) {
    return false;
  }
  float t_surface = stream_height_surface_t(features, cell, origin, direction, center, radius, normal, interval.t_near, mi, mf);
  if (!isfinite(t_surface)) {
    return false;
  }
  if (dp >= 0.0f) {
    if (t_surface < interval.t_far) {
      interval.t_far = t_surface;
      interval.far_id = -4;
    }
  } else {
    if (t_surface > interval.t_near) {
      interval.t_near = t_surface;
      interval.near_id = -4;
    }
  }
  return interval.t_far > interval.t_near;
}

inline bool stream_height_texel_sample_uses_far(FoamStreamInterval interval) {
  return interval.far_id == -4;
}

inline float stream_height_texel_sample_t(FoamStreamInterval interval) {
  return stream_height_texel_sample_uses_far(interval) ? interval.t_far : interval.t_near;
}

inline FoamStreamInterval stream_clipped_cell_interval(
    const device float* points,
    const device float* radii,
    const device int* adjacency,
    const device int* adjacency_offsets,
    uint cell,
    float3 origin,
    float3 direction,
    constant FoamStreamMetaI32& mi,
    constant FoamStreamMetaF32& mf) {
  float3 center = stream_load3(points, cell);
  float radius = radii[cell];
  FoamStreamInterval hit = stream_ray_sphere_interval(origin, direction, center, radius, mf);
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
    float3 pj = stream_load3(points, uint(neighbor_i));
    if (!stream_clip_power_face(
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

inline FoamStreamInterval stream_clipped_cell_interval_diff(
    const device float* points,
    const device float* radii,
    const device int* adjacency,
    const device int* adjacency_offsets,
    const device float* adjacency_diff,
    uint cell,
    float3 origin,
    float3 direction,
    constant FoamStreamMetaI32& mi,
    constant FoamStreamMetaF32& mf) {
  float3 center = stream_load3(points, cell);
  float radius = radii[cell];
  FoamStreamInterval hit = stream_ray_sphere_interval(origin, direction, center, radius, mf);
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
    uint diff_base = uint(edge) * 4u;
    float3 diff = float3(adjacency_diff[diff_base + 0u], adjacency_diff[diff_base + 1u], adjacency_diff[diff_base + 2u]);
    float pm_diff = adjacency_diff[diff_base + 3u];
    if (!stream_clip_power_face_diff(origin, direction, diff, pm_diff, edge - edge_begin, mf, hit)) {
      hit.hit = false;
      return hit;
    }
  }

  hit.hit = hit.t_far > hit.t_near;
  return hit;
}

inline FoamStreamEndpointGrad stream_ray_sphere_endpoint_bwd(
    float3 origin,
    float3 direction,
    float3 center,
    float radius,
    bool use_far,
    constant FoamStreamMetaF32& mf) {
  FoamStreamEndpointGrad out;
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
  float denom = max(qa * root, mf.eps);
  float3 common = (2.0f * qa * oc - qb * direction) / denom;
  if (use_far) {
    out.center = direction / qa + common;
    out.radius = 2.0f * radius / root;
  } else {
    out.center = direction / qa - common;
    out.radius = -2.0f * radius / root;
  }
  return out;
}

inline FoamStreamEndpointGrad stream_power_face_endpoint_bwd(
    float3 origin,
    float3 direction,
    float3 center,
    float radius,
    float3 adj_center,
    float adj_radius,
    constant FoamStreamMetaF32& mf) {
  FoamStreamEndpointGrad out;
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

inline void stream_route_base_endpoint_grad(
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
    constant FoamStreamMetaI32& mi,
    constant FoamStreamMetaF32& mf,
    device atomic_float* grad_points,
    device atomic_float* grad_radii) {
  if (endpoint_id == -2 || dldt == 0.0f) {
    return;
  }

  float3 center = stream_load3(points, cell);
  float radius = radii[cell];
  if (endpoint_id == -1) {
    FoamStreamEndpointGrad g = stream_ray_sphere_endpoint_bwd(origin, direction, center, radius, use_far, mf);
    stream_atomic_add3(grad_points, cell * 3u, dldt * g.center);
    atomic_fetch_add_explicit(&grad_radii[cell], dldt * g.radius, memory_order_relaxed);
    return;
  }
  if (endpoint_id < 0) {
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
  float3 adj_center = stream_load3(points, neighbor);
  float adj_radius = radii[neighbor];
  FoamStreamEndpointGrad g = stream_power_face_endpoint_bwd(origin, direction, center, radius, adj_center, adj_radius, mf);
  stream_atomic_add3(grad_points, cell * 3u, dldt * g.center);
  atomic_fetch_add_explicit(&grad_radii[cell], dldt * g.radius, memory_order_relaxed);
  stream_atomic_add3(grad_points, neighbor * 3u, dldt * g.adj_center);
  atomic_fetch_add_explicit(&grad_radii[neighbor], dldt * g.adj_radius, memory_order_relaxed);
}

inline void stream_route_height_surface_endpoint_grad(
    FoamStreamInterval base_interval,
    float dldt,
    uint cell,
    const device float* points,
    const device float* radii,
    const device int* adjacency,
    const device int* adjacency_offsets,
    const device float* features,
    float3 origin,
    float3 direction,
    constant FoamStreamMetaI32& mi,
    constant FoamStreamMetaF32& mf,
    device atomic_float* grad_features,
    device atomic_float* grad_points,
    device atomic_float* grad_radii) {
  if (!base_interval.hit || dldt == 0.0f) {
    return;
  }
  float3 center = stream_load3(points, cell);
  float radius = radii[cell];
  float3 n = stream_surface_normal(features, cell, mi);
  float dp = dot(direction, n);
  if (abs(dp) <= mf.eps) {
    return;
  }

  float inv_dp = 1.0f / dp;
  float num = dot(center - origin, n);
  uint frame_base = stream_texel_frame_base(cell, mi);
  float t_flat = num * inv_dp;
  float t_query0 = (dp >= 0.0f) ? base_interval.t_near : max(base_interval.t_near, t_flat);
  float3 hit0 = origin + direction * t_query0;
  float3 local0 = (hit0 - center) / max(radius, mf.eps);
  float3 tangent = stream_surface_tangent(features, cell, mi);
  float3 bitangent = stream_surface_bitangent(features, cell, mi);
  float2 texel_coord0 = float2(dot(local0, tangent), dot(local0, bitangent));
  uint feat_base = cell * uint(mi.feature_dim);
  uint stride = stream_texel_stride(mi);
  uint texel_count = stream_texel_count(mi);
  float height_num = 0.0f;
  float denom = 0.0f;
  for (uint s = 0u; s < texel_count; ++s) {
    uint sbase = feat_base + s * stride;
    float2 diff = texel_coord0 - float2(features[sbase + 0u], features[sbase + 1u]);
    float w = exp(-mf.texel_temperature * dot(diff, diff));
    denom += w;
    height_num += w * features[sbase + 2u];
  }
  denom = max(denom, mf.eps);
  float height = height_num / denom;

  float g_height = dldt * inv_dp;
  float g_num = dldt * inv_dp;
  float g_dp = -dldt * (num + height) * inv_dp * inv_dp;
  float2 g_texel_coord0 = float2(0.0f);
  for (uint s = 0u; s < texel_count; ++s) {
    uint sbase = feat_base + s * stride;
    float2 site = float2(features[sbase + 0u], features[sbase + 1u]);
    float2 diff = texel_coord0 - site;
    float w = exp(-mf.texel_temperature * dot(diff, diff));
    atomic_fetch_add_explicit(&grad_features[sbase + 2u], g_height * w / denom, memory_order_relaxed);
    float g_w = g_height * (features[sbase + 2u] - height) / denom;
    float2 g_diff = g_w * (-2.0f * mf.texel_temperature * w) * diff;
    g_texel_coord0 += g_diff;
    atomic_fetch_add_explicit(&grad_features[sbase + 0u], -g_diff.x, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_features[sbase + 1u], -g_diff.y, memory_order_relaxed);
  }
  float3 g_local0 = g_texel_coord0.x * tangent + g_texel_coord0.y * bitangent;
  stream_atomic_add3(grad_features, frame_base + 3u, g_texel_coord0.x * local0);
  stream_atomic_add3(grad_features, frame_base + 6u, g_texel_coord0.y * local0);
  float safe_radius = max(radius, mf.eps);
  float3 g_hit0 = g_local0 / safe_radius;
  stream_atomic_add3(grad_points, cell * 3u, -g_local0 / safe_radius);
  atomic_fetch_add_explicit(
      &grad_radii[cell], -dot(g_local0, hit0 - center) / max(safe_radius * safe_radius, mf.eps), memory_order_relaxed);

  float g_t_query0 = dot(g_hit0, direction);
  if (dp >= 0.0f || base_interval.t_near >= t_flat) {
    stream_route_base_endpoint_grad(
        base_interval.near_id,
        g_t_query0,
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
  } else {
    g_num += g_t_query0 * inv_dp;
    g_dp += -g_t_query0 * num * inv_dp * inv_dp;
  }

  stream_atomic_add3(grad_points, cell * 3u, g_num * n);
  stream_atomic_add3(grad_features, frame_base, g_num * (center - origin) + g_dp * direction);
}

inline void stream_route_endpoint_grad(
    int endpoint_id,
    float dldt,
    bool use_far,
    uint cell,
    const device float* points,
    const device float* radii,
    const device int* adjacency,
    const device int* adjacency_offsets,
    const device float* features,
    float3 origin,
    float3 direction,
    constant FoamStreamMetaI32& mi,
    constant FoamStreamMetaF32& mf,
    device atomic_float* grad_features,
    device atomic_float* grad_points,
    device atomic_float* grad_radii) {
  if (endpoint_id == -2 || dldt == 0.0f) {
    return;
  }

  float3 center = stream_load3(points, cell);
  float radius = radii[cell];
  if (endpoint_id == -3) {
    float3 n = stream_surface_normal(features, cell, mi);
    float dp = dot(direction, n);
    if (abs(dp) > mf.eps) {
      stream_atomic_add3(grad_points, cell * 3u, dldt * n / dp);
      if (mi.feature_mode == 3 || mi.feature_mode == 4) {
        uint normal_base = cell * uint(mi.feature_dim) + 4u * uint(mi.output_dim);
        if (mi.feature_mode == 4) {
          normal_base = stream_texel_frame_base(cell, mi);
        }
        float num = dot(center - origin, n);
        float inv_dp = 1.0f / dp;
        float3 grad_n = (center - origin) * inv_dp - num * direction * inv_dp * inv_dp;
        stream_atomic_add3(grad_features, normal_base, dldt * grad_n);
      }
    }
    return;
  }
  if (endpoint_id == -4) {
    FoamStreamInterval base_interval = stream_clipped_cell_interval(
        points, radii, adjacency, adjacency_offsets, cell, origin, direction, mi, mf);
    stream_route_height_surface_endpoint_grad(
        base_interval,
        dldt,
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
    return;
  }
  if (endpoint_id == -1) {
    FoamStreamEndpointGrad g = stream_ray_sphere_endpoint_bwd(origin, direction, center, radius, use_far, mf);
    stream_atomic_add3(grad_points, cell * 3u, dldt * g.center);
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
  float3 adj_center = stream_load3(points, neighbor);
  float adj_radius = radii[neighbor];
  FoamStreamEndpointGrad g = stream_power_face_endpoint_bwd(origin, direction, center, radius, adj_center, adj_radius, mf);
  stream_atomic_add3(grad_points, cell * 3u, dldt * g.center);
  atomic_fetch_add_explicit(&grad_radii[cell], dldt * g.radius, memory_order_relaxed);
  stream_atomic_add3(grad_points, neighbor * 3u, dldt * g.adj_center);
  atomic_fetch_add_explicit(&grad_radii[neighbor], dldt * g.adj_radius, memory_order_relaxed);
}

kernel void powerfoam_stream_forward(
    const device float* points [[buffer(0)]],
    const device float* radii [[buffer(1)]],
    const device float* densities [[buffer(2)]],
    const device float* features [[buffer(3)]],
    const device int* adjacency [[buffer(4)]],
    const device int* adjacency_offsets [[buffer(5)]],
    const device int* sorted_ids [[buffer(6)]],
    const device int* screen_bounds [[buffer(7)]],
    const device float* rays [[buffer(8)]],
    constant FoamStreamMetaI32& mi [[buffer(9)]],
    constant FoamStreamMetaF32& mf [[buffer(10)]],
    device float* out_features [[buffer(11)]],
    device float* out_alpha [[buffer(12)]],
    device float* out_log_t [[buffer(13)]],
    device int* out_pixel_stop [[buffer(14)]],
    uint gid [[thread_position_in_grid]]) {
  uint total = uint(mi.batch_size * mi.height * mi.width);
  if (gid >= total) {
    return;
  }

  uint pixels_per_batch = uint(mi.height * mi.width);
  uint batch = gid / pixels_per_batch;
  uint local = gid - batch * pixels_per_batch;
  uint y = local / uint(mi.width);
  uint x = local - y * uint(mi.width);
  uint out_base = gid * uint(mi.output_dim);

  for (uint f = 0u; f < uint(mi.output_dim); ++f) {
    out_features[out_base + f] = 0.0f;
  }
  out_alpha[gid] = 0.0f;
  out_log_t[gid] = 0.0f;
  out_pixel_stop[gid] = mi.cell_count;

  uint ray_base = gid * 6u;
  float3 origin = float3(rays[ray_base + 0u], rays[ray_base + 1u], rays[ray_base + 2u]);
  float3 direction = float3(rays[ray_base + 3u], rays[ray_base + 4u], rays[ray_base + 5u]);
  direction = direction / max(length(direction), mf.eps);

  float log_t = 0.0f;
  uint sorted_base = batch * uint(mi.cell_count);
  for (int order = 0; order < mi.cell_count; ++order) {
    if (exp(log_t) < mf.transmittance_threshold) {
      out_pixel_stop[gid] = order;
      break;
    }

    int cell_i = sorted_ids[sorted_base + uint(order)];
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

    FoamStreamInterval interval = stream_clipped_cell_interval(
        points, radii, adjacency, adjacency_offsets, cell, origin, direction, mi, mf);
    if (interval.hit &&
        (mi.feature_mode == 2 || mi.feature_mode == 3 || mi.feature_mode == 4 || stream_texel_has_height(mi))) {
      float3 surface_normal = stream_surface_normal(features, cell, mi);
      if (stream_texel_has_height(mi)) {
        interval.hit = stream_clip_height_surface(features, cell, origin, direction, center, radius, surface_normal, mi, mf, interval);
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
              if (mi.feature_mode == 6) {
                float2 site = float2(features[sbase + 0u], features[sbase + 1u]);
                float3 texel_world = center + radius * (site.x * tangent + site.y * bitangent);
                float3 view_vec = texel_world - origin;
                float3 view_dir = view_vec / max(length(view_vec), mf.eps);
                numer += texel_weight * stream_sv_texel_color_component(features, sbase, view_dir, f, mi, mf);
              } else {
                numer += texel_weight * features[sbase + color_offset + f];
              }
            }
            out_features[out_base + f] += weight * (numer / denom);
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
        log_t += delta;
      }
    }
  }

  out_log_t[gid] = log_t;
  out_alpha[gid] = 1.0f - exp(log_t);
}

kernel void powerfoam_stream_backward_global_atomic(
    const device float* points [[buffer(0)]],
    const device float* radii [[buffer(1)]],
    const device float* densities [[buffer(2)]],
    const device float* features [[buffer(3)]],
    const device int* adjacency [[buffer(4)]],
    const device int* adjacency_offsets [[buffer(5)]],
    const device int* sorted_ids [[buffer(6)]],
    const device int* screen_bounds [[buffer(7)]],
    const device float* rays [[buffer(8)]],
    const device float* out_log_t [[buffer(9)]],
    const device int* pixel_stop [[buffer(10)]],
    const device float* grad_out_features [[buffer(11)]],
    const device float* grad_out_alpha [[buffer(12)]],
    constant FoamStreamMetaI32& mi [[buffer(13)]],
    constant FoamStreamMetaF32& mf [[buffer(14)]],
    device atomic_float* grad_points [[buffer(15)]],
    device atomic_float* grad_radii [[buffer(16)]],
    device atomic_float* grad_densities [[buffer(17)]],
    device atomic_float* grad_features [[buffer(18)]],
    uint gid [[thread_position_in_grid]]) {
  uint total = uint(mi.batch_size * mi.height * mi.width);
  if (gid >= total) {
    return;
  }

  uint pixels_per_batch = uint(mi.height * mi.width);
  uint batch = gid / pixels_per_batch;
  uint local = gid - batch * pixels_per_batch;
  uint y = local / uint(mi.width);
  uint x = local - y * uint(mi.width);
  uint out_base = gid * uint(mi.output_dim);

  uint ray_base = gid * 6u;
  float3 origin = float3(rays[ray_base + 0u], rays[ray_base + 1u], rays[ray_base + 2u]);
  float3 direction = float3(rays[ray_base + 3u], rays[ray_base + 4u], rays[ray_base + 5u]);
  direction = direction / max(length(direction), mf.eps);

  float log_t_after = out_log_t[gid];
  float g_log_t_after = -grad_out_alpha[gid] * exp(log_t_after);
  uint sorted_base = batch * uint(mi.cell_count);
  int stop = clamp(pixel_stop[gid], 0, mi.cell_count);

  for (int order = stop - 1; order >= 0; --order) {
    int cell_i = sorted_ids[sorted_base + uint(order)];
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

    FoamStreamInterval interval = stream_clipped_cell_interval(
        points, radii, adjacency, adjacency_offsets, cell, origin, direction, mi, mf);
    if (interval.hit &&
        (mi.feature_mode == 2 || mi.feature_mode == 3 || mi.feature_mode == 4 || stream_texel_has_height(mi))) {
      float3 surface_normal = stream_surface_normal(features, cell, mi);
      if (stream_texel_has_height(mi)) {
        interval.hit = stream_clip_height_surface(features, cell, origin, direction, center, radius, surface_normal, mi, mf, interval);
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
          if (mi.feature_mode == 6) {
            float2 site = float2(features[sbase + 0u], features[sbase + 1u]);
            float3 texel_world = center + radius * (site.x * tangent + site.y * bitangent);
            float3 view_vec = texel_world - origin;
            float3 view_dir = view_vec / max(length(view_vec), mf.eps);
            numer += texel_weight * stream_sv_texel_color_component(features, sbase, view_dir, f, mi, mf);
          } else {
            numer += texel_weight * features[sbase + color_offset + f];
          }
        }
        float value = numer / denom;
        float go = grad_out_features[out_base + f];
        feature_dot_grad += go * value;
        if (mi.feature_mode != 6) {
          for (uint s = 0u; s < texel_count; ++s) {
            uint sbase = feat_base + s * stride;
            float2 diff = texel_coord - float2(features[sbase + 0u], features[sbase + 1u]);
            float texel_weight = exp(-mf.texel_temperature * dot(diff, diff));
            atomic_fetch_add_explicit(
                &grad_features[sbase + color_offset + f], go * weight * texel_weight / denom, memory_order_relaxed);
          }
        }
      }

      for (uint s = 0u; s < texel_count; ++s) {
        uint sbase = feat_base + s * stride;
        float2 site = float2(features[sbase + 0u], features[sbase + 1u]);
        float2 diff = texel_coord - site;
        float texel_weight = exp(-mf.texel_temperature * dot(diff, diff));
        float3 sv_color = float3(0.0f);
        float3 sv_color_grad = float3(0.0f);
        if (mi.feature_mode == 6) {
          float3 texel_world = center + radius * (site.x * tangent + site.y * bitangent);
          float3 view_vec = texel_world - origin;
          float3 view_dir = view_vec / max(length(view_vec), mf.eps);
          sv_color = stream_sv_texel_color(features, sbase, view_dir, mi, mf);
          sv_color_grad = weight * texel_weight / denom *
              float3(grad_out_features[out_base + 0u], grad_out_features[out_base + 1u], grad_out_features[out_base + 2u]);
          stream_route_sv_color_grad(features, sbase, view_dir, sv_color_grad, mi, mf, grad_features);
        }
        float g_w = 0.0f;
        for (uint f = 0u; f < c; ++f) {
          float numer = 0.0f;
          for (uint t = 0u; t < texel_count; ++t) {
            uint tbase = feat_base + t * stride;
            float2 tdiff = texel_coord - float2(features[tbase + 0u], features[tbase + 1u]);
            float tw = exp(-mf.texel_temperature * dot(tdiff, tdiff));
            if (mi.feature_mode == 6) {
              float2 tsite = float2(features[tbase + 0u], features[tbase + 1u]);
              float3 texel_world = center + radius * (tsite.x * tangent + tsite.y * bitangent);
              float3 view_vec = texel_world - origin;
              float3 view_dir = view_vec / max(length(view_vec), mf.eps);
              numer += tw * stream_sv_texel_color_component(features, tbase, view_dir, f, mi, mf);
            } else {
              numer += tw * features[tbase + color_offset + f];
            }
          }
          float value = numer / denom;
          float texel_value = (mi.feature_mode == 6) ? sv_color[f] : features[sbase + color_offset + f];
          g_w += grad_out_features[out_base + f] * weight * (texel_value - value) / denom;
        }
        float2 g_diff = g_w * (-2.0f * mf.texel_temperature * texel_weight) * diff;
        g_local += g_diff.x * tangent + g_diff.y * bitangent;
        stream_atomic_add3(grad_features, frame_base + 3u, g_diff.x * local_coord);
        stream_atomic_add3(grad_features, frame_base + 6u, g_diff.y * local_coord);
        atomic_fetch_add_explicit(&grad_features[sbase + 0u], -g_diff.x, memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_features[sbase + 1u], -g_diff.y, memory_order_relaxed);
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
  }
}
