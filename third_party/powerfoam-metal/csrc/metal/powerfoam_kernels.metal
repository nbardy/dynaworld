#include <metal_stdlib>
using namespace metal;

struct MetaI32 {
  int batch_size;
  int height;
  int width;
  int cell_count;
  int feature_dim;
};

struct MetaF32 {
  float near_plane;
  float alpha_threshold;
  float transmittance_threshold;
  float max_alpha;
  float eps;
};

inline float3 load3(const device float* values, uint idx) {
  uint base = idx * 3u;
  return float3(values[base + 0u], values[base + 1u], values[base + 2u]);
}

inline bool ray_sphere_interval(
    float3 origin,
    float3 direction,
    float3 center,
    float radius,
    thread float& t0,
    thread float& t1,
    constant MetaF32& mf) {
  float3 oc = origin - center;
  float a = dot(direction, direction);
  float b = 2.0f * dot(oc, direction);
  float c = dot(oc, oc) - radius * radius;
  float disc = b * b - 4.0f * a * c;
  if (disc < 0.0f || a <= mf.eps) {
    return false;
  }
  float root = sqrt(max(disc, 0.0f));
  float inv = 0.5f / a;
  t0 = (-b - root) * inv;
  t1 = (-b + root) * inv;
  if (t1 <= mf.near_plane) {
    return false;
  }
  t0 = max(t0, mf.near_plane);
  return t1 > t0;
}

inline bool clip_against_neighbor(
    float3 origin,
    float3 direction,
    float3 pi,
    float ri,
    float3 pj,
    float rj,
    thread float& t0,
    thread float& t1,
    constant MetaF32& mf) {
  float3 n = pj - pi;
  float rhs = dot(pj, pj) - dot(pi, pi) + ri * ri - rj * rj;
  float limit = rhs - 2.0f * dot(origin, n);
  float denom = 2.0f * dot(direction, n);
  if (abs(denom) <= mf.eps) {
    return limit >= -mf.eps;
  }
  float split = limit / denom;
  if (denom > 0.0f) {
    t1 = min(t1, split);
  } else {
    t0 = max(t0, split);
  }
  return t1 > t0;
}

inline void zero_features(device float* out, uint pix, uint fdim) {
  uint base = pix * fdim;
  for (uint f = 0u; f < fdim; ++f) {
    out[base + f] = 0.0f;
  }
}

inline void add_features(device float* out, const device float* features, uint pix, uint cell, uint fdim, float weight) {
  uint out_base = pix * fdim;
  uint feat_base = cell * fdim;
  for (uint f = 0u; f < fdim; ++f) {
    out[out_base + f] += weight * features[feat_base + f];
  }
}

kernel void powerfoam_rasterize_forward(
    const device float* points [[buffer(0)]],
    const device float* radii [[buffer(1)]],
    const device float* densities [[buffer(2)]],
    const device float* features [[buffer(3)]],
    const device int* adjacency [[buffer(4)]],
    const device int* offsets [[buffer(5)]],
    const device int* sorted_ids [[buffer(6)]],
    const device float* rays [[buffer(7)]],
    constant MetaI32& mi [[buffer(8)]],
    constant MetaF32& mf [[buffer(9)]],
    device float* out [[buffer(10)]],
    device float* out_alpha [[buffer(11)]],
    uint gid [[thread_position_in_grid]]) {
  uint total = uint(mi.batch_size * mi.height * mi.width);
  if (gid >= total) {
    return;
  }

  uint fdim = uint(mi.feature_dim);
  zero_features(out, gid, fdim);
  out_alpha[gid] = 0.0f;
  if (mi.cell_count <= 0) {
    return;
  }

  uint pixels_per_batch = uint(mi.height * mi.width);
  uint b = gid / pixels_per_batch;
  uint ray_base = gid * 6u;
  float3 origin = float3(rays[ray_base + 0u], rays[ray_base + 1u], rays[ray_base + 2u]);
  float3 direction = float3(rays[ray_base + 3u], rays[ray_base + 4u], rays[ray_base + 5u]);

  float transmittance = 1.0f;
  uint sorted_base = b * uint(mi.cell_count);
  for (int order = 0; order < mi.cell_count; ++order) {
    if (transmittance <= mf.transmittance_threshold) {
      break;
    }
    int cell_i = sorted_ids[sorted_base + uint(order)];
    if (cell_i < 0 || cell_i >= mi.cell_count) {
      continue;
    }

    uint cell = uint(cell_i);
    float density = max(densities[cell], 0.0f);
    if (density <= 0.0f) {
      continue;
    }
    float radius = radii[cell];
    if (radius <= 0.0f) {
      continue;
    }

    float3 center = load3(points, cell);
    float t0 = 0.0f;
    float t1 = 0.0f;
    if (!ray_sphere_interval(origin, direction, center, radius, t0, t1, mf)) {
      continue;
    }

    int begin = offsets[cell];
    int end = offsets[cell + 1u];
    if (begin < 0 || end < begin) {
      continue;
    }
    bool inside = true;
    for (int edge = begin; edge < end; ++edge) {
      int neighbor_i = adjacency[edge];
      if (neighbor_i < 0 || neighbor_i >= mi.cell_count || neighbor_i == cell_i) {
        continue;
      }
      uint neighbor = uint(neighbor_i);
      float3 pj = load3(points, neighbor);
      if (!clip_against_neighbor(origin, direction, center, radius, pj, radii[neighbor], t0, t1, mf)) {
        inside = false;
        break;
      }
    }
    if (!inside || t1 <= t0) {
      continue;
    }

    float segment_len = t1 - t0;
    float alpha = 1.0f - exp(-density * segment_len);
    alpha = clamp(alpha, 0.0f, mf.max_alpha);
    if (alpha < mf.alpha_threshold) {
      continue;
    }
    float weight = transmittance * alpha;
    add_features(out, features, gid, cell, fdim, weight);
    transmittance *= (1.0f - alpha);
  }

  out_alpha[gid] = 1.0f - transmittance;
}
