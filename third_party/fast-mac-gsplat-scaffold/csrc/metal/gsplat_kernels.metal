#include <metal_stdlib>
using namespace metal;

typedef ulong SortKey;

struct MetaI32 {
  int height;
  int width;
  int tiles_y;
  int tiles_x;
  int tile_size;
  int gaussians;
  int tile_count;
  int chunk_size;
};

struct MetaF32 {
  float alpha_threshold;
  float transmittance_threshold;
  float bg_r;
  float bg_g;
  float bg_b;
  float eps;
};

inline uint sortable_depth_bits(float z) {
  uint u = as_type<uint>(z);
  uint mask = (u & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
  return u ^ mask;
}

inline SortKey pack_key(float depth, uint gauss_id) {
  return (SortKey(sortable_depth_bits(depth)) << 32) | SortKey(gauss_id);
}

inline uint unpack_gauss_id(SortKey key) {
  return uint(key & 0xFFFFFFFFul);
}

kernel void snugbox_count(
    const device float2* means2d [[buffer(0)]],
    const device float3* conics [[buffer(1)]],
    const device float* opacities [[buffer(2)]],
    constant MetaI32& mi [[buffer(3)]],
    constant MetaF32& mf [[buffer(4)]],
    device int4* bbox_out [[buffer(5)]],
    device uint* pair_count [[buffer(6)]],
    device float* tau_out [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= uint(mi.gaussians)) return;
  float2 m = means2d[gid];
  float3 q = conics[gid];
  float op = opacities[gid];
  bbox_out[gid] = int4(0, 0, -1, -1);
  pair_count[gid] = 0;
  tau_out[gid] = 0.0f;
  if (!(op > mf.alpha_threshold) || !(q.x > 0.0f) || !(q.z > 0.0f)) return;
  float det = q.x * q.z - q.y * q.y;
  if (!(det > mf.eps)) return;
  float tau = -2.0f * log(max(mf.alpha_threshold / max(op, mf.eps), mf.eps));
  float half_x = sqrt(max(tau * q.z / det, 0.0f));
  float half_y = sqrt(max(tau * q.x / det, 0.0f));
  int x0 = max(0, int(floor(m.x - half_x - 0.5f)));
  int y0 = max(0, int(floor(m.y - half_y - 0.5f)));
  int x1 = min(mi.width - 1, int(ceil(m.x + half_x - 0.5f)));
  int y1 = min(mi.height - 1, int(ceil(m.y + half_y - 0.5f)));
  if (x0 > x1 || y0 > y1) return;
  bbox_out[gid] = int4(x0, y0, x1, y1);
  tau_out[gid] = tau;

  int tx0 = x0 / mi.tile_size;
  int ty0 = y0 / mi.tile_size;
  int tx1 = x1 / mi.tile_size;
  int ty1 = y1 / mi.tile_size;
  uint cnt = 0;
  for (int ty = ty0; ty <= ty1; ++ty) {
    float ry0 = float(ty * mi.tile_size) + 0.5f;
    float ry1 = min(float(mi.height - 1) + 0.5f, float((ty + 1) * mi.tile_size - 1) + 0.5f);
    float dy0 = ry0 - m.y;
    float dy1 = ry1 - m.y;
    for (int tx = tx0; tx <= tx1; ++tx) {
      float rx0 = float(tx * mi.tile_size) + 0.5f;
      float rx1 = min(float(mi.width - 1) + 0.5f, float((tx + 1) * mi.tile_size - 1) + 0.5f);
      float dx0 = rx0 - m.x;
      float dx1 = rx1 - m.x;
      float qmin = INFINITY;
      if (m.x >= rx0 && m.x <= rx1 && m.y >= ry0 && m.y <= ry1) qmin = 0.0f;
      if (q.z > mf.eps) {
        float dy = clamp(-(q.y / q.z) * dx0, dy0, dy1);
        qmin = min(qmin, q.x * dx0 * dx0 + 2.0f * q.y * dx0 * dy + q.z * dy * dy);
        dy = clamp(-(q.y / q.z) * dx1, dy0, dy1);
        qmin = min(qmin, q.x * dx1 * dx1 + 2.0f * q.y * dx1 * dy + q.z * dy * dy);
      }
      if (q.x > mf.eps) {
        float dx = clamp(-(q.y / q.x) * dy0, dx0, dx1);
        qmin = min(qmin, q.x * dx * dx + 2.0f * q.y * dx * dy0 + q.z * dy0 * dy0);
        dx = clamp(-(q.y / q.x) * dy1, dx0, dx1);
        qmin = min(qmin, q.x * dx * dx + 2.0f * q.y * dx * dy1 + q.z * dy1 * dy1);
      }
      qmin = min(qmin, q.x * dx0 * dx0 + 2.0f * q.y * dx0 * dy0 + q.z * dy0 * dy0);
      qmin = min(qmin, q.x * dx0 * dx0 + 2.0f * q.y * dx0 * dy1 + q.z * dy1 * dy1);
      qmin = min(qmin, q.x * dx1 * dx1 + 2.0f * q.y * dx1 * dy0 + q.z * dy0 * dy0);
      qmin = min(qmin, q.x * dx1 * dx1 + 2.0f * q.y * dx1 * dy1 + q.z * dy1 * dy1);
      if (qmin <= tau) cnt += 1;
    }
  }
  pair_count[gid] = cnt;
}

kernel void emit_pairs(
    const device float2* means2d [[buffer(0)]],
    const device float3* conics [[buffer(1)]],
    const device float* depths [[buffer(2)]],
    const device int4* bbox_in [[buffer(3)]],
    const device float* tau_in [[buffer(4)]],
    const device uint* pair_offsets [[buffer(5)]],
    constant MetaI32& mi [[buffer(6)]],
    constant MetaF32& mf [[buffer(7)]],
    device uint* tile_ids [[buffer(8)]],
    device uint* gauss_ids [[buffer(9)]],
    device ulong* pair_keys [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= uint(mi.gaussians)) return;
  int4 bb = bbox_in[gid];
  if (bb.x > bb.z || bb.y > bb.w) return;
  float2 m = means2d[gid];
  float3 q = conics[gid];
  float tau = tau_in[gid];
  uint write = pair_offsets[gid];
  for (int ty = bb.y / mi.tile_size; ty <= bb.w / mi.tile_size; ++ty) {
    float ry0 = float(ty * mi.tile_size) + 0.5f;
    float ry1 = min(float(mi.height - 1) + 0.5f, float((ty + 1) * mi.tile_size - 1) + 0.5f);
    float dy0 = ry0 - m.y;
    float dy1 = ry1 - m.y;
    for (int tx = bb.x / mi.tile_size; tx <= bb.z / mi.tile_size; ++tx) {
      float rx0 = float(tx * mi.tile_size) + 0.5f;
      float rx1 = min(float(mi.width - 1) + 0.5f, float((tx + 1) * mi.tile_size - 1) + 0.5f);
      float dx0 = rx0 - m.x;
      float dx1 = rx1 - m.x;
      float qmin = INFINITY;
      if (m.x >= rx0 && m.x <= rx1 && m.y >= ry0 && m.y <= ry1) qmin = 0.0f;
      qmin = min(qmin, q.x * dx0 * dx0 + 2.0f * q.y * dx0 * dy0 + q.z * dy0 * dy0);
      qmin = min(qmin, q.x * dx0 * dx0 + 2.0f * q.y * dx0 * dy1 + q.z * dy1 * dy1);
      qmin = min(qmin, q.x * dx1 * dx1 + 2.0f * q.y * dx1 * dy0 + q.z * dy0 * dy0);
      qmin = min(qmin, q.x * dx1 * dx1 + 2.0f * q.y * dx1 * dy1 + q.z * dy1 * dy1);
      if (qmin <= tau) {
        uint tile_id = uint(ty * mi.tiles_x + tx);
        tile_ids[write] = tile_id;
        gauss_ids[write] = gid;
        pair_keys[write] = (ulong(tile_id) << 32) | ulong(sortable_depth_bits(depths[gid]));
        write += 1;
      }
    }
  }
}

// Note: the tile_forward and tile_backward kernels below are scaffolds.
// They are structured to match the bridge ABI, but still need the final tuned implementation.

kernel void tile_forward(
    const device uint* tile_ranges [[buffer(0)]],
    const device uint* gauss_ids [[buffer(1)]],
    const device float2* means2d [[buffer(2)]],
    const device float3* conics [[buffer(3)]],
    const device float3* colors [[buffer(4)]],
    const device float* opacities [[buffer(5)]],
    constant MetaI32& mi [[buffer(6)]],
    constant MetaF32& mf [[buffer(7)]],
    device float* out_rgb [[buffer(8)]],
    device float* out_T [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {
  uint pixel = tid;
  if (pixel >= uint(mi.height * mi.width)) return;
  uint px = pixel % uint(mi.width);
  uint py = pixel / uint(mi.width);
  uint tile_x = px / uint(mi.tile_size);
  uint tile_y = py / uint(mi.tile_size);
  uint tile_id = tile_y * uint(mi.tiles_x) + tile_x;
  uint start = tile_ranges[tile_id];
  uint end = tile_ranges[tile_id + 1];

  float3 accum = float3(0.0f, 0.0f, 0.0f);
  float T = 1.0f;
  float2 p = float2(float(px) + 0.5f, float(py) + 0.5f);
  for (uint i = start; i < end; ++i) {
    uint g = gauss_ids[i];
    float2 d = p - means2d[g];
    float3 q = conics[g];
    float power = -0.5f * (q.x * d.x * d.x + 2.0f * q.y * d.x * d.y + q.z * d.y * d.y);
    if (power > 0.0f) continue;
    float alpha = min(0.99f, opacities[g] * exp(power));
    if (alpha < mf.alpha_threshold) continue;
    float w = T * alpha;
    accum += w * colors[g];
    T *= (1.0f - alpha);
    if (T < mf.transmittance_threshold) break;
  }

  out_rgb[pixel * 3 + 0] = accum.x + T * mf.bg_r;
  out_rgb[pixel * 3 + 1] = accum.y + T * mf.bg_g;
  out_rgb[pixel * 3 + 2] = accum.z + T * mf.bg_b;
  out_T[pixel] = T;
}

kernel void tile_backward(
    const device float* grad_rgb [[buffer(0)]],
    const device uint* tile_ranges [[buffer(1)]],
    const device uint* gauss_ids [[buffer(2)]],
    const device float2* means2d [[buffer(3)]],
    const device float3* conics [[buffer(4)]],
    const device float3* colors [[buffer(5)]],
    const device float* opacities [[buffer(6)]],
    const device float* out_T [[buffer(7)]],
    constant MetaI32& mi [[buffer(8)]],
    constant MetaF32& mf [[buffer(9)]],
    device atomic_float* g_means2d [[buffer(10)]],
    device atomic_float* g_conics [[buffer(11)]],
    device atomic_float* g_colors [[buffer(12)]],
    device atomic_float* g_opacities [[buffer(13)]],
    uint tid [[thread_position_in_grid]]) {
  uint pixel = tid;
  if (pixel >= uint(mi.height * mi.width)) return;
  uint px = pixel % uint(mi.width);
  uint py = pixel / uint(mi.width);
  uint tile_x = px / uint(mi.tile_size);
  uint tile_y = py / uint(mi.tile_size);
  uint tile_id = tile_y * uint(mi.tiles_x) + tile_x;
  uint start = tile_ranges[tile_id];
  uint end = tile_ranges[tile_id + 1];

  float3 go = float3(grad_rgb[pixel * 3 + 0], grad_rgb[pixel * 3 + 1], grad_rgb[pixel * 3 + 2]);
  float2 p = float2(float(px) + 0.5f, float(py) + 0.5f);

  // Recompute all local alphas for the pixel, then reverse-scan.
  // This is deliberately simple and correct-first; tune later with threadgroup staging.
  uint span = end - start;
  if (span == 0) return;

  // Small fixed scratch for scaffold purposes only.
  // Replace with chunked/threadgroup staging for the tuned version.
  const uint MAX_LOCAL = 256;
  if (span > MAX_LOCAL) return;
  thread float alpha_buf[MAX_LOCAL];
  thread float Tprev_buf[MAX_LOCAL];
  thread uint gids[MAX_LOCAL];

  float T = 1.0f;
  uint kept = 0;
  for (uint i = start; i < end && kept < MAX_LOCAL; ++i) {
    uint g = gauss_ids[i];
    float2 d = p - means2d[g];
    float3 q = conics[g];
    float power = -0.5f * (q.x * d.x * d.x + 2.0f * q.y * d.x * d.y + q.z * d.y * d.y);
    if (power > 0.0f) continue;
    float alpha = min(0.99f, opacities[g] * exp(power));
    if (alpha < mf.alpha_threshold) continue;
    gids[kept] = g;
    alpha_buf[kept] = alpha;
    Tprev_buf[kept] = T;
    T *= (1.0f - alpha);
    kept += 1;
    if (T < mf.transmittance_threshold) break;
  }

  float gT = go.x * mf.bg_r + go.y * mf.bg_g + go.z * mf.bg_b;
  for (int j = int(kept) - 1; j >= 0; --j) {
    uint g = gids[j];
    float alpha = alpha_buf[j];
    float Tprev = Tprev_buf[j];
    float3 c = colors[g];
    float dot_gc = go.x * c.x + go.y * c.y + go.z * c.z;
    float g_alpha = Tprev * dot_gc - Tprev * gT;

    atomic_fetch_add_explicit(&g_colors[g * 3 + 0], go.x * Tprev * alpha, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_colors[g * 3 + 1], go.y * Tprev * alpha, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_colors[g * 3 + 2], go.z * Tprev * alpha, memory_order_relaxed);

    // Mean/conic derivatives from q = a dx^2 + 2b dxdy + c dy^2, alpha = o exp(-0.5 q)
    float2 d = p - means2d[g];
    float dalpha_dq = -0.5f * alpha;
    float g_q = g_alpha * dalpha_dq;
    float g_a = g_q * d.x * d.x;
    float g_b = g_q * 2.0f * d.x * d.y;
    float g_c = g_q * d.y * d.y;
    float g_dx = g_q * (2.0f * conics[g].x * d.x + 2.0f * conics[g].y * d.y);
    float g_dy = g_q * (2.0f * conics[g].y * d.x + 2.0f * conics[g].z * d.y);

    atomic_fetch_add_explicit(&g_conics[g * 3 + 0], g_a, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_conics[g * 3 + 1], g_b, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_conics[g * 3 + 2], g_c, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_means2d[g * 2 + 0], -g_dx, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_means2d[g * 2 + 1], -g_dy, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_opacities[g], g_alpha * (alpha / max(opacities[g], mf.eps)), memory_order_relaxed);

    gT = dot_gc + gT * (1.0f - alpha);
  }
}
