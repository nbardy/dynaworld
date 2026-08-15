#include <metal_stdlib>
using namespace metal;

constant uint RETAINED_FIBER_MAX_DEPTH_SAMPLES = 64u;
constant uint RETAINED_FIBER_MAX_ACTIVE_ATOMS = 256u;
constant float INV_SQRT_TWO_PI = 0.3989422804014327f;

inline float quadratic_packed(const device float* q, uint atom, float3 d) {
  uint b = 6u * atom;
  return q[b] * d.x * d.x
      + 2.0f * q[b + 1u] * d.x * d.y
      + 2.0f * q[b + 2u] * d.x * d.z
      + q[b + 3u] * d.y * d.y
      + 2.0f * q[b + 4u] * d.y * d.z
      + q[b + 5u] * d.z * d.z;
}

inline float3 packed_matvec(const device float* q, uint atom, float3 d) {
  uint b = 6u * atom;
  return float3(
      q[b] * d.x + q[b + 1u] * d.y + q[b + 2u] * d.z,
      q[b + 1u] * d.x + q[b + 3u] * d.y + q[b + 4u] * d.z,
      q[b + 2u] * d.x + q[b + 4u] * d.y + q[b + 5u] * d.z
  );
}

inline void atomic_add_float(device atomic_float* output, uint index, float value) {
  atomic_fetch_add_explicit(output + index, value, memory_order_relaxed);
}

inline float3 load3(const device float* values, uint index) {
  uint b = 3u * index;
  return float3(values[b], values[b + 1u], values[b + 2u]);
}

inline void fiber_bounds(
    const device float* ma,
    const device float* q_uvt,
    const device float* depth0,
    const device float* depth_beta,
    const device float* depth_variance,
    const device float* optical_thickness,
    float3 coordinate,
    uint atom_count,
    float sigma_extent,
    float tau_threshold,
    thread float& z_min,
    thread float& z_max,
    thread uint& active_count
) {
  z_min = INFINITY;
  z_max = -INFINITY;
  active_count = 0u;
  for (uint atom = 0u; atom < atom_count; ++atom) {
    float3 delta = coordinate - load3(ma, atom);
    float qv = quadratic_packed(q_uvt, atom, delta);
    float tau = optical_thickness[atom] * exp(-0.5f * qv);
    if (!(tau > tau_threshold)) continue;
    float mean = depth0[atom] + dot(delta, load3(depth_beta, atom));
    float radius = sigma_extent * sqrt(depth_variance[atom]);
    z_min = min(z_min, mean - radius);
    z_max = max(z_max, mean + radius);
    active_count += 1u;
  }
}

kernel void retained_fiber_forward(
    const device int* fallback_mask,
    device float* output,
    const device float* ma,
    const device float* q_uvt,
    const device float* depth0,
    const device float* depth_beta,
    const device float* depth_variance,
    const device float* optical_thickness,
    const device float* color,
    const device float* times,
    constant uint& atom_count,
    constant uint& frame_count,
    constant uint& height,
    constant uint& width,
    constant uint& depth_samples,
    constant float& sigma_extent,
    constant float3& background,
    constant float& alpha_threshold,
    uint pixel_index [[thread_position_in_grid]]
) {
  uint pixel_count = frame_count * height * width;
  if (pixel_index >= pixel_count) return;
  if (fallback_mask[pixel_index] == 0) {
    uint skipped = 3u * pixel_index;
    output[skipped] = 0.0f;
    output[skipped + 1u] = 0.0f;
    output[skipped + 2u] = 0.0f;
    return;
  }
  uint frame_stride = height * width;
  uint frame = pixel_index / frame_stride;
  uint spatial = pixel_index - frame * frame_stride;
  uint y = spatial / width;
  uint x = spatial - y * width;
  float3 coordinate = float3(float(x) + 0.5f, float(y) + 0.5f, times[frame]);

  float tau_threshold = -log(max(1.0f - alpha_threshold, 1.0e-8f));
  float z_min, z_max;
  uint active_count;
  fiber_bounds(
      ma, q_uvt, depth0, depth_beta, depth_variance, optical_thickness,
      coordinate, atom_count, sigma_extent, tau_threshold, z_min, z_max,
      active_count
  );
  float dz = (z_max - z_min) / float(depth_samples);
  float transmittance = 1.0f;
  float3 accum = float3(0.0f);
  if (active_count > 0u && isfinite(dz) && dz > 0.0f) {
    for (uint sample = 0u; sample < depth_samples; ++sample) {
      float z = z_min + (float(sample) + 0.5f) * dz;
      float lambda = 0.0f;
      float3 emission = float3(0.0f);
      for (uint atom = 0u; atom < atom_count; ++atom) {
        float3 delta = coordinate - load3(ma, atom);
        float qv = quadratic_packed(q_uvt, atom, delta);
        float tau = optical_thickness[atom] * exp(-0.5f * qv);
        if (!(tau > tau_threshold)) continue;
        float variance = depth_variance[atom];
        float mean = depth0[atom] + dot(delta, load3(depth_beta, atom));
        float centered = z - mean;
        float profile = INV_SQRT_TWO_PI / sqrt(variance)
            * exp(-0.5f * centered * centered / variance);
        float density = tau * profile;
        lambda += density;
        emission += density * load3(color, atom);
      }
      if (lambda <= 0.0f) continue;
      float beta = exp(-lambda * dz);
      float alpha = 1.0f - beta;
      accum += transmittance * alpha * (emission / lambda);
      transmittance *= beta;
    }
  }
  float3 result = accum + transmittance * background;
  uint out = 3u * pixel_index;
  output[out] = result.x;
  output[out + 1u] = result.y;
  output[out + 2u] = result.z;
}

kernel void retained_fiber_vjp(
    const device int* fallback_mask,
    device atomic_float* grad_ma,
    device atomic_float* grad_q_uvt,
    device atomic_float* grad_depth0,
    device atomic_float* grad_depth_beta,
    device atomic_float* grad_depth_variance,
    device atomic_float* grad_optical_thickness,
    device atomic_float* grad_color,
    const device float* grad_output,
    const device float* ma,
    const device float* q_uvt,
    const device float* depth0,
    const device float* depth_beta,
    const device float* depth_variance,
    const device float* optical_thickness,
    const device float* color,
    const device float* times,
    constant uint& atom_count,
    constant uint& frame_count,
    constant uint& height,
    constant uint& width,
    constant uint& depth_samples,
    constant float& sigma_extent,
    constant float3& background,
    constant float& alpha_threshold,
    uint pixel_index [[thread_position_in_grid]]
) {
  uint pixel_count = frame_count * height * width;
  if (pixel_index >= pixel_count) return;
  if (fallback_mask[pixel_index] == 0) return;
  if (depth_samples == 0u || depth_samples > RETAINED_FIBER_MAX_DEPTH_SAMPLES) return;
  uint frame_stride = height * width;
  uint frame = pixel_index / frame_stride;
  uint spatial = pixel_index - frame * frame_stride;
  uint y = spatial / width;
  uint x = spatial - y * width;
  float3 coordinate = float3(float(x) + 0.5f, float(y) + 0.5f, times[frame]);
  float3 gout = load3(grad_output, pixel_index);

  float tau_threshold = -log(max(1.0f - alpha_threshold, 1.0e-8f));
  float z_min, z_max;
  uint active_count;
  fiber_bounds(
      ma, q_uvt, depth0, depth_beta, depth_variance, optical_thickness,
      coordinate, atom_count, sigma_extent, tau_threshold, z_min, z_max,
      active_count
  );
  float dz = (z_max - z_min) / float(depth_samples);
  if (active_count == 0u || !isfinite(dz) || dz <= 0.0f) return;

  float lambdas[RETAINED_FIBER_MAX_DEPTH_SAMPLES];
  float betas[RETAINED_FIBER_MAX_DEPTH_SAMPLES];
  float3 source_colors[RETAINED_FIBER_MAX_DEPTH_SAMPLES];
  float3 behind[RETAINED_FIBER_MAX_DEPTH_SAMPLES];
  for (uint sample = 0u; sample < depth_samples; ++sample) {
    float z = z_min + (float(sample) + 0.5f) * dz;
    float lambda = 0.0f;
    float3 emission = float3(0.0f);
    for (uint atom = 0u; atom < atom_count; ++atom) {
      float3 delta = coordinate - load3(ma, atom);
      float qv = quadratic_packed(q_uvt, atom, delta);
      float tau = optical_thickness[atom] * exp(-0.5f * qv);
      if (!(tau > tau_threshold)) continue;
      float variance = depth_variance[atom];
      float mean = depth0[atom] + dot(delta, load3(depth_beta, atom));
      float centered = z - mean;
      float profile = INV_SQRT_TWO_PI / sqrt(variance)
          * exp(-0.5f * centered * centered / variance);
      float density = tau * profile;
      lambda += density;
      emission += density * load3(color, atom);
    }
    lambdas[sample] = lambda;
    betas[sample] = exp(-lambda * dz);
    source_colors[sample] = lambda > 0.0f ? emission / lambda : float3(0.0f);
  }

  float3 suffix = background;
  for (int sample = int(depth_samples) - 1; sample >= 0; --sample) {
    behind[uint(sample)] = suffix;
    float beta = betas[uint(sample)];
    suffix = (1.0f - beta) * source_colors[uint(sample)] + beta * suffix;
  }

  float transmittance = 1.0f;
  for (uint sample = 0u; sample < depth_samples; ++sample) {
    float lambda = lambdas[sample];
    float beta = betas[sample];
    float alpha = 1.0f - beta;
    if (lambda > 0.0f) {
      float3 source = source_colors[sample];
      float grad_lambda = transmittance * dz * beta
          * dot(gout, source - behind[sample]);
      float3 grad_source = transmittance * alpha * gout;
      float z = z_min + (float(sample) + 0.5f) * dz;
      for (uint atom = 0u; atom < atom_count; ++atom) {
        float3 atom_ma = load3(ma, atom);
        float3 delta = coordinate - atom_ma;
        float qv = quadratic_packed(q_uvt, atom, delta);
        float gaussian_uvt = exp(-0.5f * qv);
        float tau = optical_thickness[atom] * gaussian_uvt;
        if (!(tau > tau_threshold)) continue;
        float variance = depth_variance[atom];
        float3 atom_depth_beta = load3(depth_beta, atom);
        float mean = depth0[atom] + dot(delta, atom_depth_beta);
        float centered = z - mean;
        float profile = INV_SQRT_TWO_PI / sqrt(variance)
            * exp(-0.5f * centered * centered / variance);
        float density = tau * profile;
        float3 atom_color = load3(color, atom);
        float grad_density = grad_lambda
            + dot(grad_source, atom_color - source) / lambda;
        float grad_tau = grad_density * profile;
        float grad_mean = grad_density * density * centered / variance;
        float grad_variance = grad_density * density * (
            -0.5f / variance
            + 0.5f * centered * centered / (variance * variance)
        );
        float grad_qv = -0.5f * grad_tau * tau;

        atomic_add_float(grad_optical_thickness, atom, grad_tau * gaussian_uvt);
        atomic_add_float(grad_depth0, atom, grad_mean);
        atomic_add_float(grad_depth_variance, atom, grad_variance);
        uint b3 = 3u * atom;
        atomic_add_float(grad_depth_beta, b3, grad_mean * delta.x);
        atomic_add_float(grad_depth_beta, b3 + 1u, grad_mean * delta.y);
        atomic_add_float(grad_depth_beta, b3 + 2u, grad_mean * delta.z);
        float3 grad_atom_color = grad_source * (density / lambda);
        atomic_add_float(grad_color, b3, grad_atom_color.x);
        atomic_add_float(grad_color, b3 + 1u, grad_atom_color.y);
        atomic_add_float(grad_color, b3 + 2u, grad_atom_color.z);

        uint b6 = 6u * atom;
        atomic_add_float(grad_q_uvt, b6, grad_qv * delta.x * delta.x);
        atomic_add_float(grad_q_uvt, b6 + 1u, grad_qv * 2.0f * delta.x * delta.y);
        atomic_add_float(grad_q_uvt, b6 + 2u, grad_qv * 2.0f * delta.x * delta.z);
        atomic_add_float(grad_q_uvt, b6 + 3u, grad_qv * delta.y * delta.y);
        atomic_add_float(grad_q_uvt, b6 + 4u, grad_qv * 2.0f * delta.y * delta.z);
        atomic_add_float(grad_q_uvt, b6 + 5u, grad_qv * delta.z * delta.z);
        float3 grad_delta_q = 2.0f * grad_qv * packed_matvec(q_uvt, atom, delta);
        float3 grad_atom_ma = -grad_delta_q - grad_mean * atom_depth_beta;
        atomic_add_float(grad_ma, b3, grad_atom_ma.x);
        atomic_add_float(grad_ma, b3 + 1u, grad_atom_ma.y);
        atomic_add_float(grad_ma, b3 + 2u, grad_atom_ma.z);
      }
    }
    transmittance *= beta;
  }
}

inline bool ellipsoid_box_bounds(
    const device float* ma,
    const device float* q_uvt,
    const device float* optical_thickness,
    uint atom,
    float tau_threshold,
    thread float3& atom_lower,
    thread float3& atom_upper
) {
  float optical = optical_thickness[atom];
  if (!(optical > tau_threshold)) return false;
  uint b = 6u * atom;
  float a = q_uvt[b];
  float q01 = q_uvt[b + 1u];
  float q02 = q_uvt[b + 2u];
  float d = q_uvt[b + 3u];
  float q12 = q_uvt[b + 4u];
  float f = q_uvt[b + 5u];
  float co00 = d * f - q12 * q12;
  float co11 = a * f - q02 * q02;
  float co22 = a * d - q01 * q01;
  float determinant = a * co00
      - q01 * (q01 * f - q02 * q12)
      + q02 * (q01 * q12 - q02 * d);
  if (!(determinant > 1.0e-12f)) return false;
  float cutoff = -2.0f * log(tau_threshold / optical);
  float3 inverse_diagonal = float3(co00, co11, co22) / determinant;
  if (!(cutoff >= 0.0f) || any(inverse_diagonal <= 0.0f)) return false;
  float3 half_extent = sqrt(max(cutoff * inverse_diagonal, float3(0.0f)));
  float3 center = load3(ma, atom);
  atom_lower = center - half_extent;
  atom_upper = center + half_extent;
  return all(isfinite(atom_lower)) && all(isfinite(atom_upper));
}

inline bool boxes_intersect(
    float3 lower_a,
    float3 upper_a,
    float3 lower_b,
    float3 upper_b
) {
  return all(lower_a <= upper_b) && all(lower_b <= upper_a);
}

inline float affine_box_minimum(float intercept, float3 slope, float3 lower, float3 upper) {
  float3 point = select(upper, lower, slope >= 0.0f);
  return intercept + dot(slope, point);
}

kernel void retained_fiber_certify_tiles(
    device int* fallback_tiles,
    device int* active_counts,
    device int* reason_bits,
    device float* minimum_pair_separation,
    const device float* ma,
    const device float* q_uvt,
    const device float* depth0,
    const device float* depth_beta,
    const device float* depth_variance,
    const device float* optical_thickness,
    const device float* depth_fit_error,
    constant uint& atom_count,
    constant uint& frame_count,
    constant uint& height,
    constant uint& width,
    constant uint& tile_x,
    constant uint& tile_y,
    constant uint& tile_t,
    constant uint& tiles_x,
    constant uint& tiles_y,
    constant uint& tiles_t,
    constant float& alpha_threshold,
    constant float& sigma_multiplier,
    constant float& required_gap,
    uint tile_index [[thread_position_in_grid]]
) {
  uint tile_count = tiles_x * tiles_y * tiles_t;
  if (tile_index >= tile_count) return;
  uint tile_plane = tiles_x * tiles_y;
  uint it = tile_index / tile_plane;
  uint spatial = tile_index - it * tile_plane;
  uint iy = spatial / tiles_x;
  uint ix = spatial - iy * tiles_x;
  uint x0 = ix * tile_x;
  uint y0 = iy * tile_y;
  uint t0 = it * tile_t;
  uint x1 = min(x0 + tile_x, width);
  uint y1 = min(y0 + tile_y, height);
  uint t1 = min(t0 + tile_t, frame_count);
  float time_center = 0.5f * float(frame_count - 1u);
  float3 tile_lower = float3(
      float(x0) + 0.5f,
      float(y0) + 0.5f,
      float(t0) - time_center
  );
  float3 tile_upper = float3(
      float(x1) - 0.5f,
      float(y1) - 0.5f,
      float(t1 - 1u) - time_center
  );
  float tau_threshold = -log(max(1.0f - alpha_threshold, 1.0e-8f));

  uint active_ids[RETAINED_FIBER_MAX_ACTIVE_ATOMS];
  uint active_count = 0u;
  int reasons = 0;
  for (uint atom = 0u; atom < atom_count; ++atom) {
    float3 atom_lower;
    float3 atom_upper;
    bool valid_support = ellipsoid_box_bounds(
        ma, q_uvt, optical_thickness, atom, tau_threshold,
        atom_lower, atom_upper
    );
    if (!valid_support) {
      if (optical_thickness[atom] > tau_threshold) reasons |= 2;
      continue;
    }
    if (!boxes_intersect(atom_lower, atom_upper, tile_lower, tile_upper)) continue;
    if (!(depth_variance[atom] > 0.0f) || !(depth_fit_error[atom] >= 0.0f)) {
      reasons |= 2;
      continue;
    }
    if (active_count >= RETAINED_FIBER_MAX_ACTIVE_ATOMS) {
      reasons |= 1;
      break;
    }
    active_ids[active_count++] = atom;
  }

  float worst_separation = INFINITY;
  if (reasons == 0) {
    for (uint i_active = 0u; i_active < active_count; ++i_active) {
      uint i = active_ids[i_active];
      float3 lower_i;
      float3 upper_i;
      if (!ellipsoid_box_bounds(
              ma, q_uvt, optical_thickness, i, tau_threshold,
              lower_i, upper_i)) {
        reasons |= 2;
        break;
      }
      for (uint j_active = i_active + 1u; j_active < active_count; ++j_active) {
        uint j = active_ids[j_active];
        float3 lower_j;
        float3 upper_j;
        if (!ellipsoid_box_bounds(
                ma, q_uvt, optical_thickness, j, tau_threshold,
                lower_j, upper_j)) {
          reasons |= 2;
          break;
        }
        float3 overlap_lower = max(tile_lower, max(lower_i, lower_j));
        float3 overlap_upper = min(tile_upper, min(upper_i, upper_j));
        if (any(overlap_lower > overlap_upper)) continue;

        float3 beta_i = load3(depth_beta, i);
        float3 beta_j = load3(depth_beta, j);
        float intercept_i = depth0[i] - dot(beta_i, load3(ma, i));
        float intercept_j = depth0[j] - dot(beta_j, load3(ma, j));
        float radius_i = sigma_multiplier * sqrt(depth_variance[i])
            + depth_fit_error[i];
        float radius_j = sigma_multiplier * sqrt(depth_variance[j])
            + depth_fit_error[j];
        float3 slope_ij = beta_j - beta_i;
        float gap_ij = affine_box_minimum(
            intercept_j - intercept_i - radius_i - radius_j,
            slope_ij,
            overlap_lower,
            overlap_upper
        );
        float gap_ji = affine_box_minimum(
            intercept_i - intercept_j - radius_i - radius_j,
            -slope_ij,
            overlap_lower,
            overlap_upper
        );
        float separation = max(gap_ij, gap_ji);
        worst_separation = min(worst_separation, separation);
        if (!(separation > required_gap)) {
          reasons |= 4;
          break;
        }
      }
      if (reasons != 0) break;
    }
  }

  active_counts[tile_index] = int(active_count);
  reason_bits[tile_index] = reasons;
  fallback_tiles[tile_index] = reasons == 0 ? 0 : 1;
  minimum_pair_separation[tile_index] = worst_separation;
}
