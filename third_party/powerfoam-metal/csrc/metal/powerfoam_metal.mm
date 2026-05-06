#import <Foundation/Foundation.h>

#include <ATen/ATen.h>
#include <ATen/native/mps/MetalShaderLibrary.h>
#include <torch/extension.h>
#include <torch/mps.h>

#include <memory>
#include <mutex>
#include <string>

#include "shared/common.h"

namespace powerfoam {
namespace {

using at::native::mps::DynamicMetalShaderLibrary;
using at::native::mps::MetalKernelFunction;

std::string load_shader_source() {
  NSString* metalPath = [[NSString stringWithUTF8String:__FILE__] stringByDeletingLastPathComponent];
  NSString* forwardPath = [metalPath stringByAppendingPathComponent:@"powerfoam_kernels.metal"];
  NSString* streamPath = [metalPath stringByAppendingPathComponent:@"powerfoam_streaming_kernels.metal"];
  NSString* tiledPath = [metalPath stringByAppendingPathComponent:@"powerfoam_tiled_stream_kernels.metal"];
  NSError* err = nil;
  NSString* forwardSrc = [NSString stringWithContentsOfFile:forwardPath encoding:NSUTF8StringEncoding error:&err];
  TORCH_CHECK(forwardSrc != nil, "Failed to read powerfoam_kernels.metal: ", err.localizedDescription.UTF8String);
  err = nil;
  NSString* streamSrc = [NSString stringWithContentsOfFile:streamPath encoding:NSUTF8StringEncoding error:&err];
  TORCH_CHECK(streamSrc != nil, "Failed to read powerfoam_streaming_kernels.metal: ", err.localizedDescription.UTF8String);
  err = nil;
  NSString* tiledSrc = [NSString stringWithContentsOfFile:tiledPath encoding:NSUTF8StringEncoding error:&err];
  TORCH_CHECK(tiledSrc != nil, "Failed to read powerfoam_tiled_stream_kernels.metal: ", err.localizedDescription.UTF8String);
  return std::string([forwardSrc UTF8String]) + "\n" + std::string([streamSrc UTF8String]) + "\n" +
      std::string([tiledSrc UTF8String]);
}

struct MetalPowerFoamKernels {
  std::shared_ptr<MetalKernelFunction> rasterize_forward;
  std::shared_ptr<MetalKernelFunction> stream_forward;
  std::shared_ptr<MetalKernelFunction> stream_backward;
  std::shared_ptr<MetalKernelFunction> tiled_count;
  std::shared_ptr<MetalKernelFunction> tiled_write;
  std::shared_ptr<MetalKernelFunction> tiled_emit_count;
  std::shared_ptr<MetalKernelFunction> tiled_emit_write;
  std::shared_ptr<MetalKernelFunction> tiled_forward;
  std::shared_ptr<MetalKernelFunction> tiled_aux_forward;
  std::shared_ptr<MetalKernelFunction> tiled_backward;
  std::shared_ptr<MetalKernelFunction> tiled_backward_constant_reduced;
  std::shared_ptr<MetalKernelFunction> tiled_backward_height_sv_feature_reduced;
  std::shared_ptr<MetalKernelFunction> raytrace_forward;
  std::shared_ptr<MetalKernelFunction> raytrace_height_sv_backward;
};

MetalPowerFoamKernels& kernels() {
  static std::once_flag once;
  static std::unique_ptr<DynamicMetalShaderLibrary> lib;
  static MetalPowerFoamKernels out;
  std::call_once(once, []() {
    lib = std::make_unique<DynamicMetalShaderLibrary>(load_shader_source());
    out.rasterize_forward = lib->getKernelFunction("powerfoam_rasterize_forward");
    out.stream_forward = lib->getKernelFunction("powerfoam_stream_forward");
    out.stream_backward = lib->getKernelFunction("powerfoam_stream_backward_global_atomic");
    out.tiled_count = lib->getKernelFunction("powerfoam_tiled_count_bounds_sorted");
    out.tiled_write = lib->getKernelFunction("powerfoam_tiled_write_bounds_sorted");
    out.tiled_emit_count = lib->getKernelFunction("powerfoam_tiled_emit_count_bounds");
    out.tiled_emit_write = lib->getKernelFunction("powerfoam_tiled_emit_write_bounds");
    out.tiled_forward = lib->getKernelFunction("powerfoam_tiled_forward");
    out.tiled_aux_forward = lib->getKernelFunction("powerfoam_tiled_aux_forward");
    out.tiled_backward = lib->getKernelFunction("powerfoam_tiled_backward_global_atomic");
    out.tiled_backward_constant_reduced = lib->getKernelFunction("powerfoam_tiled_backward_constant_reduced");
    out.tiled_backward_height_sv_feature_reduced =
        lib->getKernelFunction("powerfoam_tiled_backward_height_sv_feature_reduced");
    out.raytrace_forward = lib->getKernelFunction("powerfoam_raytrace_forward");
    out.raytrace_height_sv_backward = lib->getKernelFunction("powerfoam_raytrace_backward_height_sv_global_atomic");
  });
  return out;
}

template <typename Fn>
void launch(std::shared_ptr<MetalKernelFunction> fn, Fn&& body) {
  fn->runCommandBlock([&]() {
    fn->startEncoding();
    body(*fn);
  });
}

void check_float_mps(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_mps(), name, " must be on MPS");
  TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_i32_mps(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_mps(), name, " must be on MPS");
  TORCH_CHECK(t.scalar_type() == torch::kInt32, name, " must be int32");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_inputs(
    const torch::Tensor& points,
    const torch::Tensor& radii,
    const torch::Tensor& densities,
    const torch::Tensor& features,
    const torch::Tensor& adjacency,
    const torch::Tensor& offsets,
    const torch::Tensor& sorted_ids,
    const torch::Tensor& rays,
    const ParsedMeta& meta) {
  check_float_mps(points, "points");
  check_float_mps(radii, "radii");
  check_float_mps(densities, "densities");
  check_float_mps(features, "features");
  check_float_mps(rays, "rays");
  check_i32_mps(adjacency, "adjacency");
  check_i32_mps(offsets, "offsets");
  check_i32_mps(sorted_ids, "sorted_ids");

  TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "points must be [N,3]");
  TORCH_CHECK(radii.dim() == 1, "radii must be [N]");
  TORCH_CHECK(densities.dim() == 1, "densities must be [N]");
  TORCH_CHECK(features.dim() == 2, "features must be [N,F]");
  TORCH_CHECK(adjacency.dim() == 1, "adjacency must be [E]");
  TORCH_CHECK(offsets.dim() == 1, "offsets must be [N+1]");
  TORCH_CHECK(sorted_ids.dim() == 2, "sorted_ids must be [B,N]");
  TORCH_CHECK(rays.dim() == 4 && rays.size(3) == 6, "rays must be [B,H,W,6]");

  TORCH_CHECK(points.size(0) == meta.cell_count, "meta cell count mismatch");
  TORCH_CHECK(features.size(1) == meta.feature_dim, "meta feature dim mismatch");
  TORCH_CHECK(radii.size(0) == meta.cell_count, "radii size mismatch");
  TORCH_CHECK(densities.size(0) == meta.cell_count, "densities size mismatch");
  TORCH_CHECK(features.size(0) == meta.cell_count, "features size mismatch");
  TORCH_CHECK(offsets.size(0) == meta.cell_count + 1, "offsets must have N+1 entries");
  TORCH_CHECK(sorted_ids.size(0) == meta.batch_size && sorted_ids.size(1) == meta.cell_count, "sorted_ids shape mismatch");
  TORCH_CHECK(
      rays.size(0) == meta.batch_size && rays.size(1) == meta.height && rays.size(2) == meta.width,
      "rays shape mismatch");
  TORCH_CHECK(meta.batch_size > 0 && meta.height > 0 && meta.width > 0, "invalid image metadata");
  TORCH_CHECK(meta.cell_count >= 0, "invalid cell count");
  TORCH_CHECK(meta.feature_dim > 0, "feature_dim must be positive");
  TORCH_CHECK(meta.output_dim > 0, "output_dim must be positive");
  TORCH_CHECK(
      meta.feature_mode == 0 || meta.feature_mode == 1 || meta.feature_mode == 2 || meta.feature_mode == 3 ||
          meta.feature_mode == 4 || meta.feature_mode == 5 || meta.feature_mode == 6,
      "unsupported feature mode");
  if (meta.feature_mode == 0) {
    TORCH_CHECK(meta.output_dim == meta.feature_dim, "constant feature mode requires output_dim == feature_dim");
  } else if (meta.feature_mode == 1 || meta.feature_mode == 2) {
    TORCH_CHECK(meta.feature_dim == meta.output_dim * 4, "linear feature mode requires feature_dim == output_dim * 4");
  } else if (meta.feature_mode == 3) {
    TORCH_CHECK(
        meta.feature_dim == meta.output_dim * 4 + 3,
        "oriented surface-linear feature mode requires feature_dim == output_dim * 4 + 3");
  } else if (meta.feature_mode == 4) {
    const int stride = meta.output_dim + 2;
    TORCH_CHECK(
        meta.feature_dim > 9 && (meta.feature_dim - 9) % stride == 0,
        "oriented texel-surface feature mode requires feature_dim == S * (output_dim + 2) + 9");
  } else if (meta.feature_mode == 5) {
    const int stride = meta.output_dim + 3;
    TORCH_CHECK(
        meta.feature_dim > 9 && (meta.feature_dim - 9) % stride == 0,
        "oriented height texel-surface feature mode requires feature_dim == S * (output_dim + 3) + 9");
  } else {
    TORCH_CHECK(meta.output_dim == 3, "height SV texel-surface feature mode requires output_dim == 3");
    TORCH_CHECK(meta.sv_dof > 0, "height SV texel-surface feature mode requires sv_dof > 0");
    const int stride = 3 + 6 * meta.sv_dof;
    TORCH_CHECK(
        meta.feature_dim > 9 && (meta.feature_dim - 9) % stride == 0,
        "height SV texel-surface feature mode requires feature_dim == S * (3 + 6 * sv_dof) + 9");
  }
}

void check_stream_inputs(
    const torch::Tensor& points,
    const torch::Tensor& radii,
    const torch::Tensor& densities,
    const torch::Tensor& features,
    const torch::Tensor& adjacency,
    const torch::Tensor& offsets,
    const torch::Tensor& sorted_ids,
    const torch::Tensor& screen_bounds,
    const torch::Tensor& rays,
    const ParsedMeta& meta) {
  check_inputs(points, radii, densities, features, adjacency, offsets, sorted_ids, rays, meta);
  check_i32_mps(screen_bounds, "screen_bounds");
  TORCH_CHECK(screen_bounds.dim() == 3 && screen_bounds.size(2) == 4, "screen_bounds must be [B,N,4]");
  TORCH_CHECK(
      screen_bounds.size(0) == meta.batch_size && screen_bounds.size(1) == meta.cell_count,
      "screen_bounds shape mismatch");
}

void check_adjacency_diff(const torch::Tensor& adjacency_diff, const torch::Tensor& adjacency) {
  check_float_mps(adjacency_diff, "adjacency_diff");
  TORCH_CHECK(adjacency_diff.dim() == 2 && adjacency_diff.size(1) == 4, "adjacency_diff must be [E,4]");
  TORCH_CHECK(adjacency_diff.size(0) == adjacency.size(0), "adjacency_diff edge count mismatch");
}

void check_start_ids(const torch::Tensor& start_ids, const ParsedMeta& meta) {
  check_i32_mps(start_ids, "start_ids");
  const int64_t total = (int64_t)meta.batch_size * (int64_t)meta.height * (int64_t)meta.width;
  const bool batch_start = start_ids.dim() == 1 && start_ids.size(0) == meta.batch_size;
  const bool flat_ray_start = start_ids.dim() == 1 && start_ids.size(0) == total;
  const bool grid_ray_start =
      start_ids.dim() == 3 && start_ids.size(0) == meta.batch_size && start_ids.size(1) == meta.height &&
      start_ids.size(2) == meta.width;
  TORCH_CHECK(batch_start || flat_ray_start || grid_ray_start, "start_ids must be [B], [B*H*W], or [B,H,W]");
  if (meta.start_mode == 0) {
    TORCH_CHECK(batch_start, "batch raytrace start mode requires start_ids [B]");
  } else {
    TORCH_CHECK(flat_ray_start || grid_ray_start, "per-ray raytrace start mode requires start_ids [B*H*W] or [B,H,W]");
  }
}

int64_t tiled_tiles_x(const ParsedMeta& meta) {
  constexpr int64_t tile_width = 16;
  return ((int64_t)meta.width + tile_width - 1) / tile_width;
}

int64_t tiled_tiles_y(const ParsedMeta& meta) {
  constexpr int64_t tile_width = 16;
  return ((int64_t)meta.height + tile_width - 1) / tile_width;
}

int64_t tiled_total_tiles(const ParsedMeta& meta) {
  return (int64_t)meta.batch_size * tiled_tiles_x(meta) * tiled_tiles_y(meta);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> metal_rasterize_forward(
    const torch::Tensor& points,
    const torch::Tensor& radii,
    const torch::Tensor& densities,
    const torch::Tensor& features,
    const torch::Tensor& adjacency,
    const torch::Tensor& offsets,
    const torch::Tensor& sorted_ids,
    const torch::Tensor& rays,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  auto meta = parse_meta(meta_i32, meta_f32);
  check_inputs(points, radii, densities, features, adjacency, offsets, sorted_ids, rays, meta);

  auto opts_f = points.options().dtype(torch::kFloat32);
  TORCH_CHECK(meta.feature_mode == 0, "rasterize_forward only supports constant feature mode");
  auto out = torch::empty({meta.batch_size, meta.height, meta.width, meta.output_dim}, opts_f);
  auto alpha = torch::empty({meta.batch_size, meta.height, meta.width}, opts_f);

  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const uint64_t total = (uint64_t)meta.batch_size * (uint64_t)meta.height * (uint64_t)meta.width;
  launch(k.rasterize_forward, [&](MetalKernelFunction& fn) {
    fn.setArg(0, points);
    fn.setArg(1, radii);
    fn.setArg(2, densities);
    fn.setArg(3, features);
    fn.setArg(4, adjacency);
    fn.setArg(5, offsets);
    fn.setArg(6, sorted_ids);
    fn.setArg(7, rays);
    fn.setArg(8, meta_i32);
    fn.setArg(9, meta_f32);
    fn.setArg(10, out);
    fn.setArg(11, alpha);
    fn.dispatch(total, threads);
  });

  return std::make_tuple(out, alpha);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_rasterize_train_forward(
    const torch::Tensor& points,
    const torch::Tensor& radii,
    const torch::Tensor& densities,
    const torch::Tensor& features,
    const torch::Tensor& adjacency,
    const torch::Tensor& offsets,
    const torch::Tensor& sorted_ids,
    const torch::Tensor& screen_bounds,
    const torch::Tensor& rays,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  auto meta = parse_meta(meta_i32, meta_f32);
  check_stream_inputs(points, radii, densities, features, adjacency, offsets, sorted_ids, screen_bounds, rays, meta);

  auto opts_f = points.options().dtype(torch::kFloat32);
  auto opts_i = offsets.options().dtype(torch::kInt32);
  auto out = torch::empty({meta.batch_size, meta.height, meta.width, meta.output_dim}, opts_f);
  auto alpha = torch::empty({meta.batch_size, meta.height, meta.width}, opts_f);
  auto log_t = torch::empty({meta.batch_size, meta.height, meta.width}, opts_f);
  auto pixel_stop = torch::empty({meta.batch_size, meta.height, meta.width}, opts_i);

  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const uint64_t total = (uint64_t)meta.batch_size * (uint64_t)meta.height * (uint64_t)meta.width;
  launch(k.stream_forward, [&](MetalKernelFunction& fn) {
    fn.setArg(0, points);
    fn.setArg(1, radii);
    fn.setArg(2, densities);
    fn.setArg(3, features);
    fn.setArg(4, adjacency);
    fn.setArg(5, offsets);
    fn.setArg(6, sorted_ids);
    fn.setArg(7, screen_bounds);
    fn.setArg(8, rays);
    fn.setArg(9, meta_i32);
    fn.setArg(10, meta_f32);
    fn.setArg(11, out);
    fn.setArg(12, alpha);
    fn.setArg(13, log_t);
    fn.setArg(14, pixel_stop);
    fn.dispatch(total, threads);
  });

  return std::make_tuple(out, alpha, log_t, pixel_stop);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_rasterize_train_backward(
    const torch::Tensor& points,
    const torch::Tensor& radii,
    const torch::Tensor& densities,
    const torch::Tensor& features,
    const torch::Tensor& adjacency,
    const torch::Tensor& offsets,
    const torch::Tensor& sorted_ids,
    const torch::Tensor& screen_bounds,
    const torch::Tensor& rays,
    const torch::Tensor& out_log_t,
    const torch::Tensor& pixel_stop,
    const torch::Tensor& grad_out_features,
    const torch::Tensor& grad_out_alpha,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  auto meta = parse_meta(meta_i32, meta_f32);
  check_stream_inputs(points, radii, densities, features, adjacency, offsets, sorted_ids, screen_bounds, rays, meta);
  check_float_mps(out_log_t, "out_log_t");
  check_i32_mps(pixel_stop, "pixel_stop");
  check_float_mps(grad_out_features, "grad_out_features");
  check_float_mps(grad_out_alpha, "grad_out_alpha");
  TORCH_CHECK(
      out_log_t.dim() == 3 && out_log_t.size(0) == meta.batch_size && out_log_t.size(1) == meta.height &&
          out_log_t.size(2) == meta.width,
      "out_log_t shape mismatch");
  TORCH_CHECK(
      pixel_stop.dim() == 3 && pixel_stop.size(0) == meta.batch_size && pixel_stop.size(1) == meta.height &&
          pixel_stop.size(2) == meta.width,
      "pixel_stop shape mismatch");
  TORCH_CHECK(
      grad_out_features.dim() == 4 && grad_out_features.size(0) == meta.batch_size &&
          grad_out_features.size(1) == meta.height && grad_out_features.size(2) == meta.width &&
          grad_out_features.size(3) == meta.output_dim,
      "grad_out_features shape mismatch");
  TORCH_CHECK(
      grad_out_alpha.dim() == 3 && grad_out_alpha.size(0) == meta.batch_size &&
          grad_out_alpha.size(1) == meta.height && grad_out_alpha.size(2) == meta.width,
      "grad_out_alpha shape mismatch");

  auto grad_points = torch::zeros_like(points);
  auto grad_radii = torch::zeros_like(radii);
  auto grad_densities = torch::zeros_like(densities);
  auto grad_features = torch::zeros_like(features);

  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const uint64_t total = (uint64_t)meta.batch_size * (uint64_t)meta.height * (uint64_t)meta.width;
  launch(k.stream_backward, [&](MetalKernelFunction& fn) {
    fn.setArg(0, points);
    fn.setArg(1, radii);
    fn.setArg(2, densities);
    fn.setArg(3, features);
    fn.setArg(4, adjacency);
    fn.setArg(5, offsets);
    fn.setArg(6, sorted_ids);
    fn.setArg(7, screen_bounds);
    fn.setArg(8, rays);
    fn.setArg(9, out_log_t);
    fn.setArg(10, pixel_stop);
    fn.setArg(11, grad_out_features);
    fn.setArg(12, grad_out_alpha);
    fn.setArg(13, meta_i32);
    fn.setArg(14, meta_f32);
    fn.setArg(15, grad_points);
    fn.setArg(16, grad_radii);
    fn.setArg(17, grad_densities);
    fn.setArg(18, grad_features);
    fn.dispatch(total, threads);
  });

  return std::make_tuple(grad_points, grad_radii, grad_densities, grad_features);
}

torch::Tensor metal_rasterize_tiled_count(
    const torch::Tensor& screen_bounds,
    const torch::Tensor& sorted_ids,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  auto meta = parse_meta(meta_i32, meta_f32);
  check_i32_mps(screen_bounds, "screen_bounds");
  check_i32_mps(sorted_ids, "sorted_ids");
  TORCH_CHECK(screen_bounds.dim() == 3 && screen_bounds.size(2) == 4, "screen_bounds must be [B,N,4]");
  TORCH_CHECK(sorted_ids.dim() == 2, "sorted_ids must be [B,N]");
  TORCH_CHECK(
      screen_bounds.size(0) == meta.batch_size && screen_bounds.size(1) == meta.cell_count,
      "screen_bounds shape mismatch");
  TORCH_CHECK(sorted_ids.size(0) == meta.batch_size && sorted_ids.size(1) == meta.cell_count, "sorted_ids shape mismatch");

  const int64_t total_tiles = tiled_total_tiles(meta);
  auto counts = torch::empty({total_tiles + 1}, screen_bounds.options().dtype(torch::kInt32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.tiled_count, [&](MetalKernelFunction& fn) {
    fn.setArg(0, screen_bounds);
    fn.setArg(1, sorted_ids);
    fn.setArg(2, meta_i32);
    fn.setArg(3, counts);
    fn.dispatch((uint64_t)total_tiles, threads);
  });
  return counts;
}

torch::Tensor metal_rasterize_tiled_write(
    const torch::Tensor& screen_bounds,
    const torch::Tensor& sorted_ids,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  auto meta = parse_meta(meta_i32, meta_f32);
  check_i32_mps(screen_bounds, "screen_bounds");
  check_i32_mps(sorted_ids, "sorted_ids");
  check_i32_mps(tile_offsets, "tile_offsets");
  const int64_t total_tiles = tiled_total_tiles(meta);
  TORCH_CHECK(tile_offsets.dim() == 1 && tile_offsets.size(0) == total_tiles + 1, "tile_offsets shape mismatch");
  auto offsets_cpu = tile_offsets.cpu();
  const int32_t* offsets_ptr = offsets_cpu.data_ptr<int32_t>();
  int64_t candidate_count = (int64_t)offsets_ptr[total_tiles];
  TORCH_CHECK(candidate_count >= 0, "negative tiled candidate count");
  auto tile_cell_ids = torch::empty({candidate_count}, screen_bounds.options().dtype(torch::kInt32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.tiled_write, [&](MetalKernelFunction& fn) {
    fn.setArg(0, screen_bounds);
    fn.setArg(1, sorted_ids);
    fn.setArg(2, tile_offsets);
    fn.setArg(3, meta_i32);
    fn.setArg(4, tile_cell_ids);
    fn.dispatch((uint64_t)total_tiles, threads);
  });
  return tile_cell_ids;
}

torch::Tensor metal_rasterize_tiled_emit_count(
    const torch::Tensor& screen_bounds,
    const torch::Tensor& sorted_ids,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  auto meta = parse_meta(meta_i32, meta_f32);
  check_i32_mps(screen_bounds, "screen_bounds");
  check_i32_mps(sorted_ids, "sorted_ids");
  TORCH_CHECK(screen_bounds.dim() == 3 && screen_bounds.size(2) == 4, "screen_bounds must be [B,N,4]");
  TORCH_CHECK(sorted_ids.dim() == 2, "sorted_ids must be [B,N]");
  TORCH_CHECK(
      screen_bounds.size(0) == meta.batch_size && screen_bounds.size(1) == meta.cell_count,
      "screen_bounds shape mismatch");
  TORCH_CHECK(sorted_ids.size(0) == meta.batch_size && sorted_ids.size(1) == meta.cell_count, "sorted_ids shape mismatch");

  const int64_t total_tiles = tiled_total_tiles(meta);
  const int64_t total_orders = (int64_t)meta.batch_size * (int64_t)meta.cell_count;
  auto counts = torch::zeros({total_tiles + 1}, screen_bounds.options().dtype(torch::kInt32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.tiled_emit_count, [&](MetalKernelFunction& fn) {
    fn.setArg(0, screen_bounds);
    fn.setArg(1, sorted_ids);
    fn.setArg(2, meta_i32);
    fn.setArg(3, counts);
    fn.dispatch((uint64_t)total_orders, threads);
  });
  return counts;
}

std::tuple<torch::Tensor, torch::Tensor> metal_rasterize_tiled_emit_write(
    const torch::Tensor& screen_bounds,
    const torch::Tensor& sorted_ids,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  auto meta = parse_meta(meta_i32, meta_f32);
  check_i32_mps(screen_bounds, "screen_bounds");
  check_i32_mps(sorted_ids, "sorted_ids");
  check_i32_mps(tile_offsets, "tile_offsets");
  const int64_t total_tiles = tiled_total_tiles(meta);
  const int64_t total_orders = (int64_t)meta.batch_size * (int64_t)meta.cell_count;
  TORCH_CHECK(tile_offsets.dim() == 1 && tile_offsets.size(0) == total_tiles + 1, "tile_offsets shape mismatch");
  auto offsets_cpu = tile_offsets.cpu();
  const int32_t* offsets_ptr = offsets_cpu.data_ptr<int32_t>();
  int64_t candidate_count = (int64_t)offsets_ptr[total_tiles];
  TORCH_CHECK(candidate_count >= 0, "negative tiled candidate count");

  auto tile_cursors = tile_offsets.slice(0, 0, total_tiles).contiguous().clone();
  auto sort_keys = torch::empty({candidate_count}, screen_bounds.options().dtype(torch::kInt64));
  auto tile_cell_ids = torch::empty({candidate_count}, screen_bounds.options().dtype(torch::kInt32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.tiled_emit_write, [&](MetalKernelFunction& fn) {
    fn.setArg(0, screen_bounds);
    fn.setArg(1, sorted_ids);
    fn.setArg(2, tile_offsets);
    fn.setArg(3, tile_cursors);
    fn.setArg(4, meta_i32);
    fn.setArg(5, sort_keys);
    fn.setArg(6, tile_cell_ids);
    fn.dispatch((uint64_t)total_orders, threads);
  });
  return std::make_tuple(sort_keys, tile_cell_ids);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_rasterize_tiled_train_forward(
    const torch::Tensor& points,
    const torch::Tensor& radii,
    const torch::Tensor& densities,
    const torch::Tensor& features,
    const torch::Tensor& adjacency,
    const torch::Tensor& offsets,
    const torch::Tensor& adjacency_diff,
    const torch::Tensor& sorted_ids,
    const torch::Tensor& screen_bounds,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& tile_cell_ids,
    const torch::Tensor& rays,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  auto meta = parse_meta(meta_i32, meta_f32);
  check_stream_inputs(points, radii, densities, features, adjacency, offsets, sorted_ids, screen_bounds, rays, meta);
  check_adjacency_diff(adjacency_diff, adjacency);
  check_i32_mps(tile_offsets, "tile_offsets");
  check_i32_mps(tile_cell_ids, "tile_cell_ids");
  const int64_t total_tiles = tiled_total_tiles(meta);
  TORCH_CHECK(tile_offsets.dim() == 1 && tile_offsets.size(0) == total_tiles + 1, "tile_offsets shape mismatch");
  TORCH_CHECK(tile_cell_ids.dim() == 1, "tile_cell_ids must be [K]");

  auto opts_f = points.options().dtype(torch::kFloat32);
  auto opts_i = offsets.options().dtype(torch::kInt32);
  auto out = torch::empty({meta.batch_size, meta.height, meta.width, meta.output_dim}, opts_f);
  auto alpha = torch::empty({meta.batch_size, meta.height, meta.width}, opts_f);
  auto normal_distance = torch::empty({meta.batch_size, meta.height, meta.width}, opts_f);
  auto log_t = torch::empty({meta.batch_size, meta.height, meta.width}, opts_f);
  auto tile_stop = torch::empty({total_tiles}, opts_i);

  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.tiled_forward, [&](MetalKernelFunction& fn) {
    fn.setArg(0, points);
    fn.setArg(1, radii);
    fn.setArg(2, densities);
    fn.setArg(3, features);
    fn.setArg(4, adjacency);
    fn.setArg(5, offsets);
    fn.setArg(6, adjacency_diff);
    fn.setArg(7, screen_bounds);
    fn.setArg(8, tile_offsets);
    fn.setArg(9, tile_cell_ids);
    fn.setArg(10, rays);
    fn.setArg(11, meta_i32);
    fn.setArg(12, meta_f32);
    fn.setArg(13, out);
    fn.setArg(14, alpha);
    fn.setArg(15, normal_distance);
    fn.setArg(16, log_t);
    fn.setArg(17, tile_stop);
    fn.dispatch((uint64_t)total_tiles * threads, threads);
  });

  return std::make_tuple(out, alpha, normal_distance, log_t, tile_stop);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_rasterize_tiled_aux_forward(
    const torch::Tensor& points,
    const torch::Tensor& radii,
    const torch::Tensor& densities,
    const torch::Tensor& features,
    const torch::Tensor& adjacency,
    const torch::Tensor& offsets,
    const torch::Tensor& adjacency_diff,
    const torch::Tensor& sorted_ids,
    const torch::Tensor& screen_bounds,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& tile_cell_ids,
    const torch::Tensor& rays,
    const torch::Tensor& target_features,
    const torch::Tensor& depth_quantiles,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  auto meta = parse_meta(meta_i32, meta_f32);
  check_stream_inputs(points, radii, densities, features, adjacency, offsets, sorted_ids, screen_bounds, rays, meta);
  check_adjacency_diff(adjacency_diff, adjacency);
  check_i32_mps(tile_offsets, "tile_offsets");
  check_i32_mps(tile_cell_ids, "tile_cell_ids");
  check_float_mps(target_features, "target_features");
  check_float_mps(depth_quantiles, "depth_quantiles");
  const int64_t total_tiles = tiled_total_tiles(meta);
  TORCH_CHECK(tile_offsets.dim() == 1 && tile_offsets.size(0) == total_tiles + 1, "tile_offsets shape mismatch");
  TORCH_CHECK(tile_cell_ids.dim() == 1, "tile_cell_ids must be [K]");
  TORCH_CHECK(depth_quantiles.dim() == 1, "depth_quantiles must be [Q]");
  TORCH_CHECK(depth_quantiles.size(0) == meta.depth_quantile_count, "depth_quantiles count mismatch");
  TORCH_CHECK(meta.depth_quantile_count >= 1, "depth_quantile_count must be positive");
  TORCH_CHECK(
      target_features.dim() == 4 && target_features.size(0) == meta.batch_size &&
          target_features.size(1) == meta.height && target_features.size(2) == meta.width &&
          target_features.size(3) == meta.output_dim,
      "target_features shape mismatch");

  auto opts_f = points.options().dtype(torch::kFloat32);
  auto opts_i = offsets.options().dtype(torch::kInt32);
  auto normal_distance = torch::empty({meta.batch_size, meta.height, meta.width}, opts_f);
  auto normal = torch::empty({meta.batch_size, meta.height, meta.width, 3}, opts_f);
  auto depth_quantile_depths =
      torch::empty({meta.batch_size, meta.height, meta.width, meta.depth_quantile_count}, opts_f);
  auto contrib = torch::zeros({meta.batch_size, meta.cell_count}, opts_f);
  auto point_error = torch::zeros({meta.batch_size, meta.cell_count}, opts_f);
  auto visible = torch::zeros({meta.batch_size, meta.cell_count}, opts_i);

  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.tiled_aux_forward, [&](MetalKernelFunction& fn) {
    fn.setArg(0, points);
    fn.setArg(1, radii);
    fn.setArg(2, densities);
    fn.setArg(3, features);
    fn.setArg(4, adjacency);
    fn.setArg(5, offsets);
    fn.setArg(6, adjacency_diff);
    fn.setArg(7, screen_bounds);
    fn.setArg(8, tile_offsets);
    fn.setArg(9, tile_cell_ids);
    fn.setArg(10, rays);
    fn.setArg(11, meta_i32);
    fn.setArg(12, meta_f32);
    fn.setArg(13, target_features);
    fn.setArg(14, depth_quantiles);
    fn.setArg(15, normal_distance);
    fn.setArg(16, normal);
    fn.setArg(17, depth_quantile_depths);
    fn.setArg(18, contrib);
    fn.setArg(19, point_error);
    fn.setArg(20, visible);
    fn.dispatch((uint64_t)total_tiles * threads, threads);
  });

  return std::make_tuple(normal_distance, normal, depth_quantile_depths, contrib, point_error, visible);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_rasterize_tiled_train_backward(
    const torch::Tensor& points,
    const torch::Tensor& radii,
    const torch::Tensor& densities,
    const torch::Tensor& features,
    const torch::Tensor& adjacency,
    const torch::Tensor& offsets,
    const torch::Tensor& adjacency_diff,
    const torch::Tensor& sorted_ids,
    const torch::Tensor& screen_bounds,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& tile_cell_ids,
    const torch::Tensor& tile_stop,
    const torch::Tensor& rays,
    const torch::Tensor& out_log_t,
    const torch::Tensor& grad_out_features,
    const torch::Tensor& grad_out_alpha,
    const torch::Tensor& grad_out_normal_distance,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  auto meta = parse_meta(meta_i32, meta_f32);
  check_stream_inputs(points, radii, densities, features, adjacency, offsets, sorted_ids, screen_bounds, rays, meta);
  check_adjacency_diff(adjacency_diff, adjacency);
  check_i32_mps(tile_offsets, "tile_offsets");
  check_i32_mps(tile_cell_ids, "tile_cell_ids");
  check_i32_mps(tile_stop, "tile_stop");
  check_float_mps(out_log_t, "out_log_t");
  check_float_mps(grad_out_features, "grad_out_features");
  check_float_mps(grad_out_alpha, "grad_out_alpha");
  check_float_mps(grad_out_normal_distance, "grad_out_normal_distance");
  const int64_t total_tiles = tiled_total_tiles(meta);
  TORCH_CHECK(tile_offsets.dim() == 1 && tile_offsets.size(0) == total_tiles + 1, "tile_offsets shape mismatch");
  TORCH_CHECK(tile_cell_ids.dim() == 1, "tile_cell_ids must be [K]");
  TORCH_CHECK(tile_stop.dim() == 1 && tile_stop.size(0) == total_tiles, "tile_stop shape mismatch");
  TORCH_CHECK(
      out_log_t.dim() == 3 && out_log_t.size(0) == meta.batch_size && out_log_t.size(1) == meta.height &&
          out_log_t.size(2) == meta.width,
      "out_log_t shape mismatch");
  TORCH_CHECK(
      grad_out_features.dim() == 4 && grad_out_features.size(0) == meta.batch_size &&
          grad_out_features.size(1) == meta.height && grad_out_features.size(2) == meta.width &&
          grad_out_features.size(3) == meta.output_dim,
      "grad_out_features shape mismatch");
  TORCH_CHECK(
      grad_out_alpha.dim() == 3 && grad_out_alpha.size(0) == meta.batch_size &&
          grad_out_alpha.size(1) == meta.height && grad_out_alpha.size(2) == meta.width,
      "grad_out_alpha shape mismatch");
  const bool has_normal_distance_grad = grad_out_normal_distance.numel() != 0;
  if (has_normal_distance_grad) {
    TORCH_CHECK(
        grad_out_normal_distance.dim() == 3 && grad_out_normal_distance.size(0) == meta.batch_size &&
            grad_out_normal_distance.size(1) == meta.height && grad_out_normal_distance.size(2) == meta.width,
        "grad_out_normal_distance shape mismatch");
  }

  auto grad_points = torch::zeros_like(points);
  auto grad_radii = torch::zeros_like(radii);
  auto grad_densities = torch::zeros_like(densities);
  auto grad_features = torch::zeros_like(features);

  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  if (meta.feature_mode == 6 && meta.feature_dim <= 128 && !has_normal_distance_grad) {
    launch(k.tiled_backward_height_sv_feature_reduced, [&](MetalKernelFunction& fn) {
      fn.setArg(0, points);
      fn.setArg(1, radii);
      fn.setArg(2, densities);
      fn.setArg(3, features);
      fn.setArg(4, adjacency);
      fn.setArg(5, offsets);
      fn.setArg(6, adjacency_diff);
      fn.setArg(7, screen_bounds);
      fn.setArg(8, tile_offsets);
      fn.setArg(9, tile_cell_ids);
      fn.setArg(10, tile_stop);
      fn.setArg(11, rays);
      fn.setArg(12, out_log_t);
      fn.setArg(13, grad_out_features);
      fn.setArg(14, grad_out_alpha);
      fn.setArg(15, meta_i32);
      fn.setArg(16, meta_f32);
      fn.setArg(17, grad_points);
      fn.setArg(18, grad_radii);
      fn.setArg(19, grad_densities);
      fn.setArg(20, grad_features);
      fn.dispatch((uint64_t)total_tiles * threads, threads);
    });
    return std::make_tuple(grad_points, grad_radii, grad_densities, grad_features);
  }

  auto backward_kernel = k.tiled_backward;
  if (meta.feature_mode == 0) {
    backward_kernel = k.tiled_backward_constant_reduced;
  }
  launch(backward_kernel, [&](MetalKernelFunction& fn) {
    fn.setArg(0, points);
    fn.setArg(1, radii);
    fn.setArg(2, densities);
    fn.setArg(3, features);
    fn.setArg(4, adjacency);
    fn.setArg(5, offsets);
    fn.setArg(6, adjacency_diff);
    fn.setArg(7, screen_bounds);
    fn.setArg(8, tile_offsets);
    fn.setArg(9, tile_cell_ids);
    fn.setArg(10, tile_stop);
    fn.setArg(11, rays);
    fn.setArg(12, out_log_t);
    fn.setArg(13, grad_out_features);
    fn.setArg(14, grad_out_alpha);
    fn.setArg(15, grad_out_normal_distance);
    fn.setArg(16, meta_i32);
    fn.setArg(17, meta_f32);
    fn.setArg(18, grad_points);
    fn.setArg(19, grad_radii);
    fn.setArg(20, grad_densities);
    fn.setArg(21, grad_features);
    fn.dispatch((uint64_t)total_tiles * threads, threads);
  });

  return std::make_tuple(grad_points, grad_radii, grad_densities, grad_features);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_raytrace_forward(
    const torch::Tensor& points,
    const torch::Tensor& radii,
    const torch::Tensor& densities,
    const torch::Tensor& features,
    const torch::Tensor& adjacency,
    const torch::Tensor& offsets,
    const torch::Tensor& adjacency_diff,
    const torch::Tensor& start_ids,
    const torch::Tensor& rays,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  auto meta = parse_meta(meta_i32, meta_f32);
  auto dummy_sorted_ids = torch::empty({meta.batch_size, meta.cell_count}, offsets.options().dtype(torch::kInt32));
  check_inputs(points, radii, densities, features, adjacency, offsets, dummy_sorted_ids, rays, meta);
  check_adjacency_diff(adjacency_diff, adjacency);
  check_start_ids(start_ids, meta);

  auto opts_f = points.options().dtype(torch::kFloat32);
  auto opts_i = offsets.options().dtype(torch::kInt32);
  auto out = torch::empty({meta.batch_size, meta.height, meta.width, meta.output_dim}, opts_f);
  auto alpha = torch::empty({meta.batch_size, meta.height, meta.width}, opts_f);
  auto normal_distance = torch::empty({meta.batch_size, meta.height, meta.width}, opts_f);
  auto normal = torch::empty({meta.batch_size, meta.height, meta.width, 3}, opts_f);
  auto steps = torch::empty({meta.batch_size, meta.height, meta.width}, opts_i);

  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const uint64_t total = (uint64_t)meta.batch_size * (uint64_t)meta.height * (uint64_t)meta.width;
  launch(k.raytrace_forward, [&](MetalKernelFunction& fn) {
    fn.setArg(0, points);
    fn.setArg(1, radii);
    fn.setArg(2, densities);
    fn.setArg(3, features);
    fn.setArg(4, adjacency);
    fn.setArg(5, offsets);
    fn.setArg(6, adjacency_diff);
    fn.setArg(7, start_ids);
    fn.setArg(8, rays);
    fn.setArg(9, meta_i32);
    fn.setArg(10, meta_f32);
    fn.setArg(11, out);
    fn.setArg(12, alpha);
    fn.setArg(13, normal_distance);
    fn.setArg(14, normal);
    fn.setArg(15, steps);
    fn.dispatch(total, threads);
  });

  return std::make_tuple(out, alpha, normal_distance, normal, steps);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_raytrace_height_sv_backward(
    const torch::Tensor& points,
    const torch::Tensor& radii,
    const torch::Tensor& densities,
    const torch::Tensor& features,
    const torch::Tensor& adjacency,
    const torch::Tensor& offsets,
    const torch::Tensor& adjacency_diff,
    const torch::Tensor& start_ids,
    const torch::Tensor& rays,
    const torch::Tensor& grad_out_features,
    const torch::Tensor& grad_out_alpha,
    const torch::Tensor& grad_out_normal_distance,
    const torch::Tensor& grad_out_normal,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  auto meta = parse_meta(meta_i32, meta_f32);
  auto dummy_sorted_ids = torch::empty({meta.batch_size, meta.cell_count}, offsets.options().dtype(torch::kInt32));
  check_inputs(points, radii, densities, features, adjacency, offsets, dummy_sorted_ids, rays, meta);
  check_adjacency_diff(adjacency_diff, adjacency);
  check_start_ids(start_ids, meta);
  check_float_mps(grad_out_features, "grad_out_features");
  check_float_mps(grad_out_alpha, "grad_out_alpha");
  check_float_mps(grad_out_normal_distance, "grad_out_normal_distance");
  check_float_mps(grad_out_normal, "grad_out_normal");
  TORCH_CHECK(
      grad_out_features.dim() == 4 && grad_out_features.size(0) == meta.batch_size &&
          grad_out_features.size(1) == meta.height && grad_out_features.size(2) == meta.width &&
          grad_out_features.size(3) == meta.output_dim,
      "grad_out_features shape mismatch");
  TORCH_CHECK(
      grad_out_alpha.dim() == 3 && grad_out_alpha.size(0) == meta.batch_size && grad_out_alpha.size(1) == meta.height &&
          grad_out_alpha.size(2) == meta.width,
      "grad_out_alpha shape mismatch");
  const bool has_normal_distance_grad = grad_out_normal_distance.numel() != 0;
  if (has_normal_distance_grad) {
    TORCH_CHECK(
        grad_out_normal_distance.dim() == 3 && grad_out_normal_distance.size(0) == meta.batch_size &&
            grad_out_normal_distance.size(1) == meta.height && grad_out_normal_distance.size(2) == meta.width,
        "grad_out_normal_distance shape mismatch");
  }
  const bool has_normal_grad = grad_out_normal.numel() != 0;
  if (has_normal_grad) {
    TORCH_CHECK(
        grad_out_normal.dim() == 4 && grad_out_normal.size(0) == meta.batch_size &&
            grad_out_normal.size(1) == meta.height && grad_out_normal.size(2) == meta.width &&
            grad_out_normal.size(3) == 3,
        "grad_out_normal shape mismatch");
  }
  TORCH_CHECK(meta.feature_mode == 6, "raytrace height+SV backward requires feature_mode=6");
  TORCH_CHECK(meta.output_dim == 3, "raytrace height+SV backward requires output_dim=3");
  TORCH_CHECK(meta.feature_dim <= 384, "raytrace height+SV backward currently supports feature_dim <= 384");

  auto grad_normal_arg = has_normal_distance_grad
      ? grad_out_normal_distance
      : torch::zeros({meta.batch_size, meta.height, meta.width}, points.options().dtype(torch::kFloat32));
  auto grad_rendered_normal_arg = has_normal_grad
      ? grad_out_normal
      : torch::zeros({meta.batch_size, meta.height, meta.width, 3}, points.options().dtype(torch::kFloat32));
  auto grad_points = torch::zeros_like(points);
  auto grad_radii = torch::zeros_like(radii);
  auto grad_densities = torch::zeros_like(densities);
  auto grad_features = torch::zeros_like(features);

  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const uint64_t total = (uint64_t)meta.batch_size * (uint64_t)meta.height * (uint64_t)meta.width;
  launch(k.raytrace_height_sv_backward, [&](MetalKernelFunction& fn) {
    fn.setArg(0, points);
    fn.setArg(1, radii);
    fn.setArg(2, densities);
    fn.setArg(3, features);
    fn.setArg(4, adjacency);
    fn.setArg(5, offsets);
    fn.setArg(6, adjacency_diff);
    fn.setArg(7, start_ids);
    fn.setArg(8, rays);
    fn.setArg(9, grad_out_features);
    fn.setArg(10, grad_out_alpha);
    fn.setArg(11, grad_normal_arg);
    fn.setArg(12, grad_rendered_normal_arg);
    fn.setArg(13, meta_i32);
    fn.setArg(14, meta_f32);
    fn.setArg(15, grad_points);
    fn.setArg(16, grad_radii);
    fn.setArg(17, grad_densities);
    fn.setArg(18, grad_features);
    fn.dispatch(total, threads);
  });

  return std::make_tuple(grad_points, grad_radii, grad_densities, grad_features);
}

}  // namespace powerfoam
