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
  NSString* forwardPath = [metalPath stringByAppendingPathComponent:@"dynamic_powerfoam_kernels.metal"];
  NSString* streamPath = [metalPath stringByAppendingPathComponent:@"dynamic_powerfoam_streaming_kernels.metal"];
  NSError* err = nil;
  NSString* forwardSrc = [NSString stringWithContentsOfFile:forwardPath encoding:NSUTF8StringEncoding error:&err];
  TORCH_CHECK(forwardSrc != nil, "Failed to read dynamic_powerfoam_kernels.metal: ", err.localizedDescription.UTF8String);
  err = nil;
  NSString* streamSrc = [NSString stringWithContentsOfFile:streamPath encoding:NSUTF8StringEncoding error:&err];
  TORCH_CHECK(streamSrc != nil, "Failed to read dynamic_powerfoam_streaming_kernels.metal: ", err.localizedDescription.UTF8String);
  return std::string([forwardSrc UTF8String]) + "\n" + std::string([streamSrc UTF8String]);
}

struct MetalDynamicPowerFoamKernels {
  std::shared_ptr<MetalKernelFunction> rasterize_forward;
  std::shared_ptr<MetalKernelFunction> stream_forward;
  std::shared_ptr<MetalKernelFunction> stream_backward;
};

MetalDynamicPowerFoamKernels& kernels() {
  static std::once_flag once;
  static std::unique_ptr<DynamicMetalShaderLibrary> lib;
  static MetalDynamicPowerFoamKernels out;
  std::call_once(once, []() {
    lib = std::make_unique<DynamicMetalShaderLibrary>(load_shader_source());
    out.rasterize_forward = lib->getKernelFunction("dynamic_powerfoam_rasterize_forward");
    out.stream_forward = lib->getKernelFunction("dynamic_powerfoam_stream_forward");
    out.stream_backward = lib->getKernelFunction("dynamic_powerfoam_stream_backward_global_atomic");
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
          meta.feature_mode == 4,
      "unsupported feature mode");
  if (meta.feature_mode == 0) {
    TORCH_CHECK(meta.output_dim == meta.feature_dim, "constant feature mode requires output_dim == feature_dim");
  } else if (meta.feature_mode == 1 || meta.feature_mode == 2) {
    TORCH_CHECK(meta.feature_dim == meta.output_dim * 4, "linear feature mode requires feature_dim == output_dim * 4");
  } else if (meta.feature_mode == 3) {
    TORCH_CHECK(
        meta.feature_dim == meta.output_dim * 4 + 3,
        "oriented surface-linear feature mode requires feature_dim == output_dim * 4 + 3");
  } else {
    const int stride = meta.output_dim + 2;
    TORCH_CHECK(
        meta.feature_dim > 9 && (meta.feature_dim - 9) % stride == 0,
        "oriented texel-surface feature mode requires feature_dim == S * (output_dim + 2) + 9");
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

}  // namespace powerfoam
