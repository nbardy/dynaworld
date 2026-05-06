#pragma once

#include <torch/extension.h>
#include <tuple>

namespace powerfoam {

struct ParsedMeta {
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

  float near_plane;
  float alpha_threshold;
  float transmittance_threshold;
  float max_alpha;
  float eps;
  float texel_temperature;
};

inline ParsedMeta parse_meta(const torch::Tensor& meta_i32, const torch::Tensor& meta_f32) {
  TORCH_CHECK(meta_i32.numel() >= 5, "meta_i32 must have at least 5 values");
  TORCH_CHECK(meta_f32.numel() >= 5, "meta_f32 must have at least 5 values");
  auto mi = meta_i32.cpu();
  auto mf = meta_f32.cpu();
  auto* ip = mi.data_ptr<int32_t>();
  auto* fp = mf.data_ptr<float>();
  ParsedMeta out;
  out.batch_size = ip[0];
  out.height = ip[1];
  out.width = ip[2];
  out.cell_count = ip[3];
  out.feature_dim = ip[4];
  out.output_dim = meta_i32.numel() >= 6 ? ip[5] : out.feature_dim;
  out.feature_mode = meta_i32.numel() >= 7 ? ip[6] : 0;
  out.sv_dof = meta_i32.numel() >= 8 ? ip[7] : 0;
  out.depth_quantile_count = meta_i32.numel() >= 9 ? ip[8] : 1;
  out.start_mode = meta_i32.numel() >= 10 ? ip[9] : 0;
  out.near_plane = fp[0];
  out.alpha_threshold = fp[1];
  out.transmittance_threshold = fp[2];
  out.max_alpha = fp[3];
  out.eps = fp[4];
  out.texel_temperature = meta_f32.numel() >= 6 ? fp[5] : 10.0f;
  return out;
}

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
    const torch::Tensor& meta_f32);

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
    const torch::Tensor& meta_f32);

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
    const torch::Tensor& meta_f32);

torch::Tensor metal_rasterize_tiled_count(
    const torch::Tensor& screen_bounds,
    const torch::Tensor& sorted_ids,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

torch::Tensor metal_rasterize_tiled_write(
    const torch::Tensor& screen_bounds,
    const torch::Tensor& sorted_ids,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

torch::Tensor metal_rasterize_tiled_emit_count(
    const torch::Tensor& screen_bounds,
    const torch::Tensor& sorted_ids,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor> metal_rasterize_tiled_emit_write(
    const torch::Tensor& screen_bounds,
    const torch::Tensor& sorted_ids,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

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
    const torch::Tensor& meta_f32);

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
    const torch::Tensor& meta_f32);

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
    const torch::Tensor& meta_f32);

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
    const torch::Tensor& meta_f32);

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
    const torch::Tensor& meta_f32);

}  // namespace powerfoam
