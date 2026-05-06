#include <torch/extension.h>

#include "shared/common.h"

namespace powerfoam {
namespace {

std::tuple<torch::Tensor, torch::Tensor> rasterize_forward_dispatch(
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
#if defined(__APPLE__)
  if (points.device().is_mps()) {
    return metal_rasterize_forward(
        points, radii, densities, features, adjacency, offsets, sorted_ids, rays, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "powerfoam_metal.rasterize_forward: no backend available for device ", points.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize_train_forward_dispatch(
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
#if defined(__APPLE__)
  if (points.device().is_mps()) {
    return metal_rasterize_train_forward(
        points,
        radii,
        densities,
        features,
        adjacency,
        offsets,
        sorted_ids,
        screen_bounds,
        rays,
        meta_i32,
        meta_f32);
  }
#endif
  TORCH_CHECK(false, "powerfoam_metal.rasterize_train_forward: no backend available for device ", points.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize_train_backward_dispatch(
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
#if defined(__APPLE__)
  if (points.device().is_mps()) {
    return metal_rasterize_train_backward(
        points,
        radii,
        densities,
        features,
        adjacency,
        offsets,
        sorted_ids,
        screen_bounds,
        rays,
        out_log_t,
        pixel_stop,
        grad_out_features,
        grad_out_alpha,
        meta_i32,
        meta_f32);
  }
#endif
  TORCH_CHECK(false, "powerfoam_metal.rasterize_train_backward: no backend available for device ", points.device());
}

torch::Tensor rasterize_tiled_count_dispatch(
    const torch::Tensor& screen_bounds,
    const torch::Tensor& sorted_ids,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (screen_bounds.device().is_mps()) {
    return metal_rasterize_tiled_count(screen_bounds, sorted_ids, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "powerfoam_metal.rasterize_tiled_count: no backend available for device ", screen_bounds.device());
}

torch::Tensor rasterize_tiled_write_dispatch(
    const torch::Tensor& screen_bounds,
    const torch::Tensor& sorted_ids,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (screen_bounds.device().is_mps()) {
    return metal_rasterize_tiled_write(screen_bounds, sorted_ids, tile_offsets, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "powerfoam_metal.rasterize_tiled_write: no backend available for device ", screen_bounds.device());
}

torch::Tensor rasterize_tiled_emit_count_dispatch(
    const torch::Tensor& screen_bounds,
    const torch::Tensor& sorted_ids,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (screen_bounds.device().is_mps()) {
    return metal_rasterize_tiled_emit_count(screen_bounds, sorted_ids, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "powerfoam_metal.rasterize_tiled_emit_count: no backend available for device ", screen_bounds.device());
}

std::tuple<torch::Tensor, torch::Tensor> rasterize_tiled_emit_write_dispatch(
    const torch::Tensor& screen_bounds,
    const torch::Tensor& sorted_ids,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (screen_bounds.device().is_mps()) {
    return metal_rasterize_tiled_emit_write(screen_bounds, sorted_ids, tile_offsets, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "powerfoam_metal.rasterize_tiled_emit_write: no backend available for device ", screen_bounds.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize_tiled_train_forward_dispatch(
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
#if defined(__APPLE__)
  if (points.device().is_mps()) {
    return metal_rasterize_tiled_train_forward(
        points,
        radii,
        densities,
        features,
        adjacency,
        offsets,
        adjacency_diff,
        sorted_ids,
        screen_bounds,
        tile_offsets,
        tile_cell_ids,
        rays,
        meta_i32,
        meta_f32);
  }
#endif
  TORCH_CHECK(false, "powerfoam_metal.rasterize_tiled_train_forward: no backend available for device ", points.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize_tiled_train_backward_dispatch(
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
#if defined(__APPLE__)
  if (points.device().is_mps()) {
    return metal_rasterize_tiled_train_backward(
        points,
        radii,
        densities,
        features,
        adjacency,
        offsets,
        adjacency_diff,
        sorted_ids,
        screen_bounds,
        tile_offsets,
        tile_cell_ids,
        tile_stop,
        rays,
        out_log_t,
        grad_out_features,
        grad_out_alpha,
        grad_out_normal_distance,
        meta_i32,
        meta_f32);
  }
#endif
  TORCH_CHECK(false, "powerfoam_metal.rasterize_tiled_train_backward: no backend available for device ", points.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize_tiled_aux_forward_dispatch(
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
#if defined(__APPLE__)
  if (points.device().is_mps()) {
    return metal_rasterize_tiled_aux_forward(
        points,
        radii,
        densities,
        features,
        adjacency,
        offsets,
        adjacency_diff,
        sorted_ids,
        screen_bounds,
        tile_offsets,
        tile_cell_ids,
        rays,
        target_features,
        depth_quantiles,
        meta_i32,
        meta_f32);
  }
#endif
  TORCH_CHECK(false, "powerfoam_metal.rasterize_tiled_aux_forward: no backend available for device ", points.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> raytrace_forward_dispatch(
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
#if defined(__APPLE__)
  if (points.device().is_mps()) {
    return metal_raytrace_forward(
        points,
        radii,
        densities,
        features,
        adjacency,
        offsets,
        adjacency_diff,
        start_ids,
        rays,
        meta_i32,
        meta_f32);
  }
#endif
  TORCH_CHECK(false, "powerfoam_metal.raytrace_forward: no backend available for device ", points.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> raytrace_height_sv_backward_dispatch(
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
#if defined(__APPLE__)
  if (points.device().is_mps()) {
    return metal_raytrace_height_sv_backward(
        points,
        radii,
        densities,
        features,
        adjacency,
        offsets,
        adjacency_diff,
        start_ids,
        rays,
        grad_out_features,
        grad_out_alpha,
        grad_out_normal_distance,
        grad_out_normal,
        meta_i32,
        meta_f32);
  }
#endif
  TORCH_CHECK(false, "powerfoam_metal.raytrace_height_sv_backward: no backend available for device ", points.device());
}

}  // namespace
}  // namespace powerfoam

TORCH_LIBRARY(powerfoam_metal, m) {
  m.def(
      "rasterize_forward(Tensor points, Tensor radii, Tensor densities, Tensor features, Tensor adjacency, Tensor offsets, Tensor sorted_ids, Tensor rays, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor)");
  m.def(
      "rasterize_train_forward(Tensor points, Tensor radii, Tensor densities, Tensor features, Tensor adjacency, Tensor offsets, Tensor sorted_ids, Tensor screen_bounds, Tensor rays, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "rasterize_train_backward(Tensor points, Tensor radii, Tensor densities, Tensor features, Tensor adjacency, Tensor offsets, Tensor sorted_ids, Tensor screen_bounds, Tensor rays, Tensor out_log_t, Tensor pixel_stop, Tensor grad_out_features, Tensor grad_out_alpha, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "rasterize_tiled_count(Tensor screen_bounds, Tensor sorted_ids, Tensor meta_i32, Tensor meta_f32) -> Tensor");
  m.def(
      "rasterize_tiled_write(Tensor screen_bounds, Tensor sorted_ids, Tensor tile_offsets, Tensor meta_i32, Tensor meta_f32) -> Tensor");
  m.def(
      "rasterize_tiled_emit_count(Tensor screen_bounds, Tensor sorted_ids, Tensor meta_i32, Tensor meta_f32) -> Tensor");
  m.def(
      "rasterize_tiled_emit_write(Tensor screen_bounds, Tensor sorted_ids, Tensor tile_offsets, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor)");
  m.def(
      "rasterize_tiled_train_forward(Tensor points, Tensor radii, Tensor densities, Tensor features, Tensor adjacency, Tensor offsets, Tensor adjacency_diff, Tensor sorted_ids, Tensor screen_bounds, Tensor tile_offsets, Tensor tile_cell_ids, Tensor rays, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "rasterize_tiled_train_backward(Tensor points, Tensor radii, Tensor densities, Tensor features, Tensor adjacency, Tensor offsets, Tensor adjacency_diff, Tensor sorted_ids, Tensor screen_bounds, Tensor tile_offsets, Tensor tile_cell_ids, Tensor tile_stop, Tensor rays, Tensor out_log_t, Tensor grad_out_features, Tensor grad_out_alpha, Tensor grad_out_normal_distance, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "rasterize_tiled_aux_forward(Tensor points, Tensor radii, Tensor densities, Tensor features, Tensor adjacency, Tensor offsets, Tensor adjacency_diff, Tensor sorted_ids, Tensor screen_bounds, Tensor tile_offsets, Tensor tile_cell_ids, Tensor rays, Tensor target_features, Tensor depth_quantiles, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "raytrace_forward(Tensor points, Tensor radii, Tensor densities, Tensor features, Tensor adjacency, Tensor offsets, Tensor adjacency_diff, Tensor start_ids, Tensor rays, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "raytrace_height_sv_backward(Tensor points, Tensor radii, Tensor densities, Tensor features, Tensor adjacency, Tensor offsets, Tensor adjacency_diff, Tensor start_ids, Tensor rays, Tensor grad_out_features, Tensor grad_out_alpha, Tensor grad_out_normal_distance, Tensor grad_out_normal, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(powerfoam_metal, CompositeExplicitAutograd, m) {
  m.impl("rasterize_forward", powerfoam::rasterize_forward_dispatch);
  m.impl("rasterize_train_forward", powerfoam::rasterize_train_forward_dispatch);
  m.impl("rasterize_train_backward", powerfoam::rasterize_train_backward_dispatch);
  m.impl("rasterize_tiled_count", powerfoam::rasterize_tiled_count_dispatch);
  m.impl("rasterize_tiled_write", powerfoam::rasterize_tiled_write_dispatch);
  m.impl("rasterize_tiled_emit_count", powerfoam::rasterize_tiled_emit_count_dispatch);
  m.impl("rasterize_tiled_emit_write", powerfoam::rasterize_tiled_emit_write_dispatch);
  m.impl("rasterize_tiled_train_forward", powerfoam::rasterize_tiled_train_forward_dispatch);
  m.impl("rasterize_tiled_train_backward", powerfoam::rasterize_tiled_train_backward_dispatch);
  m.impl("rasterize_tiled_aux_forward", powerfoam::rasterize_tiled_aux_forward_dispatch);
  m.impl("raytrace_forward", powerfoam::raytrace_forward_dispatch);
  m.impl("raytrace_height_sv_backward", powerfoam::raytrace_height_sv_backward_dispatch);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
