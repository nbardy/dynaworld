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
  TORCH_CHECK(false, "dynamic_powerfoam_metal.rasterize_forward: no backend available for device ", points.device());
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
  TORCH_CHECK(false, "dynamic_powerfoam_metal.rasterize_train_forward: no backend available for device ", points.device());
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
  TORCH_CHECK(false, "dynamic_powerfoam_metal.rasterize_train_backward: no backend available for device ", points.device());
}

}  // namespace
}  // namespace powerfoam

TORCH_LIBRARY(dynamic_powerfoam_metal, m) {
  m.def(
      "rasterize_forward(Tensor points, Tensor radii, Tensor densities, Tensor features, Tensor adjacency, Tensor offsets, Tensor sorted_ids, Tensor rays, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor)");
  m.def(
      "rasterize_train_forward(Tensor points, Tensor radii, Tensor densities, Tensor features, Tensor adjacency, Tensor offsets, Tensor sorted_ids, Tensor screen_bounds, Tensor rays, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "rasterize_train_backward(Tensor points, Tensor radii, Tensor densities, Tensor features, Tensor adjacency, Tensor offsets, Tensor sorted_ids, Tensor screen_bounds, Tensor rays, Tensor out_log_t, Tensor pixel_stop, Tensor grad_out_features, Tensor grad_out_alpha, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(dynamic_powerfoam_metal, CompositeExplicitAutograd, m) {
  m.impl("rasterize_forward", powerfoam::rasterize_forward_dispatch);
  m.impl("rasterize_train_forward", powerfoam::rasterize_train_forward_dispatch);
  m.impl("rasterize_train_backward", powerfoam::rasterize_train_backward_dispatch);
}
