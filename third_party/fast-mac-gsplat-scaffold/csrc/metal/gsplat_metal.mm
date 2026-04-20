#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <vector>
#include <memory>
#include <mutex>

#include "shared/common.h"

namespace gsplat {
namespace {

static NSString* const kKernelSource = @R"METAL(
#include "gsplat_kernels.metal"
)METAL";

struct MetalContext {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  id<MTLComputePipelineState> snugbox_count_pso = nil;
  id<MTLComputePipelineState> emit_pairs_pso = nil;
  id<MTLComputePipelineState> tile_forward_pso = nil;
  id<MTLComputePipelineState> tile_backward_pso = nil;
};

MetalContext& metal_ctx() {
  static MetalContext ctx;
  static std::once_flag once;
  std::call_once(once, [&]() {
    ctx.device = MTLCreateSystemDefaultDevice();
    TORCH_CHECK(ctx.device != nil, "No Metal device found");
    ctx.queue = [ctx.device newCommandQueue];
    NSError* err = nil;
    NSString* metalPath = [[NSString stringWithUTF8String:__FILE__] stringByDeletingLastPathComponent];
    metalPath = [metalPath stringByAppendingPathComponent:@"gsplat_kernels.metal"];
    NSString* src = [NSString stringWithContentsOfFile:metalPath encoding:NSUTF8StringEncoding error:&err];
    TORCH_CHECK(src != nil, "Failed to read gsplat_kernels.metal: ", err.localizedDescription.UTF8String);
    ctx.library = [ctx.device newLibraryWithSource:src options:nil error:&err];
    TORCH_CHECK(ctx.library != nil, "Failed to compile Metal library: ", err.localizedDescription.UTF8String);
    auto make_pso = ^id<MTLComputePipelineState>(NSString* name) {
      id<MTLFunction> fn = [ctx.library newFunctionWithName:name];
      TORCH_CHECK(fn != nil, "Missing Metal kernel: ", name.UTF8String);
      NSError* localErr = nil;
      id<MTLComputePipelineState> pso = [ctx.device newComputePipelineStateWithFunction:fn error:&localErr];
      TORCH_CHECK(pso != nil, "Failed to create pipeline for ", name.UTF8String, ": ", localErr.localizedDescription.UTF8String);
      return pso;
    };
    ctx.snugbox_count_pso = make_pso(@"snugbox_count");
    ctx.emit_pairs_pso = make_pso(@"emit_pairs");
    ctx.tile_forward_pso = make_pso(@"tile_forward");
    ctx.tile_backward_pso = make_pso(@"tile_backward");
  });
  return ctx;
}

// NOTE:
// This file is a bridge scaffold. The MPS/MPSGraph tensor-to-MTLBuffer interop and command
// submission plumbing need to be completed on macOS, because this Linux sandbox cannot compile
// or exercise the Objective-C++ path. The tensor contracts and stage layout are already aligned.

void check_inputs(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& depths) {
  TORCH_CHECK(means2d.device().is_mps(), "means2d must be on MPS");
  TORCH_CHECK(conics.device().is_mps(), "conics must be on MPS");
  TORCH_CHECK(colors.device().is_mps(), "colors must be on MPS");
  TORCH_CHECK(opacities.device().is_mps(), "opacities must be on MPS");
  TORCH_CHECK(depths.device().is_mps(), "depths must be on MPS");
  TORCH_CHECK(means2d.scalar_type() == torch::kFloat32, "means2d must be float32");
  TORCH_CHECK(conics.scalar_type() == torch::kFloat32, "conics must be float32");
  TORCH_CHECK(colors.scalar_type() == torch::kFloat32, "colors must be float32");
  TORCH_CHECK(opacities.scalar_type() == torch::kFloat32, "opacities must be float32");
  TORCH_CHECK(depths.scalar_type() == torch::kFloat32, "depths must be float32");
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> metal_forward(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& depths,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_inputs(means2d, conics, colors, opacities, depths);
  auto meta = parse_meta(meta_i32, meta_f32);
  auto& ctx = metal_ctx();
  (void)ctx;

  auto opts_f = means2d.options().dtype(torch::kFloat32);
  auto opts_i = means2d.options().dtype(torch::kInt32);

  torch::Tensor out = torch::zeros({meta.height, meta.width, 3}, opts_f);

  // Aux bundle is a tensor pack placeholder for the Python autograd wrapper.
  // Layout:
  // [0] pair_count prefix total N
  // [1] reserved
  torch::Tensor aux = torch::zeros({2}, opts_i);

  // Stage outputs to persist for backward. In a real build you will likely carry:
  // bbox [G,4], tau [G], tile_ids [N], gauss_ids [N], tile_ranges [T+1], out_T [H*W]
  // either as explicit tensors returned through aux packing or as an opaque handle.

  // This scaffold intentionally stops at the API boundary because tensor<->MTLBuffer interop,
  // GPU scan, GPU sort, and command submission require building on macOS.
  TORCH_CHECK(false,
      "metal_forward scaffold generated successfully, but the macOS-specific MTLBuffer/MPS interop "
      "and command submission still need to be completed on an Apple machine.");

  return std::make_tuple(out, aux);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& depths,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& aux) {
  check_inputs(means2d, conics, colors, opacities, depths);
  auto opts = means2d.options().dtype(torch::kFloat32);
  auto g_means2d = torch::zeros_like(means2d, opts);
  auto g_conics = torch::zeros_like(conics, opts);
  auto g_colors = torch::zeros_like(colors, opts);
  auto g_opacities = torch::zeros_like(opacities, opts);
  auto g_depths = torch::zeros_like(depths, opts);

  TORCH_CHECK(false,
      "metal_backward scaffold generated successfully, but the macOS-specific MTLBuffer/MPS interop "
      "and command submission still need to be completed on an Apple machine.");

  return std::make_tuple(g_means2d, g_conics, g_colors, g_opacities, g_depths);
}

}  // namespace gsplat
