from __future__ import annotations

"""
Metal-first projected 2D Gaussian rasterizer for MLX.

This module focuses on the real bottleneck: differentiable rasterization after 3D Gaussians
have already been projected into 2D conics.

What it implements:
- depth sort of Gaussians (separate sorting stage 1)
- SnugBox-style tight opacity-aware screen-space bounds
- exact tile / ellipse intersection tests via min-q-over-rectangle
- pair count + prefix-sum + pair emit
- per-tile forward rasterization with threadgroup staging
- low-memory backward via recompute + reverse alpha scan
- atomic gradient accumulation to per-Gaussian buffers

What it intentionally does not differentiate through:
- depth sort permutation
- tile/pixel support truncation
- bbox clipping

This is the right place to start for a Metal backend because projection can stay in MLX/PyTorch,
while the rasterizer core is where memory and wall time blow up.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

try:
    import mlx.core as mx
except Exception as exc:  # pragma: no cover - syntax-only on non-Apple machines
    mx = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _require_mlx() -> None:
    if mx is None:
        raise ImportError("mlx is required to run this module. Install MLX on Apple silicon.") from _IMPORT_ERROR


@dataclass(frozen=True)
class MetalRasterConfig:
    height: int
    width: int
    tile_size: int = 16
    chunk_size: int = 32
    alpha_threshold: float = 1.0 / 255.0
    transmittance_threshold: float = 1e-4
    background: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    dtype: str = "float32"

    @property
    def tiles_x(self) -> int:
        return (self.width + self.tile_size - 1) // self.tile_size

    @property
    def tiles_y(self) -> int:
        return (self.height + self.tile_size - 1) // self.tile_size

    @property
    def tile_count(self) -> int:
        return self.tiles_x * self.tiles_y


_KERNEL_CACHE: Dict[str, object] = {}


def _cache_key(name: str, cfg: MetalRasterConfig) -> str:
    return (
        f"{name}|H={cfg.height}|W={cfg.width}|T={cfg.tile_size}|"
        f"C={cfg.chunk_size}|ATH={cfg.alpha_threshold}|"
        f"TTH={cfg.transmittance_threshold}"
    )


def _kernel_name(name: str, cfg: MetalRasterConfig) -> str:
    alpha_tag = int(round(float(cfg.alpha_threshold) * 1_000_000_000))
    trans_tag = int(round(float(cfg.transmittance_threshold) * 1_000_000_000))
    return f"{name}_h{cfg.height}_w{cfg.width}_t{cfg.tile_size}_c{cfg.chunk_size}_ath{alpha_tag}_tth{trans_tag}"


def _meta_i32(cfg: MetalRasterConfig, g: int):
    _require_mlx()
    return mx.array(
        [
            int(cfg.height),
            int(cfg.width),
            int(cfg.tiles_y),
            int(cfg.tiles_x),
            int(cfg.tile_size),
            int(g),
            int(cfg.tile_count),
            int(cfg.chunk_size),
        ],
        mx.int32,
    )


def _meta_f32(cfg: MetalRasterConfig):
    _require_mlx()
    return mx.array(
        [
            float(cfg.alpha_threshold),
            float(cfg.transmittance_threshold),
            float(cfg.background[0]),
            float(cfg.background[1]),
            float(cfg.background[2]),
            1e-8,
        ],
        mx.float32,
    )


SNUGBOX_COUNT_SRC = r"""
#define EPS 1e-8f
uint gid = thread_position_in_grid.x;
uint G = (uint)meta_i32[5];
if (gid >= G) return;

float mx0 = means2d[gid * 2 + 0];
float my0 = means2d[gid * 2 + 1];
float a = conics[gid * 3 + 0];
float b = conics[gid * 3 + 1];
float c = conics[gid * 3 + 2];
float op = opacities[gid];

int H = meta_i32[0];
int W = meta_i32[1];
int tiles_y = meta_i32[2];
int tiles_x = meta_i32[3];
int TILE = meta_i32[4];
float alpha_thr = meta_f32[0];

bbox_out[gid * 4 + 0] = 0;
bbox_out[gid * 4 + 1] = 0;
bbox_out[gid * 4 + 2] = -1;
bbox_out[gid * 4 + 3] = -1;
pair_count[gid] = 0;
tau_out[gid] = 0.0f;

if (!(op > alpha_thr) || !(a > 0.0f) || !(c > 0.0f)) {
    return;
}

float det = a * c - b * b;
if (!(det > EPS)) {
    return;
}

float tau = -2.0f * metal::log(metal::max(alpha_thr / metal::max(op, EPS), EPS));
float half_x = metal::sqrt(metal::max(tau * c / det, 0.0f));
float half_y = metal::sqrt(metal::max(tau * a / det, 0.0f));

int px0 = metal::max(0, (int)metal::floor(mx0 - half_x - 0.5f));
int py0 = metal::max(0, (int)metal::floor(my0 - half_y - 0.5f));
int px1 = metal::min(W - 1, (int)metal::ceil(mx0 + half_x - 0.5f));
int py1 = metal::min(H - 1, (int)metal::ceil(my0 + half_y - 0.5f));

if (px0 > px1 || py0 > py1) {
    return;
}

bbox_out[gid * 4 + 0] = px0;
bbox_out[gid * 4 + 1] = py0;
bbox_out[gid * 4 + 2] = px1;
bbox_out[gid * 4 + 3] = py1;
tau_out[gid] = tau;

int tx0 = metal::max(0, px0 / TILE);
int ty0 = metal::max(0, py0 / TILE);
int tx1 = metal::min(tiles_x - 1, px1 / TILE);
int ty1 = metal::min(tiles_y - 1, py1 / TILE);

uint count = 0;
for (int ty = ty0; ty <= ty1; ++ty) {
    float ry0 = (float)(ty * TILE) + 0.5f;
    float ry1 = metal::min((float)(H - 1) + 0.5f, (float)((ty + 1) * TILE - 1) + 0.5f);
    float dy0 = ry0 - my0;
    float dy1 = ry1 - my0;
    for (int tx = tx0; tx <= tx1; ++tx) {
        float rx0 = (float)(tx * TILE) + 0.5f;
        float rx1 = metal::min((float)(W - 1) + 0.5f, (float)((tx + 1) * TILE - 1) + 0.5f);
        float dx0 = rx0 - mx0;
        float dx1 = rx1 - mx0;

        float qmin = INFINITY;
        if (mx0 >= rx0 && mx0 <= rx1 && my0 >= ry0 && my0 <= ry1) {
            qmin = 0.0f;
        }

        float dy_star = 0.0f;
        float dx_star = 0.0f;
        float q = 0.0f;

        if (c > EPS) {
            dy_star = metal::clamp(-(b / c) * dx0, dy0, dy1);
            q = a * dx0 * dx0 + 2.0f * b * dx0 * dy_star + c * dy_star * dy_star;
            qmin = metal::min(qmin, q);
            dy_star = metal::clamp(-(b / c) * dx1, dy0, dy1);
            q = a * dx1 * dx1 + 2.0f * b * dx1 * dy_star + c * dy_star * dy_star;
            qmin = metal::min(qmin, q);
        }
        if (a > EPS) {
            dx_star = metal::clamp(-(b / a) * dy0, dx0, dx1);
            q = a * dx_star * dx_star + 2.0f * b * dx_star * dy0 + c * dy0 * dy0;
            qmin = metal::min(qmin, q);
            dx_star = metal::clamp(-(b / a) * dy1, dx0, dx1);
            q = a * dx_star * dx_star + 2.0f * b * dx_star * dy1 + c * dy1 * dy1;
            qmin = metal::min(qmin, q);
        }

        q = a * dx0 * dx0 + 2.0f * b * dx0 * dy0 + c * dy0 * dy0;
        qmin = metal::min(qmin, q);
        q = a * dx0 * dx0 + 2.0f * b * dx0 * dy1 + c * dy1 * dy1;
        qmin = metal::min(qmin, q);
        q = a * dx1 * dx1 + 2.0f * b * dx1 * dy0 + c * dy0 * dy0;
        qmin = metal::min(qmin, q);
        q = a * dx1 * dx1 + 2.0f * b * dx1 * dy1 + c * dy1 * dy1;
        qmin = metal::min(qmin, q);

        if (qmin <= tau) {
            count += 1;
        }
    }
}
pair_count[gid] = (int)count;
"""

EMIT_PAIRS_SRC = r"""
#define EPS 1e-8f
uint gid = thread_position_in_grid.x;
uint G = (uint)meta_i32[5];
if (gid >= G) return;

int H = meta_i32[0];
int W = meta_i32[1];
int tiles_y = meta_i32[2];
int tiles_x = meta_i32[3];
int TILE = meta_i32[4];

int px0 = bbox_in[gid * 4 + 0];
int py0 = bbox_in[gid * 4 + 1];
int px1 = bbox_in[gid * 4 + 2];
int py1 = bbox_in[gid * 4 + 3];
if (px0 > px1 || py0 > py1) return;

float mx0 = means2d[gid * 2 + 0];
float my0 = means2d[gid * 2 + 1];
float a = conics[gid * 3 + 0];
float b = conics[gid * 3 + 1];
float c = conics[gid * 3 + 2];
float tau = tau_in[gid];

int tx0 = metal::max(0, px0 / TILE);
int ty0 = metal::max(0, py0 / TILE);
int tx1 = metal::min(tiles_x - 1, px1 / TILE);
int ty1 = metal::min(tiles_y - 1, py1 / TILE);

uint out_base = pair_offsets[gid];
uint w = 0;
for (int ty = ty0; ty <= ty1; ++ty) {
    float ry0 = (float)(ty * TILE) + 0.5f;
    float ry1 = metal::min((float)(H - 1) + 0.5f, (float)((ty + 1) * TILE - 1) + 0.5f);
    float dy0 = ry0 - my0;
    float dy1 = ry1 - my0;
    for (int tx = tx0; tx <= tx1; ++tx) {
        float rx0 = (float)(tx * TILE) + 0.5f;
        float rx1 = metal::min((float)(W - 1) + 0.5f, (float)((tx + 1) * TILE - 1) + 0.5f);
        float dx0 = rx0 - mx0;
        float dx1 = rx1 - mx0;

        float qmin = INFINITY;
        if (mx0 >= rx0 && mx0 <= rx1 && my0 >= ry0 && my0 <= ry1) {
            qmin = 0.0f;
        }
        float dy_star = 0.0f;
        float dx_star = 0.0f;
        float q = 0.0f;
        if (c > EPS) {
            dy_star = metal::clamp(-(b / c) * dx0, dy0, dy1);
            q = a * dx0 * dx0 + 2.0f * b * dx0 * dy_star + c * dy_star * dy_star;
            qmin = metal::min(qmin, q);
            dy_star = metal::clamp(-(b / c) * dx1, dy0, dy1);
            q = a * dx1 * dx1 + 2.0f * b * dx1 * dy_star + c * dy_star * dy_star;
            qmin = metal::min(qmin, q);
        }
        if (a > EPS) {
            dx_star = metal::clamp(-(b / a) * dy0, dx0, dx1);
            q = a * dx_star * dx_star + 2.0f * b * dx_star * dy0 + c * dy0 * dy0;
            qmin = metal::min(qmin, q);
            dx_star = metal::clamp(-(b / a) * dy1, dx0, dx1);
            q = a * dx_star * dx_star + 2.0f * b * dx_star * dy1 + c * dy1 * dy1;
            qmin = metal::min(qmin, q);
        }
        q = a * dx0 * dx0 + 2.0f * b * dx0 * dy0 + c * dy0 * dy0;
        qmin = metal::min(qmin, q);
        q = a * dx0 * dx0 + 2.0f * b * dx0 * dy1 + c * dy1 * dy1;
        qmin = metal::min(qmin, q);
        q = a * dx1 * dx1 + 2.0f * b * dx1 * dy0 + c * dy0 * dy0;
        qmin = metal::min(qmin, q);
        q = a * dx1 * dx1 + 2.0f * b * dx1 * dy1 + c * dy1 * dy1;
        qmin = metal::min(qmin, q);

        if (qmin <= tau) {
            uint out_idx = out_base + w;
            tile_ids[out_idx] = (uint)(ty * tiles_x + tx);
            gauss_ids[out_idx] = gid;
            w += 1;
        }
    }
}
"""

TILE_HIST_SRC = r"""
uint gid = thread_position_in_grid.x;
uint N = (uint)tile_ids_shape[0];
if (gid >= N) return;
uint tid = tile_ids[gid];
atomic_fetch_add_explicit(&tile_counts[tid], 1, memory_order_relaxed);
"""

TILE_FORWARD_SRC = r"""
#define EPS 1e-8f
uint tile_x = threadgroup_position_in_grid.x;
uint tile_y = threadgroup_position_in_grid.y;
uint lx = thread_position_in_threadgroup.x;
uint ly = thread_position_in_threadgroup.y;

int H = meta_i32[0];
int W = meta_i32[1];
int tiles_y = meta_i32[2];
int tiles_x = meta_i32[3];
int TILE = meta_i32[4];
int CHUNK = meta_i32[7];
float alpha_thr = meta_f32[0];
float trans_thr = meta_f32[1];
float bg0 = meta_f32[2];
float bg1 = meta_f32[3];
float bg2 = meta_f32[4];

if (tile_x >= (uint)tiles_x || tile_y >= (uint)tiles_y) return;
uint px = tile_x * (uint)TILE + lx;
uint py = tile_y * (uint)TILE + ly;
if (px >= (uint)W || py >= (uint)H) return;
uint tile_id = tile_y * (uint)tiles_x + tile_x;
uint begin = tile_ranges[tile_id];
uint end = tile_ranges[tile_id + 1];

threadgroup float tg_mx[64];
threadgroup float tg_my[64];
threadgroup float tg_a[64];
threadgroup float tg_b[64];
threadgroup float tg_c[64];
threadgroup float tg_op[64];
threadgroup float tg_r[64];
threadgroup float tg_g[64];
threadgroup float tg_bl[64];

float fx = (float)px + 0.5f;
float fy = (float)py + 0.5f;
float3 acc = float3(0.0f, 0.0f, 0.0f);
float T = 1.0f;
uint lidx = ly * (uint)TILE + lx;

for (uint base = begin; base < end; base += (uint)CHUNK) {
    uint chunk = metal::min((uint)CHUNK, end - base);
    if (lidx < chunk) {
        uint g = gauss_ids[base + lidx];
        tg_mx[lidx] = means2d[g * 2 + 0];
        tg_my[lidx] = means2d[g * 2 + 1];
        tg_a[lidx] = conics[g * 3 + 0];
        tg_b[lidx] = conics[g * 3 + 1];
        tg_c[lidx] = conics[g * 3 + 2];
        tg_op[lidx] = opacities[g];
        tg_r[lidx] = colors[g * 3 + 0];
        tg_g[lidx] = colors[g * 3 + 1];
        tg_bl[lidx] = colors[g * 3 + 2];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < chunk; ++j) {
        float dx = fx - tg_mx[j];
        float dy = fy - tg_my[j];
        float q = tg_a[j] * dx * dx + 2.0f * tg_b[j] * dx * dy + tg_c[j] * dy * dy;
        float alpha = tg_op[j] * metal::exp(-0.5f * q);
        alpha = metal::clamp(alpha, 0.0f, 0.9999f);
        if (alpha > alpha_thr) {
            float w = T * alpha;
            acc += w * float3(tg_r[j], tg_g[j], tg_bl[j]);
            T *= (1.0f - alpha);
            if (T < trans_thr) break;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (T < trans_thr) break;
}

uint pix = py * (uint)W + px;
out_rgb[pix * 3 + 0] = acc.x + T * bg0;
out_rgb[pix * 3 + 1] = acc.y + T * bg1;
out_rgb[pix * 3 + 2] = acc.z + T * bg2;
out_T[pix] = T;
"""

TILE_BACKWARD_SRC = r"""
#define EPS 1e-8f
uint tile_x = threadgroup_position_in_grid.x;
uint tile_y = threadgroup_position_in_grid.y;
uint lx = thread_position_in_threadgroup.x;
uint ly = thread_position_in_threadgroup.y;
uint lane = thread_index_in_simdgroup;

int H = meta_i32[0];
int W = meta_i32[1];
int tiles_y = meta_i32[2];
int tiles_x = meta_i32[3];
int TILE = meta_i32[4];
int CHUNK = meta_i32[7];
float alpha_thr = meta_f32[0];
float bg0 = meta_f32[2];
float bg1 = meta_f32[3];
float bg2 = meta_f32[4];

if (tile_x >= (uint)tiles_x || tile_y >= (uint)tiles_y) return;
uint px = tile_x * (uint)TILE + lx;
uint py = tile_y * (uint)TILE + ly;
if (px >= (uint)W || py >= (uint)H) return;
uint tile_id = tile_y * (uint)tiles_x + tile_x;
uint begin = tile_ranges[tile_id];
uint end = tile_ranges[tile_id + 1];

threadgroup float tg_mx[64];
threadgroup float tg_my[64];
threadgroup float tg_a[64];
threadgroup float tg_b[64];
threadgroup float tg_c[64];
threadgroup float tg_op[64];
threadgroup float tg_r[64];
threadgroup float tg_g[64];
threadgroup float tg_bl[64];
threadgroup uint tg_gid[64];

float fx = (float)px + 0.5f;
float fy = (float)py + 0.5f;
uint pix = py * (uint)W + px;
float3 gout = float3(cotangent[pix * 3 + 0], cotangent[pix * 3 + 1], cotangent[pix * 3 + 2]);
float gT = gout.x * bg0 + gout.y * bg1 + gout.z * bg2;
float Tcur = final_T[pix];
uint lidx = ly * (uint)TILE + lx;

for (int base = (int)end - 1; base >= (int)begin; base -= CHUNK) {
    uint chunk = (uint)metal::min((int)CHUNK, base - (int)begin + 1);
    uint chunk_begin = (uint)(base - (int)chunk + 1);
    if (lidx < chunk) {
        uint g = gauss_ids[chunk_begin + lidx];
        tg_gid[lidx] = g;
        tg_mx[lidx] = means2d[g * 2 + 0];
        tg_my[lidx] = means2d[g * 2 + 1];
        tg_a[lidx] = conics[g * 3 + 0];
        tg_b[lidx] = conics[g * 3 + 1];
        tg_c[lidx] = conics[g * 3 + 2];
        tg_op[lidx] = opacities[g];
        tg_r[lidx] = colors[g * 3 + 0];
        tg_g[lidx] = colors[g * 3 + 1];
        tg_bl[lidx] = colors[g * 3 + 2];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int jj = (int)chunk - 1; jj >= 0; --jj) {
        uint g = tg_gid[jj];
        float dx = fx - tg_mx[jj];
        float dy = fy - tg_my[jj];
        float q = tg_a[jj] * dx * dx + 2.0f * tg_b[jj] * dx * dy + tg_c[jj] * dy * dy;
        float e = metal::exp(-0.5f * q);
        float alpha = metal::clamp(tg_op[jj] * e, 0.0f, 0.9999f);
        float one_minus = metal::max(1.0f - alpha, 1e-6f);

        float gm0 = 0.0f, gm1 = 0.0f;
        float gc0 = 0.0f, gc1 = 0.0f, gc2 = 0.0f;
        float gq0 = 0.0f, gq1 = 0.0f, gq2 = 0.0f;
        float gop = 0.0f;

        if (alpha > alpha_thr) {
            float Tprev = Tcur / one_minus;
            float3 col = float3(tg_r[jj], tg_g[jj], tg_bl[jj]);
            float3 gcol = gout * (Tprev * alpha);
            float galpha = (gout.x * (Tprev * col.x) + gout.y * (Tprev * col.y) + gout.z * (Tprev * col.z)) - gT * Tprev;
            float gq = -0.5f * galpha * alpha;
            gop = galpha * e;
            gm0 = gq * (-(2.0f * tg_a[jj] * dx + 2.0f * tg_b[jj] * dy));
            gm1 = gq * (-(2.0f * tg_b[jj] * dx + 2.0f * tg_c[jj] * dy));
            gq0 = gq * (dx * dx);
            gq1 = gq * (2.0f * dx * dy);
            gq2 = gq * (dy * dy);
            gc0 = gcol.x;
            gc1 = gcol.y;
            gc2 = gcol.z;
            gT = (gout.x * (alpha * col.x) + gout.y * (alpha * col.y) + gout.z * (alpha * col.z)) + gT * one_minus;
            Tcur = Tprev;
        }

        float sm0 = simd_sum(gm0);
        float sm1 = simd_sum(gm1);
        float sc0 = simd_sum(gc0);
        float sc1 = simd_sum(gc1);
        float sc2 = simd_sum(gc2);
        float sq0 = simd_sum(gq0);
        float sq1 = simd_sum(gq1);
        float sq2 = simd_sum(gq2);
        float sop = simd_sum(gop);

        if (lane == 0) {
            atomic_fetch_add_explicit(&means_grad[g * 2 + 0], sm0, memory_order_relaxed);
            atomic_fetch_add_explicit(&means_grad[g * 2 + 1], sm1, memory_order_relaxed);
            atomic_fetch_add_explicit(&conics_grad[g * 3 + 0], sq0, memory_order_relaxed);
            atomic_fetch_add_explicit(&conics_grad[g * 3 + 1], sq1, memory_order_relaxed);
            atomic_fetch_add_explicit(&conics_grad[g * 3 + 2], sq2, memory_order_relaxed);
            atomic_fetch_add_explicit(&colors_grad[g * 3 + 0], sc0, memory_order_relaxed);
            atomic_fetch_add_explicit(&colors_grad[g * 3 + 1], sc1, memory_order_relaxed);
            atomic_fetch_add_explicit(&colors_grad[g * 3 + 2], sc2, memory_order_relaxed);
            atomic_fetch_add_explicit(&opacities_grad[g], sop, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
"""


def _kernel(name: str, cfg: MetalRasterConfig, *, input_names, output_names, source: str, atomic_outputs: bool = False):
    _require_mlx()
    key = _cache_key(name, cfg)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = mx.fast.metal_kernel(
            name=_kernel_name(name, cfg),
            input_names=list(input_names),
            output_names=list(output_names),
            source=source,
            atomic_outputs=atomic_outputs,
        )
    return _KERNEL_CACHE[key]


def _exclusive_cumsum_1d(x):
    return mx.cumsum(x, axis=0, inclusive=False)


def _pad_with_last(x, last_value):
    return mx.concatenate([x, mx.array([last_value], x.dtype)], axis=0)


def _snugbox_and_count(means2d, conics, opacities, cfg: MetalRasterConfig):
    g = int(means2d.shape[0])
    kernel = _kernel(
        "snugbox_count",
        cfg,
        input_names=["means2d", "conics", "opacities", "meta_i32", "meta_f32"],
        output_names=["bbox_out", "pair_count", "tau_out"],
        source=SNUGBOX_COUNT_SRC,
    )
    meta_i32 = _meta_i32(cfg, g)
    meta_f32 = _meta_f32(cfg)
    outputs = kernel(
        inputs=[means2d, conics, opacities, meta_i32, meta_f32],
        grid=(g, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(g, 4), (g,), (g,)],
        output_dtypes=[mx.int32, mx.int32, mx.float32],
        template=[],
    )
    return outputs[0], outputs[1], outputs[2]


def _emit_pairs(means2d, conics, bbox, tau, offsets, cfg: MetalRasterConfig):
    g = int(means2d.shape[0])
    if g > 0:
        total_pairs_arr = offsets[-1] + 0
        mx.eval(total_pairs_arr)
        total_pairs = int(total_pairs_arr.item())
    else:
        total_pairs = 0
    if g > 0:
        # exact total = last exclusive offset + last count; caller must provide full total separately in practice
        pass
    kernel = _kernel(
        "emit_pairs",
        cfg,
        input_names=["means2d", "conics", "bbox_in", "tau_in", "pair_offsets", "meta_i32"],
        output_names=["tile_ids", "gauss_ids"],
        source=EMIT_PAIRS_SRC,
    )
    raise RuntimeError("Use _emit_pairs_from_counts(), which computes exact output allocation.")


def _emit_pairs_from_counts(means2d, conics, bbox, tau, counts, cfg: MetalRasterConfig):
    g = int(means2d.shape[0])
    offsets = _exclusive_cumsum_1d(counts)
    total_pairs_arr = offsets[-1] + counts[-1] if g > 0 else mx.array(0, mx.int32)
    if g > 0:
        mx.eval(total_pairs_arr)
    total_pairs = int(total_pairs_arr.item()) if g > 0 else 0
    if total_pairs == 0:
        empty_u32 = mx.zeros((0,), dtype=mx.uint32)
        return empty_u32, empty_u32, offsets, total_pairs

    kernel = _kernel(
        "emit_pairs",
        cfg,
        input_names=["means2d", "conics", "bbox_in", "tau_in", "pair_offsets", "meta_i32"],
        output_names=["tile_ids", "gauss_ids"],
        source=EMIT_PAIRS_SRC,
    )
    meta_i32 = _meta_i32(cfg, g)
    outputs = kernel(
        inputs=[means2d, conics, bbox, tau, offsets, meta_i32],
        grid=(g, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(total_pairs,), (total_pairs,)],
        output_dtypes=[mx.uint32, mx.uint32],
        template=[],
    )
    return outputs[0], outputs[1], offsets, total_pairs


def _tile_histogram(sorted_tile_ids, cfg: MetalRasterConfig):
    kernel = _kernel(
        "tile_histogram",
        cfg,
        input_names=["tile_ids"],
        output_names=["tile_counts"],
        source=TILE_HIST_SRC,
        atomic_outputs=True,
    )
    out = kernel(
        inputs=[sorted_tile_ids],
        grid=(int(sorted_tile_ids.shape[0]), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(cfg.tile_count,)],
        output_dtypes=[mx.int32],
        init_value=0,
    )
    return out[0]


def _raster_forward(means2d, conics, colors, opacities, gauss_ids, tile_ranges, cfg: MetalRasterConfig):
    g = int(means2d.shape[0])
    kernel = _kernel(
        "tile_forward",
        cfg,
        input_names=["means2d", "conics", "colors", "opacities", "gauss_ids", "tile_ranges", "meta_i32", "meta_f32"],
        output_names=["out_rgb", "out_T"],
        source=TILE_FORWARD_SRC,
    )
    meta_i32 = _meta_i32(cfg, g)
    meta_f32 = _meta_f32(cfg)
    outputs = kernel(
        inputs=[means2d, conics, colors, opacities, gauss_ids, tile_ranges, meta_i32, meta_f32],
        grid=(cfg.tiles_x * cfg.tile_size, cfg.tiles_y * cfg.tile_size, 1),
        threadgroup=(cfg.tile_size, cfg.tile_size, 1),
        output_shapes=[(cfg.height * cfg.width, 3), (cfg.height * cfg.width,)],
        output_dtypes=[mx.float32, mx.float32],
        template=[],
    )
    image = mx.reshape(outputs[0], (cfg.height, cfg.width, 3))
    final_t = mx.reshape(outputs[1], (cfg.height, cfg.width))
    return image, final_t


def _raster_backward(
    means2d,
    conics,
    colors,
    opacities,
    gauss_ids,
    tile_ranges,
    final_t,
    cotangent,
    cfg: MetalRasterConfig,
):
    g = int(means2d.shape[0])
    kernel = _kernel(
        "tile_backward",
        cfg,
        input_names=[
            "means2d",
            "conics",
            "colors",
            "opacities",
            "gauss_ids",
            "tile_ranges",
            "final_T",
            "cotangent",
            "meta_i32",
            "meta_f32",
        ],
        output_names=["means_grad", "conics_grad", "colors_grad", "opacities_grad"],
        source=TILE_BACKWARD_SRC,
        atomic_outputs=True,
    )
    meta_i32 = _meta_i32(cfg, g)
    meta_f32 = _meta_f32(cfg)
    outputs = kernel(
        inputs=[
            means2d,
            conics,
            colors,
            opacities,
            gauss_ids,
            tile_ranges,
            mx.reshape(final_t, (cfg.height * cfg.width,)),
            mx.reshape(cotangent, (cfg.height * cfg.width, 3)),
            meta_i32,
            meta_f32,
        ],
        grid=(cfg.tiles_x * cfg.tile_size, cfg.tiles_y * cfg.tile_size, 1),
        threadgroup=(cfg.tile_size, cfg.tile_size, 1),
        output_shapes=[means2d.shape, conics.shape, colors.shape, opacities.shape],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32],
        init_value=0,
    )
    return outputs[0], outputs[1], outputs[2], outputs[3]


def _forward_impl(means2d, conics, colors, opacities, depths, cfg: MetalRasterConfig):
    # promote to fp32 for stable accumulation and atomics
    means2d = means2d.astype(mx.float32)
    conics = conics.astype(mx.float32)
    colors = colors.astype(mx.float32)
    opacities = opacities.astype(mx.float32)
    depths = depths.astype(mx.float32)

    # separate sorting stage 1: global depth sort
    depth_perm = mx.argsort(depths, axis=0)
    means_s = means2d[depth_perm]
    conics_s = conics[depth_perm]
    colors_s = colors[depth_perm]
    opac_s = opacities[depth_perm]

    bbox, counts, tau = _snugbox_and_count(means_s, conics_s, opac_s, cfg)
    tile_ids, gauss_ids, _, total_pairs = _emit_pairs_from_counts(means_s, conics_s, bbox, tau, counts, cfg)

    if total_pairs == 0:
        image = mx.broadcast_to(
            mx.array(cfg.background, mx.float32),
            (cfg.height, cfg.width, 3),
        )
        final_t = mx.ones((cfg.height, cfg.width), mx.float32)
        aux = {
            "means_s": means_s,
            "conics_s": conics_s,
            "colors_s": colors_s,
            "opac_s": opac_s,
            "depth_perm": depth_perm,
            "inv_depth_perm": mx.argsort(depth_perm, axis=0),
            "sorted_gauss_ids": gauss_ids,
            "tile_ranges": mx.zeros((cfg.tile_count + 1,), mx.uint32),
            "final_t": final_t,
        }
        return image, aux

    # separate sorting stage 2: tile-local ordering; gauss_ids are already depth-rank indices.
    keys = tile_ids.astype(mx.int64) * int(means_s.shape[0]) + gauss_ids.astype(mx.int64)
    pair_perm = mx.argsort(keys, axis=0)
    sorted_tile_ids = tile_ids[pair_perm]
    sorted_gauss_ids = gauss_ids[pair_perm]

    tile_counts = _tile_histogram(sorted_tile_ids, cfg)
    tile_offsets = _exclusive_cumsum_1d(tile_counts)
    tile_ranges = _pad_with_last(tile_offsets.astype(mx.uint32), mx.array(total_pairs, mx.uint32))

    image, final_t = _raster_forward(means_s, conics_s, colors_s, opac_s, sorted_gauss_ids, tile_ranges, cfg)
    aux = {
        "means_s": means_s,
        "conics_s": conics_s,
        "colors_s": colors_s,
        "opac_s": opac_s,
        "depth_perm": depth_perm,
        "inv_depth_perm": mx.argsort(depth_perm, axis=0),
        "sorted_gauss_ids": sorted_gauss_ids,
        "tile_ranges": tile_ranges,
        "final_t": final_t,
    }
    return image, aux


def make_projected_gaussian_rasterizer(cfg: MetalRasterConfig) -> Callable:
    _require_mlx()

    @mx.custom_function
    def rasterize_projected(means2d, conics, colors, opacities, depths):
        image, _ = _forward_impl(means2d, conics, colors, opacities, depths, cfg)
        return image

    @rasterize_projected.vjp
    def rasterize_projected_vjp(primals, cotangent, _output):
        means2d, conics, colors, opacities, depths = primals
        _, aux = _forward_impl(means2d, conics, colors, opacities, depths, cfg)

        if int(aux["sorted_gauss_ids"].shape[0]) == 0:
            z_means = mx.zeros_like(means2d).astype(mx.float32)
            z_conics = mx.zeros_like(conics).astype(mx.float32)
            z_colors = mx.zeros_like(colors).astype(mx.float32)
            z_opac = mx.zeros_like(opacities).astype(mx.float32)
            z_depth = mx.zeros_like(depths).astype(mx.float32)
            return z_means, z_conics, z_colors, z_opac, z_depth

        g_means_s, g_conics_s, g_colors_s, g_opac_s = _raster_backward(
            aux["means_s"],
            aux["conics_s"],
            aux["colors_s"],
            aux["opac_s"],
            aux["sorted_gauss_ids"],
            aux["tile_ranges"],
            aux["final_t"],
            cotangent.astype(mx.float32),
            cfg,
        )

        inv = aux["inv_depth_perm"]
        g_means = g_means_s[inv]
        g_conics = g_conics_s[inv]
        g_colors = g_colors_s[inv]
        g_opac = g_opac_s[inv]
        g_depth = mx.zeros_like(depths).astype(mx.float32)  # no gradients through sort order
        return g_means, g_conics, g_colors, g_opac, g_depth

    return rasterize_projected


def example_usage() -> str:
    return """
import mlx.core as mx
from mlx_projected_gaussian_rasterizer import MetalRasterConfig, make_projected_gaussian_rasterizer

cfg = MetalRasterConfig(height=384, width=384, tile_size=16, chunk_size=32)
rasterize = make_projected_gaussian_rasterizer(cfg)

G = 512
means2d = mx.random.normal((G, 2)).astype(mx.float32) * 40 + mx.array([192.0, 192.0], mx.float32)
conics  = mx.broadcast_to(mx.array([0.02, 0.0, 0.02], mx.float32), (G, 3))
colors  = mx.random.uniform((G, 3)).astype(mx.float32)
opacity = mx.full((G,), 0.2, mx.float32)
depths  = mx.arange(G, dtype=mx.float32)

image = rasterize(means2d, conics, colors, opacity, depths)
mx.eval(image)
""".strip()


__all__ = [
    "MetalRasterConfig",
    "make_projected_gaussian_rasterizer",
    "example_usage",
]
