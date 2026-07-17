from __future__ import annotations

import torch


SPARSE_VISUAL_PIXEL_SOURCES = {"stratified_grid", "stratified_patch_grid", "stratified_patch_grid_phase"}
SPARSE_VISUAL_PATCH_PIXEL_SOURCES = {"stratified_patch_grid", "stratified_patch_grid_phase"}
SPARSE_VISUAL_LOSS_BASES = {"pixel", "patch_mean", "target_area_mean"}
SPARSE_VISUAL_COMPOSITIONS = {"black", "target_background"}
SPARSE_VISUAL_LOSS_VJP_MODES = {
    "autograd",
    "manual_hidden",
    "manual_hidden_fastgelu",
    "manual_hidden_star_only",
    "manual_hidden_star_only_fastgelu",
    "manual_hidden64",
    "manual_hidden64_fastgelu",
    "manual_hidden64_star_only",
    "manual_hidden64_star_only_fastgelu",
    "manual_linear",
    "native_hidden_star_only",
    "native_hidden64_star_only",
    "native_hidden_target_area_star_only",
    "native_hidden64_target_area_star_only",
    "native_hidden_target_area_star_only_vec4_wt",
    "native_hidden64_target_area_star_only_vec4_wt",
    "native_hidden_target_area_colorizer_vec4_wt",
    "native_hidden64_target_area_colorizer_vec4_wt",
    "native_hidden_target_area_colorizer_simdreduce_vec4_wt",
    "native_hidden64_target_area_colorizer_simdreduce_vec4_wt",
}
NATIVE_PIXEL_SPARSE_VISUAL_LOSS_VJP_MODES = {
    "native_hidden_star_only",
    "native_hidden64_star_only",
}
NATIVE_TARGET_AREA_SPARSE_VISUAL_LOSS_VJP_MODES = {
    "native_hidden_target_area_star_only",
    "native_hidden64_target_area_star_only",
    "native_hidden_target_area_star_only_vec4_wt",
    "native_hidden64_target_area_star_only_vec4_wt",
    "native_hidden_target_area_colorizer_vec4_wt",
    "native_hidden64_target_area_colorizer_vec4_wt",
    "native_hidden_target_area_colorizer_simdreduce_vec4_wt",
    "native_hidden64_target_area_colorizer_simdreduce_vec4_wt",
}
NATIVE_SPARSE_VISUAL_LOSS_VJP_MODES = (
    NATIVE_PIXEL_SPARSE_VISUAL_LOSS_VJP_MODES | NATIVE_TARGET_AREA_SPARSE_VISUAL_LOSS_VJP_MODES
)


def _stratified_indices(count: int, size: int, *, device: torch.device) -> torch.Tensor:
    if count <= 0 or size <= 0:
        raise ValueError("count and size must be positive")
    if count > size:
        raise ValueError(f"cannot draw {count} stratified samples from size {size}")
    values = (torch.arange(count, device=device, dtype=torch.float32) + 0.5) * (float(size) / float(count))
    return values.floor().to(dtype=torch.int64).clamp_(0, size - 1)


def _stratified_patch_indices(
    count: int,
    size: int,
    *,
    patch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if patch_size > size:
        raise ValueError(f"patch_size={patch_size} cannot exceed size={size}")
    centers = _stratified_indices(count, size, device=device)
    starts = (centers - (patch_size // 2)).clamp_(0, size - patch_size)
    offsets = torch.arange(patch_size, device=device, dtype=torch.int64)
    return (starts[:, None] + offsets[None, :]).reshape(-1).contiguous()


def _stratified_phase_patch_indices(
    count: int,
    size: int,
    *,
    patch_size: int,
    phase: int,
    phase_count: int,
    device: torch.device,
) -> torch.Tensor:
    if count <= 0 or size <= 0:
        raise ValueError("count and size must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if phase_count <= 0:
        raise ValueError("phase_count must be positive")
    if size % count != 0:
        raise ValueError("stratified_patch_grid_phase requires size divisible by sample count")
    cell_size = size // count
    if patch_size * phase_count > cell_size:
        raise ValueError("patch_size * phase_count cannot exceed the stratified cell size")
    phase_offset = (int(phase) % int(phase_count)) * int(patch_size)
    cell_starts = torch.arange(count, device=device, dtype=torch.int64) * int(cell_size)
    offsets = phase_offset + torch.arange(patch_size, device=device, dtype=torch.int64)
    return (cell_starts[:, None] + offsets[None, :]).reshape(-1).contiguous()


def _sparse_visual_patch_phase_for_step(
    *,
    pixel_source: str,
    global_step: int,
    patch_phase_shape: tuple[int, int],
) -> tuple[int, int]:
    if pixel_source != "stratified_patch_grid_phase":
        return (0, 0)
    phase_h, phase_w = int(patch_phase_shape[0]), int(patch_phase_shape[1])
    if phase_h <= 0 or phase_w <= 0:
        raise ValueError("sparse_visual.patch_phase_shape must be positive")
    phase = int(global_step) % (phase_h * phase_w)
    return (phase // phase_w, phase % phase_w)


def _sparse_visual_local_frame_ids_for_chunk(
    *,
    render_frames: int,
    frame_start: int,
    chunk_frames: int,
    sample_grid_shape: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    frame_ids = _stratified_indices(int(sample_grid_shape[0]), render_frames, device=device)
    return frame_ids[(frame_ids >= frame_start) & (frame_ids < frame_start + chunk_frames)] - frame_start


def _sparse_visual_pixel_ids_for_chunk(
    *,
    pixel_source: str,
    chunk_frames: int,
    height: int,
    width: int,
    render_frames: int,
    frame_start: int,
    sample_grid_shape: tuple[int, int, int],
    patch_shape: tuple[int, int] = (1, 1),
    patch_phase: tuple[int, int] = (0, 0),
    patch_phase_shape: tuple[int, int] = (1, 1),
    device: torch.device,
) -> torch.Tensor:
    if pixel_source not in SPARSE_VISUAL_PIXEL_SOURCES:
        expected = ", ".join(sorted(SPARSE_VISUAL_PIXEL_SOURCES))
        raise ValueError(f"sparse_visual.pixel_source must be one of: {expected}")
    local_frames = _sparse_visual_local_frame_ids_for_chunk(
        render_frames=render_frames,
        frame_start=frame_start,
        chunk_frames=chunk_frames,
        sample_grid_shape=sample_grid_shape,
        device=device,
    )
    if int(local_frames.numel()) == 0:
        return torch.empty((0,), device=device, dtype=torch.int32)
    if pixel_source == "stratified_patch_grid_phase":
        row_ids = _stratified_phase_patch_indices(
            int(sample_grid_shape[1]),
            height,
            patch_size=int(patch_shape[0]),
            phase=int(patch_phase[0]),
            phase_count=int(patch_phase_shape[0]),
            device=device,
        )
        col_ids = _stratified_phase_patch_indices(
            int(sample_grid_shape[2]),
            width,
            patch_size=int(patch_shape[1]),
            phase=int(patch_phase[1]),
            phase_count=int(patch_phase_shape[1]),
            device=device,
        )
    elif pixel_source in SPARSE_VISUAL_PATCH_PIXEL_SOURCES:
        row_ids = _stratified_patch_indices(
            int(sample_grid_shape[1]),
            height,
            patch_size=int(patch_shape[0]),
            device=device,
        )
        col_ids = _stratified_patch_indices(
            int(sample_grid_shape[2]),
            width,
            patch_size=int(patch_shape[1]),
            device=device,
        )
    else:
        row_ids = _stratified_indices(int(sample_grid_shape[1]), height, device=device)
        col_ids = _stratified_indices(int(sample_grid_shape[2]), width, device=device)
    ids = (
        local_frames[:, None, None] * (height * width)
        + row_ids[None, :, None] * width
        + col_ids[None, None, :]
    )
    return ids.reshape(-1).to(dtype=torch.int32).contiguous()


def _sparse_visual_loss_sample_count(
    pixel_count: int,
    *,
    loss_basis: str,
    patch_shape: tuple[int, int] = (1, 1),
) -> int:
    if loss_basis == "pixel":
        return int(pixel_count)
    if loss_basis in {"patch_mean", "target_area_mean"}:
        patch_area = int(patch_shape[0]) * int(patch_shape[1])
        if patch_area <= 0:
            raise ValueError("sparse_visual.patch_shape must be positive")
        if int(pixel_count) % patch_area != 0:
            raise ValueError(f"{loss_basis} sparse visual pixel count must divide by patch area")
        return int(pixel_count) // patch_area
    expected = ", ".join(sorted(SPARSE_VISUAL_LOSS_BASES))
    raise ValueError(f"sparse_visual.loss_basis must be one of: {expected}")


__all__ = [
    "NATIVE_PIXEL_SPARSE_VISUAL_LOSS_VJP_MODES",
    "NATIVE_SPARSE_VISUAL_LOSS_VJP_MODES",
    "NATIVE_TARGET_AREA_SPARSE_VISUAL_LOSS_VJP_MODES",
    "SPARSE_VISUAL_COMPOSITIONS",
    "SPARSE_VISUAL_LOSS_BASES",
    "SPARSE_VISUAL_LOSS_VJP_MODES",
    "SPARSE_VISUAL_PATCH_PIXEL_SOURCES",
    "SPARSE_VISUAL_PIXEL_SOURCES",
    "_sparse_visual_local_frame_ids_for_chunk",
    "_sparse_visual_loss_sample_count",
    "_sparse_visual_patch_phase_for_step",
    "_sparse_visual_pixel_ids_for_chunk",
    "_stratified_indices",
]
