"""One-update material-bar coordination for ragged paper batches.

The paper sampler is ragged in camera view: a logical step contains ``B``
arbitrary ``(view, frame)`` observations, while every compiled WorldFoam world
is view-local.  This module owns the missing outer reduction.  It streams one
``B_p x K`` target block at a time, accepts one compact ``[S_b,4]`` material
bar from an injected view-local executor, and index-adds every block directly
into one caller-owned global ``[S,4]`` bar under the single
``P * B * 3`` denominator.

The coordinator deliberately knows nothing about a particular native shader.
That keeps the lifecycle testable on CPU and lets both the fixed-word and
kinetic precompiled-length backends implement the same request contract.  The
warm path checks sealed object identity plus tensor layout/version metadata;
it never content-hashes, copies to CPU, or reads a device scalar.  Exact
coverage uses one integer cursor per ``(view, spatial block)``, not a list of
sample blocks, so coordinator state is independent of temporal partition
density.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from paper_ragged_track_staging import (
    PaperRaggedTrackBatch,
    PaperRaggedTrackTargetStageBlock,
    PaperRaggedViewTrackGroup,
)

_BLOCK_SEAL = object()
_PROGRAM_SEAL = object()
_REQUEST_SEAL = object()
_RESULT_SEAL = object()


@dataclass(frozen=True)
class PaperRaggedMaterialSpatialBlock:
    """One view-local track interval and its compact-to-global site map."""

    block_id: str
    view_index: int
    track_start: int
    track_end: int
    world_token: Any = field(repr=False)
    world_generation_id: str
    source_site_ids_i64: torch.Tensor = field(repr=False)
    global_site_count: int
    site_mapping_id: str
    _source_site_signature: tuple[object, ...] = field(repr=False)
    _world_token_identity: int = field(repr=False)
    _seal: object = field(default=None, repr=False)

    @property
    def track_count(self) -> int:
        return self.track_end - self.track_start

    @property
    def compact_site_count(self) -> int:
        return int(self.source_site_ids_i64.numel())

    def assert_current(self) -> None:
        if self._seal is not _BLOCK_SEAL:
            raise ValueError("ragged material spatial block was not sealed by its preparer")
        if not self.block_id.strip() or self.track_start < 0 or self.track_end <= self.track_start:
            raise ValueError("ragged material spatial block metadata is invalid")
        if not self.world_generation_id.strip() or not self.site_mapping_id.strip():
            raise ValueError("ragged material spatial block provenance is missing")
        if id(self.world_token) != self._world_token_identity:
            raise ValueError("ragged material spatial block world token identity changed")
        _assert_world_token_current(self.world_token, self.world_generation_id)
        if _tensor_signature(self.source_site_ids_i64) != self._source_site_signature:
            raise ValueError("ragged material compact-to-global site mapping is stale")
        if (
            self.source_site_ids_i64.dtype != torch.int64
            or self.source_site_ids_i64.ndim != 1
            or not self.source_site_ids_i64.is_contiguous()
            or self.compact_site_count < 1
            or self.global_site_count < 1
        ):
            raise ValueError("ragged material site mapping layout changed")


def prepare_paper_ragged_material_spatial_block(
    *,
    block_id: str,
    view_index: int,
    track_start: int,
    track_end: int,
    world_token: Any,
    world_generation_id: str,
    source_site_ids: torch.Tensor,
    global_site_count: int,
    device: torch.device | str,
) -> PaperRaggedMaterialSpatialBlock:
    """Cold-seal one block; all content inspection happens at this boundary."""

    if not block_id.strip():
        raise ValueError("ragged material block_id must be nonempty")
    if view_index < 0 or track_start < 0 or track_end <= track_start:
        raise ValueError("ragged material view and track interval must be valid")
    if not world_generation_id.strip() or global_site_count < 1:
        raise ValueError("ragged material world generation and global site count must be valid")
    source = torch.as_tensor(source_site_ids)
    if source.ndim != 1 or source.numel() < 1 or source.dtype not in {torch.int32, torch.int64}:
        raise ValueError("source_site_ids must be a nonempty int32/int64 vector")
    source_cpu = source.detach().to(device="cpu", dtype=torch.int64).contiguous()
    source_values = tuple(int(value) for value in source_cpu.tolist())
    if min(source_values) < 0 or max(source_values) >= global_site_count:
        raise IndexError("source_site_ids contains a row outside the global material table")
    world_source_ids = _world_source_site_ids(world_token)
    if world_source_ids is not None:
        world_source_values = tuple(
            int(value)
            for value in world_source_ids.detach().to(device="cpu", dtype=torch.int64).tolist()
        )
        if world_source_values != source_values:
            raise ValueError("ragged material site mapping does not match its world token")
    source_device = source_cpu.to(device=torch.device(device), dtype=torch.int64).contiguous()
    _assert_world_token_current(world_token, world_generation_id)
    mapping_id = _digest_parts(
        "paper-ragged-material-site-map-v1",
        block_id,
        view_index,
        track_start,
        track_end,
        global_site_count,
        source_values,
    )
    block = PaperRaggedMaterialSpatialBlock(
        block_id=block_id,
        view_index=int(view_index),
        track_start=int(track_start),
        track_end=int(track_end),
        world_token=world_token,
        world_generation_id=world_generation_id,
        source_site_ids_i64=source_device,
        global_site_count=int(global_site_count),
        site_mapping_id=mapping_id,
        _source_site_signature=_tensor_signature(source_device),
        _world_token_identity=id(world_token),
        _seal=_BLOCK_SEAL,
    )
    block.assert_current()
    return block


@dataclass(frozen=True)
class PaperRaggedMaterialViewProgram:
    """A complete nonoverlapping spatial tiling for one view-local world."""

    view_index: int
    global_track_count: int
    global_site_count: int
    blocks: tuple[PaperRaggedMaterialSpatialBlock, ...]
    generation_id: str
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        if self._seal is not _PROGRAM_SEAL:
            raise ValueError("ragged material view program was not sealed by its preparer")
        if self.global_track_count < 1 or self.global_site_count < 1 or not self.blocks:
            raise ValueError("ragged material view program metadata is invalid")
        next_track = 0
        block_ids: set[str] = set()
        for block in self.blocks:
            block.assert_current()
            if block.view_index != self.view_index:
                raise ValueError("ragged material block belongs to a different view world")
            if block.global_site_count != self.global_site_count:
                raise ValueError("ragged material blocks disagree on the global site table")
            if block.block_id in block_ids:
                raise ValueError("ragged material block ids must be unique within a view")
            if block.track_start != next_track:
                raise ValueError("ragged material blocks must tile tracks without gaps or overlaps")
            next_track = block.track_end
            block_ids.add(block.block_id)
        if next_track != self.global_track_count:
            raise ValueError("ragged material blocks must cover every pixel track exactly once")
        expected_generation = _view_program_generation_id(
            self.view_index,
            self.global_track_count,
            self.global_site_count,
            self.blocks,
        )
        if self.generation_id != expected_generation:
            raise ValueError("ragged material view program generation is stale")


def prepare_paper_ragged_material_view_program(
    *,
    view_index: int,
    global_track_count: int,
    global_site_count: int,
    blocks: Sequence[PaperRaggedMaterialSpatialBlock],
) -> PaperRaggedMaterialViewProgram:
    """Cold-seal one view program after validating its exact track tiling."""

    normalized_blocks = tuple(blocks)
    program = PaperRaggedMaterialViewProgram(
        view_index=int(view_index),
        global_track_count=int(global_track_count),
        global_site_count=int(global_site_count),
        blocks=normalized_blocks,
        generation_id=_view_program_generation_id(
            int(view_index),
            int(global_track_count),
            int(global_site_count),
            normalized_blocks,
        ),
        _seal=_PROGRAM_SEAL,
    )
    program.assert_current()
    return program


@dataclass(frozen=True)
class PaperRaggedMaterialBarRequest:
    """One ephemeral target block bound to one exact world/site mapping."""

    step_generation_id: str
    request_generation_id: str
    loss_normalization_id: str
    global_track_count: int
    global_observation_count: int
    global_loss_element_count: int
    global_loss_scale: float
    group: PaperRaggedViewTrackGroup = field(repr=False)
    block: PaperRaggedMaterialSpatialBlock = field(repr=False)
    staged: PaperRaggedTrackTargetStageBlock = field(repr=False)
    local_sample_start: int
    local_sample_end: int
    logical_sample_start: int
    logical_sample_end: int
    _staged_tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    _seal: object = field(default=None, repr=False)

    @property
    def view_index(self) -> int:
        return self.group.view_index

    @property
    def target_rgb(self) -> torch.Tensor:
        return self.staged.targets

    @property
    def world_token(self) -> Any:
        return self.block.world_token

    @property
    def source_site_ids_i64(self) -> torch.Tensor:
        return self.block.source_site_ids_i64

    @property
    def compact_site_count(self) -> int:
        return self.block.compact_site_count

    @property
    def local_rgb_element_count(self) -> int:
        return self.block.track_count * (self.local_sample_end - self.local_sample_start) * 3

    def assert_current(self) -> None:
        if self._seal is not _REQUEST_SEAL:
            raise ValueError("ragged material request was not sealed by its coordinator")
        self.block.assert_current()
        tensors = _staged_identity_tensors(self.staged)
        if tuple(_tensor_signature(tensor) for tensor in tensors) != self._staged_tensor_signatures:
            raise ValueError("ragged material request target/identity tensors are stale")
        normalization = self.staged.normalization
        if (
            not self.loss_normalization_id.strip()
            or normalization.global_track_count != self.global_track_count
            or normalization.global_sample_count != self.global_observation_count
            or normalization.global_rgb_element_count != self.global_loss_element_count
            or normalization.block_track_count != self.block.track_count
            or normalization.block_sample_count != self.local_sample_end - self.local_sample_start
            or self.logical_sample_start != self.group.logical_sample_start + self.local_sample_start
            or self.logical_sample_end != self.group.logical_sample_start + self.local_sample_end
        ):
            raise ValueError("ragged material request loss normalization or range changed")


@dataclass(frozen=True)
class PaperRaggedCompactMaterialBarResult:
    """Executor output sealed to the exact request that generated it."""

    request: PaperRaggedMaterialBarRequest = field(repr=False)
    loss_f32: torch.Tensor
    grad_compact_site_rgba_f32: torch.Tensor
    _tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        if self._seal is not _RESULT_SEAL:
            raise ValueError("ragged compact material result was not sealed by its factory")
        self.request.assert_current()
        tensors = (self.loss_f32, self.grad_compact_site_rgba_f32)
        if tuple(_tensor_signature(tensor) for tensor in tensors) != self._tensor_signatures:
            raise ValueError("ragged compact material result tensors are stale")
        if (
            self.loss_f32.dtype != torch.float32
            or tuple(self.loss_f32.shape) != (1,)
            or not self.loss_f32.is_contiguous()
            or self.loss_f32.requires_grad
        ):
            raise ValueError("ragged compact material loss must be detached contiguous float32[1]")
        gradient = self.grad_compact_site_rgba_f32
        if (
            gradient.device != self.loss_f32.device
            or gradient.dtype != torch.float32
            or tuple(gradient.shape) != (self.request.compact_site_count, 4)
            or not gradient.is_contiguous()
            or gradient.requires_grad
        ):
            raise ValueError("ragged compact material bar has the wrong device/layout")


def seal_paper_ragged_compact_material_bar_result(
    request: PaperRaggedMaterialBarRequest,
    *,
    loss_f32: torch.Tensor,
    grad_compact_site_rgba_f32: torch.Tensor,
) -> PaperRaggedCompactMaterialBarResult:
    """Seal an executor result without synchronizing or copying device data."""

    if not isinstance(request, PaperRaggedMaterialBarRequest):
        raise TypeError("ragged compact material result requires a coordinator request")
    request.assert_current()
    loss = loss_f32
    gradient = grad_compact_site_rgba_f32
    if not isinstance(loss, torch.Tensor) or not isinstance(gradient, torch.Tensor):
        raise TypeError("ragged compact material loss and bar must be tensors")
    result = PaperRaggedCompactMaterialBarResult(
        request=request,
        loss_f32=loss,
        grad_compact_site_rgba_f32=gradient,
        _tensor_signatures=tuple(_tensor_signature(tensor) for tensor in (loss, gradient)),
        _seal=_RESULT_SEAL,
    )
    result.assert_current()
    return result


@dataclass
class PaperRaggedMaterialBarStepLedger:
    """Constant-per-program coverage state and one global material bar."""

    batch: PaperRaggedTrackBatch
    programs: tuple[PaperRaggedMaterialViewProgram, ...]
    global_grad_site_rgba_f32: torch.Tensor
    loss_f32: torch.Tensor
    step_generation_id: str
    batch_tensor_signatures: tuple[tuple[object, ...], ...]
    state_tensor_signatures: tuple[tuple[object, ...], ...]
    provider_identities: tuple[int, int]
    next_logical_sample_by_block: dict[tuple[int, str], int]
    active_request: PaperRaggedMaterialBarRequest | None = None
    request_count: int = 0
    consumed_rgb_element_count: int = 0
    compact_site_rows_accumulated: int = 0
    peak_staged_target_bytes: int = 0
    finalized: bool = False
    authorization_issued: bool = False

    @property
    def global_site_count(self) -> int:
        return int(self.global_grad_site_rgba_f32.shape[0])

    @property
    def global_loss_element_count(self) -> int:
        return self.batch.global_rgb_element_count


def begin_paper_ragged_material_bar_step(
    batch: PaperRaggedTrackBatch,
    *,
    programs: Sequence[PaperRaggedMaterialViewProgram],
    global_grad_site_rgba_f32: torch.Tensor,
    loss_f32: torch.Tensor | None = None,
) -> PaperRaggedMaterialBarStepLedger:
    """Bind and zero the only global material bar for one logical paper step."""

    if not isinstance(batch, PaperRaggedTrackBatch):
        raise TypeError("ragged material coordination requires a PaperRaggedTrackBatch")
    gradient = global_grad_site_rgba_f32
    if not isinstance(gradient, torch.Tensor):
        raise TypeError("global_grad_site_rgba_f32 must be a caller-owned tensor")
    if (
        gradient.dtype != torch.float32
        or gradient.ndim != 2
        or int(gradient.shape[1]) != 4
        or int(gradient.shape[0]) < 1
        or not gradient.is_contiguous()
        or gradient.requires_grad
    ):
        raise ValueError("global material bar must be detached contiguous float32 [site_count,4]")
    loss = (
        torch.zeros((1,), dtype=torch.float32, device=gradient.device)
        if loss_f32 is None
        else loss_f32
    )
    if (
        not isinstance(loss, torch.Tensor)
        or loss.device != gradient.device
        or loss.dtype != torch.float32
        or tuple(loss.shape) != (1,)
        or not loss.is_contiguous()
        or loss.requires_grad
    ):
        raise ValueError("ragged material loss must be detached contiguous float32[1] on the bar device")
    if loss.untyped_storage().data_ptr() == gradient.untyped_storage().data_ptr():
        raise ValueError("ragged material loss and global bar must own distinct storage")

    normalized_programs = tuple(programs)
    groups_by_view = {group.view_index: group for group in batch.groups}
    programs_by_view: dict[int, PaperRaggedMaterialViewProgram] = {}
    for program in normalized_programs:
        if not isinstance(program, PaperRaggedMaterialViewProgram):
            raise TypeError("programs must contain PaperRaggedMaterialViewProgram values")
        program.assert_current()
        if program.view_index in programs_by_view:
            raise ValueError("ragged material step received duplicate view programs")
        if program.view_index not in groups_by_view:
            raise ValueError("ragged material program has no matching paper view group")
        if program.global_track_count != batch.pixel_count:
            raise ValueError("ragged material view program changed the global pixel-track count")
        if program.global_site_count != int(gradient.shape[0]):
            raise ValueError("ragged material view program changed the caller global site table")
        if any(block.source_site_ids_i64.device != gradient.device for block in program.blocks):
            raise ValueError("ragged material site maps and global bar must share one device")
        programs_by_view[program.view_index] = program
    if set(programs_by_view) != set(groups_by_view):
        raise ValueError("ragged material step requires exactly one program for every active view")

    batch_tensors = _batch_identity_tensors(batch)
    gradient.zero_()
    loss.zero_()
    step_generation_id = _step_generation_id(batch, normalized_programs, int(gradient.shape[0]))
    cursors = {
        (program.view_index, block.block_id): groups_by_view[program.view_index].logical_sample_start
        for program in normalized_programs
        for block in program.blocks
    }
    ledger = PaperRaggedMaterialBarStepLedger(
        batch=batch,
        programs=normalized_programs,
        global_grad_site_rgba_f32=gradient,
        loss_f32=loss,
        step_generation_id=step_generation_id,
        batch_tensor_signatures=tuple(_tensor_signature(tensor) for tensor in batch_tensors),
        state_tensor_signatures=tuple(_tensor_signature(tensor) for tensor in (gradient, loss)),
        provider_identities=(
            id(batch.groups[0].staging_plan.target_provider),
            id(batch.groups[0].staging_plan.ray_provider),
        ),
        next_logical_sample_by_block=cursors,
    )
    _assert_ledger_current(ledger)
    return ledger


def stage_next_paper_ragged_material_bar_request(
    ledger: PaperRaggedMaterialBarStepLedger,
    *,
    view_index: int,
    block_id: str,
    local_sample_start: int,
    local_sample_end: int,
) -> PaperRaggedMaterialBarRequest:
    """Stage one bounded target rectangle at the exact next coverage cursor."""

    _assert_ledger_current(ledger)
    if ledger.finalized:
        raise ValueError("ragged material step was already finalized")
    if ledger.active_request is not None:
        raise ValueError("ragged material coordinator permits only one in-flight target block")
    group = _group_for_view(ledger.batch, int(view_index))
    program = _program_for_view(ledger.programs, int(view_index))
    block = _block_by_id(program, block_id)
    start = int(local_sample_start)
    end = int(local_sample_end)
    if not 0 <= start < end <= group.observation_count:
        raise ValueError("ragged material local sample range is invalid")
    logical_start = group.logical_sample_start + start
    logical_end = group.logical_sample_start + end
    key = (group.view_index, block.block_id)
    expected_start = ledger.next_logical_sample_by_block[key]
    if logical_start < expected_start:
        raise ValueError("duplicate or overlapping ragged logical sample coverage")
    if logical_start > expected_start:
        raise ValueError("gap in ragged logical sample coverage")
    staged = group.stage_targets(
        track_start=block.track_start,
        track_end=block.track_end,
        sample_start=start,
        sample_end=end,
    )
    if (
        staged.logical_sample_start != logical_start
        or staged.logical_sample_end != logical_end
        or staged.normalization.global_rgb_element_count != ledger.global_loss_element_count
    ):
        raise ValueError("ragged target staging changed the registered logical range or denominator")
    request_generation_id = _digest_parts(
        "paper-ragged-material-request-v1",
        ledger.step_generation_id,
        ledger.request_count,
        group.view_index,
        block.block_id,
        block.world_generation_id,
        block.site_mapping_id,
        block.track_start,
        block.track_end,
        logical_start,
        logical_end,
    )
    request = PaperRaggedMaterialBarRequest(
        step_generation_id=ledger.step_generation_id,
        request_generation_id=request_generation_id,
        loss_normalization_id=ledger.batch.loss_normalization_id,
        global_track_count=ledger.batch.pixel_count,
        global_observation_count=ledger.batch.observation_count,
        global_loss_element_count=ledger.global_loss_element_count,
        global_loss_scale=1.0 / float(ledger.global_loss_element_count),
        group=group,
        block=block,
        staged=staged,
        local_sample_start=start,
        local_sample_end=end,
        logical_sample_start=logical_start,
        logical_sample_end=logical_end,
        _staged_tensor_signatures=tuple(
            _tensor_signature(tensor) for tensor in _staged_identity_tensors(staged)
        ),
        _seal=_REQUEST_SEAL,
    )
    request.assert_current()
    ledger.active_request = request
    ledger.peak_staged_target_bytes = max(
        ledger.peak_staged_target_bytes,
        request.target_rgb.numel() * request.target_rgb.element_size(),
    )
    return request


@torch.no_grad()
def consume_paper_ragged_compact_material_bar_result(
    ledger: PaperRaggedMaterialBarStepLedger,
    request: PaperRaggedMaterialBarRequest,
    result: PaperRaggedCompactMaterialBarResult,
) -> None:
    """Accumulate one compact bar directly into the sole global site bar."""

    _assert_ledger_current(ledger)
    if ledger.finalized:
        raise ValueError("ragged material step was already finalized")
    if ledger.active_request is not request:
        raise ValueError("ragged compact result has a stale or mismatched request token")
    if not isinstance(result, PaperRaggedCompactMaterialBarResult) or result.request is not request:
        raise ValueError("ragged compact result changed world, site-map, denominator, or request provenance")
    request.assert_current()
    result.assert_current()
    if request.step_generation_id != ledger.step_generation_id:
        raise ValueError("ragged compact result belongs to a different logical step")
    if result.loss_f32.device != ledger.loss_f32.device:
        raise ValueError("ragged compact result and global accumulator must share one device")
    if (
        result.grad_compact_site_rgba_f32.untyped_storage().data_ptr()
        == ledger.global_grad_site_rgba_f32.untyped_storage().data_ptr()
    ):
        raise ValueError("compact material bar must not alias the global accumulator")

    ledger.global_grad_site_rgba_f32.index_add_(
        0,
        request.source_site_ids_i64,
        result.grad_compact_site_rgba_f32,
    )
    ledger.loss_f32.add_(result.loss_f32)
    key = (request.view_index, request.block.block_id)
    ledger.next_logical_sample_by_block[key] = request.logical_sample_end
    ledger.request_count += 1
    ledger.consumed_rgb_element_count += request.local_rgb_element_count
    ledger.compact_site_rows_accumulated += request.compact_site_count
    ledger.active_request = None
    ledger.state_tensor_signatures = tuple(
        _tensor_signature(tensor) for tensor in (ledger.global_grad_site_rgba_f32, ledger.loss_f32)
    )
    _assert_ledger_current(ledger)


@dataclass(frozen=True)
class PaperRaggedMaterialBarStepResult:
    """Final views of the caller-owned global bar and scalar loss."""

    step_generation_id: str
    loss_normalization_id: str
    loss_f32: torch.Tensor
    grad_global_site_rgba_f32: torch.Tensor
    accounting: dict[str, int | float | str | bool]
    _tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)

    def assert_current(self) -> None:
        tensors = (self.loss_f32, self.grad_global_site_rgba_f32)
        if tuple(_tensor_signature(tensor) for tensor in tensors) != self._tensor_signatures:
            raise ValueError("final ragged material result buffers changed before optimizer authorization")


@dataclass
class PaperRaggedMaterialOptimizerAuthorization:
    """Single-use permission issued only after exact global coverage."""

    result: PaperRaggedMaterialBarStepResult
    consumed: bool = False

    def consume(self, optimizer_update: Callable[[PaperRaggedMaterialBarStepResult], None]) -> None:
        if self.consumed:
            raise ValueError("ragged material optimizer authorization was already consumed")
        if not callable(optimizer_update):
            raise TypeError("optimizer_update must be callable")
        self.result.assert_current()
        self.consumed = True
        optimizer_update(self.result)


def finalize_paper_ragged_material_bar_step(
    ledger: PaperRaggedMaterialBarStepLedger,
) -> PaperRaggedMaterialOptimizerAuthorization:
    """Prove full disjoint coverage, then issue exactly one update capability."""

    _assert_ledger_current(ledger)
    if ledger.finalized:
        raise ValueError("ragged material step was already finalized")
    if ledger.active_request is not None:
        raise ValueError("ragged material step cannot finalize with an in-flight target block")
    for program in ledger.programs:
        group = _group_for_view(ledger.batch, program.view_index)
        for block in program.blocks:
            if ledger.next_logical_sample_by_block[(program.view_index, block.block_id)] != group.logical_sample_end:
                raise ValueError("ragged material step cannot finalize with missing logical sample coverage")
    if ledger.consumed_rgb_element_count != ledger.global_loss_element_count:
        raise ValueError("ragged material step coverage does not equal the global P*B*3 denominator")
    if ledger.authorization_issued:
        raise ValueError("ragged material optimizer authorization was already issued")
    ledger.finalized = True
    ledger.authorization_issued = True
    state_signatures = tuple(
        _tensor_signature(tensor) for tensor in (ledger.loss_f32, ledger.global_grad_site_rgba_f32)
    )
    result = PaperRaggedMaterialBarStepResult(
        step_generation_id=ledger.step_generation_id,
        loss_normalization_id=ledger.batch.loss_normalization_id,
        loss_f32=ledger.loss_f32,
        grad_global_site_rgba_f32=ledger.global_grad_site_rgba_f32,
        accounting={
            "global_track_count": ledger.batch.pixel_count,
            "global_observation_count": ledger.batch.observation_count,
            "global_loss_element_count": ledger.global_loss_element_count,
            "global_site_count": ledger.global_site_count,
            "loss_normalization_id": ledger.batch.loss_normalization_id,
            "active_view_count": ledger.batch.active_view_count,
            "view_spatial_block_count": len(ledger.next_logical_sample_by_block),
            "sample_block_result_count": ledger.request_count,
            "compact_site_rows_accumulated": ledger.compact_site_rows_accumulated,
            "consumed_rgb_element_count": ledger.consumed_rgb_element_count,
            "global_site_gradient_buffer_count": 1,
            "global_site_gradient_buffer_bytes": (
                ledger.global_grad_site_rgba_f32.numel()
                * ledger.global_grad_site_rgba_f32.element_size()
            ),
            "per_view_global_gradient_buffers": 0,
            "persistent_target_tensor_bytes": 0,
            "persistent_explicit_ray_tensor_bytes": 0,
            "peak_in_flight_target_blocks": 1,
            "peak_staged_target_bytes": ledger.peak_staged_target_bytes,
            "coverage_state_complexity": "O(active_views * spatial_blocks_per_view)",
            "sample_partition_records_retained": 0,
            "optimizer_update_authorization_count": 1,
            "global_denominator_preserved": True,
            "view_time_cartesian_tensor_allocated": False,
        },
        _tensor_signatures=state_signatures,
    )
    result.assert_current()
    return PaperRaggedMaterialOptimizerAuthorization(result=result)


@dataclass(frozen=True)
class PaperRaggedMaterialBarRunResult:
    """Completed coordinator result after its one update callback."""

    step: PaperRaggedMaterialBarStepResult
    optimizer_update_callback_count: int = 1


def run_paper_ragged_material_bar_step(
    batch: PaperRaggedTrackBatch,
    *,
    programs: Sequence[PaperRaggedMaterialViewProgram],
    global_grad_site_rgba_f32: torch.Tensor,
    executor: Callable[[PaperRaggedMaterialBarRequest], PaperRaggedCompactMaterialBarResult],
    optimizer_update: Callable[[PaperRaggedMaterialBarStepResult], None],
    sample_block_size: int,
    loss_f32: torch.Tensor | None = None,
    view_order: Sequence[int] | None = None,
) -> PaperRaggedMaterialBarRunResult:
    """Stream all ragged blocks and invoke one update only after finalization."""

    if sample_block_size < 1:
        raise ValueError("sample_block_size must be positive")
    if not callable(executor):
        raise TypeError("executor must be callable")
    ledger = begin_paper_ragged_material_bar_step(
        batch,
        programs=programs,
        global_grad_site_rgba_f32=global_grad_site_rgba_f32,
        loss_f32=loss_f32,
    )
    canonical_views = tuple(group.view_index for group in batch.groups)
    execution_views = canonical_views if view_order is None else tuple(int(value) for value in view_order)
    if len(execution_views) != len(canonical_views) or set(execution_views) != set(canonical_views):
        raise ValueError("view_order must be an exact permutation of active paper views")
    for view_index in execution_views:
        group = _group_for_view(batch, view_index)
        program = _program_for_view(ledger.programs, view_index)
        for sample_start in range(0, group.observation_count, sample_block_size):
            sample_end = min(group.observation_count, sample_start + sample_block_size)
            for block in program.blocks:
                request = stage_next_paper_ragged_material_bar_request(
                    ledger,
                    view_index=view_index,
                    block_id=block.block_id,
                    local_sample_start=sample_start,
                    local_sample_end=sample_end,
                )
                result = executor(request)
                consume_paper_ragged_compact_material_bar_result(ledger, request, result)
                del result, request
    authorization = finalize_paper_ragged_material_bar_step(ledger)
    authorization.consume(optimizer_update)
    return PaperRaggedMaterialBarRunResult(step=authorization.result)


def _assert_ledger_current(ledger: PaperRaggedMaterialBarStepLedger) -> None:
    if not isinstance(ledger, PaperRaggedMaterialBarStepLedger):
        raise TypeError("ragged material ledger has the wrong type")
    batch_tensors = _batch_identity_tensors(ledger.batch)
    if tuple(_tensor_signature(tensor) for tensor in batch_tensors) != ledger.batch_tensor_signatures:
        raise ValueError("ragged paper batch identity tensors changed during material reduction")
    if (
        id(ledger.batch.groups[0].staging_plan.target_provider),
        id(ledger.batch.groups[0].staging_plan.ray_provider),
    ) != ledger.provider_identities:
        raise ValueError("ragged paper target/ray provider identity changed")
    for program in ledger.programs:
        program.assert_current()
    state_tensors = (ledger.global_grad_site_rgba_f32, ledger.loss_f32)
    if tuple(_tensor_signature(tensor) for tensor in state_tensors) != ledger.state_tensor_signatures:
        raise ValueError("ragged material global loss/bar buffers changed outside the coordinator")
    if ledger.active_request is not None:
        ledger.active_request.assert_current()


def _assert_world_token_current(world_token: Any, expected_generation_id: str) -> None:
    if world_token is None:
        raise TypeError("ragged material block requires an opaque world token")
    assert_current = getattr(world_token, "assert_current", None)
    if callable(assert_current):
        assert_current()
    actual_generation = getattr(world_token, "generation_digest", expected_generation_id)
    if actual_generation != expected_generation_id:
        raise ValueError("ragged material block world generation is stale or mismatched")


def _world_source_site_ids(world_token: Any) -> torch.Tensor | None:
    direct = getattr(world_token, "source_site_ids_i64", None)
    if isinstance(direct, torch.Tensor):
        return direct
    topology = getattr(world_token, "topology", None)
    nested = getattr(topology, "source_site_ids_i64", None)
    return nested if isinstance(nested, torch.Tensor) else None


def _group_for_view(batch: PaperRaggedTrackBatch, view_index: int) -> PaperRaggedViewTrackGroup:
    matches = tuple(group for group in batch.groups if group.view_index == view_index)
    if len(matches) != 1:
        raise ValueError("ragged material view has no unique paper group")
    return matches[0]


def _program_for_view(
    programs: tuple[PaperRaggedMaterialViewProgram, ...],
    view_index: int,
) -> PaperRaggedMaterialViewProgram:
    matches = tuple(program for program in programs if program.view_index == view_index)
    if len(matches) != 1:
        raise ValueError("ragged material view has no unique world program")
    return matches[0]


def _block_by_id(
    program: PaperRaggedMaterialViewProgram,
    block_id: str,
) -> PaperRaggedMaterialSpatialBlock:
    matches = tuple(block for block in program.blocks if block.block_id == block_id)
    if len(matches) != 1:
        raise ValueError("ragged material spatial block id is not registered")
    return matches[0]


def _batch_identity_tensors(batch: PaperRaggedTrackBatch) -> tuple[torch.Tensor, ...]:
    return (
        batch.pixel_indices,
        *(
            tensor
            for group in batch.groups
            for tensor in (
                group.batch_positions,
                group.staging_plan.pixel_indices,
                group.staging_plan.sample_indices,
                group.staging_plan.sample_times,
            )
        ),
    )


def _staged_identity_tensors(block: PaperRaggedTrackTargetStageBlock) -> tuple[torch.Tensor, ...]:
    return (
        block.batch_positions,
        block.pixel_indices,
        block.sample_indices,
        block.frame_indices,
        block.sample_times,
        block.targets,
    )


def _view_program_generation_id(
    view_index: int,
    global_track_count: int,
    global_site_count: int,
    blocks: tuple[PaperRaggedMaterialSpatialBlock, ...],
) -> str:
    return _digest_parts(
        "paper-ragged-material-view-program-v1",
        view_index,
        global_track_count,
        global_site_count,
        tuple(
            (
                block.block_id,
                block.track_start,
                block.track_end,
                block.world_generation_id,
                block.site_mapping_id,
            )
            for block in blocks
        ),
    )


def _step_generation_id(
    batch: PaperRaggedTrackBatch,
    programs: tuple[PaperRaggedMaterialViewProgram, ...],
    global_site_count: int,
) -> str:
    return _digest_parts(
        "paper-ragged-material-step-v1",
        batch.loss_normalization_id,
        batch.pixel_count,
        batch.observation_count,
        global_site_count,
        tuple((sample.view_index, sample.frame_index) for sample in batch.batch.samples),
        tuple(
            (
                group.view_index,
                group.logical_sample_start,
                group.logical_sample_end,
                tuple(group.batch_positions.tolist()),
            )
            for group in batch.groups
        ),
        tuple(sorted((program.view_index, program.generation_id) for program in programs)),
    )


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        tuple(tensor.shape),
        str(tensor.dtype),
        str(tensor.device),
        tuple(tensor.stride()),
        int(tensor.storage_offset()),
        int(tensor.untyped_storage().data_ptr()),
        int(getattr(tensor, "_version", 0)),
    )


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


__all__ = [
    "PaperRaggedCompactMaterialBarResult",
    "PaperRaggedMaterialBarRequest",
    "PaperRaggedMaterialBarRunResult",
    "PaperRaggedMaterialBarStepLedger",
    "PaperRaggedMaterialBarStepResult",
    "PaperRaggedMaterialOptimizerAuthorization",
    "PaperRaggedMaterialSpatialBlock",
    "PaperRaggedMaterialViewProgram",
    "begin_paper_ragged_material_bar_step",
    "consume_paper_ragged_compact_material_bar_result",
    "finalize_paper_ragged_material_bar_step",
    "prepare_paper_ragged_material_spatial_block",
    "prepare_paper_ragged_material_view_program",
    "run_paper_ragged_material_bar_step",
    "seal_paper_ragged_compact_material_bar_result",
    "stage_next_paper_ragged_material_bar_request",
]
