"""Flat, compact fixed-word blocks for streamed native WorldFoam steps.

The compiler may discover owner words with Python objects, but the training
step should retain only flat CSR tensors for the selected spatial block.  This
module compacts tracks, referenced boundaries, and every site needed either as
an owner or as an endpoint of a referenced power boundary.  It deliberately
contains no frame/sample axis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from compiled_transfer_adjoint import FAR_CUT_ID, NEAR_CUT_ID, StableCellWord


@dataclass(frozen=True)
class PreparedWorldFoamTrackBlock:
    """Launch-ready fixed-topology CSR for one contiguous track block."""

    source_track_ids: torch.Tensor
    source_boundary_ids: torch.Tensor
    source_site_ids: torch.Tensor
    word_offsets_i32: torch.Tensor
    word_owner_i32: torch.Tensor
    word_left_incidence_i32: torch.Tensor
    word_right_incidence_i32: torch.Tensor
    track_incidence_offsets_i32: torch.Tensor
    incidence_boundary_i32: torch.Tensor
    boundary_site_pairs_i32: torch.Tensor

    @property
    def track_count(self) -> int:
        return int(self.source_track_ids.numel())

    @property
    def boundary_count(self) -> int:
        return int(self.source_boundary_ids.numel())

    @property
    def site_count(self) -> int:
        return int(self.source_site_ids.numel())

    @property
    def incidence_count(self) -> int:
        return int(self.incidence_boundary_i32.numel())

    @property
    def word_count(self) -> int:
        return int(self.word_owner_i32.numel())

    @property
    def resident_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.source_track_ids,
                self.source_boundary_ids,
                self.source_site_ids,
                self.word_offsets_i32,
                self.word_owner_i32,
                self.word_left_incidence_i32,
                self.word_right_incidence_i32,
                self.track_incidence_offsets_i32,
                self.incidence_boundary_i32,
                self.boundary_site_pairs_i32,
            )
        )


def prepare_worldfoam_track_block(
    words: Sequence[StableCellWord],
    boundary_site_pairs: torch.Tensor,
    *,
    site_count: int,
    track_start: int,
    track_end: int,
) -> PreparedWorldFoamTrackBlock:
    """Compact a contiguous slice of fixed words into local CSR indices.

    Positive cut ids index rows of ``boundary_site_pairs``.  The two physical
    sentinel ids remain negative in the word arrays.  Every nonnegative word
    cut is recoded to a row-local incidence id, while each incidence points to
    a compact block-local boundary row.  Owner and boundary endpoint sites are
    recoded into one compact site table.
    """

    words_tuple = tuple(words)
    if not words_tuple:
        raise ValueError("words must be non-empty")
    if track_start < 0 or track_end <= track_start or track_end > len(words_tuple):
        raise ValueError("expected 0 <= track_start < track_end <= len(words)")
    if site_count < 1:
        raise ValueError("site_count must be positive")
    pairs = torch.as_tensor(boundary_site_pairs, dtype=torch.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("boundary_site_pairs must have shape [B,2]")
    if pairs.numel() and (int(pairs.min().item()) < 0 or int(pairs.max().item()) >= site_count):
        raise ValueError("boundary_site_pairs contain an out-of-range site id")

    selected_words = words_tuple[track_start:track_end]
    _validate_selected_words(
        selected_words,
        site_count=site_count,
        boundary_count=int(pairs.shape[0]),
    )
    referenced_boundary_ids = sorted(
        {
            int(cut_id)
            for word in selected_words
            for cut_id in torch.cat((word.left_cut_ids, word.right_cut_ids)).tolist()
            if int(cut_id) >= 0
        }
    )
    referenced_owner_ids = {int(owner_id) for word in selected_words for owner_id in word.owners.tolist()}
    referenced_site_ids = set(referenced_owner_ids)
    for boundary_id in referenced_boundary_ids:
        referenced_site_ids.update(int(value) for value in pairs[boundary_id].tolist())

    source_boundary_ids = torch.tensor(referenced_boundary_ids, dtype=torch.int64)
    source_site_ids = torch.tensor(sorted(referenced_site_ids), dtype=torch.int64)
    boundary_to_local = {boundary_id: local_id for local_id, boundary_id in enumerate(referenced_boundary_ids)}
    site_to_local = {site_id: local_id for local_id, site_id in enumerate(source_site_ids.tolist())}

    word_offsets = [0]
    word_owner: list[int] = []
    word_left_incidence: list[int] = []
    word_right_incidence: list[int] = []
    track_incidence_offsets = [0]
    incidence_boundary: list[int] = []
    for word in selected_words:
        row_boundaries = sorted(
            {int(cut_id) for cut_id in torch.cat((word.left_cut_ids, word.right_cut_ids)).tolist() if int(cut_id) >= 0}
        )
        row_boundary_to_incidence = {boundary_id: local_id for local_id, boundary_id in enumerate(row_boundaries)}
        incidence_boundary.extend(boundary_to_local[boundary_id] for boundary_id in row_boundaries)
        track_incidence_offsets.append(len(incidence_boundary))
        word_owner.extend(site_to_local[int(owner_id)] for owner_id in word.owners.tolist())
        word_left_incidence.extend(
            _recode_cut_id(int(cut_id), row_boundary_to_incidence) for cut_id in word.left_cut_ids.tolist()
        )
        word_right_incidence.extend(
            _recode_cut_id(int(cut_id), row_boundary_to_incidence) for cut_id in word.right_cut_ids.tolist()
        )
        word_offsets.append(len(word_owner))

    compact_pairs = (
        torch.tensor(
            [
                [site_to_local[int(site_id)] for site_id in pairs[boundary_id].tolist()]
                for boundary_id in referenced_boundary_ids
            ],
            dtype=torch.int32,
        )
        if referenced_boundary_ids
        else torch.empty((0, 2), dtype=torch.int32)
    )
    block = PreparedWorldFoamTrackBlock(
        source_track_ids=torch.arange(track_start, track_end, dtype=torch.int64),
        source_boundary_ids=source_boundary_ids.contiguous(),
        source_site_ids=source_site_ids.contiguous(),
        word_offsets_i32=torch.tensor(word_offsets, dtype=torch.int32),
        word_owner_i32=torch.tensor(word_owner, dtype=torch.int32),
        word_left_incidence_i32=torch.tensor(word_left_incidence, dtype=torch.int32),
        word_right_incidence_i32=torch.tensor(word_right_incidence, dtype=torch.int32),
        track_incidence_offsets_i32=torch.tensor(track_incidence_offsets, dtype=torch.int32),
        incidence_boundary_i32=torch.tensor(incidence_boundary, dtype=torch.int32),
        boundary_site_pairs_i32=compact_pairs.contiguous(),
    )
    _validate_prepared_block(block)
    return block


def gather_prepared_rows(values: torch.Tensor, source_ids: torch.Tensor) -> torch.Tensor:
    """Gather a compact block without changing trailing dimensions."""

    values_tensor = torch.as_tensor(values)
    ids = torch.as_tensor(source_ids, dtype=torch.int64, device=values_tensor.device)
    if values_tensor.ndim < 1:
        raise ValueError("values must have at least one dimension")
    if ids.ndim != 1:
        raise ValueError("source_ids must be one-dimensional")
    if ids.numel() and (int(ids.min().item()) < 0 or int(ids.max().item()) >= values_tensor.shape[0]):
        raise ValueError("source_ids contain an out-of-range row")
    return values_tensor.index_select(0, ids).contiguous()


def scatter_prepared_rows(
    compact_grad: torch.Tensor,
    source_ids: torch.Tensor,
    *,
    output_rows: int,
) -> torch.Tensor:
    """Allocate a global result and scatter-add compact rows into it.

    This convenience helper is useful for one-shot references.  Streamed
    spatial training should call :func:`accumulate_prepared_rows_` with a
    caller-owned buffer so every ``B_p`` block shares one allocation.
    """

    compact = torch.as_tensor(compact_grad)
    if output_rows < 0:
        raise ValueError("output_rows must be nonnegative")
    output = torch.zeros(
        (output_rows, *compact.shape[1:]),
        dtype=compact.dtype,
        device=compact.device,
    )
    return accumulate_prepared_rows_(output, compact, source_ids)


def accumulate_prepared_rows_(
    output: torch.Tensor,
    compact_grad: torch.Tensor,
    source_ids: torch.Tensor,
) -> torch.Tensor:
    """Index-add compact rows into an existing global gradient buffer.

    Returning ``output`` follows PyTorch's in-place helper convention and
    makes it easy for callers to assert that storage identity never changes.
    Shared source rows are intentionally summed across calls.
    """

    global_grad = torch.as_tensor(output)
    compact = torch.as_tensor(compact_grad)
    ids = torch.as_tensor(source_ids, dtype=torch.int64, device=global_grad.device)
    if global_grad.ndim < 1:
        raise ValueError("output must have at least one dimension")
    if compact.ndim < 1 or ids.ndim != 1 or compact.shape[0] != ids.numel():
        raise ValueError("compact_grad rows must match one-dimensional source_ids")
    if tuple(compact.shape[1:]) != tuple(global_grad.shape[1:]):
        raise ValueError("compact_grad trailing dimensions must match output")
    if compact.dtype != global_grad.dtype or compact.device != global_grad.device:
        raise ValueError("compact_grad must match output dtype and device")
    if global_grad.requires_grad:
        raise ValueError("caller-owned gradient output must not require autograd")
    if ids.numel() and (int(ids.min().item()) < 0 or int(ids.max().item()) >= global_grad.shape[0]):
        raise ValueError("source_ids contain an out-of-range output row")
    global_grad.index_add_(0, ids, compact)
    return global_grad


def _recode_cut_id(cut_id: int, boundary_to_incidence: dict[int, int]) -> int:
    if cut_id < 0:
        return cut_id
    try:
        return boundary_to_incidence[cut_id]
    except KeyError as error:
        raise ValueError("word cut is missing from its row incidence table") from error


def _validate_selected_words(
    words: tuple[StableCellWord, ...],
    *,
    site_count: int,
    boundary_count: int,
) -> None:
    for word in words:
        owners = torch.as_tensor(word.owners, dtype=torch.int64).reshape(-1)
        left = torch.as_tensor(word.left_cut_ids, dtype=torch.int64).reshape(-1)
        right = torch.as_tensor(word.right_cut_ids, dtype=torch.int64).reshape(-1)
        if owners.numel() < 1 or left.shape != owners.shape or right.shape != owners.shape:
            raise ValueError("each word must contain aligned non-empty owner/cut rows")
        if int(owners.min().item()) < 0 or int(owners.max().item()) >= site_count:
            raise ValueError("word contains an out-of-range owner id")
        if int(left[0].item()) != NEAR_CUT_ID or int(right[-1].item()) != FAR_CUT_ID:
            raise ValueError("each word must begin at near and end at far")
        if owners.numel() > 1 and not torch.equal(right[:-1], left[1:]):
            raise ValueError("adjacent word segments must share the same cut id")
        if bool(torch.any(right == NEAR_CUT_ID).item()) or bool(torch.any(left[1:] == NEAR_CUT_ID).item()):
            raise ValueError("only the first segment may use the near cut")
        if bool(torch.any(left == FAR_CUT_ID).item()) or bool(torch.any(right[:-1] == FAR_CUT_ID).item()):
            raise ValueError("only the final segment may use the far cut")
        if owners.numel() > 1 and bool(torch.any(right[:-1] < 0).item()):
            raise ValueError("every nonfinal segment must end at an active boundary")
        if bool(torch.any(left == right).item()):
            raise ValueError("word segments must have distinct left and right cuts")
        cut_ids = torch.cat((left, right))
        if bool(torch.any(cut_ids < FAR_CUT_ID).item()):
            raise ValueError("word uses an unknown negative cut sentinel")
        positive = cut_ids[cut_ids >= 0]
        if positive.numel() and int(positive.max().item()) >= boundary_count:
            raise ValueError("word contains an out-of-range boundary id")


def _validate_prepared_block(block: PreparedWorldFoamTrackBlock) -> None:
    if int(block.word_offsets_i32[0].item()) != 0:
        raise AssertionError("word offsets must start at zero")
    if int(block.word_offsets_i32[-1].item()) != block.word_count:
        raise AssertionError("word offsets must end at word_count")
    if int(block.track_incidence_offsets_i32[0].item()) != 0:
        raise AssertionError("incidence offsets must start at zero")
    if int(block.track_incidence_offsets_i32[-1].item()) != block.incidence_count:
        raise AssertionError("incidence offsets must end at incidence_count")
    for track_id in range(block.track_count):
        word_start = int(block.word_offsets_i32[track_id].item())
        word_end = int(block.word_offsets_i32[track_id + 1].item())
        incidence_start = int(block.track_incidence_offsets_i32[track_id].item())
        incidence_end = int(block.track_incidence_offsets_i32[track_id + 1].item())
        row_incidence_count = incidence_end - incidence_start
        for cuts in (block.word_left_incidence_i32, block.word_right_incidence_i32):
            active = cuts[word_start:word_end]
            active = active[active >= 0]
            if active.numel() and (int(active.min().item()) < 0 or int(active.max().item()) >= row_incidence_count):
                raise AssertionError("word incidence id escaped its track row")
