from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import worldfoam_spatial_major_heldout_evaluator as subject
from paper_training_protocol import PaperRGBMetricAccumulator
from worldfoam_native4d_public_quality_row import PixelChunkRequest, _canonical_bytes


CAMERAS = 2
FRAMES = 3
HEIGHT = 2
WIDTH = 3
PIXELS = HEIGHT * WIDTH


def _target(camera: int, frame: int, pixels: tuple[int, ...]) -> torch.Tensor:
    rows = [
        [
            (camera * 31 + frame * 13 + pixel * 5 + channel * 2) / 255.0
            for channel in range(3)
        ]
        for pixel in pixels
    ]
    return torch.tensor(rows, dtype=torch.float32).contiguous()


def _prediction(camera: int, frame: int, pixels: tuple[int, ...]) -> torch.Tensor:
    return (_target(camera, frame, pixels) * 0.8 + 0.05).contiguous()


class _Session:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls: list[tuple[int, tuple[int, ...]]] = []
        self.target_calls: list[tuple[int, tuple[int, ...]]] = []
        self.fail_on_call = fail_on_call

    def maximum_heldout_tracks_per_cross_time_block(self) -> int:
        return 2

    def render_heldout_track_block_across_frames(
        self,
        *,
        camera_index: int,
        pixel_ids: tuple[int, ...],
    ) -> torch.Tensor:
        self.calls.append((camera_index, pixel_ids))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("injected spatial replay failure")
        return torch.stack(
            [
                _prediction(camera_index, frame, pixel_ids)
                for frame in range(FRAMES)
            ]
        ).contiguous()

    def read_heldout_target_track_block_across_frames(
        self,
        *,
        camera_index: int,
        pixel_ids: tuple[int, ...],
    ):
        self.target_calls.append((camera_index, pixel_ids))
        target = torch.stack(
            [_target(camera_index, frame, pixel_ids) for frame in range(FRAMES)]
        ).contiguous()
        source_only = target.numel() * target.element_size() * 2
        returned = target.numel() * target.element_size()
        payload = {
            "schema_version": 1,
            "kind": subject.TARGET_READ_KIND,
            "camera_index": camera_index,
            "pixel_ids": pixel_ids,
            "pixel_ids_sha256": subject._canonical_sha256(pixel_ids),
            "track_count": len(pixel_ids),
            "frame_count": FRAMES,
            "observation_count": len(pixel_ids) * FRAMES,
            "selection_mode": "direct_pixels",
            "source_provenance": "fake-rgb8-pixel-time-v1",
            "source_only_visible_peak_logical_tensor_bytes_upper_bound": source_only,
            "returned_target_tensor_bytes": returned,
            "source_plus_returned_target_peak_logical_tensor_bytes_upper_bound": (
                source_only + returned
            ),
            "transient_mapped_address_space_bytes": 0,
            "requested_unique_mapped_page_count": 0,
            "requested_mapped_page_bytes_upper_bound": 0,
            "mapping_closed_before_return": True,
            "full_frame_materialization_count": 0,
            "ray_tensor_bytes": 0,
        }
        return target, {
            **payload,
            "generation_digest": subject._canonical_sha256(payload),
        }

    def heldout_spatial_major_receipt(self):
        track_count = sum(len(pixels) for _camera, pixels in self.calls)
        payload = {
            "schema_version": 1,
            "kind": subject.SESSION_REPLAY_KIND,
            "camera_count": CAMERAS,
            "frame_count": FRAMES,
            "image_height": HEIGHT,
            "image_width": WIDTH,
            "cross_time_track_block_size": 2,
            "render_call_count": len(self.calls),
            "cold_track_compile_count": track_count,
            "complete_camera_record_validation_count": track_count * FRAMES,
            "admitted_site_reference_upper_bound": track_count * 7,
            "native_bundle_count": len(self.calls),
            "native_tracks_per_bundle_limit": 2,
            "expected_native_bundle_count": len(self.calls),
            "native_sample_count": track_count * FRAMES,
            "native_prediction_target_observation_read_count": track_count * FRAMES,
            "spatial_target_staging_call_count": len(self.target_calls),
            "spatial_target_staging_observation_count": sum(
                len(pixels) * FRAMES for _camera, pixels in self.target_calls
            ),
            "spatial_target_staging_peak_logical_bytes": max(
                (
                    len(pixels) * FRAMES * 3 * 4 * 3
                    for _camera, pixels in self.target_calls
                ),
                default=0,
            ),
            "prediction_receipt_chain_sha256": "1" * 64,
            "target_receipt_chain_sha256": "2" * 64,
            "target_ray_tensor_bytes": 0,
            "full_pixel_full_temporal": True,
            "frame_major_recompile_per_time_used": False,
            "prediction_spool_dtype": "float32",
        }
        return {**payload, "generation_digest": subject._canonical_sha256(payload)}


class _Media:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.frames: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.abort_count = 0

    def add_frame(self, prediction, target) -> None:
        self.frames.append((prediction.clone(), target.clone()))

    def finish(self, *, expected_frame_count: int) -> Path:
        assert len(self.frames) == expected_frame_count
        self.path.write_bytes(b"fake-media")
        return self.path

    def abort(self) -> None:
        self.abort_count += 1
        self.path.unlink(missing_ok=True)


def _context():
    return SimpleNamespace(
        protocol=SimpleNamespace(
            dataset=SimpleNamespace(
                frame_count=FRAMES,
                heldout_cameras=tuple(f"camera-{index}" for index in range(CAMERAS)),
            ),
            final_stage=SimpleNamespace(
                image_size=SimpleNamespace(height=HEIGHT, width=WIDTH),
            ),
        ),
        work_plan=SimpleNamespace(
            maximum_pixels_per_chunk=4,
            heldout_maximum_pixels_per_chunk=4,
        ),
    )


def _lpips(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float((prediction - target).abs().mean().item())


def _reference_metrics() -> tuple[dict[str, float], float]:
    accumulator = PaperRGBMetricAccumulator()
    lpips = 0.0
    for camera in range(CAMERAS):
        for frame in range(FRAMES):
            prediction = _prediction(camera, frame, tuple(range(PIXELS))).reshape(
                HEIGHT,
                WIDTH,
                3,
            )
            target = _target(camera, frame, tuple(range(PIXELS))).reshape(
                HEIGHT,
                WIDTH,
                3,
            )
            accumulator.update(prediction.unsqueeze(0), target.unsqueeze(0))
            lpips += _lpips(prediction.unsqueeze(0), target.unsqueeze(0))
    return accumulator.metrics(prefix="heldout_eval"), lpips / (CAMERAS * FRAMES)


def _run_once(tmp_path: Path, suffix: str):
    spool = tmp_path / f"spool-{suffix}"
    session = _Session()
    media = _Media(tmp_path / f"media-{suffix}.mp4")
    result = subject.evaluate_worldfoam_spatial_major_final_checkpoint(
        _context(),
        session=session,
        media_sink=media,
        maximum_render_call_count=10,
        spool_directory=spool,
        minimum_free_bytes_after_spool=0,
        write_superblock_track_limit=4,
        lpips_metric=_lpips,
    )
    return result, session, media, spool


def test_spatial_major_evaluator_is_exact_bounded_and_deterministic(tmp_path):
    first, session, media, spool = _run_once(tmp_path, "first")
    second, *_rest = _run_once(tmp_path, "second")

    expected_calls = [
        (camera, pixels)
        for camera in range(CAMERAS)
        for pixels in ((0, 1), (2, 3), (4, 5))
    ]
    assert session.calls == expected_calls
    assert session.target_calls == expected_calls
    assert all(len(pixels) <= 2 for _camera, pixels in session.calls)
    assert len(media.frames) == CAMERAS * FRAMES
    assert not tuple(spool.iterdir())

    reference, reference_lpips = _reference_metrics()
    assert first.evaluation.frame_count == CAMERAS * FRAMES
    assert first.evaluation.pixel_count == CAMERAS * FRAMES * PIXELS
    assert first.evaluation.pixel_chunk_count == (
        CAMERAS * FRAMES * math.ceil(PIXELS / 4)
    )
    common_coverage = hashlib.sha256()
    for camera in range(CAMERAS):
        for frame in range(FRAMES):
            for pixel_start in range(0, PIXELS, 4):
                request = PixelChunkRequest(
                    split="heldout",
                    step=None,
                    sample_slot=None,
                    camera_index=camera,
                    frame_index=frame,
                    pixel_start=pixel_start,
                    pixel_count=min(4, PIXELS - pixel_start),
                    image_height=HEIGHT,
                    image_width=WIDTH,
                )
                common_coverage.update(_canonical_bytes(request.as_dict()))
                common_coverage.update(b"\n")
    assert first.evaluation.coverage_sha256 == common_coverage.hexdigest()
    assert first.evaluation.metrics["heldout_eval_l1"] == pytest.approx(
        reference["heldout_eval_l1"]
    )
    assert first.evaluation.metrics["heldout_eval_psnr"] == pytest.approx(
        reference["heldout_eval_psnr"]
    )
    assert first.evaluation.metrics["heldout_eval_ssim"] == pytest.approx(
        reference["heldout_eval_ssim"]
    )
    assert first.evaluation.metrics["heldout_eval_lpips"] == pytest.approx(
        reference_lpips
    )
    assert subject.validate_spatial_replay_receipt(first.spatial_replay_receipt)
    assert (
        first.spatial_replay_receipt
        == second.spatial_replay_receipt
    )
    assert first.spatial_replay_receipt["spatial_track_count"] == CAMERAS * PIXELS
    assert first.spatial_replay_receipt["target_pixel_count"] == (
        CAMERAS * FRAMES * PIXELS
    )
    assert first.spatial_replay_receipt[
        "total_target_source_observation_read_count"
    ] == (
        2 * CAMERAS * FRAMES * PIXELS
    )
    assert first.spatial_replay_receipt[
        "total_target_observation_traversal_count"
    ] == (3 * CAMERAS * FRAMES * PIXELS)


def test_spatial_major_evaluator_fails_closed_without_session_seam(tmp_path):
    media = _Media(tmp_path / "media.mp4")
    with pytest.raises(TypeError, match="spatial-major prediction/target replay"):
        subject.evaluate_worldfoam_spatial_major_final_checkpoint(
            _context(),
            session=object(),
            media_sink=media,
            maximum_render_call_count=10,
            spool_directory=tmp_path / "spool",
            minimum_free_bytes_after_spool=0,
            write_superblock_track_limit=4,
            lpips_metric=_lpips,
        )
    assert not (tmp_path / "spool").exists()
    assert media.abort_count == 0


def test_spatial_major_evaluator_cleans_spool_and_media_after_failure(tmp_path):
    spool = tmp_path / "spool"
    media = _Media(tmp_path / "media.mp4")
    with pytest.raises(RuntimeError, match="injected spatial replay failure"):
        subject.evaluate_worldfoam_spatial_major_final_checkpoint(
            _context(),
            session=_Session(fail_on_call=2),
            media_sink=media,
            maximum_render_call_count=10,
            spool_directory=spool,
            minimum_free_bytes_after_spool=0,
            write_superblock_track_limit=4,
            lpips_metric=_lpips,
        )
    assert spool.is_dir()
    assert not tuple(spool.iterdir())
    assert media.abort_count == 1
    assert not media.path.exists()


def test_spatial_major_evaluator_rejects_call_budget_before_spool(tmp_path):
    media = _Media(tmp_path / "media.mp4")
    with pytest.raises(MemoryError, match="frozen call bound"):
        subject.evaluate_worldfoam_spatial_major_final_checkpoint(
            _context(),
            session=_Session(),
            media_sink=media,
            maximum_render_call_count=5,
            spool_directory=tmp_path / "spool",
            minimum_free_bytes_after_spool=0,
            write_superblock_track_limit=4,
            lpips_metric=_lpips,
        )
    assert not (tmp_path / "spool").exists()
    assert media.abort_count == 0


def test_production_session_distinguishes_128_track_calls_from_13_track_bundles():
    from worldfoam_native4d_public_quality_executor import (
        WorldFoamPublicQualitySession,
    )

    session = object.__new__(WorldFoamPublicQualitySession)
    session._heldout_provider = SimpleNamespace(
        view_count=1,
        frame_count=300,
        height=1,
        width=260,
    )
    session.inputs = SimpleNamespace(
        maximum_tracks_per_bundle=128,
        maximum_observations_per_bundle=4096,
    )
    session._training_finalized = True
    session._heldout_pilot_prepared = False
    session.state = SimpleNamespace(site_count=1024)
    session._heldout_spatial_major_call_count = 3
    session._heldout_spatial_major_track_count = 260
    # ceil(128/13) + ceil(128/13) + ceil(4/13) = 21.
    session._heldout_spatial_major_native_bundle_count = 21
    session._heldout_spatial_major_native_sample_count = 260 * 300
    session._heldout_spatial_major_prediction_target_read_count = 260 * 300
    session._heldout_spatial_major_target_staging_call_count = 3
    session._heldout_spatial_major_target_staging_observation_count = 260 * 300
    session._heldout_spatial_major_target_staging_peak_logical_bytes = 1
    session._heldout_spatial_major_prediction_receipt_chain_sha256 = "1" * 64
    session._heldout_spatial_major_target_receipt_chain_sha256 = "2" * 64

    assert session.maximum_heldout_tracks_per_cross_time_block() == 128
    assert session._heldout_native_tracks_per_bundle_limit() == 13
    receipt = session.heldout_spatial_major_receipt()
    assert receipt["render_call_count"] == 3
    assert receipt["native_tracks_per_bundle_limit"] == 13
    assert receipt["native_bundle_count"] == 21
    assert receipt["expected_native_bundle_count"] == 21
