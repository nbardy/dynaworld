from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import generate_worldfoam_public_quality_assets as asset_generator
import run_worldfoam_public_quality_ablation as runner
import verify_worldfoam_public_quality_ablation as verifier


SCENES = {
    "coffee_martini": ("cam04", "cam09", "cam06"),
    "cook_spinach": ("cam14", "cam18", "cam16"),
    "cut_roasted_beef": ("cam14", "cam18", "cam16"),
}


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture_contract(root: Path) -> tuple[Path, dict]:
    scenes = []
    for scene, (train_a, train_b, heldout) in SCENES.items():
        manifest_path = Path("src/dataset_configs") / f"{scene}.jsonl"
        sample_id = f"neural3d_{scene}_fixture"
        manifest = {
            "dataset": "neural_3d_video",
            "scene": scene,
            "frame_count": 300,
            "sample_id": sample_id,
            "dataset_scene_dir": f"data/external/{scene}",
            "train_cameras": [train_a, train_b],
            "heldout_cameras": [heldout],
        }
        manifest_file = root / manifest_path
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        protocol_path = (
            Path("src/train_configs/paper_protocols") / f"{scene}_fixed.jsonc"
        )
        protocol = {
            "name": f"{scene}_fixed",
            "enabled": True,
            "dataset": {
                "manifest": str(manifest_path),
                "sample_id": sample_id,
                "train_cameras": [train_a, train_b],
                "heldout_cameras": [heldout],
                "frame_count": 300,
                "fps": 30.0,
            },
            "steps": 300,
            "max_train_seconds": 86400.0,
            "frames_per_step": 4,
            "same_time_count": 2,
            "local_time_count": 1,
            "local_time_radius": 8,
            "sampler_seed_offset": 7001,
            "stages": [
                {
                    "label": "fixed_512w",
                    "until_step": 300,
                    "image_size": [384, 512],
                    "primitive_count": 1024,
                    "frames_per_step": 4,
                    "lr_multiplier": 0.35,
                }
            ],
        }
        _write_json(root / protocol_path, protocol)
        scenes.append({"scene": scene, "protocol": str(protocol_path)})
    config = {
        "schema_version": 1,
        "name": "worldfoam_native4d_g4_public_quality_v1",
        "artifact_kind": verifier.ARTIFACT_KIND,
        "output_root": "outputs/g4_fixture",
        "dataset_family": "neural_3d_video",
        "device": "mps",
        "seeds": [17, 29, 43],
        "scenes": scenes,
        "routes": [
            {
                "route": "worldfoam_native4d",
                "lane": "worldfoam_native4d",
                "execution_mode": "compiled_shared_adjoint",
                "backend": "metal_real_native",
                "same_representation_group": "worldfoam_retained_depth_v1",
            },
            {
                "route": "worldfoam_framewise_replay",
                "lane": "worldfoam_native4d",
                "execution_mode": "framewise_same_representation",
                "backend": "metal_real_native",
                "same_representation_group": "worldfoam_retained_depth_v1",
            },
            {
                "route": "world_tubes",
                "lane": "world_tubes",
                "execution_mode": "selected_time_uvt_replay",
                "backend": "star_uvt_metal",
                "same_representation_group": "world_tubes_v1",
            },
            {
                "route": "dynamic_3dgs",
                "lane": "dynamic_3dgs",
                "execution_mode": "per_frame_dynamic_splats",
                "backend": "fast_mac_metal",
                "same_representation_group": "dynamic_3dgs_v1",
            },
        ],
        "public_protocol": {
            "dataset_frame_count": 300,
            "image_size": [384, 512],
            "optimizer_steps": 300,
            "frames_per_step": 4,
            "primitive_count": 1024,
            "target_pixel_budget": 235929600,
            "require_calibrated_multiview": True,
            "require_final_checkpoint_heldout_evaluation": True,
            "require_full_temporal_heldout_evaluation": True,
            "require_identical_evaluator_within_scene_seed": True,
            "require_identical_sample_schedule_within_scene_seed": True,
            "require_wandb_run_file": True,
            "require_clean_source": True,
        },
        "acceptance": {
            "minimum_native_heldout_psnr": 13.0,
            "minimum_native_heldout_ssim": 0.15,
            "minimum_mean_psnr_delta_vs_framewise_replay_db": -1.0,
            "minimum_worst_psnr_delta_vs_framewise_replay_db": -2.0,
            "minimum_mean_ssim_delta_vs_framewise_replay": -0.03,
            "maximum_mean_lpips_delta_vs_framewise_replay": 0.05,
            "maximum_mean_psnr_deficit_vs_best_gaussian_db": 4.0,
            "maximum_worst_psnr_deficit_vs_best_gaussian_db": 6.0,
        },
        "execution_guard": {
            "minimum_available_memory_bytes": 8589934592,
            "maximum_swap_used_bytes": 2147483648,
            "minimum_disk_free_bytes": 34359738368,
            "maximum_load_1m_per_logical_cpu": 0.75,
            "run_one_fresh_process_at_a_time": True,
            "abort_before_first_row_if_any_route_is_unavailable": True,
        },
    }
    config_path = root / "src/train_configs/paper_protocols/g4.jsonc"
    _write_json(config_path, config)
    return config_path, config


def _identity(root: Path, path: Path, **extra) -> dict:
    absolute = root / path
    absolute.parent.mkdir(parents=True, exist_ok=True)
    absolute.write_bytes((str(path) + " fixture payload").encode())
    return {
        "path": str(path),
        "sha256": verifier.file_sha256(absolute),
        "bytes": absolute.stat().st_size,
        **extra,
    }


def _accepted_artifact(root: Path, config_path: Path, config: dict) -> dict:
    receipt = verifier.validate_contract(config, config_path=config_path)
    route_specs = {row["route"]: row for row in config["routes"]}
    rows = []
    source_commit = "a" * 40
    for scene in SCENES:
        scene_receipt = receipt["scenes"][scene]
        for seed in config["seeds"]:
            schedule = verifier.canonical_sha256({"scene": scene, "seed": seed})
            evaluator = verifier.canonical_sha256({"evaluator": scene, "seed": seed})
            worldfoam_representation = verifier.canonical_sha256(
                {"worldfoam": scene, "seed": seed}
            )
            for route in verifier.REQUIRED_ROUTES:
                route_spec = route_specs[route]
                prefix = Path("outputs/g4_fixture") / scene / f"seed_{seed}" / route
                psnr = {
                    "worldfoam_native4d": 20.0,
                    "worldfoam_framewise_replay": 20.0,
                    "world_tubes": 21.0,
                    "dynamic_3dgs": 22.0,
                }[route]
                checkpoint = _identity(
                    root, prefix / "checkpoint_final.pt", step=300
                )
                attestation = {
                    "real_native": True,
                    "native_extension_attested": False,
                    "fake_native": False,
                    "source_only": False,
                    "procedural_target": False,
                    "public_target_provider": True,
                    "heldout_evaluator": True,
                    "full_geometry_trainable": True,
                    "compiled_shared_adjoint": route == "worldfoam_native4d",
                    "same_representation_framewise_replay": route
                    == "worldfoam_framewise_replay",
                }
                raw = {
                    "schema_version": 1,
                    "row_kind": verifier.ROW_KIND,
                    "row_id": f"{scene}/seed_{seed}/{route}",
                    "scene": scene,
                    "seed": seed,
                    "route": route,
                    "lane": route_spec["lane"],
                    "execution_mode": route_spec["execution_mode"],
                    "backend": route_spec["backend"],
                    "protocol_path": scene_receipt["protocol_path"],
                    "protocol_sha256": scene_receipt["protocol_sha256"],
                    "dataset_manifest_path": scene_receipt["manifest_path"],
                    "dataset_manifest_sha256": scene_receipt["manifest_sha256"],
                    "sample_id": scene_receipt["sample_id"],
                    "train_cameras": scene_receipt["train_cameras"],
                    "heldout_cameras": scene_receipt["heldout_cameras"],
                    "frame_count": 300,
                    "image_size": [384, 512],
                    "optimizer_steps": 300,
                    "frames_per_step": 4,
                    "primitive_state_temporal_scope": (
                        "per_frame"
                        if route == "dynamic_3dgs"
                        else "shared_across_time"
                    ),
                    "target_pixel_budget": 235929600,
                    "sample_schedule_sha256": schedule,
                    "evaluator_sha256": evaluator,
                    "representation_sha256": (
                        worldfoam_representation
                        if route.startswith("worldfoam_")
                        else verifier.canonical_sha256(
                            {"route": route, "scene": scene, "seed": seed}
                        )
                    ),
                    "source_commit": source_commit,
                    "source_dirty": False,
                    "public_quality_evidence": True,
                    "paper_evidence_eligible": True,
                    "proxy_or_test_artifact": False,
                    "measurement_is_simulated": False,
                    "smoke": False,
                    "dataset_is_public": True,
                    "calibrated_multiview": True,
                    "final_checkpoint_evaluation": True,
                    "full_temporal_heldout_evaluation": True,
                    "route_attestation": attestation,
                    "checkpoint": checkpoint,
                    "heldout_media": _identity(
                        root,
                        prefix / "heldout.mp4",
                        camera_ids=scene_receipt["heldout_cameras"],
                        frame_count=300,
                    ),
                    "wandb_run_file": _identity(
                        root,
                        prefix / "run.wandb",
                        run_id=f"{scene}-{seed}-{route}",
                        mode="offline",
                    ),
                    "metrics": {
                        "heldout_eval_psnr": psnr,
                        "heldout_eval_ssim": 0.6,
                        "heldout_eval_lpips": 0.2,
                        "heldout_eval_l1": 0.1,
                    },
                    "cost": {
                        "optimizer_steps": 300,
                        "target_pixels": 235929600,
                        "rasterized_pixels": 235929600,
                        "parameter_count": 1024,
                        "parameter_bytes": 65536,
                        "serialized_checkpoint_bytes": checkpoint["bytes"],
                        "final_active_primitive_count_per_render": 1024,
                        "stored_primitive_state_count": (
                            307200 if route == "dynamic_3dgs" else 1024
                        ),
                        "process_lifetime_peak_rss_through_checkpoint_bytes": 1024,
                        "sampled_peak_mps_driver_during_training_and_checkpoint_bytes": 1024,
                        "training_and_checkpoint_elapsed_s": 10.0,
                        "process_lifetime_peak_rss_through_heldout_evaluation_bytes": 2048,
                        "sampled_peak_mps_driver_through_heldout_evaluation_bytes": 2048,
                        "executor_dataset_and_model_setup_elapsed_s": 1.0,
                        "heldout_evaluation_elapsed_s": 2.0,
                        "full_row_through_heldout_evaluation_elapsed_s": 13.0,
                    },
                }
                raw_path = prefix / "g4_row.json"
                _write_json(root / raw_path, raw)
                rows.append(
                    {
                        **raw,
                        "receipt": {
                            "path": str(raw_path),
                            "sha256": verifier.file_sha256(root / raw_path),
                            "bytes": (root / raw_path).stat().st_size,
                        },
                    }
                )
    artifact = {
        "schema_version": 1,
        "artifact_kind": verifier.ARTIFACT_KIND,
        "status": "measured",
        "public_quality_evidence": True,
        "proxy_or_test_artifact": False,
        "measurement_is_simulated": False,
        "matrix_config": str(config_path.relative_to(root)),
        "matrix_config_sha256": verifier.file_sha256(config_path),
        "source_commit": source_commit,
        "rows": rows,
        "acceptance": verifier.compute_acceptance(rows, config),
        "artifact_sha256": "",
    }
    artifact["artifact_sha256"] = verifier.artifact_sha256(artifact)
    return artifact


def test_checked_in_g4_plan_is_exact_and_fails_before_execution() -> None:
    config = verifier.load_contract()
    plan = runner.finalize_plan(runner.build_plan(config))
    assert plan["expected_row_count"] == 36
    assert plan["fresh_process_count"] == 36
    assert len(plan["rows"]) == 36
    assert plan["runtime_ready"] is False
    assert plan["abort_before_first_row"] is True
    assert "public_native4d_spatial_compile_reuse_unimplemented" in plan[
        "runtime_blockers"
    ]
    assert "public_native4d_runtime_capability_receipt_missing" in plan[
        "runtime_blockers"
    ]
    assert "production_full_geometry_adapter_is_synthetic_only" not in plan[
        "runtime_blockers"
    ]
    assert "mapped_public_target_binding_is_train_only" not in plan[
        "runtime_blockers"
    ]
    assert "public_native4d_row_worker_missing" not in plan["runtime_blockers"]
    assert "worldfoam" not in {row["route"] for row in plan["rows"]}
    assert {row["route"] for row in plan["rows"]} == set(verifier.REQUIRED_ROUTES)


def test_execute_with_runtime_blockers_launches_no_subprocess(monkeypatch) -> None:
    config = verifier.load_contract()
    plan = runner.finalize_plan(
        runner.build_plan(config, allow_local_mps_execution=True)
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("G4 must abort before launching the first row")

    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    with pytest.raises(RuntimeError, match="aborted before first row"):
        runner.execute_plan(
            plan,
            config=config,
            config_path=verifier.DEFAULT_CONFIG,
            output_path=verifier.DEFAULT_ARTIFACT,
            allow_local_mps_execution=True,
        )


def test_verifier_accepts_only_complete_hash_bound_public_matrix(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, config = _fixture_contract(tmp_path)
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    artifact = _accepted_artifact(tmp_path, config_path, config)
    report = verifier.verify_artifact(artifact, config_path=config_path)
    assert report["accepted"] is True, report["failures"]
    assert report["observed_row_count"] == 36
    assert report["observed_scene_count"] == 3
    assert report["observed_seed_count"] == 3
    assert report["observed_route_count"] == 4


def test_assets_are_deterministic_and_require_the_accepted_matrix(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, config = _fixture_contract(tmp_path)
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    artifact = _accepted_artifact(tmp_path, config_path, config)
    artifact_path = tmp_path / "outputs/g4_fixture/worldfoam_public_quality_ablation.json"
    _write_json(artifact_path, artifact)
    output_dir = tmp_path / "paper_assets"
    first = asset_generator.write_assets(artifact_path, config_path, output_dir)
    first_bytes = {
        path.name: path.read_bytes() for path in output_dir.iterdir() if path.is_file()
    }
    second = asset_generator.write_assets(artifact_path, config_path, output_dir)
    second_bytes = {
        path.name: path.read_bytes() for path in output_dir.iterdir() if path.is_file()
    }
    assert first == second
    assert first_bytes == second_bytes
    assert asset_generator.verify_asset_dir(
        output_dir, artifact_path, config_path
    ) == []
    assert b"WorldFoam compiled" in (output_dir / "g4_public_quality_table.tex").read_bytes()
    assert b"G4 public held-out PSNR" in (output_dir / "g4_public_quality.svg").read_bytes()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda artifact: artifact["rows"][0].__setitem__(
                "proxy_or_test_artifact", True
            ),
            "proxy_or_test_artifact is not publication eligible",
        ),
        (
            lambda artifact: artifact["rows"][0]["route_attestation"].__setitem__(
                "fake_native", True
            ),
            "route attestation fake_native is not False",
        ),
        (
            lambda artifact: artifact["rows"][1].__setitem__(
                "representation_sha256", "f" * 64
            ),
            "compiled and replay WorldFoam changed representation",
        ),
        (
            lambda artifact: artifact["rows"].pop(),
            "artifact matrix is missing 1 row",
        ),
    ],
)
def test_verifier_rejects_rebound_proxy_fake_or_incomplete_rows(
    tmp_path: Path, monkeypatch, mutate, message: str
) -> None:
    config_path, config = _fixture_contract(tmp_path)
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    artifact = _accepted_artifact(tmp_path, config_path, config)
    mutate(artifact)
    # Rebind every retained raw receipt and the top-level digest. Semantic
    # checks must still reject the change even after all attacker-controlled
    # hashes are internally consistent.
    for row in artifact["rows"]:
        receipt_path = tmp_path / row["receipt"]["path"]
        _write_json(
            receipt_path,
            {key: row[key] for key in verifier.ROW_KEYS},
        )
        row["receipt"]["sha256"] = verifier.file_sha256(receipt_path)
        row["receipt"]["bytes"] = receipt_path.stat().st_size
    artifact["acceptance"] = verifier.compute_acceptance(artifact["rows"], config)
    artifact["artifact_sha256"] = verifier.artifact_sha256(artifact)
    report = verifier.verify_artifact(artifact, config_path=config_path)
    assert report["accepted"] is False
    assert any(message in failure for failure in report["failures"])
