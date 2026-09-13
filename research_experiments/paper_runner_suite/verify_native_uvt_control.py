"""Independently recompute the existing frozen-world tolerances from saved tensors.

Reload bounded CPU targets, use NumPy for image/loss/gradient math, and verify
identity, resource receipts, timing arithmetic and offline backing. Camera
calibration uses the producer's Metal arithmetic; no rendering is performed.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from config_utils import load_config_file
from multicam_video_data import load_multicam_video_bundle
from paper_multicam_targets import PaperMulticamTargetProvider
from paper_local_resources import configure_local_mps
from research_experiments.paper_runner_suite.run_unified_paper_ablation import FROZEN_WORLD_ACCEPTANCE, validate_frozen_world_evidence

def verify_control(out: Path, launch_out: Path, prior: dict, full_rgb: np.ndarray) -> dict:
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report = read(out / 'report.json')
    frozen = load_config_file(report['config']['frozen_config'])
    assert report['status'] == 'complete' and report['publication_eligible'] is False
    assert report['acceptance'] == FROZEN_WORLD_ACCEPTANCE == prior['acceptance']
    assert report['checkpoint']['sha256'] == frozen['expected_checkpoint_sha256'] == sha(report['checkpoint']['path'])
    assert report['world_state_after_sha256'] == frozen['expected_world_state_sha256'] == prior['checkpoint']['world_state_sha256']
    assert report['camera'] == prior['heldout_camera'] == 'cam06'
    frames = len(report['frame_indices'])
    full_frames = frozen['full_frames']
    assert full_frames == 32
    assert report['frame_indices'] == prior['frame_indices'] == [int(np.floor(i*31/(frames-1)+0.5)) for i in range(frames)]
    validate_frozen_world_evidence(prior, expected_frames=frames, expected_full_frames=full_frames,
        expected_image_size=(96, 128), expected_heldout_camera='cam06', expected_active_tubes=2048)
    assert report['image_size'] == prior['image_size'] == [96, 128]
    assert report['config']['frame_batch_sizes'][0] == 1
    assert set(report['correctness']) == set(report['timing']) == set(map(str, report['config']['frame_batch_sizes']))
    if 'selected_frame_count' in report['config']:
        assert frames == report['config']['selected_frame_count']
        assert all('execution_work' in row for row in report['correctness'].values())
    meta = read(Path(frozen['source_run']) / 'run_meta.json')
    baseline = load_config_file(meta['baseline_config'])
    configure_local_mps(load_config_file(frozen['resource_protocol']), 'mps')
    bundle = load_multicam_video_bundle(
        data_cfg=meta['config_data'], camera_cfg={**baseline['camera'], 'rig_init': 'neural_3d_video'},
        target_size=(96, 128), device=torch.device('mps'), frame_device=torch.device('cpu'), defer_video_frames=True,
    )
    provider = PaperMulticamTargetProvider(bundle.heldout_frame_sources, cache_capacity_frames=8)
    payloads = {}
    for batch, row in report['correctness'].items():
        assert row['artifact_sha256'] == sha(row['artifact'])
        payloads[batch] = torch.load(row['artifact'], map_location='cpu', weights_only=True)
        assert payloads[batch]['rgb'].shape == (frames, 96, 128, 3)
        assert payloads[batch]['rgb'].dtype == torch.float32
        assert set(payloads[batch]['covered']) == set(report['checkpoint']['parameter_names'])
        assert set(payloads[batch]['gradients']) == set(report['checkpoint']['parameter_names'])
        for name, tensor in payloads[batch]['gradients'].items():
            assert list(tensor.shape) == report['checkpoint']['parameter_shapes'][name]
            assert torch.isfinite(tensor).all() and tensor.dtype == torch.float32
        assert torch.isfinite(payloads[batch]['rgb']).all()

    # Seven-frame verification requests differ from all producer batching layouts.
    nchw_hash = hashlib.sha256()
    nhwc_hash = hashlib.sha256(b'torch.float32' + json.dumps([frames, 96, 128, 3]).encode())
    losses = {batch: 0.0 for batch in payloads}
    for start in range(0, frames, 7):
        stop = min(start + 7, frames)
        targets = provider.select_view_frames([0] * (stop - start), report['frame_indices'][start:stop]).numpy()
        nchw_hash.update(targets.tobytes())
        targets = np.ascontiguousarray(targets.transpose(0, 2, 3, 1))
        nhwc_hash.update(targets.tobytes())
        for batch, payload in payloads.items():
            diff = payload['rgb'][start:stop].numpy().astype(np.float64) - targets
            losses[batch] += float(np.sqrt(diff * diff + 1e-6).sum()) / (frames * 96 * 128 * 3)
    assert nhwc_hash.hexdigest() == prior['contract_hashes']['target_frames_sha256']

    def tensor_hash(tensor):
        return hashlib.sha256(str(tensor.dtype).encode() + json.dumps(list(tensor.shape)).encode() + tensor.cpu().numpy().tobytes()).hexdigest()

    K = bundle.heldout_K[0] if bundle.heldout_K.ndim == 3 else bundle.heldout_K[0, 0]
    w2c = bundle.heldout_w2c[0, 0]
    lens = 'pinhole' if bundle.heldout_lens_models is None else bundle.heldout_lens_models[0]
    dist = 'none' if bundle.heldout_distortions is None else tensor_hash(bundle.heldout_distortions[0])
    camera_hash = hashlib.sha256((tensor_hash(K) + tensor_hash(w2c) + dist).encode())
    camera_hash.update(json.dumps({'heldout_camera': bundle.heldout_camera_names[0], 'heldout_lens_model': lens,
        'camera_projection': frozen['camera_projection']}, sort_keys=True, separators=(',', ':')).encode())
    assert camera_hash.hexdigest() == prior['contract_hashes']['camera_program_sha256']


    reference = payloads['1']
    results = {}
    for batch, payload in payloads.items():
        assert payload['target_sha256'] == nchw_hash.hexdigest()
        assert abs(payload['loss'] - losses[batch]) <= FROZEN_WORLD_ACCEPTANCE['loss_absolute_delta']
        assert abs(losses[batch] - prior['loss']['replay']) <= FROZEN_WORLD_ACCEPTANCE['loss_absolute_delta']
        pixel_error = float(np.abs(payload['rgb'].numpy() - reference['rgb'].numpy()).max())
        assert pixel_error <= FROZEN_WORLD_ACCEPTANCE['image_max_abs_error']
        a = np.concatenate([reference['gradients'][k].numpy().ravel().astype(np.float64) for k in reference['gradients']])
        b = np.concatenate([payload['gradients'][k].numpy().ravel().astype(np.float64) for k in reference['gradients']])
        global_error = float(np.linalg.norm(a-b) / max(np.sqrt(a@a + b@b), 1e-12))
        parameter_errors = {}
        for name in reference['gradients']:
            x = reference['gradients'][name].numpy().astype(np.float64)
            y = payload['gradients'][name].numpy().astype(np.float64)
            parameter_errors[name] = float(np.linalg.norm(x-y) / max(np.linalg.norm(x)+np.linalg.norm(y), 1e-12))
        assert min(np.linalg.norm(a), np.linalg.norm(b)) > FROZEN_WORLD_ACCEPTANCE['min_world_vjp_l2_norm']
        assert global_error <= FROZEN_WORLD_ACCEPTANCE['gradient_global_normalized_l2_error']
        assert max(parameter_errors.values()) <= FROZEN_WORLD_ACCEPTANCE['gradient_max_parameter_normalized_l2_error']
        recorded = report['correctness'][batch]
        assert recorded['accepted'] and all(recorded['checks'].values())
        assert pixel_error == recorded['max_rgb_error']
        np.testing.assert_allclose(global_error, recorded['gradient']['global_normalized_l2_error'], rtol=1e-10, atol=1e-15)
        for name, error in parameter_errors.items():
            np.testing.assert_allclose(error, recorded['gradient']['per_parameter_normalized_l2_error'][name], rtol=1e-10, atol=1e-15)
        if 'execution_work' in recorded:
            width = int(batch)
            expected_ids = sorted({j for i in report['frame_indices'] for j in range((i//width)*width, min((i//width+1)*width, full_frames))})
            expected_work = {'projection_calls': len({i//width for i in report['frame_indices']}),
                'rendered_frames': len(expected_ids), 'rendered_rgb_values': len(expected_ids)*96*128*3,
                'loss_frames': frames, 'rendered_frame_indices': expected_ids}
            assert recorded['execution_work'] == expected_work
            assert all(sample['execution_work'] == expected_work for sample in report['timing'][batch]['samples'])
        cross_run_rgb_error = float(np.abs(payload['rgb'].numpy() - full_rgb[report['frame_indices']]).max())
        assert cross_run_rgb_error <= FROZEN_WORLD_ACCEPTANCE['image_max_abs_error']
        values = report['timing'][batch]
        assert len(values['samples']) == frozen['timing_repeats'] == 3 and frozen['timing_warmups'] == 1
        for sample in values['samples']:
            assert set(sample['phases_s']) == {'project_bin_render', 'target_cpu_load', 'target_transfer', 'loss', 'backward'}
            assert all(np.isfinite(value) and value >= 0 for value in sample['phases_s'].values())
        for key in ['evaluator_plus_backward_s', 'full_s']:
            samples = [sum(row['phases_s'].values()) if key == 'full_s' else
                       row['phases_s']['project_bin_render'] + row['phases_s']['backward'] for row in values['samples']]
            assert np.isfinite(samples).all() and min(samples) > 0
            summary = values[key]
            for metric, expected in [('min', min(samples)), ('p25', np.percentile(samples, 25)), ('median', np.median(samples)),
                                     ('p75', np.percentile(samples, 75)), ('max', max(samples)), ('mean', np.mean(samples))]:
                np.testing.assert_allclose(summary[metric], expected, rtol=1e-12, atol=1e-15)
            assert summary['count'] == 3
        results[batch] = {'execution_work': recorded.get('execution_work'), 'cross_run_rgb_error': cross_run_rgb_error, 'loss_recomputed_float64': losses[batch], 'max_rgb_error': pixel_error,
            'global_gradient_error': global_error, 'parameter_gradient_errors': parameter_errors,
            'EB_median_s': values['evaluator_plus_backward_s']['median'], 'full_median_s': values['full_s']['median']}

    receipt = read(launch_out / 'evaluate_native_batches_resource_receipt.json')
    assert not receipt['guard_tripped']
    assert receipt['local_resources']['limits'] == read('outputs/benchmarks/2026-09-13_direct_slice_cells/evaluate_device_chunks_resource_receipt.json')['local_resources']['limits']
    limits = receipt['local_resources']['limits']
    assert receipt['local_resources']['peak_process_tree_and_launcher_rss_bytes'] < limits['process_tree_rss_limit_bytes']
    assert receipt['local_resources']['peak_host_swap_growth_bytes'] <= limits['max_swap_growth_bytes']
    for row in [*report['correctness'].values(), *(s for t in report['timing'].values() for s in t['samples'])]:
        assert row['memory']['memory_sample_count'] > 0
        assert row['memory']['sampled_peak_current_allocated_bytes'] < limits['mps_allocator_limit_bytes']
    accounting = report['target_provider_accounting']
    assert accounting['cache_capacity_frames'] == 8 and accounting['full_video_tensor_materialization_count'] == 0
    assert not accounting['full_source_resident'] and accounting['output_device'] == 'cpu'
    assert max(accounting[k] for k in ['peak_request_frame_count','peak_cache_resident_frames','peak_decode_batch_frames']) <= 8
    for path, digest in read(launch_out / 'evaluate_native_batches_source_sha256.json').items():
        assert sha(launch_out / 'after' / Path(path).relative_to(Path.cwd())) == digest, path
    native = read(launch_out / 'native_extension_identity.json')
    assert sha(native['path']) == native['sha256']
    wandb = read(out / 'wandb_identity.json')
    assert wandb['mode'] == 'offline' and read(Path(wandb['dir']) / 'report.json') == report
    for row in report['correctness'].values():
        assert sha(Path(wandb['dir']) / Path(row['artifact']).name) == row['artifact_sha256']
    result = {'accepted': True, 'scope': report['scope'], 'publication_eligible': False, 'rows': results,
        'native': native, 'wandb': wandb, 'resources': receipt,
        'target_sha256_nchw': nchw_hash.hexdigest(), 'legacy_target_sha256_nhwc': nhwc_hash.hexdigest(),
        'legacy_camera_sha256': camera_hash.hexdigest(),
        'EB_batch1_over_batch32': results['1']['EB_median_s'] / results['32']['EB_median_s'],
        'full_batch1_over_batch32': results['1']['full_median_s'] / results['32']['full_median_s'],
        'report_sha256': sha(out / 'report.json')}
    return result


def main(output_dir: str) -> None:
    torch.set_num_threads(2)
    out = Path(output_dir).resolve()
    read = lambda p: json.loads(Path(p).read_text())
    previous = Path('outputs/benchmarks/2026-09-13_native_uvt_batch_control/attempt03')
    old_report = read(previous / 'report.json')
    identity = old_report['correctness']['1']
    assert hashlib.sha256(Path(identity['artifact']).read_bytes()).hexdigest() == identity['artifact_sha256']
    full_rgb = torch.load(identity['artifact'], map_location='cpu', weights_only=True)['rgb'].numpy()
    reference_path = Path('outputs/benchmarks/2026-09-13_grouped_target_provider/report.json')
    references = {r['frame_count']: r for r in read(reference_path)['rows']}
    if (out / 'progress.json').exists():
        progress = read(out / 'progress.json')
        assert progress['status'] == 'complete'
        configured = read(out / 'resolved_runtime_config.json')['frame_counts']
        assert [r['frames'] for r in progress['rows']] == configured
        reports = [Path(r['path']).resolve() for r in progress['rows']]
    else:
        reports = [out / 'report.json']
    results = []
    for path in reports:
        frames = len(read(path)['frame_indices'])
        row = verify_control(path.parent, out, references[frames], full_rgb)
        row.update(frame_count=frames, report_path=str(path))
        results.append(row)
        print(json.dumps({'F': frames, 'rows': row['rows']}), flush=True)
    result = {'accepted': True, 'publication_eligible': False, 'rows': results,
        'reference_report': str(reference_path),
        'reference_report_sha256': hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        'scope': 'native selected-loss control; full native frame work is charged'}
    (out / 'report_validation.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
