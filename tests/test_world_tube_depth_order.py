"""Protect image/color-gradient agreement at a float32 depth-order boundary."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

STAR = Path(__file__).resolve().parents[1] / 'third_party/fast-mac-gsplat/variants/star_uvt_v0'
sys.path.insert(0, str(STAR))
from torch_gsplat_bridge_star_uvt import UVTRenderConfig, brute_force_render_uvt_tubes
from torch_gsplat_bridge_star_uvt.projective_trace import (
    render_projective_trace_cell_atlas_reference,
    slice_projective_trace_cell_atlas_frames,
    uvt_tubes_to_projective_trace_cell_atlas,
)


def _fixture(spatial_depth=False):
    # Depth values from the learned F4 failure; strongly different colors make
    # its one-ULP ordering error visible without the original video/checkpoint.
    tensors = dict(
        ma=[[.5, .5, -.48883867263793945], [.5, .5, 1.4888386726379395]],
        q_uvt=[[.7, 0, 0, .9, 0, .04], [.7, 0, 0, .9, 0, .04]],
        depth0=[2.180581569671631, 2.180581569671631],
        depth_beta=[[0, 0, -.006274801678955555], [0, 0, .006274800281971693]],
        opacity=[.48, .48], color=[[.9, .2, .1], [.1, .3, .9]],
    )
    if spatial_depth:
        tensors['depth_beta'][0][0] = .01
        tensors['depth_beta'][1][0] = -.01
    return {k:torch.tensor(v, dtype=torch.float32, requires_grad=True) for k,v in tensors.items()}


def _compile(inputs, times):
    return uvt_tubes_to_projective_trace_cell_atlas(
        **inputs, times=times, sigma_px=1.0, image_width=2, image_height=2,
        tile_size=8, alpha_threshold=1/255, require_isotropic_spatial=False,
        auto_support_padding_from_alpha=True, allow_depth_affine_uv=True,
        stratify_visibility=True, mark_visibility_fallback=True,
    )


def _render(atlas, times):
    return render_projective_trace_cell_atlas_reference(
        atlas, times, image_width=2, image_height=2, tile_size=8, sigma_px=1.0,
        alpha_cutoff=1/255, transmittance_cutoff=1e-4, allow_fallback_cells=True,
    )


@pytest.mark.parametrize('spatial_depth', [False, True])
def test_centered_depth_preserves_rgb_and_vjp_through_frame_slicing(spatial_depth):
    inputs = _fixture(spatial_depth)
    times = torch.arange(4, dtype=torch.float32) - 1.5
    atlas = _compile(inputs, times)
    expected = brute_force_render_uvt_tubes(**inputs, config=UVTRenderConfig(height=2, width=2, frames=4))
    actual = torch.cat([_render(slice_projective_trace_cell_atlas_frames(atlas, start=f, stop=f+1), times[f:f+1]) for f in range(4)])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    # A colored cotangent catches order-dependent gradient reassignment.
    cotangent = torch.linspace(-1, 1, actual.numel()).reshape_as(actual)
    differentiable = tuple(inputs[k] for k in ['ma','q_uvt','opacity','color'])
    reference_grads = torch.autograd.grad((expected*cotangent).sum(), differentiable, retain_graph=True)
    actual_grads = torch.autograd.grad((actual*cotangent).sum(), differentiable)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual_grad, reference_grad, rtol=2e-5, atol=2e-6)


def test_source_id_tie_break_survives_trace_table_reordering():
    inputs = _fixture()
    times = torch.arange(4, dtype=torch.float32) - 1.5
    atlas = _compile(inputs, times)
    index = torch.tensor([1, 0])
    reordered = replace(atlas,
        **{name:getattr(atlas,name).index_select(0,index) for name in ['coeffs','opacity','color','opacity_time_coeffs','spatial_precision_uv','depth_affine_uv','depth_reference_uvt']},
        source_primitive_ids=(1,0),
        active_start=tuple(reversed(atlas.active_start)), active_stop=tuple(reversed(atlas.active_stop)),
        cells=[replace(cell, primitive_ids=tuple(1-i for i in cell.primitive_ids), ordered_primitive_ids=tuple(1-i for i in cell.ordered_primitive_ids)) for cell in atlas.cells],
    )
    torch.testing.assert_close(_render(reordered,times),_render(atlas,times),rtol=0,atol=0)


def test_stale_polynomial_edit_is_rejected_and_world_reprojection_updates_depth():
    inputs = _fixture()
    times = torch.arange(4, dtype=torch.float32) - 1.5
    atlas = _compile(inputs, times)
    coefficients = atlas.coeffs.clone()
    coefficients[0,6] += .01
    with pytest.raises(ValueError, match='depth reference is stale'):
        _render(replace(atlas,coeffs=coefficients),times)
    with torch.no_grad():
        inputs['depth0'][0] += .01
    updated = _render(_compile(inputs,times),times)
    expected = brute_force_render_uvt_tubes(**inputs,config=UVTRenderConfig(height=2,width=2,frames=4))
    torch.testing.assert_close(updated,expected,rtol=1e-5,atol=1e-6)


def test_retained_artifact_counts_source_depth_state(tmp_path):
    from research_project.benchmarks.multicam_heldout_compare import _write_frozen_atlas_storage
    atlas = _compile(_fixture(), torch.arange(4, dtype=torch.float32)-1.5)
    with_source = _write_frozen_atlas_storage(atlas, out_dir=tmp_path/'with', frame_count=4)
    without_source = _write_frozen_atlas_storage(replace(atlas,depth_reference_uvt=None), out_dir=tmp_path/'without', frame_count=4)
    assert with_source['tensor_payload_bytes'] - without_source['tensor_payload_bytes'] == 2*7*4
    assert with_source['tensor_count'] == without_source['tensor_count'] + 1


def test_live_uvt_cache_update_refreshes_source_depth():
    from star_uvt_projective_interval_backend import make_projective_cell_interval_live_atlas_from_uvt_tubes
    from research_project.trainer_harness.tile_metal_autograd import ProjectiveCellIntervalTrainerState
    inputs = _fixture()
    times = torch.arange(4, dtype=torch.float32)-1.5
    atlas = _compile(inputs,times)
    with torch.no_grad():
        inputs['depth0'][0] += .01
    live = make_projective_cell_interval_live_atlas_from_uvt_tubes(
        **inputs, cfg={'data':{'max_frames':4,'target_size':2}, 'feature_uvt':{'feature_dim':3,'alpha_threshold':1/255,'max_alpha':.99,'projective_interval':{'enabled':True,'allow_anisotropic_spatial_precision':True}}},
        reference_atlas=atlas,
    )
    state = ProjectiveCellIntervalTrainerState(atlas=live,times=times,config=UVTRenderConfig(height=2,width=2,frames=4),sigma_px=1.0,image_width=2,image_height=2,tile_size=8,allow_ambiguous_fallback=True,fallback_render_mode='mixed')
    state.refresh(force=True)
    expected = brute_force_render_uvt_tubes(**inputs,config=state.config)
    torch.testing.assert_close(_render(state.atlas,times),expected,rtol=1e-5,atol=1e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='local Metal required')
@pytest.mark.parametrize('spatial_depth', [False, True])
def test_metal_mixed_fallback_matches_native_rgb_and_parameter_vjp(spatial_depth):
    from research_project.benchmarks import multicam_heldout_compare as c
    inputs = {k:v.detach().to('mps').requires_grad_() for k,v in _fixture(spatial_depth).items()}
    times = torch.arange(4, device='mps', dtype=torch.float32)-1.5
    config = UVTRenderConfig(height=2,width=2,frames=4)
    atlas = _compile(inputs,times)
    actual = c.ProjectiveCellIntervalTrainerState(atlas=atlas,times=times,config=config,sigma_px=1.0,image_width=2,image_height=2,tile_size=8,fallback_render_mode='mixed').render()
    expected = c.render_projected_sequence(c.ProjectedTubeSequence(**inputs),config,backend='metal_tile',reduction_mode='index_add',sample_emission_mode='direct_atomic').rgb
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    cotangent = torch.linspace(-1,1,actual.numel(),device='mps').reshape_as(actual)
    parameters = tuple(inputs[k] for k in ['ma','q_uvt','opacity','color'])
    reference_grads = torch.autograd.grad((expected*cotangent).sum(),parameters,retain_graph=True)
    actual_grads = torch.autograd.grad((actual*cotangent).sum(),parameters)
    for actual_grad, reference_grad in zip(actual_grads,reference_grads):
        torch.testing.assert_close(actual_grad,reference_grad,rtol=2e-5,atol=2e-6)
