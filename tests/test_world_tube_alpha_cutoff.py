"""Preserve source alpha membership when completing the spacetime square."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

STAR = Path(__file__).resolve().parents[1] / 'third_party/fast-mac-gsplat/variants/star_uvt_v0'
sys.path.insert(0, str(STAR))
from torch_gsplat_bridge_star_uvt import UVTRenderConfig
from torch_gsplat_bridge_star_uvt.projective_trace import (
    render_projective_trace_cell_atlas_reference,
    slice_projective_trace_cell_atlas_frames,
    uvt_tubes_to_projective_trace_cell_atlas,
)
from torch_gsplat_bridge_star_uvt.rasterize import _quadratic


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason='local Metal required'))])
@pytest.mark.parametrize('opacity_delta', [-4e-6, 0., 4e-6])
@pytest.mark.parametrize('fallback', [False, True])
def test_uvt_cutoff_preserves_pixel_and_vjp_through_live_update_and_slice(device, opacity_delta, fallback):
    # Retained F32 counterexample, trace 771 at frame 1 / pixel (44,43).
    # Completing the square puts alpha on the other side of 1/255 in float32.
    inputs = dict(
        ma=torch.tensor([[30.740467071533203, 4.4010467529296875, -12.813916206359863]], device=device, requires_grad=True),
        q_uvt=torch.tensor([[.007928503677248955, -.001524341874755919, -.004785473458468914, .007458716630935669, .022616731002926826, .2930063307285309]], device=device, requires_grad=True),
        depth0=torch.tensor([6.6099324226379395], device=device),
        depth_beta=torch.tensor([[0., 0., -.14591820538043976]], device=device),
        opacity=torch.tensor([.4176533818244934*(1+opacity_delta)], device=device, requires_grad=True),
        color=torch.tensor([[.41100722551345825, .39113593101501465, .3601039946079254]], device=device, requires_grad=True),
    )
    times=torch.tensor([-15.5,-14.5],device=device)
    atlas=uvt_tubes_to_projective_trace_cell_atlas(**{**inputs, 'q_uvt':inputs['q_uvt']*1.0001},times=times,sigma_px=1.,image_width=48,image_height=48,tile_size=8,alpha_threshold=1/255,require_isotropic_spatial=False,auto_support_padding_from_alpha=True,allow_depth_affine_uv=True,temporal_mode='centered')
    from star_uvt_projective_interval_backend import make_projective_cell_interval_live_atlas_from_uvt_tubes
    cfg={'data':{'max_frames':2,'target_size':48},'feature_uvt':{'feature_dim':3,'tile_t':1,'tile_capacity':128,'alpha_threshold':1/255,'max_alpha':1.,'projective_interval':{'enabled':True,'sigma_px':1.,'tile_size':8,'allow_anisotropic_spatial_precision':True}}}
    atlas=make_projective_cell_interval_live_atlas_from_uvt_tubes(**inputs,cfg=cfg,reference_atlas=atlas)
    if fallback:
        atlas=replace(atlas,cells=[replace(cell,fallback=True,fallback_reasons=('test',)) for cell in atlas.cells])
    chunk=slice_projective_trace_cell_atlas_frames(atlas,start=1,stop=2)
    if device=='mps':
        from research_project.trainer_harness.tile_metal_autograd import ProjectiveCellIntervalTrainerState
        state=ProjectiveCellIntervalTrainerState(atlas=chunk,times=times[1:],config=UVTRenderConfig(height=48,width=48,frames=1),sigma_px=1.,image_width=48,image_height=48,tile_size=8,fallback_render_mode='mixed')
        actual=state.render()[0,43,44]
        if not fallback:
            from torch_gsplat_bridge_star_uvt.projective_trace import render_projective_trace_cell_interval_atlas_rows_metal
            row_image=render_projective_trace_cell_interval_atlas_rows_metal(chunk,times[1:],torch.ones((1,48),device=device),state.config,sigma_px=1.)
            torch.testing.assert_close(row_image[43,44],actual,rtol=0,atol=0)
    else:
        actual=render_projective_trace_cell_atlas_reference(chunk,times[1:],image_width=48,image_height=48,tile_size=8,sigma_px=1.,alpha_cutoff=1/255,allow_fallback_cells=True)[0,43,44]
    alpha=inputs['opacity']*torch.exp(-.5*_quadratic(inputs['q_uvt'],inputs['ma'].new_tensor([44.5,43.5,-14.5])-inputs['ma']))
    expected=(torch.where(alpha>=1/255,alpha,torch.zeros_like(alpha))[:,None]*inputs['color'])[0]
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-8)
    parameters=tuple(inputs[k] for k in ['ma','q_uvt','opacity','color'])
    expected_grads=torch.autograd.grad(expected.sum(),parameters,retain_graph=True)
    actual_grads=torch.autograd.grad(actual.sum(),parameters)
    for value,reference in zip(actual_grads,expected_grads):
        torch.testing.assert_close(value,reference,rtol=2e-5,atol=2e-7)
