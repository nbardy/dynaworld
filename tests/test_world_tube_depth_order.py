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
    ProjectiveTraceCellTraceAtlas,
    ProjectiveTraceTileTimeCell,
    mark_projective_trace_cell_visibility_fallbacks,
    pack_projective_trace_tile_time_bins,
    projective_trace_cell_atlas_fallback_tile_sample_mask,
    render_projective_trace_cell_atlas_reference,
    slice_projective_trace_cell_atlas_frames,
    uvt_tubes_to_projective_trace_cell_atlas,
)


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason='local Metal required'))])
@pytest.mark.parametrize('origin', [0., 8192.])
@pytest.mark.parametrize('fallback', [False, True])
def test_centered_temporal_envelope_keeps_value_and_adjoint_under_time_translation(device, origin, fallback):
    # A change of time origin must not destroy the temporal Gaussian or VJP.
    # All offsets are exactly representable at either origin in float32.
    center = torch.tensor(origin+.125,device=device,requires_grad=True)
    precision = torch.tensor(.5,device=device,requires_grad=True)
    ma = torch.stack((center.new_tensor(1.5),center.new_tensor(1.5),center)).reshape(1,3)
    q = torch.stack((precision.new_tensor(.75),precision.new_tensor(0.),precision.new_tensor(0.),precision.new_tensor(1.25),precision.new_tensor(0.),precision)).reshape(1,6)
    opacity = torch.tensor([.7],device=device,requires_grad=True)
    color = torch.tensor([[.8,.3,.2]],device=device,requires_grad=True)
    inputs = dict(ma=ma,q_uvt=q,depth0=torch.ones(1,device=device),depth_beta=torch.zeros((1,3),device=device),opacity=opacity,color=color)
    times = torch.tensor([origin-.5,origin,origin+.5],device=device)
    atlas = uvt_tubes_to_projective_trace_cell_atlas(**inputs,times=times,sigma_px=1.,image_width=3,image_height=3,tile_size=8,alpha_threshold=1/255,require_isotropic_spatial=False,auto_support_padding_from_alpha=True,allow_depth_affine_uv=True,temporal_mode='centered')
    # Cached live updates and frame slicing must carry the encoding as well.
    from star_uvt_projective_interval_backend import make_projective_cell_interval_live_atlas_from_uvt_tubes
    cfg = {'data':{'max_frames':3,'target_size':3},'feature_uvt':{'feature_dim':3,'tile_t':1,'tile_capacity':128,'alpha_threshold':1/255,'max_alpha':1.,'projective_interval':{'enabled':True,'sigma_px':1.,'tile_size':8,'allow_anisotropic_spatial_precision':True}}}
    atlas = make_projective_cell_interval_live_atlas_from_uvt_tubes(**inputs,cfg=cfg,reference_atlas=atlas)
    assert atlas.opacity_time_centered
    if fallback:
        atlas = replace(atlas,cells=[replace(cell,fallback=True,fallback_reasons=('test',)) for cell in atlas.cells])
    actual = []
    for frame in range(len(times)):
        chunk = slice_projective_trace_cell_atlas_frames(atlas,start=frame,stop=frame+1)
        if device == 'mps':
            from research_project.trainer_harness.tile_metal_autograd import ProjectiveCellIntervalTrainerState
            state = ProjectiveCellIntervalTrainerState(atlas=chunk,times=times[frame:frame+1],config=UVTRenderConfig(height=3,width=3,frames=1),sigma_px=1.,image_width=3,image_height=3,tile_size=8,fallback_render_mode='mixed')
            actual.append(state.render())
        else:
            actual.append(render_projective_trace_cell_atlas_reference(chunk,times[frame:frame+1],image_width=3,image_height=3,tile_size=8,sigma_px=1.,alpha_cutoff=1/255,allow_fallback_cells=True))
    actual = torch.cat(actual)
    yy,xx = torch.meshgrid(torch.arange(3,device=device)+.5,torch.arange(3,device=device)+.5,indexing='ij')
    qv = .75*(xx-1.5).square()+1.25*(yy-1.5).square()+precision*(times[:,None,None]-center).square()
    expected = opacity*torch.exp(-.5*qv)[...,None]*color
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    cotangent = torch.linspace(-.8,1.2,actual.numel(),device=device).reshape_as(actual)
    parameters = (center,precision,opacity,color)
    reference_grads = torch.autograd.grad((expected*cotangent).sum(),parameters,retain_graph=True)
    actual_grads = torch.autograd.grad((actual*cotangent).sum(),parameters)
    for value,reference in zip(actual_grads,reference_grads):
        torch.testing.assert_close(value,reference,rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason='local Metal required'))])
def test_mixed_size_tubes_preserve_images_and_gradients_without_false_tile_overflow(device):
    # One broad tube must not give forty small tubes its support footprint.
    # An inactive first row also exercises source-row mapping after filtering.
    centers = [[4.+8*x,4.+8*y,0.] for y in range(5) for x in range(8)]
    ma = torch.tensor([[32.,20.,0.],*centers,[32.,20.,0.]],device=device)
    count = len(ma)
    precision = torch.tensor([[.8,.1,4.]]*count,device=device)
    precision[[0,-1]] = torch.tensor([.004,0.,.03],device=device)
    velocity = torch.stack((torch.arange(count,device=device)%2*1.8-.9,torch.full((count,),.1,device=device)),dim=1)
    cross = -torch.stack((precision[:,0]*velocity[:,0]+precision[:,1]*velocity[:,1],precision[:,1]*velocity[:,0]+precision[:,2]*velocity[:,1]),dim=1)
    q = torch.stack((precision[:,0],precision[:,1],cross[:,0],precision[:,2],cross[:,1],.025-(cross*velocity).sum(1)),dim=1)
    opacity = torch.full((count,),.45,device=device)
    opacity[0] = .001
    inputs = dict(ma=ma.requires_grad_(),q_uvt=q.requires_grad_(),depth0=1+torch.arange(count,device=device)*.1,depth_beta=torch.zeros((count,3),device=device),opacity=opacity.requires_grad_(),color=torch.linspace(.1,.9,count*3,device=device).reshape(count,3).requires_grad_())
    frames = [0,3,5,6]
    times = torch.tensor(frames,device=device,dtype=torch.float32)-3
    config = UVTRenderConfig(height=40,width=64,frames=7,transmittance_threshold=0.)
    atlas = uvt_tubes_to_projective_trace_cell_atlas(**inputs,times=times,sigma_px=1.,image_width=config.width,image_height=config.height,tile_size=config.tile_x,alpha_threshold=config.alpha_threshold,require_isotropic_spatial=False,auto_support_padding_from_alpha=True,allow_depth_affine_uv=True,mark_visibility_fallback=True)
    assert 0 not in atlas.source_primitive_ids
    bins = pack_projective_trace_tile_time_bins(atlas.cells,image_width=config.width,image_height=config.height,frames=len(times),tile_x=config.tile_x,tile_y=config.tile_y,tile_t=len(times),tile_capacity=32,allow_fallback_cells=True)
    assert not bins.tile_overflow.any(), 'a distant broad tube inflated unrelated tile lists'
    from star_uvt_feature_tube_model import FeatureTubeRenderConfig, dense_render_feature_tubes
    expected, _alpha = dense_render_feature_tubes(**{('feature' if k=='color' else k):v for k,v in inputs.items()},config=FeatureTubeRenderConfig(frames=config.frames,height=config.height,width=config.width,feature_dim=3,alpha_threshold=config.alpha_threshold,max_alpha=1.))
    expected = expected.permute(0,2,3,1)[frames]
    if device == 'mps':
        from research_project.trainer_harness.tile_metal_autograd import ProjectiveCellIntervalTrainerState
        state = ProjectiveCellIntervalTrainerState(atlas=atlas,times=times,config=replace(config,frames=len(times)),sigma_px=1.,image_width=config.width,image_height=config.height,tile_size=config.tile_x,fallback_render_mode='mixed')
        actual = state.render()
    else:
        actual = render_projective_trace_cell_atlas_reference(atlas,times,image_width=config.width,image_height=config.height,tile_size=config.tile_x,sigma_px=1.,alpha_cutoff=config.alpha_threshold,transmittance_cutoff=0.,allow_fallback_cells=True)
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
    cotangent = torch.linspace(-.8,1.2,actual.numel(),device=device).reshape_as(actual)/actual.numel()
    parameters = tuple(inputs[k] for k in ['ma','q_uvt','opacity','color'])
    expected_grads = torch.autograd.grad((expected*cotangent).sum(),parameters,retain_graph=True)
    actual_grads = torch.autograd.grad((actual*cotangent).sum(),parameters)
    for actual_grad,expected_grad in zip(actual_grads,expected_grads):
        torch.testing.assert_close(actual_grad,expected_grad,rtol=2e-5,atol=2e-6)


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason='local Metal required'))])
@pytest.mark.parametrize('case', ['temporal_anisotropic','opaque','tile_stop','partial_stop'])
def test_batched_fallback_matches_scalar_compositing_and_gradients(device,case):
    count = 5
    coeffs = torch.zeros((count,9),device=device)
    coeffs[:,0] = 1.5
    coeffs[:,3] = 1.5
    coeffs[:,6] = torch.arange(count,device=device)+1
    precision = torch.tensor([.8,.15,.6],device=device).repeat(count,1)
    opacity = torch.tensor([0.,.003,.7,.999,1.1] if case=='temporal_anisotropic' else [1.1 if case=='opaque' else .8]*count,device=device)
    if case in {'opaque','tile_stop'}:
        precision[:] = torch.tensor([1e-4,0,1e-4],device=device)
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=coeffs.requires_grad_(),opacity=opacity.requires_grad_(),
        color=torch.linspace(.1,.9,count*3,device=device).reshape(count,3).requires_grad_(),
        spatial_precision_uv=precision.requires_grad_(),
        opacity_time_coeffs=torch.tensor([.1,-.05,.02] if case=='temporal_anisotropic' else [0.,0.,0.],device=device).repeat(count,1).requires_grad_(),
        cells=[ProjectiveTraceTileTimeCell(tile_u=0,tile_v=0,start=0,stop=2,primitive_ids=tuple(range(count)),ordered_primitive_ids=tuple(range(count)),depth_intervals=tuple((i+1.,i+1.) for i in range(count)),fallback=True,fallback_reasons=('test_order',))],
        source_window_indices=(0,)*count,source_primitive_ids=tuple(range(count)),active_start=(0,)*count,active_stop=(2,)*count,
    )
    times = torch.arange(2,device=device,dtype=torch.float32)
    args = dict(image_width=5,image_height=3,tile_size=8,sigma_px=1.,allow_fallback_cells=True,alpha_cutoff=1/255,transmittance_cutoff=.01 if case=='tile_stop' else .1)
    expected = render_projective_trace_cell_atlas_reference(atlas,times,**args)
    actual = render_projective_trace_cell_atlas_reference(atlas,times,**args,fallback_tiles_only=True)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    cotangent = torch.linspace(-.8,1.2,actual.numel(),device=device).reshape_as(actual)
    parameters = tuple(getattr(atlas,k) for k in ['coeffs','opacity','color','spatial_precision_uv','opacity_time_coeffs'])
    expected_grads = torch.autograd.grad((expected*cotangent).sum(),parameters,retain_graph=True)
    actual_grads = torch.autograd.grad((actual*cotangent).sum(),parameters)
    for actual_grad,expected_grad in zip(actual_grads,expected_grads):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad,expected_grad,rtol=2e-5,atol=2e-6)


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason='local Metal required'))])
def test_sparse_fallback_keeps_all_tile_contributors_and_their_gradients(device):
    # Only the red cell at time 1 is flagged. Green still contributes behind it;
    # blue belongs to a different tile and must stay on the native route.
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor([[4.,0,0,4,0,0,1,0,0],[4.,0,0,4,0,0,2,0,0],[12.,0,0,4,0,0,3,0,0]],device=device,requires_grad=True),
        opacity=torch.tensor([.5,.4,.6],device=device,requires_grad=True),
        color=torch.eye(3,device=device,requires_grad=True),
        cells=[ProjectiveTraceTileTimeCell(tile_u=tile,tile_v=0,start=start,stop=stop,primitive_ids=(trace,),ordered_primitive_ids=(trace,),depth_intervals=((trace+1.,trace+1.),),fallback=fallback,fallback_reasons=('depth_event',) if fallback else ()) for trace,tile,start,stop,fallback in [(0,0,0,1,False),(0,0,1,2,True),(1,0,0,2,False),(2,1,0,2,False)]],
        source_window_indices=(0,0,0),source_primitive_ids=(0,1,2),active_start=(0,0,0),active_stop=(2,2,2),
    )
    times = torch.arange(2,device=device,dtype=torch.float32)
    render_args = dict(image_width=16,image_height=8,tile_size=8,sigma_px=1.,allow_fallback_cells=True,alpha_cutoff=1/255,transmittance_cutoff=1e-4)
    full = render_projective_trace_cell_atlas_reference(atlas,times,**render_args)
    sparse = render_projective_trace_cell_atlas_reference(atlas,times,**render_args,fallback_tiles_only=True)
    mask = torch.zeros_like(full,dtype=torch.bool)
    mask[1,:,:8] = True
    expected = torch.where(mask,full,torch.zeros_like(full))
    torch.testing.assert_close(sparse,expected,rtol=0,atol=0)
    assert sparse[1,:,:8,1].sum() > 0
    parameters = (atlas.coeffs,atlas.opacity,atlas.color)
    cotangent = torch.linspace(-1,1,full.numel(),device=device).reshape_as(full)
    expected_grads = torch.autograd.grad((expected*cotangent).sum(),parameters,retain_graph=True)
    actual_grads = torch.autograd.grad((sparse*cotangent).sum(),parameters,retain_graph=True)
    for actual, reference in zip(actual_grads,expected_grads):
        torch.testing.assert_close(actual,reference,rtol=1e-6,atol=1e-7)
    assert actual_grads[-1][1].abs().sum() > 0
    assert not actual_grads[-1][2].any()
    if device == 'mps':
        from research_project.trainer_harness.tile_metal_autograd import ProjectiveCellIntervalTrainerState
        state = ProjectiveCellIntervalTrainerState(atlas=atlas,times=times,config=UVTRenderConfig(height=8,width=16,frames=2),sigma_px=1.,image_width=16,image_height=8,tile_size=8,fallback_render_mode='mixed')
        mixed = state.render()
        torch.testing.assert_close(mixed,full,rtol=1e-5,atol=1e-6)
        expected_grads = torch.autograd.grad((full*cotangent).sum(),parameters,retain_graph=True)
        actual_grads = torch.autograd.grad((mixed*cotangent).sum(),parameters)
        for actual, reference in zip(actual_grads,expected_grads):
            torch.testing.assert_close(actual,reference,rtol=2e-5,atol=2e-6)


def _segmented_atlas(device='cpu'):
    # Trace 0 is continuous; trace 1 disappears for two samples. Duplicate
    # support must neither consume extra capacity nor fill that real gap.
    intervals = [(0,0,2),(0,2,4),(0,1,3),(1,0,1),(1,3,4)]
    return ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor([[.5,.02,0,.5,0,0,1,0,0],[.7,0,0,.5,.01,0,2,0,0]],device=device,requires_grad=True),
        opacity=torch.tensor([.5,.4],device=device,requires_grad=True),
        color=torch.tensor([[.9,.1,.2],[.1,.3,.8]],device=device,requires_grad=True),
        cells=[ProjectiveTraceTileTimeCell(tile_u=0,tile_v=0,start=start,stop=stop,primitive_ids=(i,),ordered_primitive_ids=(i,),depth_intervals=((i+1.,i+1.),),fallback=False,fallback_reasons=()) for i,start,stop in intervals],
        source_window_indices=(0,0),source_primitive_ids=(0,1),active_start=(0,0),active_stop=(4,4),
    )


@pytest.mark.parametrize('centered', [False, True])
def test_retained_atlas_preserves_temporal_envelope_interpretation(tmp_path, centered):
    # Identical coefficient bytes have different meanings in the two encodings.
    # Losing the flag would silently change a restored atlas's opacity over time.
    from research_project.benchmarks.multicam_heldout_compare import _write_frozen_atlas_storage
    from research_experiments.paper_runner_suite.frozen_atlas_storage import verify_retained_storage_artifact

    atlas = replace(_segmented_atlas(), opacity_time_centered=centered,
                    opacity_time_coeffs=torch.tensor([[0., .125, .5]]).repeat(2, 1))
    identity = _write_frozen_atlas_storage(atlas, out_dir=tmp_path, frame_count=4)
    header = verify_retained_storage_artifact(identity, expected_frame_count=4,
                                             expected_trace_count=2, expected_cell_count=len(atlas.cells))
    assert header['topology'].get('opacity_time_centered', False) is centered


@pytest.mark.parametrize('tile_t', [1,2,4])
def test_interval_union_preserves_sample_membership_without_false_overflow(tile_t):
    bins = pack_projective_trace_tile_time_bins(
        _segmented_atlas().cells,image_width=2,image_height=2,frames=4,
        tile_x=8,tile_y=8,tile_t=tile_t,tile_capacity=3,
    )
    assert not bins.tile_overflow.any()
    for frame, expected in enumerate([{0,1},{0},{0},{0,1}]):
        tile = frame//tile_t
        actual = []
        for slot in range(int(bins.tile_counts[tile])):
            offset = tile*3+slot
            if bins.tile_active_start[offset] <= frame < bins.tile_active_stop[offset]:
                actual.append(int(bins.tile_primitive_ids[offset]))
        assert len(actual) == len(expected)
        assert set(actual) == expected


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason='local Metal required'))])
@pytest.mark.parametrize('capacity', [2, 3, 5])
@pytest.mark.parametrize('empty', [False, True])
def test_interval_buffers_preserve_overflow_counts_gaps_and_empty_tile_padding(device, capacity, empty):
    # Native consumers require the true count even on overflow, and negative
    # ids/zero bounds in padding. A real temporal gap must survive packing.
    bins = pack_projective_trace_tile_time_bins(
        [] if empty else _segmented_atlas().cells,
        image_width=16, image_height=8, frames=4, tile_x=8, tile_y=8,
        tile_t=4, tile_capacity=capacity, device=device,
    )
    count = 0 if empty else 3
    used = min(count, capacity)
    expected = {
        'tile_counts': [count, 0],
        'tile_overflow': [int(count > capacity), 0],
        'tile_primitive_ids': [0, 1, 1][:used] + [-1] * (2 * capacity - used),
        'tile_active_start': [0, 0, 3][:used] + [0] * (2 * capacity - used),
        'tile_active_stop': [4, 1, 4][:used] + [0] * (2 * capacity - used),
    }
    for name, values in expected.items():
        actual = getattr(bins, name)
        assert actual.is_contiguous() and actual.dtype == torch.int32
        torch.testing.assert_close(actual, torch.tensor(values, dtype=torch.int32, device=device), rtol=0, atol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='local Metal required')
def test_metal_interval_union_preserves_rgb_and_vjp_with_real_gaps():
    from research_project.trainer_harness.tile_metal_autograd import render_projective_cell_interval_atlas_metal_backward
    atlas = _segmented_atlas('mps')
    times = torch.arange(4,device='mps',dtype=torch.float32)
    expected = _render(atlas,times)
    actual = render_projective_cell_interval_atlas_metal_backward(atlas,times,UVTRenderConfig(height=2,width=2,frames=4),sigma_px=1.0)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    cotangent = torch.linspace(-1,1,actual.numel(),device='mps').reshape_as(actual)
    parameters = (atlas.coeffs,atlas.opacity,atlas.color)
    expected_grads = torch.autograd.grad((expected*cotangent).sum(),parameters,retain_graph=True)
    actual_grads = torch.autograd.grad((actual*cotangent).sum(),parameters)
    for actual_grad, expected_grad in zip(actual_grads,expected_grads):
        torch.testing.assert_close(actual_grad,expected_grad,rtol=2e-5,atol=2e-6)


@pytest.mark.parametrize('explicit_zero_depth', [False, True])
@pytest.mark.parametrize('inherited_fallback', [False, True])
def test_single_ambiguous_time_does_not_mark_the_whole_cell(inherited_fallback, explicit_zero_depth):
    # Depths touch the ambiguity band only at t=1; the entire four-sample
    # interval has one valid spatial support and one unchanged physical order.
    times = torch.arange(4, dtype=torch.float32)
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor([[.5,0,0,.5,0,0,1,0,0],[.5,0,0,.5,0,0,1.0200001,-.04,.02]],dtype=torch.float32),
        opacity=torch.tensor([.5,.5]),color=torch.tensor([[1.,0,0],[0.,0,1.]]),
        cells=[ProjectiveTraceTileTimeCell(tile_u=0,tile_v=0,start=0,stop=4,primitive_ids=(0,1),ordered_primitive_ids=(0,1),depth_intervals=((1.,1.),(1.,1.09)),fallback=inherited_fallback,fallback_reasons=('unresolved_projection',) if inherited_fallback else ())],
        source_window_indices=(0,0),source_primitive_ids=(0,1),active_start=(0,0),active_stop=(4,4),
        depth_affine_uv=torch.zeros((2,6)) if explicit_zero_depth else None,
    )
    marked = mark_projective_trace_cell_visibility_fallbacks(atlas,times,depth_epsilon=1e-6,image_width=2,image_height=2,tile_size=8)
    mask = projective_trace_cell_atlas_fallback_tile_sample_mask(marked,frames=4,image_width=2,image_height=2,tile_size=8)
    assert mask[:,0,0].tolist() == ([True]*4 if inherited_fallback else [False,True,False,False])
    if inherited_fallback:
        assert all('unresolved_projection' in cell.fallback_reasons for cell in marked.cells)
    # Changing fallback segmentation must preserve every pixel contribution.
    torch.testing.assert_close(_render(marked,times),_render(atlas,times),rtol=0,atol=0)


@pytest.mark.parametrize('slope_kind,expected', [('zero',[False,False]),('constant',[True,True]),('time',[False,True])])
def test_spatial_depth_crossing_is_checked_even_if_initial_slope_is_zero(slope_kind,expected):
    atlas = _segmented_atlas()
    coefficients = torch.tensor([[4.,0,0,4.,0,0,1.,0,0],[4.,0,0,4.,0,0,2.,0,0]])
    slopes = torch.zeros((2,6))
    if slope_kind != 'zero':
        slopes[1,0 if slope_kind=='constant' else 1] = .5
    cell = ProjectiveTraceTileTimeCell(tile_u=0,tile_v=0,start=0,stop=2,primitive_ids=(0,1),ordered_primitive_ids=(0,1),depth_intervals=((1.,1.),(2.,2.)),fallback=False,fallback_reasons=())
    atlas = replace(atlas,coeffs=coefficients,depth_affine_uv=slopes,cells=[cell],active_stop=(2,2))
    times = torch.arange(2,dtype=torch.float32)
    marked = mark_projective_trace_cell_visibility_fallbacks(atlas,times,image_width=8,image_height=8,tile_size=8)
    mask = projective_trace_cell_atlas_fallback_tile_sample_mask(marked,frames=2,image_width=8,image_height=8,tile_size=8)
    assert mask[:,0,0].tolist() == expected


@pytest.mark.parametrize('tile_u', [-1,1])
def test_zero_spatial_depth_still_rejects_tiles_outside_the_image(tile_u):
    atlas = _segmented_atlas()
    atlas = replace(atlas,depth_affine_uv=torch.zeros((2,6)),cells=[replace(cell,tile_u=tile_u) for cell in atlas.cells])
    with pytest.raises(ValueError,match='tile coordinates'):
        mark_projective_trace_cell_visibility_fallbacks(atlas,torch.arange(4,dtype=torch.float32),image_width=8,image_height=8,tile_size=8)


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
        **{name:getattr(atlas,name).index_select(0,index) for name in ['coeffs','opacity','color','opacity_time_coeffs','spatial_precision_uv','depth_affine_uv','depth_reference_uvt','alpha_cutoff_reference_uvt']},
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


@pytest.mark.parametrize('field,width', [('depth_reference_uvt',7), ('alpha_cutoff_reference_uvt',9)])
def test_retained_artifact_counts_source_depth_state(tmp_path,field,width):
    from research_project.benchmarks.multicam_heldout_compare import _write_frozen_atlas_storage
    atlas = _compile(_fixture(), torch.arange(4, dtype=torch.float32)-1.5)
    with_source = _write_frozen_atlas_storage(atlas, out_dir=tmp_path/'with', frame_count=4)
    without_source = _write_frozen_atlas_storage(replace(atlas,**{field:None}), out_dir=tmp_path/'without', frame_count=4)
    assert with_source['tensor_payload_bytes'] - without_source['tensor_payload_bytes'] == 2*width*4
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
