"""Print a float64 fixture from the canonical SPD4 compiler (CPU only).

Run from the dynaworld root; stdout is the retained JSON fixture. The browser
implementation does not import or generate this independent PyTorch reference.
"""
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from research_experiments.spd4_world_tubes.compiler import pushforward_world_atoms
from research_experiments.spd4_world_tubes.model import AffineRayGauge, WorldAtomBatch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
width, height = 24, 18
aspect, filtering = width / height, (0.3 / height) ** 2
rows = []
for i in range(2):
    rows.append([
        -0.07 + i * 0.13, 0.02 - i * 0.06, 1.4 + i * 0.13,
        math.log(0.45 + i * 0.1), 0.04 - i * 0.07, -0.02 + i * 0.05, 0.08 - i * 0.14, 0.43 + i * 0.13,
        0, 0, 0, 0, math.log(0.13), math.log(0.08), math.log(0.2), 0,
        0.15, -0.2 + i * 0.1, 0.07, 0.97, 0.7 - i * 0.5, 0.2 + i * 0.5, 0.4, 0.4,
    ])
p = torch.tensor(rows, requires_grad=True)
cameras = []
for i in range(2):
    a = i * 0.17
    cameras.append({"name": f"cam{i}", "role": "train", "worldToCamera": [
        math.cos(a), 0, math.sin(a), -i * 0.1, 0, 1, 0, i * 0.02,
        -math.sin(a), 0, math.cos(a), i * 0.03, 0, 0, 0, 1], "intrinsics": [0.8, 0.9, 0.5, 0.5]})


def compile_atoms(params, camera):
    packed = []
    camera_matrix = torch.tensor(camera["worldToCamera"]).reshape(4, 4)
    rc = camera_matrix[:3, :3]
    fx, fy, cx, cy = camera["intrinsics"]
    for atom in params:
        mean = atom[:3] + atom[4:7] * (2 * atom[7] - 1)
        cp = rc @ mean + camera_matrix[:3, 3]
        quat = atom[16:20] / (atom[16:20].square().sum() + 1e-16).sqrt()
        x, y, z, w = quat
        rotation = torch.stack([
            torch.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)]),
            torch.stack([2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)]),
            torch.stack([2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]),
        ])
        spatial = rotation @ torch.diag(torch.exp(2 * atom[12:15])) @ rotation.T
        tilt, variance = 2 * atom[4:7], torch.exp(2 * atom[3])
        covariance = torch.cat([
            torch.cat([spatial + variance * tilt[:, None] * tilt[None, :], (variance * tilt)[:, None]], dim=1),
            torch.cat([variance * tilt, variance[None]])[None, :],
        ])
        zero = cp[0] * 0
        j = torch.stack([
            torch.stack([aspect*fx/cp[2], zero, -aspect*fx*cp[0]/cp[2]**2]),
            torch.stack([zero, fy/cp[2], -fy*cp[1]/cp[2]**2]),
        ])
        matrix = torch.eye(4)
        matrix[:2, :3] = j @ rc
        matrix[2, :3] = rc[2]
        origin = torch.cat([mean, atom[7:8]])
        projected_mean = torch.stack([aspect*(fx*cp[0]/cp[2]+cx), fy*cp[1]/cp[2]+cy, cp[2], atom[7]])
        gauge = AffineRayGauge(matrix, projected_mean - matrix @ origin, torch.tensor(1.0))
        # Independent pixel measurement noise, expressed in source coordinates
        # so the canonical compiler can retain its unmodified API.
        inverse = torch.linalg.inv(matrix)
        covariance = covariance + inverse @ torch.diag(torch.tensor([filtering, filtering, 0, 0])) @ inverse.T
        source = WorldAtomBatch(origin[None], covariance[None], atom[23].sigmoid()[None], atom[20:23][None])
        trace = pushforward_world_atoms(source, gauge)
        packed.append(torch.cat([trace.ma[0], torch.ones(1), trace.q_uvt[0], trace.depth_variance,
                                 torch.zeros(1), trace.depth0, trace.depth_beta[0], atom[20:23], atom[23].sigmoid()[None]]))
    return torch.stack(packed)


traces = torch.stack([compile_atoms(p, camera) for camera in cameras])
samples, predictions = [], []
for view in range(2):
    for t in [0.0, 0.13, 0.61, 1.0]:
        for y, x in [(8, 11), (9, 12), (7, 10), (10, 13)]:
            a = torch.tensor([(x + 0.5) / height, (y + 0.5) / height, t])
            candidates = []
            for index, trace in enumerate(traces[view]):
                delta = a - trace[:3]
                q = trace[4:10]
                quad = q[0]*delta[0]**2 + 2*q[1]*delta[0]*delta[1] + 2*q[2]*delta[0]*delta[2] + q[3]*delta[1]**2 + 2*q[4]*delta[1]*delta[2] + q[5]*delta[2]**2
                raw = trace[19] * torch.exp(-0.5 * quad)
                depth = trace[12] + trace[13:16] @ delta
                if raw.detach().item() >= 1/255 and depth.detach().item() > 0.1:
                    candidates.append((depth.detach().item(), index, raw.clamp(max=0.99)))
            rgb, trans = torch.zeros(3), torch.ones(())
            for _, index, alpha in sorted(candidates):
                rgb = rgb + trans * alpha * traces[view, index, 16:19]
                trans = trans * (1-alpha)
            predictions.append(rgb)
            samples.append([*a.tolist(), view, 0.35, 0.25, 0.45, 0])
loss = ((torch.stack(predictions) - torch.tensor(samples)[:, 4:7])**2).mean()
gradient = torch.autograd.grad(loss, p)[0]
print(json.dumps({"provenance": "canonical float64 pushforward_world_atoms + torch.autograd, CPU",
                  "width": width, "height": height, "params": p.detach().flatten().tolist(), "cameras": cameras,
                  "traces": traces.detach().flatten().tolist(), "samples": samples,
                  "rgb": torch.stack(predictions).detach().flatten().tolist(), "loss": loss.item(),
                  "gradients": gradient.flatten().tolist()}, indent=2))
