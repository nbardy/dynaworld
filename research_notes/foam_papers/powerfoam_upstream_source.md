# PowerFoam Upstream Source Pin

Date: 2026-05-03

This is the current upstream reference pin for local PowerFoam parity work.
The upstream repo is not vendored into this tree; this file records the exact
scratch-clone revision used when checking official training and renderer
semantics.

## Upstream

- Remote: `https://github.com/theialab/powerfoam`
- Scratch clone: `/tmp/powerfoam_official`
- Commit: `96392252ebd0059fe6ca98881b62e12295d9242f`
- Subject: `GC to clear pytorch cache`
- Author: `Shrisudhan <shrisudhan07@gmail.com>`
- Author date: `Fri May 1 23:49:01 2026 -0700`
- Committer: `Shrisudhan <shrisudhan07@gmail.com>`
- Commit date: `Fri May 1 23:49:01 2026 -0700`

Verification commands:

```bash
git -C /tmp/powerfoam_official rev-parse HEAD
git -C /tmp/powerfoam_official log -1 --format=fuller
git -C /tmp/powerfoam_official remote -v
git -C /tmp/powerfoam_official status --short
```

The scratch clone was clean when inspected. A prior scan note from
2026-04-30 recorded commit `25d6f7b`; treat that as historical context only,
not the current source pin.

## Local Uses

- Official LR schedule behavior was checked against
  `/tmp/powerfoam_official/powerfoam/scheduling.py`.
- Official parameter group names, cosine schedule calls, and warmup choices
  were checked against `/tmp/powerfoam_official/powerfoam/scene.py`.
- Official regularizer loss schedule shape was checked against
  `/tmp/powerfoam_official/train.py`.

When adding parity fixtures, include this commit hash in the fixture metadata
so the expected outputs can be traced back to a real upstream state.

## Runtime Parity Limitation On This Mac

The current local machine cannot execute an official PowerFoam runtime fixture
directly:

- `.venv` does not currently include `warp`.
- `torch.cuda.is_available()` is `False`.
- The upstream renderer/trainer paths call CUDA-specific APIs such as
  `torch.cuda.current_stream()`, `torch.cuda.synchronize()`, and
  `device="cuda"` initialization in `train.py`, `test.py`,
  `powerfoam/rasterize.py`, `powerfoam/raytrace.py`,
  `powerfoam/color_fn.py`, and `powerfoam/bvh.py`.

The local fixture
`research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_origin_parity_v1.json`
therefore remains a local Torch-reference/Metal parity fixture, not an
official CUDA/Warp output fixture. Closing that P0 item requires running the
same tiny scene on a CUDA host with `warp-lang` and the upstream checkout at
commit `96392252ebd0059fe6ca98881b62e12295d9242f`.
