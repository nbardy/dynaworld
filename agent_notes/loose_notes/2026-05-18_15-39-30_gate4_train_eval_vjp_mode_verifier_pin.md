# Gate4 train/eval VJP mode verifier pin

After adding the owner-update train/eval autograd mode, the train/eval verifier
still had a narrow coverage gap: it checked the top-level `vjp_mode`, but it
did not reject per-frame rows that drifted to another mode. The tests also did
not explicitly prove that the new owner-update train/eval mode was accepted.

Updated:

```text
research_experiments/world_foam_lane2/verify_gate4_affine_train_eval.py
research_experiments/world_foam_lane2/test_verify_gate4_affine_train_eval.py
```

Verifier contract added:

- every row's `vjp_mode` must match the top-level artifact `vjp_mode`
- verifier output echoes `vjp_mode` and `gradient_scope`

New train/eval verifier tests:

- accepts `direct_atomic_grad_only_ownerupdate`
- rejects a wrong required top-level `vjp_mode`
- rejects row-level `vjp_mode` drift

Regenerated verifier JSONs:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_gate4_affine_train_eval.py \
  --artifact research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_ownerupdate_gradonly_render32_site12_2_4_8_16.json \
  --vjp-mode direct_atomic_grad_only_ownerupdate \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_ownerupdate_gradonly_render32_site12_2_4_8_16_verifier.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_gate4_affine_train_eval.py \
  --artifact research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_gradonly_repeat20_render32_site12_2_4_8_16.json \
  --require-median-timing \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_gradonly_repeat20_render32_site12_2_4_8_16_verifier.json
```

Both verifier artifacts are status `ok`, failures `[]`, and now include the
checked mode/scope:

- normal repeat20: `direct_atomic_grad_only`,
  `frozen_geometry_site_rgba_only_mixed_num32_den16_vjp`
- owner-update: `direct_atomic_grad_only_ownerupdate`,
  `frozen_geometry_site_rgba_only_mixed_num32_den16_vjp_direct_atomic_grad_only_ownerupdate`

Focused and combined gates:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/verify_gate4_affine_train_eval.py \
  research_experiments/world_foam_lane2/test_verify_gate4_affine_train_eval.py

PYTHONPATH=research_experiments/world_foam_lane2:src/train \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_train_eval -v

PYTHONPATH=research_experiments/world_foam_lane2:src/train \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_tape_bridge \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_train_eval -v
```

Results:

- focused train/eval verifier tests: 13 passed
- combined Gate4 suite: 26 passed

Interpretation: the saved owner-update train/eval evidence is now protected
against accidental mode drift. This does not change the runtime conclusion:
owner-update is correctness-positive but not the speed path.
