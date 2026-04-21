# Bounded random init follow-up

The 8192-splat Taichi overfit run looked too smooth and did not sharpen as much
as expected after increasing splat count. One plausible cause is that the
previous "small init" was too clustered:

- `token_init_std=0.01`
- `head_output_init_std=0.002`
- `rotation_init=identity`
- fixed opacity bias
- final xyz bias sampled per sibling slot, not per token

Because the xyz final bias has shape `[gaussians_per_token, 3]`, it creates
only 64 anchor patterns, which are then reused across all 128 tokens. With tiny
token/head randomness, the initial decoded splats are much closer to 64 repeated
groups than to 8192 independent means. AdamW weight decay then also pulls token
and head variation back toward those shared biases.

Change made:

- Add `position_init_extent_coverage` for the xyz
  bias initialization. For xy, sample desired normalized positions uniformly and
  apply `atanh` so `tanh(raw)` starts uniform in decoded space. For z, sample a
  uniform normalized depth interval and apply `logit` so `sigmoid(raw)` starts
  uniform in decoded depth. Older configs using `position_init_raw_jitter` are
  migrated once during config normalization.
- Set the active 8192 config to `token_init_std=0.3`,
  `head_output_init_std=0.06`, and `rotation_init=random`. A quick init-stat
  probe showed that `0.1/0.03` fixed xyz spread but left RGB almost constant
  (`rgb std ~= 0.002`), while `0.3/0.06` preserved xyz spread and gave modest
  RGB/opacity diversity.
- Turn off weight decay for this local overfit probe. Weight decay is useful for
  generalization experiments, but here it regularizes against the diversity we
  are trying to test.

I did not add a learnable tanh slope yet. A learned bound temperature is a real
architecture knob, but it also becomes a global scene-scale knob that can fight
camera/depth normalization. The cleaner first test is bounded uniform decoded
init plus enough random MLP variation to break repeated-slot symmetry.
