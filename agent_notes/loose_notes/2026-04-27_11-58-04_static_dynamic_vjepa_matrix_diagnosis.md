# Static/Dynamic V-JEPA Matrix Diagnosis

## Question

Nicholas asked what is really going on after noticing that the recent residual/free-bank matrix excluded the strongest static/dynamic + V-JEPA lane.

## Evidence Checked

- `agent_notes/best_tweaks.md`
- `agent_notes/loose_notes/2026-04-26_19-06-17_static_dynamic_vjepa_feature_ablation.md`
- `agent_notes/loose_notes/2026-04-26_18-44-57_residual_free_bank_architecture_matrix.md`
- local W&B summaries under `wandb/run-*{mybv736f,oaor6um2,fja3e512,vph578vo,leunxckm,2a0vmenl,1shque5e}/files`
- the actual configs for the strong static/dynamic lane and the recent matrix cells

## Confirmed Local W&B Summaries

Strong lane:

| run | variant/backend | split | cross | steps config | Eval/Loss | SSIM | temporal adj ratio | decoded XYZ adj | camera adj rot |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `oaor6um2` | learned_time_orbit_path / precomputed V-JEPA 2.1 ViT-B/384 | 96/32 | 4 | 250 | 0.0881 | 0.6109 | 0.6322 | 0.0945 | 0.1309 deg |
| `mybv736f` | learned_time_orbit_path / precomputed V-JEPA 2.1 ViT-B/384 | 96/32 | 4 | 1000, interrupted around 525 | 0.0547 | 0.7836 | 0.8009 | 0.1305 | 0.1827 deg |

Recent matrix anchors:

| run | variant/backend | split | cross | init | Eval/Loss | SSIM | temporal adj ratio |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| `fja3e512` | unconditioned_tokens / none | none | 1 | strong RGB/uniform, token 0.8, head 0.12 | 0.1289 | 0.4009 | 0.0956 |
| `leunxckm` | learned_time_orbit_path / local | none | 1 | weak token 0.3, head 0.06, no RGB uniform | 0.1485 | 0.2946 | 0.0247 |
| `vph578vo` | learned_time_orbit_path / HF V-JEPA fpc16/256 SSV2 | none | 1 | weak token 0.3, head 0.06, no RGB uniform | 0.1601 | 0.2690 | 0.0261 |
| `2a0vmenl` | residual_free_bank / local | none | 1 | free-bank base, weak residual token/head | 0.1590 | 0.3106 | 0.0235 |
| `1shque5e` | residual_free_bank / HF V-JEPA fpc16/256 SSV2 | none | 1 | free-bank base, weak residual token/head | 0.1497 | 0.3397 | 0.0407 |

## Diagnosis

The recent matrix is not a fair falsification of the strong recipe because it changed several important variables at once:

- no static/dynamic split in the plain V-JEPA cells
- one cross-attention layer instead of four
- weaker token/head init in local and V-JEPA conditioned cells
- HF V-JEPA fpc16/256 SSV2 classification checkpoint instead of the precomputed V-JEPA 2.1 ViT-B/384 torchhub feature path
- 250-step plain conditioned cells compared against a strong-init unconditioned token control

The strongest current explanation is an interaction:

1. Static/dynamic split fixes the optimization channel. Local split already improved `0.1410 -> 0.1195` loss and `0.3401 -> 0.4287` SSIM compared with the clean cross1 temporal baseline.
2. V-JEPA helps only when the decoder has a useful place to put the information. Plain cross1 V-JEPA stayed nearly frozen; static/dynamic + cross4 V-JEPA produced much more pixel motion and lower loss.
3. Strong RGB/uniform and larger token/head init are not incidental. The unconditioned token matrix winner had strong init, while matrix local/V-JEPA cells did not.
4. The best run also moved the camera more, so the source-view result may be partly camera-path compensation. This must be controlled before calling it a true dynamic-world win.

## What This Means

Do not run more plain 128px matrix cells as the main path. The next useful tests should be matched around the strong recipe:

- static/dynamic local strong-init cross4
- static/dynamic V-JEPA strong-init cross4
- static/dynamic unconditioned strong-init cross4
- static/dynamic V-JEPA strong-init cross4 with reduced camera freedom
- same cells at 256px end-to-end or held-out/scene-distinct validation

The current belief should be phrased narrowly:

> Static/dynamic + strong init + enough cross-attention made V-JEPA features usable on the same-source dog-clip overfit. It is not yet evidence that V-JEPA alone helps, and it is not yet evidence of novel-view or held-out generalization.
