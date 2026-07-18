# Renderer Lane Taxonomy Correction

## Context

The 2026-07-17 closeout used the phrase "Gauged UVT theory lane is closed."
That was easy to read as dropping the camera-gauge and ray-fiber paper. The
repo evidence says the opposite: those objects are the mathematical core of
the World Tubes draft and are also reused by WorldFoam.

## Corrected Model

The stopped work is open-ended theory and naming proliferation without a
replayable compiler failure. The retained hierarchy is:

```text
Gauged camera-ray bundle       shared mathematics
World Tubes                    primary Gaussian/compiler paper method
STAR UVT                       implementation family
projective STAR UVT / PRT      internal moving-camera interval implementation
WorldFoam                      retained-depth optical-transfer second paper
Gate4/owner-run/cutwalk        WorldFoam prototype implementation family
PowerFoam                      ancestor/baseline, not a synonym
```

The operator-order split remains the important theorem:

```text
World Tubes: pi_* Gamma^* w, then visibility certificates/compositing
WorldFoam: retain lambda(u,v,t,z), then transmittance/product-integral scan
```

Visibility does not generally commute with depth marginalization. This makes
WorldFoam a meaningful second method without invalidating World Tubes.

## Current In-Flight Engineering Observed On Disk

Uncommitted work predating this note is extending the shared paper protocol:

```text
typed dataset and protocol contracts
full 300-frame coffee_martini manifest
progressive and fixed matched-pixel configs
paper budget/coverage accounting
selected-time World Tubes paper batches instead of full-sequence raster replay
PowerFoam final-stage primitive-count validation
one unified World Tubes / WorldFoam / dynamic-3DGS paper-ablation runner
```

Those files were not modified or committed by this taxonomy pass.

## Decision

Finish the shared paper protocol, then run World Tubes across camera triplets
and scenes. Package the World Tubes paper if quality parity and sublinear
world-side scaling survive. Re-evaluate WorldFoam on the same breadth table;
only then spend on native optical-transfer Metal parity. Keep broader
world-token, V-JEPA/F32, browser, Softmax-GS, and PowerFoam backlog claims
separate.

The canonical detailed map is:

```text
research_notes/renderer_lane_taxonomy.md
```
