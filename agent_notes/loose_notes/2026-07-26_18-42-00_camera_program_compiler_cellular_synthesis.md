# Camera-Program Compiler And Cellular Backend Synthesis

Date: 2026-07-26 18:42:00 +0900

## Trigger

Audited the ChatGPT Pro export:

```text
/Users/nicholasbardy/.codex/attachments/
59da4296-f678-4d80-a5cb-8ec2526c3360/pasted-text.txt
```

The request was to preserve its useful information in research notes and
compare it with the current archive and World Tubes/STAR UVT paper.

## Work Performed

- Read the 1,609-line export in full and extracted its claims, proposed
  theorems, research prompt, experiments, and kill criteria.
- Compared it with the canonical renderer taxonomy, SPD(4) representation
  audit, native-motion/shared-raster derivation, World Tubes manuscript,
  WorldFoam manuscript, depth-fiber cross-track note, and the same-day
  log-FEM audit.
- Checked current paper evidence and canonical baseline routing.
- Verified the named prior-art boundary against primary paper pages for native
  4DGS, Spacetime Gaussian Feature Splatting, Disentangled4DGS, Radiant Foam,
  Power Foam, Radiance Meshes, DiffTetVR, simplex space-time meshes, and 3DGUT.
- Wrote the durable synthesis:
  `research_notes/camera_program_compiler_and_cellular_backend_synthesis.md`.

## Main Conclusions

1. The export validates the current paper spine rather than replacing it:
   compile a dynamic world through a continuous camera program into reusable
   sensor-time forward and adjoint state.
2. Public naming should remain World Tubes in Gauged Camera Space, implemented
   by projective STAR UVT. The camera-program compiler/adjoint is the umbrella.
3. World Tubes and WorldFoam are sibling operator lowerings, not literally
   interchangeable backends of the current STAR compiler. Early depth
   marginalization and retained-depth transfer have different IRs and
   semantics.
4. Full SPD(4) and the full linear Gaussian tube are exactly equivalent. The
   fact that the tube is a real 4D field is correct but not novel.
5. The export's claim that changing intrinsics/extrinsics within a sequence is
   not compiled is stale relative to the current projective/orbit route.
6. The decisive paper baseline is identical learned world parameters rendered
   through per-time replay versus the compiled atlas.
7. The most useful new cellular distinction is direct nonnegative P1
   extinction versus the existing P1/P2 log-extinction proposal. Direct P1
   produces affine ray density and exponential-of-quadratic transmittance with
   truncated Gaussian moment integrals.
8. Cellular work remains a bounded toy/second-paper lane until it proves event
   sharing, fixed-stratum VJP, reverse-storage scaling, and quality per byte.

## Validation

No code or benchmark was changed. Documentation validation and final
`git diff --check` were run after writing.

