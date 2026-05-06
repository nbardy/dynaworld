# PowerFoam Full Reproduction TODO

User asked to turn the foam implementation audit into a TODO covering what we
already have in Metal shaders and Torch code, what is left for full PowerFoam,
and how to test acceptance.

Added:

```text
TODO/powerfoam_full_reproduction_todo.md
```

The TODO splits the work into:

- current Torch inventory
- current Metal inventory
- dynamic/feature foam fork inventory
- definition of done for "full PowerFoam"
- P0-P8 implementation checklist
- acceptance gates for math, forward parity, backward parity, adjacency,
  trainer smoke, small overfit, held-out eval, performance, and baseline rows

Also linked the TODO from:

```text
research_notes/foam_papers/foam_implementation_status_2026-05-02.md
```

Important preserved conclusion:

- `third_party/powerfoam-metal` is a partial trainable Metal raster/backward
  core, not full PowerFoam.
- `src/train/powerfoam_direct.py` is the closest local paper-math reference,
  but not the official scalable system.
- dynamic/feature foam are Dynaworld experiments, not upstream PowerFoam.
- RadFoam/Radiant Foam remains unimplemented locally.

