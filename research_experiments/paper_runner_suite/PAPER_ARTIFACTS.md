# World Tubes paper artifacts

`generate_world_tubes_paper_artifacts.py` is the submission-facing artifact
generator. The tables and SVG files emitted directly by training/matrix
runners are diagnostic intermediates; do not copy them into the manuscript.

The generator is CPU-only and does not import Torch or a renderer. It reads:

1. the expected public matrix, its completed canonical `matrix_summary.json`,
   and each schema-v2 `run_summary.json`;
2. the verifier-accepted frozen-world wrapper `summary.json`;
3. the verifier-accepted variable-camera closure/death `summary.json`;
4. the verifier-produced theorem-table `summary.json`; and
5. the learned-world moving-camera-density report, which remains a separate
   fail-closed requirement even after the synthetic closure/death curve is
   accepted.

When `--run-root` is omitted, the generator reads the selected matrix
config's canonical `output_root`. This prevents a full-breadth matrix from
silently consuming the seven-row submission directory, or vice versa.

Generate the current honest state:

```bash
python3 \
  research_experiments/paper_runner_suite/generate_world_tubes_paper_artifacts.py \
  --allow-incomplete
```

The accepted clean variable-camera artifact is a paper-freeze schema-v2
report. The current variable-camera runner uses a different schema-v1 contract,
so it must never decode or relabel those older accepted bytes. After regenerating
the base bundle, import the frozen component through the pinned compatibility
verifier:

```bash
.venv/bin/python \
  research_experiments/paper_runner_suite/import_world_tubes_variable_camera_schema_v2.py
```

That importer verifies the exact raw SHA-256, clean start/finish commits,
implementation-source manifest, every handoff `SHA256SUMS` target, and
byte-identical Markdown/TeX/SVG outputs. It stores the raw artifact and a
portable receipt under
`artifacts/paper_evidence/world_tubes_variable_camera_schema_v2_clean/`, then
changes only that component in the current ledger. It preserves all other
validation results and restores the moving-camera-density gate if an older
bundle omitted it. The dirty schema-v1 178/179-degree diagnostic remains
excluded.

The exact frozen SVG remains unchanged at
`generated/schema_v2/variable_camera_closure_death.svg`. Because its
right-edge death label clips under portable rasterization, the importer also
writes the separately receipted
`figures/world_tubes_variable_camera_closure_death_publication.svg`. That
display derivative changes only the label anchor; its receipt binds the exact
evidence-SVG digest and records that no data or plot geometry changed.

Incomplete components produce explicit Markdown, TeX, and SVG placeholders.
They never produce partial numeric tables. The command records every missing
or rejected runtime input in:

```text
research_notes/gauged_uvt_trace_atlas/paper/generated/schema_v2/
  evidence_ledger.json
  missing_runtime_inputs.json
```

After every required run is accepted, omit `--allow-incomplete`. The command
then fails unless the entire bundle is submission-ready:

```bash
python3 \
  research_experiments/paper_runner_suite/generate_world_tubes_paper_artifacts.py
```

Verify an honest incomplete bundle independently:

```bash
python3 \
  research_experiments/paper_runner_suite/generate_world_tubes_paper_artifacts.py \
  --verify-dir \
  research_notes/gauged_uvt_trace_atlas/paper/generated/schema_v2 \
  --allow-incomplete
```

The strict evidence-bundle gate intentionally omits `--allow-incomplete`.
Its `submission_ready` compatibility field is explicitly scoped by
`readiness_scope=evidence_artifact_bundle_only`; it never means that venue
conversion or PDF inspection is complete.

Verify the imported component independently, without reading the external
paper-freeze tree:

```bash
.venv/bin/python \
  research_experiments/paper_runner_suite/import_world_tubes_variable_camera_schema_v2.py \
  --verify-local
```

The manuscript-consumable files are `theorem_table.tex`,
`public_context_table.tex`, `frozen_scaling_table.tex`,
`variable_camera_table.tex`, and the four corresponding SVG figures.
`artifact_manifest.json` binds every generated file by byte count and SHA-256.
The four TeX tables are wired into `WORLD_TUBES_PAPER_DRAFT.md` with
`\input{...}` and therefore remain live when Pandoc regenerates the working
TeX. Schema-v1 public numbers and their historical plot are forbidden from
both manuscript sources.

Verify the complete working-manuscript package without importing Torch:

```bash
python3 \
  research_experiments/paper_runner_suite/generate_world_tubes_paper_artifacts.py \
  --verify-manuscript \
  --allow-incomplete
```

Omit `--allow-incomplete` for the final gate. In strict mode the verifier also
rejects the generic Pandoc `article` class: venue conversion is a separate
required packaging step. The SVGs remain deterministic source figures; export
accepted plots to the venue's required PDF/PNG format during that conversion
and visually inspect the rendered PDF.

The public component is accepted only when `matrix_summary.json` has the exact
matrix name, run count, ordered run keys, and lane count declared by the
matrix config. Every embedded summary—after removing the runner-added
`run_summary_path`—must equal its retained `run_summary.json` exactly. This
keeps the runner's deep canonical validation in the submission chain.

The frozen component does not trust reported medians. It requires publication
warmups/repeats, checks the exact raw timing-sample key set and lengths,
recomputes min/quartiles/median/max/mean, and verifies every forward/backward,
compile-plus-total, and per-frame algebraic identity.

The theorem component does not trust the theorem summary's
`all_sources_verified` bit. It reopens exact retained reports whose SHA-256
digests are pinned in the generator and rederives every emitted row.
Single-shot bounded-fixture forward/backward timing ratios are explicitly
excluded; only structural trace-count reuse remains in the theorem table.
Publication speed belongs exclusively to the warmed, repeated frozen-world
component.
