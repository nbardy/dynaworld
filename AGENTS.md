# Dynaworld Agent Guide

## Project Skills

Project-local Codex skills live under `.agents/skills/`. Use them when their
names or trigger phrases match the user request.

- `.agents/skills/deep-critical-thought/SKILL.md`: Use for durable thinking
  documents, "continue" expansion passes, branching/backtracking analysis,
  proofs, derivations, red-team hypothesis challenges, and rigorous agent notes.

## Baselines

Current baselines are tracked in `BASELINES.md` at the repo root. It is the
canonical standings table: configs, W&B run ids, train/eval splits, step
counts, wall-clock times, and PSNR/SSIM/L1 per baseline category. Read it
before claiming a model "beats baseline" or proposing a new one, and update
it (append a dated row, do not overwrite) whenever a baseline is re-run.

## Agent Notes

There are two agent memory layers. Use both, but do not blur them.

### Loose Notes

Put raw session journals in `agent_notes/loose_notes/`.

These are append-only progress logs. They are not curated knowledge and they are
not only for what worked. They are the long-form lab notebook for the messy
process:

- what we did
- why we thought it made sense at the time
- ideas and assumptions that turned out wrong
- commands, experiments, and benchmarks that changed our mind
- bugs, surprises, failed attempts, and fixes
- stray technical thoughts that may matter later
- handoff context for future agents

Write one loose note per meaningful session or work chunk. Do not silently
rewrite old history; add a new note when the understanding changes. Small typo
fixes are fine, but the journal should preserve the trail.

Loose note filenames must use searchable datetime-first slugs:

```text
agent_notes/loose_notes/{YYYY-MM-DD_HH-MM-SS}_{topic_slug}.md
```

Examples:

```text
agent_notes/loose_notes/2026-04-20_12-57-46_trainer_interface_cleanup.md
agent_notes/loose_notes/2026-04-20_13-20-10_jsonc_config_migration.md
```

### Key Learnings

Put the dense memory bank in exactly this file:

```text
agent_notes/key_learnings.md
```

`key_learnings.md` is for the most surprising things we learned by trying and
failing. It is not a second journal and it is not a changelog. Add only
unexpected, high-signal lessons that changed our model of the project.

Rules for `key_learnings.md`:

- keep it under 200 lines
- recompress older bullets instead of letting it grow
- prefer dense bullets over prose
- include failures and surprises, not obvious facts
- if a point only records what happened, put it in `loose_notes/`
- if a point changes how future agents should reason, compress it into `key_learnings.md`

Use `research_notes/` for more curated research writeups, paper notes, or
durable conclusions. Use `agent_notes/loose_notes/` for raw chronology and
decision history.

### Strategic docs (read before proposing architecture)

`research_notes/` has two strategic subfolders that supersede ad-hoc
derivation. Read these before re-inventing:

- `research_notes/meta_philosophy/` — north star, problem doc with
  failure modes F1–F7, mistakes-to-not-repeat log (append-only), and
  the required-XML driver prompt for external LLMs. Start with that
  folder's `README.md`.
- `research_notes/framing_the_problem/` — three framings of the
  novel-view bottleneck. Framing 1 is information-theoretic (use to
  derive losses); framing 2 is the self-sufficiency / generative-
  reconstruction contract (use to audit architectures for frame-local
  leaks); **framing 3 is the patched bitter-lesson predictive-quotient
  baseline and the current default** — start there when proposing
  anything new.
  That folder's `README.md` explains when to use which.
- `research_notes/training_contract_v1.md` — operational contract for
  patched framing 3: sampler, model signatures, baseline losses,
  diagnostics, escape hatches, support assumptions, and export tripwires.
- `research_notes/three_architectures_for_novel_view_synthesis.md` —
  concrete A/B/C candidates cross-referenced against all three framings.

See `research_notes/README.md` for the full navigation index.

## Build & Run Conventions

### Always launch from the dynaworld root

The trainer launch is:

```bash
PYTHONPATH=src/train uv run python src/train/train_video_token_implicit_dynamic.py <path/to/config.jsonc>
```

Run it from `/Users/nicholasbardy/git/gsplats_browser/dynaworld`. The
`PYTHONPATH=src/train` is required so that local modules (`config_utils`,
`runtime_types`, `colorize`, `feature_pca_viz`, etc.) resolve.

### Fast-mac variant pyproject gotcha

`third_party/fast-mac-gsplat/variants/v5/` and `variants/v5_features/` (and the
other variant forks) ship a `pyproject.toml` with only a `[build-system]`
table — no `[project]` table. That is intentional; they're built with
`setup.py build_ext --inplace`, not installed as Python projects.

The consequence: if `uv run` is invoked with its CWD inside one of those
variant directories, `uv` walks up looking for a project, hits the bare
pyproject, and aborts with `error: No `project` table found in: ...`.

Three rules to avoid this:

1. **Don't `cd` into a variant directory and then keep working.** The
   `Bash` tool persists CWD between commands. If you must `cd` to build,
   `cd` back to the dynaworld root immediately afterward. Better: build
   with absolute paths so CWD never changes.
2. **Don't put a variant directory on `PYTHONPATH`.** The dynaworld
   wrapper (`src/train/renderers/fast_mac.py`) injects the right variant
   onto `sys.path` at runtime via `_ensure_fast_mac_v5_on_path()` /
   `_ensure_fast_mac_v5_features_on_path()`. The trainer launch only
   needs `PYTHONPATH=src/train`.
3. **To build a new variant**, the canonical recipe is:

   ```bash
   ( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/<variant>
     uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
   ```

   The subshell parens keep the CWD change scoped — the parent shell
   stays at the dynaworld root.

### Smoke-test rules (don't trust py_compile alone)

`py_compile` only catches syntax and import errors. It does not catch:

- tuple-arity mismatches (`a, b, c = func()` when `func` now returns 4 values)
- dict-key mismatches (`d["foo"]` after `foo` was renamed in the producer)
- attribute renames on dataclasses
- inheritance / override signature mismatches

Any edit that touches a function's return signature, a dataclass field, a
config key, or a method override **must** be followed by a runtime smoke
before declaring the change done. The smoke must exercise the actual call
graph, not just `import` the module.

For the trainer, the canonical 1-step F=3 smoke is:

```bash
PYTHONPATH=src/train WANDB_MODE=offline /Users/nicholasbardy/git/gsplats_browser/dynaworld/.venv/bin/python \
  src/train/train_video_token_implicit_dynamic.py /tmp/smoke.jsonc
```

with `/tmp/smoke.jsonc` a copy of a fast config patched to `train.steps: 1`.
The 1-step smoke fires `val_log(0, initial_result)` (because `0 % video_log_every == 0`)
and `val_log(1, ...)` (via `always_log_last_step`), so it exercises both
training and validation paths in under 10 s.

For F=32 / feature splatting, mirror with the F32 config; the smoke also
exercises the colorize MLP and PCA video paths.

Rule of thumb: if the change spans more than one file, run the smoke after
**all** files are edited, not after each one. Mid-cascade states are
broken by construction.

### Test quality rules

Do not add tests just to prove an implementation detail that is unlikely to
catch a real regression. A test must protect a behavior contract the user cares
about, or it should not exist.

Avoid brittle tests that monkeypatch internals such as `torch.linalg.svd` only
to count calls, inspect helper mechanics, or assert a particular implementation
shape. Those tests can pass while the actual feature is broken, and they make
future refactors noisy.

Before adding a test, write down the failure it is supposed to catch. If the
failure is "we might change the helper implementation" rather than "the logged
video is missing frames", "F=32 no longer reaches the feature rasterizer", or
"the configured smoke path crashes", prefer a runtime smoke, artifact check, or
no test.

### Renderer dispatch

`src/train/renderers/fast_mac.py` dispatches by `rgbs.shape[-1]`:

- `F == 3` → `v5` (clamps output to `[0, 1]` for direct loss)
- `F != 3` → `v5_features` (raw features; downstream `FeatureToColor`
  applies sigmoid)

Both `.so` files must be built for the active Python (currently 3.11).
Check with `ls third_party/fast-mac-gsplat/variants/v5*/torch_gsplat_bridge_*/_C.cpython-311-darwin.so`.

## Config Style

Training hyperparameters should be defined once, in checked-in JSONC files under `src/train_configs/`.

- Do not add environment-variable fanout for every knob.
- Do not mirror full config defaults in large Python dictionaries.
- Do not add argparse blocks that duplicate the config schema.
- Shell scripts should choose a config file and call the trainer with that path.
- Python trainers should accept a config dict or config path, normalize only runtime concerns such as `Path` values, and fail loudly when required keys are missing.
- If a backward-compatible default is needed for an older config, apply it once during config load/normalization. Do not scatter `cfg.get("key", magic_number)` across model construction, logging, or train-loop code.
- Runtime code should read normalized configs with explicit keys. Repeated `.get(..., default)` at use sites is a smell unless the value is truly optional and `None` has semantic meaning.
- For status prints, prefer small dictionary/summary helpers that iterate over named keys. Do not hand-build long f-string chains that duplicate config defaults or quietly drift from the schema.

Keep code lean by passing config sections through warm paths instead of destructuring and rebuilding the same data repeatedly. When a boundary needs renamed constructor parameters, keep that mapping in one small factory function close to the boundary.

Prefer JSONC (`*.jsonc`) for train configs so experimental notes can live next to the values they explain.

### Anti-patterns to eliminate

These five patterns repeatedly grow line count without adding clarity. They are mechanical to spot and mechanical to fix. When you find one, fix it; when you write new code, don't introduce one.

The unifying rule: **never name an intermediate value just to thread it through a function boundary**. Pass the canonical container (`cfg`, the config-bearing object, the colorize module) down to the leaf, and read `container["key"]` or `container.attr` at the actual use site.

**P1 — Local cfg destructure**
```python
# BAD: shuffle. Local alias used 1-2 times then thrown away.
render_cfg = cfg["render"]
render_size = int(render_cfg["render_size"])
return render_gaussian_frames(..., height=render_size, tile_size=render_cfg["tile_size"], ...)

# GOOD: leaf reads its own keys.
return render_gaussian_frames(cfg, ...)        # leaf takes cfg, reads cfg["render"]["..."] inside
```
Detector: `^\s*\w+ = \w+\["[^"]+"\]\s*$` followed by ≤3 uses inside the same function.

**P2 — `self.X` cfg alias used <3 times**
```python
# BAD: hoist into __init__, used in only one or two methods.
self.render_cfg = self.cfg["render"]
self.render_size = int(self.render_cfg["render_size"])
```
Detector: `self.X = self.cfg[...]` or `self.X = self.foo_cfg[...]` in `__init__`, then count `self.X` references across the class. ≤2 uses → kill the alias and read from `self.cfg["section"]["key"]` at the use site. (Heavily-used aliases — e.g. `self.model_cfg` with 30+ refs — are fine; the detector is the ratio of declarations to uses.)

**P3 — Kwargs-forwarding pyramid**
```python
# BAD: every layer copies the same cfg-derived kwargs.
def foo(*, render_size, feature_pca_log, fps, tile_size, ...):  # 8 kwargs
    bar(render_size=render_size, feature_pca_log=feature_pca_log, ...)  # 6 forwarded
```
Detector: a function whose signature has ≥4 cfg-derived kwargs. Fix: take `cfg` (or the relevant subsection) as first positional; let the function read its own keys. Each layer drops the wall.

**P4 — Wrapper-then-unwrap at the call site**
```python
# BAD: re-extract attributes that already live on the object you have.
colorize_view_dirs(features, cameras,
    view_condition=self.colorize.view_condition,
    detach=self.colorize.detach_view_condition,
)
```
Detector: a call site where `kwarg=obj.attr` AND the function only uses that attribute. Fix: pass `obj` itself, read `obj.attr` inside. If the attribute lives somewhere else (factory wrapper, sibling alias), move it onto the object that gets passed around.

**P5 — Validation duplicated at the hoist site**
```python
# BAD: re-validate at __init__ what resolve_config should have validated once.
self.recon_backward_strategy = self.train_cfg["recon_backward_strategy"]
if self.recon_backward_strategy not in {"framewise", "microbatch", "batched"}:
    raise ValueError(...)
self.temporal_microbatch_size = int(self.train_cfg["temporal_microbatch_size"])
if self.temporal_microbatch_size < 1:
    raise ValueError(...)
```
Detector: `raise ValueError` immediately after a `self.X = ...` self-assign in `__init__`. Fix: move the `raise` into `resolve_config` (or whatever normalizes the config once at boot). Drop the hoist if it's also a P2.

### Detectors as one-liners

```bash
# P1: function-scope cfg aliases
grep -nE '^\s+\w+_cfg = \w+\["[^"]+"\]' src/train/

# P2: self.X = ... cfg hoists in __init__ (then count usage manually)
grep -nE '^        self\.[a-z_]+ = (self\.cfg|self\.[a-z_]+_cfg|int\(self\.|bool\(self\.|float\(self\.)' \
    src/train/train_*.py

# P3: kwargs walls (functions with ≥5 args)
awk '/^def [a-z_]/ { in_def=1; nargs=0 }
     in_def && /^[ ]+[a-z*]/ { nargs++ }
     in_def && /\)\s*->/ { in_def=0; if (nargs>=5) print FILENAME":"NR" args="nargs" "$0 }' \
    src/train/**/*.py

# P4: kwarg=self.obj.attr at call sites
grep -rnE '^\s+\w+=self\.\w+\.\w+,?$' src/train/

# P5: raise immediately after a self-assign in __init__
grep -B1 -A2 -nE '^        if self\.\w+' src/train/train_*.py | grep -A2 'raise'
```

When you fix one, run the smoke gate (1-step F=3 + F=32 + multicam-ult — see "Smoke-test rules") before declaring it done. The change must reduce LOC; if it doesn't, you've reshuffled rather than fixed.
