# World Tubes ICLR 2027 fail-closed venue scaffold

## Scope

This work created only the source/provenance scaffold that is truthful before
the official ICLR 2027 kit and missing paper evidence exist. It ran no trainer,
Metal/MPS workload, native build, or numeric ablation. Tests in this session
protect package behavior; they are not experimental evidence.

## Changes

- Added `paper/venue/iclr2027/package_manifest.json`, `main.tex`, and a README.
- Recorded the official archive URL and the observed 2026-08-15 HTTP 404, with
  null archive/style hashes and no substituted ICLR 2026 style.
- Bound the concise manuscript, bibliography, schema-v2 evidence manifest, and
  AI-use statement by exact path, byte count, and SHA-256 in both the manifest
  and TeX comments.
- Made the TeX scaffold stop explicitly and consume only the accepted theorem
  fragment. Missing frozen/public evidence and the dirty-source variable-camera
  CPU candidate are not included or promoted.
- Strengthened the strict verifier to require a retained template archive, a
  matching archive hash, explicit official-template/package-ready lifecycle
  states, and machine-readable source bindings in addition to TeX comments.
- Added regression coverage for archive tampering, stale source bindings, and
  the checked-in scaffold's expected rejection.

## Validation

- Focused package suite: `5 passed`.
- Broader World Tubes artifact/package suite: `41 passed`.
- Python compilation and JSON parsing passed.
- The real strict audit returns `accepted=false` for the intended reasons:
  unavailable official template, incomplete/placeholder evidence, absent
  official PDF/build recorder/visual QA, and dirty repositories. It reports no
  stale or missing source binding after the final manuscript claim audit.

The scaffold must remain rejected until the real ICLR 2027 archive exists and
the frozen/public evidence, clean-source candidate promotion, official build,
and page-by-page PDF QA are complete.
