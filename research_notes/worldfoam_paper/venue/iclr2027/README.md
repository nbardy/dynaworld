# WorldFoam ICLR 2027 venue scaffold

This directory is a fail-closed scaffold, not a submission package.

As checked on 2026-08-15, the archive URL named by the official ICLR 2027
author guide returned HTTP 404 and the public master-template repository did
not yet contain an ICLR 2027 style. Consequently this directory contains no
style archive, no style file, no compiled PDF, and no invented provenance
hashes. An ICLR 2026 style must not be copied or renamed here.

`package_manifest.json` and `main.tex` bind the current concise manuscript,
bibliography, foundation evidence manifest, and AI-use statement by their exact
byte counts and SHA-256 digests. The TeX entry point contains an intentional
error and consumes only `synthetic_visibility_table.tex`, the generated table
backed by the accepted float64 CPU G0/G3 suite. It does not consume the current
G4 public-quality or G6 native-memory placeholders, and it does not turn the
foundation bundle's limited acceptance into submission readiness.

The strict gate is expected to fail until all of the following are true:

1. the official `iclr2027.zip` exists locally and its actual archive/style
   hashes and retrieval date replace the null fields;
2. G4 public quality and G6 fresh-process native memory/work are independently
   measured, verified, and promoted by a regenerated evidence bundle;
3. all required portable figures and generated tables are consumed;
4. the official anonymous TeX package compiles within the nine-page main-text
   limit with embedded non-Type-3 fonts; and
5. every page of that exact PDF has a hash-bound visual-QA record.

Run the strict audit from the repository root:

```bash
python3 research_experiments/paper_runner_suite/verify_worldfoam_iclr_package.py
```

A nonzero exit while this scaffold is present is correct behavior.
