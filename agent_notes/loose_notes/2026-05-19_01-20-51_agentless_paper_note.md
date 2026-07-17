# Agentless Paper Note

Created `alpha_evolve/papers/notes/015_agentless.md` as the fifteenth paper
note in the AlphaEvolve/agentic-code-evolution survey.

Sources inspected:

- arXiv paper `2407.01489`
- official `OpenAutoCoder/Agentless` README
- implementation files for localization and repair

Main takeaway for local `alpha_evolve` work:

- Build an Agentless-style staged baseline before adding a complex autonomous
  Codex agent loop.
- The baseline should localize, sample multiple patches with `codex exec`, run
  syntax/regression/generated-test validation, and rank patches.
- The note adds microlibs for `agentless_baseline_runner`,
  `repo_structure_summarizer`, `file_localizer`, `edit_location_sampler`,
  `patch_sampler`, `generated_repro_test_builder`, `regression_selector`,
  `patch_ranker`, and `benchmark_sanitizer`.

Queue state after this note:

- Papers 001-015 have first-pass detailed notes.
- Papers 016-020 remain queued, with OpenHands next.
