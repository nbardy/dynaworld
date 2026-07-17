# CTO Hourly Review Thread Type

## Context

The user asked for a new evergreen thread type called `CTO Code Reviewer Super
Autist`: an hourly fresh-eyes review over recent Git commits plus the current
on-disk dirty tree, with a cynical code-quality stance and strong attention to
the project's key patterns.

## Implementation

- Added `.agents/thread_types/cto_code_reviewer_super_autist/PROMPT.md` as the
  durable role prompt.
- Added `.agents/thread_types/cto_code_reviewer_super_autist/collect_packet.sh`
  to gather the hourly evidence packet:
  - branch, HEAD, status, and submodule state
  - commits in the configured lookback window
  - staged and unstaged summaries plus changed file lists
  - untracked review candidates, excluding noisy generated artifact trees
  - `git diff --check` and `git diff --cached --check`
  - AGENTS anti-pattern detectors for P1-P5
- Added `.agents/thread_types/cto_code_reviewer_super_autist/run_hourly_review.sh`
  to collect the packet, invoke `codex exec` in read-only mode, and write the
  resulting review under ignored `outputs/code_reviews/reports/`.
- Added a LaunchAgent plist at
  `.agents/thread_types/cto_code_reviewer_super_autist/com.dynaworld.cto-code-reviewer-super-autist.plist`
  with `StartInterval=3600` and `RunAtLoad=true`.
- Added `install_launchd.sh` and `uninstall_launchd.sh` helpers so scheduling
  can be turned on and off deliberately.
- Added `.agents/thread_types/README.md` so future agents can discover the
  evergreen thread type.

## Operating Contract

The hourly reviewer is review-only. It must not mutate the repo, launch paid
training, stage files, commit, or run long jobs. The runner enforces this by
calling Codex with `--sandbox read-only --ask-for-approval never`; the wrapper
writes the final markdown report after Codex exits.

The reviewer should treat old-but-still-dirty files as current risk, not as
outside the review scope. It should lead with findings, then missing proof,
pattern drift, dirty-state risks, and explicitly inspected no-issue areas.

## Scheduling

The plist is checked in but not loaded by this session. Loading it would start
hourly Codex calls, which can spend model time. Install deliberately with:

```bash
.agents/thread_types/cto_code_reviewer_super_autist/install_launchd.sh
```

Stop it with:

```bash
.agents/thread_types/cto_code_reviewer_super_autist/uninstall_launchd.sh
```

Manual packet-only smoke:

```bash
.agents/thread_types/cto_code_reviewer_super_autist/run_hourly_review.sh --packet-only
```
