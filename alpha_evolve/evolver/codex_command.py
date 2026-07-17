"""Command construction for non-interactive Codex candidate runs."""

from __future__ import annotations

from pathlib import Path


def build_codex_exec_command(
    *,
    worktree: str | Path,
    prompt: str,
    final_message_path: str | Path,
    sandbox: str = "danger-full-access",
    ask_for_approval: str = "never",
    json_events: bool = True,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> list[str]:
    """Return a `codex exec` argv list.

    The local CLI currently treats `-p` as a profile flag, not prompt input.
    Keep prompt text as the final positional argument.
    """

    command = [
        "codex",
        "exec",
        "--cd",
        str(worktree),
        "--sandbox",
        sandbox,
        "--ask-for-approval",
        ask_for_approval,
        "--output-last-message",
        str(final_message_path),
    ]
    if json_events:
        command.append("--json")
    if model:
        command.extend(["--model", model])
    if reasoning_effort:
        command.extend(["--reasoning-effort", reasoning_effort])
    command.append(prompt)
    return command
