"""Inspect and reclaim disk for the Blender / Sintel synthetic-render bundle.

This dataset is currently a hand-curated, hand-rendered set of `.blend` outputs
under `data/blender_synthetic/`. It does NOT have a download stage; the
production tree comes from the Sintel DVD ISO on archive.org and Blender 2.79b
from the Blender release server. The pipeline here is therefore deliberately
minimal: list what is on disk, validate that nothing critical was nuked, and
flag the safely-reclaimable files.

Reclaimability rules (do NOT relax these without re-reading the agent memory at
`~/.claude/projects/.../sintel_asset_access_2026_04.md` and
`blender_synthetic_pipeline_facts.md`):

  - `sintel/_iso/Sintel_DATA.iso`   PRESERVE. The 4-DVD `DATA` ISO on
    archive.org is the only free public source for the Sintel production tree
    (the original Durian SVN is dead). Re-acquiring it is a slow archive.org
    download. NEVER auto-delete.
  - `sintel/_iso/Sintel_PAL.iso`    RECLAIMABLE on opt-in. PAL DVD ISO. Also on
    archive.org. We don't currently use it for blends; the `DATA` disc is what
    matters.
  - `sintel/_iso/*.mkv` etc.        RECLAIMABLE. Finished-movie pixels — useful
    only as a reference watch, not for rendering. Re-downloadable.
  - `sintel/renders/`               RECLAIMABLE on opt-in. Scratch render
    outputs we produced locally. Re-renderable.
  - `_blender_2_79b/blender-2.79b-macOS.zip`            RECLAIMABLE. Re-fetchable from
    https://download.blender.org/release/Blender2.79/.
  - `_blender_2_79b/blender-2.79b-macOS-10.6/`          RECLAIMABLE on opt-in. The
    extracted Blender app — re-extractable from the zip above.
  - `_blender_2_79b/__MACOSX/`                           RECLAIMABLE. macOS unzip
    cruft. Always safe.

Anything we'd need to re-acquire from a paid source (Blender Studio
subscription) we never have on disk in the first place, so it does not appear
in this list.
"""
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_DIR.parent


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


@dataclass(frozen=True)
class ReclaimEntry:
    path: Path
    bytes_total: int
    reason: str
    safe_to_auto_delete: bool


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def file_count(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return 1
    return sum(1 for child in path.rglob("*") if child.is_file())


# Single source of truth for which files are safe to delete.
# `safe_to_auto_delete=False` means even `--execute` will print but skip; the
# file must be removed by hand (e.g. Sintel_PAL.iso, full extracted Blender app)
# to make ISO retention require deliberate human action.
def reclaim_candidates(root: Path) -> list[ReclaimEntry]:
    sintel = root / "sintel"
    blender = root / "_blender_2_79b"
    candidates: list[ReclaimEntry] = []

    macosx = blender / "__MACOSX"
    candidates.append(
        ReclaimEntry(
            path=macosx,
            bytes_total=directory_bytes(macosx),
            reason="macOS unzip cruft (always safe to delete)",
            safe_to_auto_delete=True,
        )
    )

    candidates.append(
        ReclaimEntry(
            path=blender / "blender-2.79b-macOS.zip",
            bytes_total=directory_bytes(blender / "blender-2.79b-macOS.zip"),
            reason="Blender 2.79b zip; re-fetchable from download.blender.org/release/Blender2.79/",
            safe_to_auto_delete=True,
        )
    )

    candidates.append(
        ReclaimEntry(
            path=blender / "blender-2.79b-macOS-10.6",
            bytes_total=directory_bytes(blender / "blender-2.79b-macOS-10.6"),
            reason="extracted Blender 2.79b app; re-extractable from blender-2.79b-macOS.zip",
            safe_to_auto_delete=False,
        )
    )

    candidates.append(
        ReclaimEntry(
            path=sintel / "renders",
            bytes_total=directory_bytes(sintel / "renders"),
            reason="local scratch render outputs; re-renderable from sintel production blends",
            safe_to_auto_delete=False,
        )
    )

    candidates.append(
        ReclaimEntry(
            path=sintel / "_iso" / "Sintel.2010.1080p.mkv",
            bytes_total=directory_bytes(sintel / "_iso" / "Sintel.2010.1080p.mkv"),
            reason="finished movie file; not used for rendering; re-downloadable",
            safe_to_auto_delete=True,
        )
    )

    candidates.append(
        ReclaimEntry(
            path=sintel / "_iso" / "Sintel_PAL.iso",
            bytes_total=directory_bytes(sintel / "_iso" / "Sintel_PAL.iso"),
            reason="PAL DVD ISO; also on archive.org; not the production-blend source",
            safe_to_auto_delete=False,
        )
    )

    return candidates


def critical_assets(root: Path) -> list[Path]:
    # Anything in this list MUST exist for the dataset to remain useful and
    # MUST never be deleted by this tool. validate-local fails if missing.
    return [
        root / "sintel" / "_iso" / "Sintel_DATA.iso",
    ]


def inspect(root: Path) -> None:
    sintel = root / "sintel"
    blender = root / "_blender_2_79b"
    print(f"Blender synthetic root: {root}")
    print(f"  exists: {root.exists()}")
    print(f"  total_bytes: {directory_bytes(root)}")
    print()
    print("sintel/_iso/")
    iso_dir = sintel / "_iso"
    if iso_dir.exists():
        for item in sorted(iso_dir.iterdir()):
            print(f"  {item.name}\t{item.stat().st_size} bytes\t{item.stat().st_size / 1_000_000.0:.1f} MB")
    print()
    print("sintel/renders/")
    renders_dir = sintel / "renders"
    if renders_dir.exists():
        for item in sorted(renders_dir.iterdir()):
            if item.is_dir():
                print(f"  {item.name}\tfiles={file_count(item)}\tbytes={directory_bytes(item)}")
    print()
    print("_blender_2_79b/")
    if blender.exists():
        for item in sorted(blender.iterdir()):
            label = "dir" if item.is_dir() else "file"
            print(f"  [{label}] {item.name}\tbytes={directory_bytes(item)}")


def validate_local(root: Path) -> None:
    failures: list[str] = []
    for path in critical_assets(root):
        if not path.exists():
            failures.append(f"missing critical asset (re-acquire from archive.org sintel-dvd): {path}")
            continue
        size = path.stat().st_size
        if size < 1_000_000_000:  # ISO should be ~7.5 GB; a truncated ISO is worse than missing
            failures.append(f"critical asset truncated ({size} bytes, expected ~8 GB): {path}")
            continue
        print(f"  ok\t{path}\t{size} bytes")
    blender_app = root / "_blender_2_79b" / "blender-2.79b-macOS-10.6" / "blender.app"
    if blender_app.exists():
        print(f"  ok\t{blender_app} (re-extractable from blender-2.79b-macOS.zip)")
    else:
        # Not a critical failure: the user may have already reclaimed the
        # extracted Blender app. The zip alone is enough to re-extract.
        print(f"  note\tBlender 2.79 app not extracted (zip only): {blender_app}")
    if failures:
        raise RuntimeError(
            "Blender synthetic validate-local failed:\n  " + "\n  ".join(failures)
        )
    print("validate-local: critical assets present.")


def cleanup(root: Path, *, execute: bool, include_protected: bool) -> None:
    candidates = reclaim_candidates(root)
    present = [c for c in candidates if c.bytes_total > 0]
    auto = [c for c in present if c.safe_to_auto_delete]
    protected = [c for c in present if not c.safe_to_auto_delete]
    print(f"Blender synthetic cleanup ({'EXECUTE' if execute else 'DRY-RUN'}):")
    auto_total = 0
    for entry in auto:
        print(f"  reclaim\t{entry.path}\t{entry.bytes_total / 1_000_000.0:.1f} MB\t{entry.reason}")
        auto_total += entry.bytes_total
    protected_total = 0
    for entry in protected:
        flag = "reclaim" if include_protected else "skip"
        print(
            f"  {flag}\t{entry.path}\t{entry.bytes_total / 1_000_000.0:.1f} MB"
            f"\t(opt-in --include-protected) {entry.reason}"
        )
        protected_total += entry.bytes_total
    for path in critical_assets(root):
        size = directory_bytes(path)
        print(f"  PRESERVE\t{path}\t{size / 1_000_000.0:.1f} MB\t(never auto-deletable: archive.org-only source)")
    print(
        f"  TOTAL auto-reclaimable: {auto_total} bytes ({auto_total / 1_000_000.0:.1f} MB) "
        f"across {len(auto)} entries"
    )
    if include_protected:
        print(
            f"  TOTAL with --include-protected: {auto_total + protected_total} bytes "
            f"({(auto_total + protected_total) / 1_000_000.0:.1f} MB)"
        )
    if execute:
        targets = list(auto) + (list(protected) if include_protected else [])
        for entry in targets:
            if entry.path.is_file():
                entry.path.unlink()
            else:
                shutil.rmtree(entry.path)
            print(f"  deleted: {entry.path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and reclaim disk for the Blender/Sintel synthetic bundle.")
    parser.add_argument("stage", choices=("inspect", "validate-local", "cleanup"))
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/blender_synthetic"),
        help="Root directory of the bundle (default: data/blender_synthetic).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="cleanup only: actually delete reclaimable files. Default is dry-run.",
    )
    parser.add_argument(
        "--include-protected",
        action="store_true",
        help=(
            "cleanup only: also include opt-in entries (PAL ISO, extracted Blender app, "
            "renders/) in the deletion set. The DATA ISO is NEVER included."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_path(args.root)
    if args.stage == "inspect":
        inspect(root)
    elif args.stage == "validate-local":
        validate_local(root)
    elif args.stage == "cleanup":
        cleanup(root, execute=args.execute, include_protected=args.include_protected)


if __name__ == "__main__":
    main()
