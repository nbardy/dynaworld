from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path


DYNAWORLD = Path(__file__).resolve().parents[2]
SLAB_TOOLS = (
    DYNAWORLD
    / "third_party"
    / "fast-mac-gsplat"
    / "variants"
    / "world_foam_lane2_fused_slab_v0"
    / "tools"
)
if str(SLAB_TOOLS) not in sys.path:
    sys.path.insert(0, str(SLAB_TOOLS))

import smoke_fused_slab_affine_realray_mps as smoke  # noqa: E402


class SmokeFusedSlabAffineRealrayCliTests(unittest.TestCase):
    def test_rejects_ownerupdate_without_per_track_layout(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            smoke.parse_args(["--include-ownerupdate"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--include-ownerupdate requires --layout per-track", stderr.getvalue())

    def test_accepts_ownerupdate_with_per_track_layout(self) -> None:
        args = smoke.parse_args(["--layout", "per-track", "--include-ownerupdate"])

        self.assertEqual(args.layout, "per-track")
        self.assertTrue(args.include_ownerupdate)

    def test_default_layout_remains_tiled_for_non_ownerupdate_smokes(self) -> None:
        args = smoke.parse_args([])

        self.assertEqual(args.layout, "tiled")
        self.assertFalse(args.include_ownerupdate)


if __name__ == "__main__":
    unittest.main()
