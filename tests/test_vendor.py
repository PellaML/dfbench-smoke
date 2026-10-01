"""Vendored organizer files must stay byte-identical to the recorded commit."""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
VENDOR = PROJECT / "src" / "dfbench_smoke" / "vendor"
UPSTREAM = json.loads((VENDOR / "UPSTREAM.json").read_text(encoding="utf-8"))
# The package marker is ours, not an upstream file.
LOCAL_FILES = {"__init__.py"}


class VendorProvenanceTest(unittest.TestCase):
    def test_commit_and_repository_are_recorded(self):
        self.assertRegex(UPSTREAM["commit"], r"^[0-9a-f]{40}$")
        self.assertTrue(UPSTREAM["repository"].startswith("https://github.com/"))

    def test_each_vendored_file_matches_recorded_hash(self):
        self.assertTrue(UPSTREAM["files"])
        for entry in UPSTREAM["files"]:
            with self.subTest(file=entry["file"]):
                self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
                # Hash raw bytes so newline conversion on checkout is caught.
                digest = hashlib.sha256((VENDOR / entry["file"]).read_bytes()).hexdigest()
                self.assertEqual(digest, entry["sha256"])

    def test_file_names_match_their_upstream_paths(self):
        for entry in UPSTREAM["files"]:
            with self.subTest(file=entry["file"]):
                self.assertEqual(Path(entry["source_path"]).name, entry["file"])
                self.assertFalse(Path(entry["file"]).is_absolute())
                self.assertNotIn("..", Path(entry["file"]).parts)

    def test_no_unlisted_python_sources_in_vendor(self):
        listed = {entry["file"] for entry in UPSTREAM["files"]}
        present = {path.name for path in VENDOR.glob("*.py")}
        self.assertEqual(present - LOCAL_FILES, listed)
        self.assertEqual(len(listed), len(UPSTREAM["files"]))

    def test_vendor_package_marker_has_no_imports(self):
        source = (VENDOR / "__init__.py").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"^\s*(import|from)\s", source, re.MULTILINE))

    def test_licence_file_is_present_and_mit(self):
        licence = PROJECT / UPSTREAM["license_file"]
        self.assertTrue(licence.is_file())
        self.assertIn("MIT License", licence.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
