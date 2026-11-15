"""Regression checks for the real frozen team CSVs, without image access."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from resnet.import_clip_splits import convert

ROOT = Path(__file__).resolve().parents[2]

class SharedManifestTests(unittest.TestCase):
    def test_lossless_frozen_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "converted"
            audit = convert(ROOT / "manifests/splits_v1", dest)
            self.assertEqual(audit["counts"]["train.csv"], 28400)
            self.assertEqual(audit["counts"]["val.csv"], 4000)
            self.assertEqual(audit["unique_test_images"], 6000)
            fixture = json.loads((ROOT / "resnet/tests/frozen_manifest_hashes.json").read_text())
            self.assertEqual(audit["source_manifest_sha256"], fixture["source"])
            self.assertEqual(audit["converted_manifest_sha256"], fixture["converted"])
            with self.assertRaises(FileExistsError):
                convert(ROOT / "manifests/splits_v1", dest)
