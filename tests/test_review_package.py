"""CPU-only regression checks for the portable manual-review source bundle."""

from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import package_review as packaging


class ReviewPackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="emmy-review-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in packaging.REQUIRED_FILES:
            self.write(name, "source\n")
        for name in packaging.SOURCE_TREES:
            self.write(name + "/example.md", "review material\n")
        for name in packaging.FIXTURE_PATHS:
            self.write(name if Path(name).suffix else name + "/case/meta.json", "{}\n")

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_roundtrip_deterministic_inventory_extensions_and_failed_evidence(self):
        self.write("ptx_tensor.py", "tensor extension\n")
        self.write("ptx_bulk.py", "copy extension\n")
        self.write("tests/__pycache__/ignored.pyc", "cache")
        self.write("tests/.env", "not for distribution")
        self.write("artifacts/new/summary.json", '{"pass":false,"checks":3}\n')
        self.write("artifacts/new/example.npz", "fixture bytes")
        one = packaging.package_review(self.root, self.root / "release-one", includes=["artifacts/new/example.npz"], evidence=["artifacts/new/summary.json"], update_latest=True)
        two = packaging.package_review(self.root, self.root / "release-two", includes=["artifacts/new/example.npz"], evidence=["artifacts/new/summary.json"], update_latest=True)
        self.assertEqual(one, two)
        first = self.root / "release-one" / one["zip"]
        self.assertEqual(first.read_bytes(), (self.root / "release-two" / two["zip"]).read_bytes())
        self.assertEqual(hashlib.sha256(first.read_bytes()).hexdigest(), one["zip_sha256"])
        self.assertEqual(json.loads((self.root / "release-one/latest.json").read_text()), one)
        with zipfile.ZipFile(first) as archive:
            manifest = json.loads(archive.read(packaging.MANIFEST_MEMBER))
            self.assertEqual(set(archive.namelist()), set(manifest["files"]) | {packaging.MANIFEST_MEMBER})
            self.assertEqual(manifest["evidence"][0]["reported_status"], {"pass": False})
            self.assertEqual(set(manifest["runtime_sha256"]), set(packaging.RUNTIME_FILES + packaging.OPTIONAL_RUNTIME_FILES))
            self.assertFalse(any("__pycache__" in n or ".env" in n for n in archive.namelist()))
            extracted = self.root / "unpacked"
            archive.extractall(extracted)
            for name, record in manifest["files"].items():
                self.assertEqual(hashlib.sha256((extracted / name).read_bytes()).hexdigest(), record["sha256"])

    def test_escaping_hidden_cache_and_binary_paths_are_rejected(self):
        for name in ("../escape.json", "..\\escape.json", "C:/secret.json", "/tmp/secret.json", ".git/config", "tests/__pycache__/cache.py", "."):
            with self.subTest(name=name), self.assertRaises(packaging.ReviewPackagingError):
                packaging.collect_files(self.root, includes=[name])
        self.write("artifacts/run.cubin", "binary")
        with self.assertRaisesRegex(packaging.ReviewPackagingError, "not allowlisted"):
            packaging.collect_files(self.root, includes=["artifacts/run.cubin"])

    def test_required_input_missing_and_size_limit_fail_before_publication(self):
        (self.root / "SUBMISSION.md").unlink()
        with self.assertRaisesRegex(packaging.ReviewPackagingError, "SUBMISSION.md"):
            packaging.package_review(self.root, self.root / "release")
        self.assertFalse((self.root / "release").exists())
        self.write("SUBMISSION.md", "source\n")
        with self.assertRaisesRegex(packaging.ReviewPackagingError, "exceeds"):
            packaging.package_review(self.root, self.root / "release", max_bytes=1)
        self.assertFalse((self.root / "release").exists())

    def test_links_and_changed_sources_are_rejected(self):
        original_check = Path.is_symlink
        target = self.root / "tests/example.md"
        with patch.object(Path, "is_symlink", lambda path: path == target or original_check(path)):
            with self.assertRaisesRegex(packaging.ReviewPackagingError, "links and junctions"):
                packaging.collect_files(self.root)
        original_verify = packaging.verify_archive
        def mutate_after_verification(*args):
            original_verify(*args)
            self.write("compile.py", "changed during packaging\n")
        with patch.object(packaging, "verify_archive", mutate_after_verification):
            with self.assertRaisesRegex(packaging.ReviewPackagingError, "source changed"):
                packaging.package_review(self.root, self.root / "release")
        self.assertFalse((self.root / "release").exists())

    def test_evidence_requires_json_object_and_retains_exact_bytes(self):
        report = self.write("artifacts/report.json", "[]\n")
        with self.assertRaisesRegex(packaging.ReviewPackagingError, "JSON object"):
            packaging.collect_files(self.root, evidence=["artifacts/report.json"])
        report.write_bytes(b'\xef\xbb\xbf{ "pass": true }\r\n')
        files, manifest = packaging.collect_files(self.root, evidence=["artifacts/report.json"])
        self.assertEqual(files["artifacts/report.json"], report.read_bytes())
        self.assertEqual(manifest["evidence"][0]["sha256"], hashlib.sha256(report.read_bytes()).hexdigest())

    def test_retained_archive_cannot_be_replaced(self):
        first = packaging.package_review(self.root, self.root / "release")
        path = self.root / "release" / first["zip"]
        path.write_bytes(b"different retained bytes")
        with self.assertRaisesRegex(packaging.ReviewPackagingError, "refusing to overwrite"):
            packaging.package_review(self.root, self.root / "release")
        self.assertEqual(path.read_bytes(), b"different retained bytes")


if __name__ == "__main__":
    unittest.main()
