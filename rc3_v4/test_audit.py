"""Integrity regression tests: false hashes must fail the aggregate and CLI."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from .audit import audit_manifest, main, sha256


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'source.py'
        self.source.write_text('original\n', encoding='utf-8')
        self.manifest = self.root/'manifest.json'
        self.write_manifest({'source_sha256': {'source.py': sha256(self.source)}})

    def write_manifest(self, value):
        self.manifest.write_text(json.dumps(value), encoding='utf-8')

    def audit(self, **kwargs):
        return audit_manifest(self.manifest, self.root, **kwargs)

    def test_valid_source_passes(self):
        result = self.audit()
        self.assertTrue(result['all_passed'])
        self.assertEqual(result['matched_files'], 1)

    def test_changed_source_fails_report_and_cli(self):
        self.source.write_text('changed\n', encoding='utf-8')
        self.assertFalse(self.audit()['all_passed'])
        output = self.root/'report.json'
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(['--manifest', str(self.manifest), '--root', str(self.root), '--output', str(output)])
        self.assertEqual(code, 2)
        report = json.loads(output.read_text(encoding='utf-8'))
        self.assertFalse(report['all_passed'])
        self.assertTrue(report['errors'])

    def test_missing_file_and_malformed_manifest_fail(self):
        self.source.unlink()
        self.assertFalse(self.audit()['all_passed'])
        self.manifest.write_text('{bad json', encoding='utf-8')
        self.assertFalse(self.audit()['all_passed'])

    def test_empty_or_wrong_schema_fails(self):
        for value in ({}, {'files': []}, {'source_sha256': {}}, {'source_sha256': []}):
            self.write_manifest(value)
            self.assertFalse(self.audit()['all_passed'])

    def test_package_checks_bytes_and_no_traversal(self):
        self.write_manifest({'files': [{'path': 'source.py', 'sha256': sha256(self.source), 'bytes': 0}]})
        self.assertFalse(self.audit()['all_passed'])
        self.write_manifest({'files': [{'path': '../outside', 'sha256': '0'*64}]})
        result = self.audit()
        self.assertFalse(result['all_passed'])
        self.assertIn('traversal', result['errors'][0])

    def test_explicit_windows_root_relocation(self):
        self.write_manifest({'source_sha256': {r'Z:\old-project\source.py': sha256(self.source)}})
        self.assertFalse(self.audit()['all_passed'])
        self.assertTrue(self.audit(relocate_from=r'Z:\old-project')['all_passed'])

    def test_old_false_source_check_cannot_pass(self):
        prior = self.root/'old-report.json'
        prior.write_text(json.dumps({'all_passed': True, 'errors': [],
            'source_checks': [{'path': 'source.py', 'unchanged': False}]}), encoding='utf-8')
        self.assertFalse(self.audit(prior_report=prior)['all_passed'])


if __name__ == '__main__':
    unittest.main()
