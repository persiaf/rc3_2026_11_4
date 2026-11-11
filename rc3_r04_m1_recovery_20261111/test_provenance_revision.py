"""Standard-library-only synthetic tests; do not run before engineering debit.

No torch import, model, optimizer, saved trajectory or scientific retry occurs.
"""
import hashlib
import importlib.machinery
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest

from .provenance_revision import RevisedTracedProvenance,classify_project_source


class VirtualOps(types.ModuleType):
    __file__='_ops.py'
    def __getattr__(self,name):
        raise AssertionError('Dynamic module hook must not be invoked: '+name)


class VirtualClasses(types.ModuleType):
    __file__='_classes.py'
    def __getattr__(self,name):
        raise AssertionError('Dynamic module hook must not be invoked: '+name)


class ModuleProvenanceRevisionTests(unittest.TestCase):
    def setUp(self):
        parent=Path(__file__).resolve().parent
        self.root=Path(tempfile.mkdtemp(prefix='synthetic_module_identity_',dir=parent)).resolve()
        assert self.root.is_relative_to(parent)
        self.output=self.root/'trace';self.output.mkdir()
        self.source=self.root/'registered.py';self.source.write_bytes(b'VALUE = 1\n')
        self.other=self.root/'unregistered.py';self.other.write_bytes(b'VALUE = 2\n')
        self.manifest=dict(source_sha256={'registered.py':hashlib.sha256(self.source.read_bytes()).hexdigest()},artifact_sha256={})

    def tearDown(self):
        # Only individually enumerated synthetic files are removed; no recursive
        # delete, no old evidence or unregistered workspace path is touched.
        assert self.root.is_relative_to(Path(__file__).resolve().parent)
        for path in self.output.iterdir():
            assert path.is_file() and path.resolve().is_relative_to(self.root)
            path.unlink()
        self.output.rmdir();self.source.unlink();self.other.unlink();self.root.rmdir()

    def module(self,name,path):
        spec=importlib.util.spec_from_file_location(name,str(path))
        value=types.ModuleType(name);value.__dict__.update(__spec__=spec,__loader__=spec.loader,__file__=str(path))
        return value

    def verifier(self):
        return RevisedTracedProvenance(self.root,self.output,self.manifest,lambda:None,'synthetic-engineering-only')

    def test_virtual_torch_style_ops_class_attr_not_a_repo_source(self):
        module=VirtualOps('torch.ops')
        self.assertEqual(classify_project_source(module,self.root)['kind'],'skip')
        self.assertEqual(self.verifier().check_loaded_modules({'torch.ops':module}),[])

    def test_virtual_torch_style_classes_class_attr_not_a_repo_source(self):
        module=VirtualClasses('torch.classes')
        self.assertEqual(classify_project_source(module,self.root)['kind'],'skip')
        self.assertEqual(self.verifier().check_loaded_modules({'torch.classes':module}),[])

    def test_virtual_instance_label_without_file_is_not_source(self):
        module=VirtualOps('virtual');module.__dict__['__file__']='does_not_exist.py'
        self.assertEqual(classify_project_source(module,self.root)['kind'],'skip')

    def test_real_registered_SourceFileLoader_is_hashed(self):
        verifier=self.verifier();source=self.module('registered',self.source)
        self.assertEqual(verifier.check_loaded_modules({'registered':source}),[dict(module='registered',path='registered.py')])
        self.assertEqual(verifier.verified, self.manifest['source_sha256'])

    def test_real_unregistered_SourceFileLoader_still_rejected(self):
        source=self.module('unregistered',self.other)
        with self.assertRaisesRegex(ValueError,'Unregistered actual dependency'):
            self.verifier().check_loaded_modules({'unregistered':source})

    def test_real_project_source_missing_spec_fails_closed(self):
        module=types.ModuleType('bad');module.__dict__['__file__']=str(self.source)
        with self.assertRaisesRegex(ValueError,'no ModuleSpec'):classify_project_source(module,self.root)

    def test_real_project_source_unsupported_loader_fails_closed(self):
        module=self.module('bad',self.source);module.__dict__['__spec__'].loader=None
        with self.assertRaisesRegex(ValueError,'unsupported source loader'):classify_project_source(module,self.root)

    def test_source_origin_loader_mismatch_fails_closed(self):
        module=self.module('bad',self.source);module.__dict__['__spec__'].origin=str(self.other)
        with self.assertRaisesRegex(ValueError,'disagree'):classify_project_source(module,self.root)

    def test_changed_registered_source_bytes_fail_exact_hash(self):
        self.source.write_bytes(b'VALUE = 3\n')
        with self.assertRaisesRegex(ValueError,'Source identity differs'):
            self.verifier().check_loaded_modules({'registered':self.module('registered',self.source)})


if __name__=='__main__':unittest.main()
