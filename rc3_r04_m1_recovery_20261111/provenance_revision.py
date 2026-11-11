"""NOT USED BY THE FAILED SCIENTIFIC ATTEMPT: source-module classifier repair.

This additive candidate preserves the failed bound provenance.py unchanged.
Only standard-library synthetic engineering may exercise it under a separately
registered remaining engineering budget. It does not authorize another M1 run.
"""
from __future__ import annotations

import importlib.machinery
from pathlib import Path
import sys
import types

from .provenance import TracedProvenance


def _project_path(path, root):
    if not isinstance(path, (str, bytes, Path)) or isinstance(path, bytes):
        return None
    value=Path(path)
    # A source spec/loader may supply a relative path, but only explicit source
    # provenance permits rooting it. Virtual __file__ labels alone never do.
    resolved=(value if value.is_absolute() else root/value).resolve()
    if resolved.suffix.lower()!='.py' or not resolved.is_relative_to(root):
        return None
    if any(part in ('.venv','site-packages','__pycache__') for part in resolved.relative_to(root).parts):
        return None
    return resolved


def classify_project_source(module, root):
    """Identify actual project Python sources without invoking module hooks.

    Real project sources require a ModuleSpec and SourceFileLoader whose paths
    agree. A real in-project __file__ with missing/unsupported provenance fails
    closed; it is not silently ignored. Unregistered valid source modules are
    returned for the existing exact-SHA allowlist to reject.
    """
    root=Path(root).resolve()
    if not isinstance(module,types.ModuleType):
        return dict(kind='skip',reason='not_a_python_module_object')
    # Avoid getattr(module, '__file__'/'__spec__') and subclass __getattr__.
    namespace=types.ModuleType.__getattribute__(module,'__dict__')
    spec=namespace.get('__spec__')
    hint=namespace.get('__file__')
    hint_project=_project_path(hint,root)
    hint_real=hint_project is not None and hint_project.is_file()
    if not isinstance(spec,importlib.machinery.ModuleSpec):
        if hint_real:
            raise ValueError('Real project source has no ModuleSpec: '+str(hint_project))
        return dict(kind='skip',reason='virtual_or_non_source_namespace_without_ModuleSpec')
    loader=spec.loader
    if not isinstance(loader,importlib.machinery.SourceFileLoader):
        origin_project=_project_path(spec.origin,root)
        if hint_real or origin_project is not None:
            raise ValueError('Project source has unsupported source loader: '+str(origin_project or hint_project))
        return dict(kind='skip',reason='non_SourceFileLoader_namespace_builtin_or_extension')
    # Invoke the standard loader method directly, not a namespace/module hook.
    loader_filename=importlib.machinery.SourceFileLoader.get_filename(loader,spec.name)
    origin_project=_project_path(spec.origin,root)
    loader_project=_project_path(loader_filename,root)
    if origin_project is None and loader_project is None and not hint_real:
        return dict(kind='skip',reason='external_source_module')
    if origin_project is None or loader_project is None or origin_project!=loader_project:
        raise ValueError('Project ModuleSpec origin and SourceFileLoader path disagree')
    if hint is not None and hint_project!=origin_project:
        raise ValueError('Project module __file__ and source specification disagree')
    if not spec.has_location:
        raise ValueError('Project source ModuleSpec has no file location')
    if not origin_project.is_file():
        raise ValueError('Project source specification does not identify a real file: '+str(origin_project))
    return dict(kind='source',path=origin_project,module_name=spec.name,
                reason='real_ModuleSpec_SourceFileLoader_consistent_source')


class RevisedTracedProvenance(TracedProvenance):
    """Candidate only: preserve exact-SHA failure for real unregistered sources."""
    def check_loaded_modules(self, modules=None):
        actual=sys.modules if modules is None else modules
        checked=[];skipped={}
        for name,module in list(actual.items()):
            result=classify_project_source(module,self.root)
            if result['kind']!='source':
                reason=result['reason'];skipped[reason]=skipped.get(reason,0)+1
                continue
            source=result['path']
            self.hash_file(source)
            relative=source.relative_to(self.root).as_posix()
            self.imported.add(relative);checked.append(dict(module=name,path=relative))
        self.emit('loaded_modules_checked',count=len(checked),skipped_by_reason=skipped,
                  classifier_revision='ModuleSpec_SourceFileLoader_v1',used_for_scientific_attempt=False)
        return checked
