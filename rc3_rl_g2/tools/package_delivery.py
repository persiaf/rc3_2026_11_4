"""Build/verify a portable G2 source-and-evidence ZIP; smoke is opt-in.

The builder only reads source files. Extraction is exclusive and restricted to
this workspace's rc3_rl_g2/outputs/portable_test, never the source root.
No Torch, NumPy or planning code is imported by build/extract/verify.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
PORTABLE_ROOT = ROOT / 'outputs' / 'portable_test'
MANIFEST = 'DELIVERY_MANIFEST.json'
DEPENDENCIES = 'FROZEN_DEPENDENCIES.json'
EXCLUDED_DIRS = {'.venv', '__pycache__', 'prepared_train', 'portable_test', 'portable_check', '.git'}
EXCLUDED_SUFFIXES = {'.pyc', '.pyo', '.pending', '.pend', '.tmp', '.lock'}
MAX_MEMBER_BYTES = 1024 ** 3
MAX_TOTAL_BYTES = 10 * 1024 ** 3


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def encoded(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def member_name(name):
    if not isinstance(name, str) or '\\' in name or ':' in name:
        raise ValueError('invalid package member name')
    p = PurePosixPath(name)
    if p.is_absolute() or any(part in ('', '.', '..') for part in name.split('/')):
        raise ValueError('package member path traversal')
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {f'{base}{i}' for base in ('COM', 'LPT') for i in range(1, 10)}
    if any(part.endswith((' ', '.')) or part.split('.')[0].upper() in reserved for part in p.parts):
        raise ValueError('unsafe Windows package member')
    return p.as_posix()


def inside(base, path, allow_equal=False):
    base = Path(base).resolve(); path = Path(path).resolve()
    if not path.is_relative_to(base) or (path == base and not allow_equal):
        raise ValueError(f'path must stay inside {base}: {path}')
    return path


def walk_files(base):
    base = Path(base).resolve()
    for current, dirs, names in os.walk(base, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS)
        for directory in dirs:
            p = Path(current) / directory
            if p.is_symlink() or not p.resolve().is_relative_to(base):
                raise ValueError('symlink/reparse directory is not packaged: ' + str(p))
        for name in sorted(names):
            p = Path(current) / name
            if p.suffix.lower() in EXCLUDED_SUFFIXES:
                continue
            if p.is_symlink() or not p.resolve().is_relative_to(base):
                raise ValueError('symlink/reparse file is not packaged: ' + str(p))
            if p.is_file():
                yield p


def manifest_entries(value, prefix=''):
    result = {}
    for item in value['files']:
        name = member_name(prefix + item['path'])
        if name in result:
            raise ValueError('duplicate frozen manifest path')
        result[name] = item
    return result


def frozen_inventory():
    before = read(ROOT / 'outputs/freeze_before.json')
    archives = {}
    for name in ('rc3_v5_delivery.zip', 'rc3_rl_research/outputs/rc3_rl_research_g01.zip'):
        path = inside(WORKSPACE, WORKSPACE / name)
        actual = digest(path)
        if actual != before['archive_sha256'][name]:
            raise ValueError('frozen archive changed: ' + name)
        archives[name] = actual
    with zipfile.ZipFile(WORKSPACE / 'rc3_v5_delivery.zip') as archive:
        v5_raw = archive.read('PACKAGE_MANIFEST.json')
    with zipfile.ZipFile(WORKSPACE / 'rc3_rl_research/outputs/rc3_rl_research_g01.zip') as archive:
        g01_raw = archive.read('rc3_rl_research/PACKAGE_MANIFEST.json')
    if g01_raw != (WORKSPACE / 'rc3_rl_research/PACKAGE_MANIFEST.json').read_bytes():
        raise ValueError('G01 manifest differs from its frozen archive')
    inventory = manifest_entries(json.loads(v5_raw))
    inventory.update(manifest_entries(json.loads(g01_raw), 'rc3_rl_research/'))
    python_inventory = {name: item for name, item in inventory.items()
                        if name.endswith('.py') and name.split('/')[0].startswith('rc3_')}
    metadata = {'delivery_metadata/frozen_v5_PACKAGE_MANIFEST.json': v5_raw,
                'delivery_metadata/frozen_g01_PACKAGE_MANIFEST.json': g01_raw}
    return python_inventory, archives, metadata


def import_roots(path):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
        elif isinstance(node, ast.Call) and node.args:
            is_import = (isinstance(node.func, ast.Name) and node.func.id == '__import__') or (
                isinstance(node.func, ast.Attribute) and node.func.attr == 'import_module')
            if is_import and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                names.append(node.args[0].value)
        roots.update(name.split('.')[0] for name in names if name.startswith('rc3_'))
    return roots


def discover_dependencies(g2_files, inventory):
    roots = set()
    for path in g2_files:
        if path.suffix == '.py':
            roots.update(import_roots(path))
    roots.discard('rc3_rl_g2'); done = set(); selected = {}; edges = {}
    while roots - done:
        package = sorted(roots - done)[0]
        names = sorted(name for name in inventory if name.split('/')[0] == package)
        if not names:
            raise ValueError('local dependency lacks frozen source inventory: ' + package)
        dependencies = set()
        for name in names:
            path = inside(WORKSPACE, WORKSPACE / name)
            if not path.is_file():
                raise ValueError('missing frozen dependency: ' + name)
            item = inventory[name]
            actual = digest(path)
            if actual != item['sha256'] or path.stat().st_size != item['bytes']:
                raise ValueError('frozen dependency bytes changed: ' + name)
            selected[name] = path
            dependencies.update(import_roots(path))
        dependencies -= {package, 'rc3_rl_g2'}
        edges[package] = sorted(dependencies); roots.update(dependencies); done.add(package)
    return selected, edges


def final_guards():
    receipt = read(ROOT / 'outputs/training/ppo_receipt.json')
    if receipt.get('completed') is not True or receipt.get('actual_actions') != 32768:
        raise ValueError('packaging requires the completed registered 32768-action run')
    formal = read(ROOT / 'outputs/formal_source_manifest.json')
    if len(formal['source_sha256']) != 18:
        raise ValueError('expected 18 bound production sources')
    for name, expected in formal['source_sha256'].items():
        if digest(inside(ROOT, ROOT / name)) != expected:
            raise ValueError('formal production source changed: ' + name)
    freeze = read(ROOT / 'outputs/freeze_after.json')
    if freeze.get('all_passed') is not True or freeze.get('phase') != 'final':
        raise ValueError('run tools.freeze_audit --phase final before final packaging')
    return receipt


PORTABLE_README = '''# G2 可迁移源码、模型与证据包

解压根目录同时包含 rc3_rl_g2 与所需七个旧依赖包。旧依赖仅复制原清单中的
Python 源码原字节，不含旧训练二进制、旧完整实验和机器人网格。FROZEN_DEPENDENCIES.json
保存逐文件 SHA256、原归档身份及依赖闭包；DELIVERY_MANIFEST.json 保存全部 payload
的 SHA256/大小（不对清单自身循环求哈希）。原始两份冻结清单置于 delivery_metadata。

在装有 NumPy、SciPy、PyTorch、threadpoolctl 的 Python 环境中，从解压根目录运行：

    python -B -m rc3_rl_g2.tools.package_delivery verify --directory .
    python -B -m rc3_rl_g2.runtime_g2 --checkpoint rc3_rl_g2/outputs/training/ppo_32768.pt --nx 3 --ny 3 --seed 20261103 --output my_plan.json
    python -B -m rc3_rl_g2.tools.audit_training --output my_audit.json

训练入口及完整接口见 rc3_rl_g2/README_G2.md。已有训练账本应保留，复现完整训练应
在另一个工作副本中按协议组织新输出；模型/训练检查点保留全部源码绑定。

verify 是当前交付子集的完整性复验。原 tools.freeze_audit 的四个归档和 G01 全部
279 payload 检查需要原工作区；本包保留其最终回执，不冒称携带旧完整实验。
准备缓存、虚拟环境、__pycache__、临时文件和打包自身均排除。历史 debug 记录保留其
原目录和性质，不纳入正式训练动作数。只从可信来源读取本项目 pickle/PyTorch 文件。
'''


def build(archive_path):
    final_guards()
    archive_path = inside(WORKSPACE, archive_path)
    if archive_path.suffix.lower() != '.zip' or archive_path.exists():
        raise ValueError('new .zip output required; existing packages are not overwritten')
    if archive_path.is_relative_to(PORTABLE_ROOT.resolve()):
        raise ValueError('archive output cannot be inside the portable extraction tree')
    receipt_path = ROOT / 'outputs/delivery_package_receipt.json'
    files = [p for p in walk_files(ROOT) if p.resolve() not in
             {archive_path, receipt_path.resolve()} and not p.name.startswith('rc3_rl_g2_delivery')]
    inventory, archives, generated = frozen_inventory()
    dependencies, edges = discover_dependencies(files, inventory)
    sources = {p.relative_to(WORKSPACE).as_posix(): p for p in files}
    sources.update(dependencies)
    dependency_rows = [dict(path=name, bytes=p.stat().st_size, sha256=digest(p),
                           frozen_manifest_sha256=inventory[name]['sha256'], original_bytes=True)
                       for name, p in sorted(dependencies.items())]
    generated[DEPENDENCIES] = encoded(dict(schema='rc3-g2-frozen-source-subset-v1',
        scope='Unmodified original Python byte copies only; historical experiment binaries omitted.',
        frozen_archives=archives, dependency_packages=sorted(edges), dependency_edges=edges,
        files=dependency_rows))
    generated['PORTABLE_README.md'] = PORTABLE_README.encode('utf-8')
    rows = [dict(path=name, bytes=path.stat().st_size, sha256=digest(path),
                 kind='frozen_dependency_source' if name in dependencies else 'g2_source_or_evidence')
            for name, path in sorted(sources.items())]
    rows.extend(dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), kind='package_metadata')
                for name, raw in sorted(generated.items()))
    rows.sort(key=lambda row: row['path'])
    manifest = dict(schema='rc3-g2-portable-delivery-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        files=rows, payload_count=len(rows), payload_bytes=sum(row['bytes'] for row in rows),
        manifest_self_hash_excluded=True, excludes=sorted(EXCLUDED_DIRS | EXCLUDED_SUFFIXES),
        excluded_other=['archive itself', 'delivery_package_receipt.json', 'prior rc3_rl_g2_delivery* files'],
        dependency_copy_policy='Frozen Python sources copied byte-for-byte; no legacy binary experiment data.',
        completed_ppo_actions=32768)
    if manifest['payload_bytes'] > MAX_TOTAL_BYTES or any(row['bytes'] > MAX_MEMBER_BYTES for row in rows):
        raise ValueError('delivery exceeds explicit extraction size limits')
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    pending = archive_path.with_name(archive_path.name + '.' + uuid.uuid4().hex + '.pending')
    with zipfile.ZipFile(pending, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for row in rows:
            name = member_name(row['path']); h = hashlib.sha256(); size = 0
            with z.open(name, 'w', force_zip64=True) as target:
                if name in generated:
                    raw = generated[name]; target.write(raw); h.update(raw); size = len(raw)
                else:
                    with sources[name].open('rb') as source:
                        for chunk in iter(lambda: source.read(1024 * 1024), b''):
                            target.write(chunk); h.update(chunk); size += len(chunk)
            if h.hexdigest() != row['sha256'] or size != row['bytes']:
                raise ValueError('source mutated during packaging: ' + name)
        z.writestr(MANIFEST, encoded(manifest))
    # Windows rename fails if another publisher won the same output path.
    if archive_path.exists():
        raise FileExistsError(archive_path)
    pending.rename(archive_path)
    directory = extract(archive_path)
    result = dict(schema='rc3-g2-package-receipt-v1', archive=str(archive_path),
        sha256=digest(archive_path), bytes=archive_path.stat().st_size,
        payload_count=len(rows), payload_bytes=manifest['payload_bytes'],
        frozen_dependency_files=len(dependency_rows), frozen_dependency_packages=sorted(edges),
        portable_directory=str(directory), payload_verified=True, smoke_executed=False)
    receipt_path.write_bytes(encoded(result))
    return result


def checked_rows(manifest):
    if manifest.get('schema') != 'rc3-g2-portable-delivery-v1':
        raise ValueError('unknown delivery manifest')
    rows = {}; folded = set()
    for row in manifest['files']:
        name = member_name(row['path'])
        if name == MANIFEST or name.casefold() in folded:
            raise ValueError('duplicate/self-referential delivery path')
        if type(row['bytes']) is not int or not 0 <= row['bytes'] <= MAX_MEMBER_BYTES:
            raise ValueError('invalid delivery member size')
        if not isinstance(row['sha256'], str) or len(row['sha256']) != 64:
            raise ValueError('invalid delivery member digest')
        rows[name] = row; folded.add(name.casefold())
    if len(rows) != manifest['payload_count'] or sum(r['bytes'] for r in rows.values()) != manifest['payload_bytes']:
        raise ValueError('delivery counts disagree')
    if manifest['payload_bytes'] > MAX_TOTAL_BYTES:
        raise ValueError('delivery too large')
    return rows


def verify(directory):
    directory = Path(directory).resolve()
    rows = checked_rows(read(directory / MANIFEST))
    for name, row in rows.items():
        path = inside(directory, directory / name)
        if not path.is_file() or path.is_symlink() or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('payload does not match manifest: ' + name)
    dependency = read(directory / DEPENDENCIES)
    for row in dependency['files']:
        if row['original_bytes'] is not True or row['sha256'] != row['frozen_manifest_sha256']:
            raise ValueError('legacy source is not a frozen byte copy')
        if rows[row['path']]['sha256'] != row['sha256']:
            raise ValueError('dependency and payload manifests disagree')
    return dict(all_passed=True, payload_count=len(rows), frozen_dependency_files=len(dependency['files']),
                directory=str(directory), scope='Included payload integrity; not the omitted full legacy archives/experiments.')


def extract(archive_path):
    archive_path = Path(archive_path).resolve()
    inside(ROOT, PORTABLE_ROOT)
    archive_sha = digest(archive_path)
    destination = inside(PORTABLE_ROOT, PORTABLE_ROOT / (archive_path.stem + '-' + archive_sha[:12]))
    if destination.exists():
        raise FileExistsError('exclusive extraction directory already exists: ' + str(destination))
    with zipfile.ZipFile(archive_path) as z:
        members = z.infolist(); names = [member_name(item.filename) for item in members]
        if len({n.casefold() for n in names}) != len(names):
            raise ValueError('duplicate archive member')
        if MANIFEST not in names or z.getinfo(MANIFEST).file_size > 16 * 1024 * 1024:
            raise ValueError('missing or oversized delivery manifest')
        manifest = json.loads(z.read(MANIFEST)); rows = checked_rows(manifest)
        if set(names) != set(rows) | {MANIFEST}:
            raise ValueError('archive members differ from payload manifest')
        for item in members:
            if stat.S_ISLNK(item.external_attr >> 16) or item.is_dir():
                raise ValueError('archive links/directories are not accepted')
            if item.filename != MANIFEST and item.file_size != rows[item.filename]['bytes']:
                raise ValueError('archive member size differs')
        # Only after all names and declared sizes pass do any filesystem writes occur.
        PORTABLE_ROOT.mkdir(parents=True, exist_ok=True); destination.mkdir(exist_ok=False)
        for item in members:
            path = inside(destination, destination / item.filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            with z.open(item) as source, path.open('xb') as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
    verify(destination)
    return destination


def validate_plan(path, output):
    # Explicit smoke subcommand only; never imported by packaging or verification.
    from rc3_fast.solver import prepare
    from rc3_baseline.validator import validate
    data = read(path); problem = prepare(data['scene'])
    checks = {name: validate(data['scene'], plan, problem.sparse) for name, plan in data['plans'].items()}
    if not checks or not all(row['valid_model_plan'] for row in checks.values()):
        raise ValueError('independent portable plan validation failed')
    Path(output).write_bytes(encoded(checks))


def smoke(directory, python):
    inside(ROOT, PORTABLE_ROOT)
    directory = inside(PORTABLE_ROOT, directory)
    verified = verify(directory)
    output = inside(directory, directory / ('portable_smoke_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')))
    output.mkdir(exist_ok=False)
    checkpoint = 'rc3_rl_g2/outputs/deployment/policy_ppo.pt'
    packages = read(directory / DEPENDENCIES)['dependency_packages'] + ['rc3_rl_g2']
    provenance_code = ('import importlib,json,pathlib; r=pathlib.Path.cwd().resolve(); result={}; '
        'names=' + repr(packages) + ';\n'
        'for name in names:\n'
        ' m=importlib.import_module(name); paths=[pathlib.Path(m.__file__).resolve()] if m.__file__ else [pathlib.Path(p).resolve() for p in m.__path__]; '
        'assert all(p.is_relative_to(r) for p in paths),(name,paths); result[name]=[str(p) for p in paths]\n'
        'print(json.dumps(result))')
    relative = output.relative_to(directory).as_posix()
    commands = [
        ('import_provenance', ['-c', provenance_code]),
        ('train_help', ['-m', 'rc3_rl_g2.train_ppo', '--help']),
        ('contract_tests', ['-m', 'unittest', 'rc3_rl_g2.test_env_g2', 'rc3_rl_g2.test_collector_contract', '-v']),
        ('runtime_stop', ['-m', 'rc3_rl_g2.runtime_g2', '--method', 'stop', '--nx', '2', '--ny', '2',
                          '--seed', '20261103', '--mode', 'mixed', '--output', relative + '/stop.json']),
        ('runtime_learned', ['-m', 'rc3_rl_g2.runtime_g2', '--checkpoint', checkpoint, '--nx', '2', '--ny', '2',
                             '--seed', '20261103', '--mode', 'mixed', '--output', relative + '/learned.json']),
        ('independent_stop', ['-m', 'rc3_rl_g2.tools.package_delivery', 'validate-plan', '--plan', relative + '/stop.json', '--output', relative + '/stop_check.json']),
        ('independent_learned', ['-m', 'rc3_rl_g2.tools.package_delivery', 'validate-plan', '--plan', relative + '/learned.json', '--output', relative + '/learned_check.json'])]
    env = os.environ.copy(); env['PYTHONPATH'] = str(directory); env['PYTHONNOUSERSITE'] = '1'
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    results = []
    for name, args in commands:
        command = [str(python), '-B'] + args
        completed = subprocess.run(command, cwd=directory, env=env, capture_output=True, timeout=180)
        (output / (name + '.log')).write_bytes(completed.stdout + b'\nSTDERR:\n' + completed.stderr)
        results.append(dict(name=name, command=command, returncode=completed.returncode))
        if completed.returncode:
            (output / 'receipt.json').write_bytes(encoded(dict(all_passed=False, checks=results)))
            raise RuntimeError('portable smoke failed: ' + name + '; see ' + str(output))
    result = dict(all_passed=True, before=verified, after=verify(directory), checks=results,
        output=str(output), scope='Portable imports, training CLI, targeted contracts, tiny STOP/learned plans and fresh validators only; no training or performance rerun.')
    (output / 'receipt.json').write_bytes(encoded(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('build'); p.add_argument('--output', type=Path, default=WORKSPACE / 'rc3_rl_g2_delivery.zip')
    p = commands.add_parser('verify'); p.add_argument('--directory', type=Path, default=Path('.'))
    p = commands.add_parser('extract'); p.add_argument('--archive', type=Path, required=True)
    p = commands.add_parser('smoke'); p.add_argument('--directory', type=Path, required=True); p.add_argument('--python', type=Path, default=Path(sys.executable))
    p = commands.add_parser('validate-plan'); p.add_argument('--plan', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'build': result = build(args.output)
    elif args.command == 'verify': result = verify(args.directory)
    elif args.command == 'extract': result = dict(portable_directory=str(extract(args.archive)))
    elif args.command == 'smoke': result = smoke(args.directory, args.python.resolve())
    else:
        validate_plan(args.plan, args.output); result = dict(all_passed=True, output=str(args.output))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
