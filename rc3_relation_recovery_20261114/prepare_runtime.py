"""Create the explicitly versioned runtime fork; never modify frozen sources."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(__file__).resolve().parent


def main():
    source = ROOT / 'rc3_relation_drl_20261113/runtime.py'
    text = source.read_text(encoding='utf8')
    text = text.replace('from .actor import RelationActor,tensor_hash',
                        'from rc3_relation_drl_20261113.actor import RelationActor,tensor_hash')
    text = text.replace('from .common import ROOT,HOME,H_ID,put,read,sha,digest',
                        'from .common import ROOT,HOME,H_ID,put,read,sha,digest,replace_retry\nfrom .runtime_boundary import persist_request')
    start, end = text.index('def request_record('), text.index('\n\ndef collect(')
    text = text[:start] + '''def request_record(ledger,rid,value,tick):
    value['full_request_s']=persist_request(ledger,rid,value,tick)
    if 'timing' in value:value['timing']['complete_request_s']=value['full_request_s']
    return value
''' + text[end:]
    text = text.replace('os.replace(tmp,path)', 'replace_retry(tmp,path)')
    text = text.replace("f'{rid}_teacher_{i}'", "f'recovery_v1_{rid}_teacher_{i}'")
    destination = HOME / 'runtime.py'
    if destination.exists():
        raise RuntimeError('Versioned fork already exists; edit its explicit patch instead of recreating')
    destination.write_text(text, encoding='utf8', newline='\n')
    old = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(source.read_text(encoding='utf8')).body if isinstance(node, ast.FunctionDef)}
    new = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
    changed = [name for name in old if old[name] != new[name]]
    assert set(changed) == {'request_record', 'sl_update', 'checkpoint'}
    normalized = ast.parse(text.replace("f'recovery_v1_{rid}_teacher_{i}'", "f'{rid}_teacher_{i}'"))
    sl = next(n for n in normalized.body if isinstance(n, ast.FunctionDef) and n.name == 'sl_update')
    assert ast.dump(sl, include_attributes=False) == old['sl_update']
    (HOME / 'RUNTIME_DIFF_REVIEW.json').write_text(json.dumps(dict(
        original='rc3_relation_drl_20261113/runtime.py', fork='rc3_relation_recovery_20261114/runtime.py',
        changed_functions=changed, unchanged_functions=[x for x in old if x not in changed],
        sl_loss_and_backward_AST_unchanged_except_request_ID_prefix=True,
        optimizer_function_AST_unchanged=True, model_import='original exact RelationActor',
        scientific_execution=False), indent=2) + '\n', encoding='utf8')
    print(dict(changed=changed, unchanged=len(old)-len(changed)))


if __name__ == '__main__':
    main()
