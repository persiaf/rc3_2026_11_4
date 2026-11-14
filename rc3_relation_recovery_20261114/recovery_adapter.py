"""Future authorized M2a integration, no CLI and no automatic science dispatch.

The proposal and two new approvals must be bound before any restore. The outer
supervisor and its remaining-wall enforcement are specified in the proposal;
the current turn never calls these model/update hooks.
"""
from pathlib import Path
from .persistence import read, sha, digest
from .runtime_boundary import require_recovery_authorization, commit_update

ROOT = Path(__file__).resolve().parents[1]


def checked_ref(ref):
    path = (ROOT / ref['path']).resolve()
    if not path.is_relative_to(ROOT) or sha(path) != ref['sha256']:
        raise ValueError('Changed bound recovery reference')
    return path


def restore_sl238(actor, opt, ledger, proposal, approval, stage_authorization):
    require_recovery_authorization(proposal, approval, stage_authorization)
    audit = read(checked_ref(proposal['sl238_audit']))
    for name, expected in read(checked_ref(proposal['new_source_binding']))['files'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Recovery implementation changed after approval: ' + name)
    from .runtime import state_digest, set_rng, rng_state, tensor_hash
    import torch
    saved = torch.load(checked_ref(audit['checkpoint']), map_location='cpu', weights_only=False)
    if (ledger.identity != saved['identity'] or ledger.cursor != 238 or saved['cursor'] != 238
            or saved['engineering_only'] is not False or saved['config'] != actor.config
            or ledger.pending or ledger.counts != audit['original_counts']
            or set(ledger.ids) - set(saved['request_ids']) != set(audit['paid_after_checkpoint'])
            or not set(saved['request_ids']).issubset(ledger.ids)):
        raise ValueError('Exact historical model/counter/query boundary required')
    if (state_digest(saved['optimizer']) != audit['checkpoint']['optimizer_sha256']
            or state_digest(saved['rng']) != audit['checkpoint']['rng_sha256']):
        raise ValueError('Archived Adam/RNG bytes differ')
    actor.load_state_dict(saved['model'], strict=True)
    opt.load_state_dict(saved['optimizer'])
    opt.zero_grad(set_to_none=True)
    set_rng(saved['rng'])
    if (tensor_hash(actor) != audit['checkpoint']['model_sha256']
            or state_digest(opt.state_dict()) != audit['checkpoint']['optimizer_sha256']
            or state_digest(rng_state()) != audit['checkpoint']['rng_sha256']):
        raise ValueError('Restored state differs')
    # Ledger counts and consumed IDs deliberately remain 1910 and all old IDs.
    return saved


def save_update_boundary(actor, opt, ledger, cursor):
    from .runtime import checkpoint
    ref = checkpoint(actor, opt, ledger, ledger.folder / f'current_slot{cursor % 2}.pt',
                     cursor, engineering_only=False)
    state = dict(schema='relation-M2a-recovery-state-v1', cursor=cursor, checkpoint=ref,
                 checkpoint_sha256=ref['sha256'], next_action='COMPLETE' if cursor == 512 else 'UPDATE')
    commit_update(ledger, cursor, ROOT / ref['path'], state)
    return state


def original_origin(audit, original_events):
    """Pure metadata construction; not a reset and not an execution permit."""
    if (audit['original_unique_request_debits'] != 1910 or audit['committed_teacher_debits'] != 1904
            or audit['original_counts']['optimizer_completed'] != 238):
        raise ValueError('Unexpected archived origin')
    ids = [event['id'] for event in original_events if event['event'] == 'begin']
    if len(ids) != len(set(ids)) or len(ids) != 1910 + 238:
        raise ValueError('Original query journal differs')
    return dict(counts=audit['original_counts'], ids=sorted(ids), cursor=238, elapsed_s=audit['historical_wall_s'],
                original_resource=audit['resource'], original_events=audit['journal'],
                historical_overlap_preserved=audit['overlapping_complete_failed_id'])
