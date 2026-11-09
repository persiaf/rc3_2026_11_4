"""Explicit scientific model identity, independent of generic pipeline kind."""
from .common import require


def actor_identity(role, entry):
    require(entry['role'] == entry['cursor'] == entry['provenance']['role'] == role,
            'Scientific role/actor/source conflict')
    update = 256 if role == 'SL256' else 128
    require(role in ('SL256', 'B128', 'C128', 'G128'), 'Unknown frozen endpoint')
    require(entry['provenance']['training_checkpoint_path'].endswith(f'/update_{update:04}.pt'),
            'Intermediate or foreign checkpoint refused')
    return dict(study_method=role, checkpoint_role=role, actor_file=entry['path'],
        actor_sha256=entry['checkpoint_sha256'], tensor_sha256=entry['model_tensor_sha256'],
        training_run=entry['provenance']['training_checkpoint_path'].rsplit('/checkpoints/', 1)[0]
            if role != 'SL256' else entry['provenance']['training_checkpoint_path'].rsplit('/', 1)[0],
        fixed_update=update, training_checkpoint=entry['provenance']['training_checkpoint_path'],
        training_checkpoint_sha256=entry['provenance']['training_checkpoint_sha256'])


def label_result(row, role):
    require(row['request']['study_method'] == role, 'Request/study method conflict')
    row['pipeline_method'] = row['request']['method']
    if role in ('Fixed3', 'Detour16'):
        require(row['weight'] is None, 'Native method acquired neural weights')
        row.update(study_method=role, checkpoint_role=None, actor_file=None,
                   actor_sha256=None, tensor_sha256=None, training_run=None, fixed_update=None)
    else:
        row.update(actor_identity(role, row['weight']))
    row['training_weight_role'] = row['checkpoint_role']
    row['plan']['pipeline_method'] = row['plan']['method']
    row['plan']['method'] = role
    row['plan']['model_identity'] = {k: row[k] for k in
        ('study_method','checkpoint_role','actor_file','actor_sha256','tensor_sha256','training_run','fixed_update')}
    return row
