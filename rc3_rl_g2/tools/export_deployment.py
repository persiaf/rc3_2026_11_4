"""Export the predeclared final policies without optimizer or worker state."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]

def main():
    import torch
    from rc3_rl_g2.checkpoints import load_model,save_model,sha256
    torch.set_num_threads(1)
    receipt=json.loads((ROOT/'outputs/training/ppo_receipt.json').read_text(encoding='utf-8'))
    if receipt.get('actual_actions')!=32768 or not receipt.get('completed'):
        raise ValueError('Completed registered PPO run required')
    destination=ROOT/'outputs/deployment';destination.mkdir(parents=True,exist_ok=True)
    outputs=[]
    for source_name,target_name,role in [
        ('ppo_32768.pt','policy_ppo.pt','Predeclared 32768-action research endpoint; not selected by validation'),
        ('bc_initial.pt','policy_bc_frozen.pt','Same initial BC used by both continuation branches'),
        ('bc_continued_2048.pt','policy_bc_continued.pt','Predeclared 2048-extra-update supervised control')]:
        source=ROOT/'outputs/training'/source_name
        model,payload=load_model(source,'cpu');model.eval()
        metadata={k:payload[k] for k in ['protocol_sha256','parent_bc_sha256','actual_actions','optimizer_updates','updates'] if k in payload}
        metadata.update(role=role,source_checkpoint_sha256=sha256(source),
                        scope='Static 2D proxy reachability; local-search selector; no physical IK',
                        export_contains='Architecture and model parameters only; no optimizer, RNG or worker state')
        if source_name.startswith('ppo_') and (ROOT/'outputs/interruption_receipt.json').exists():
            metadata.update(interruption_receipt_sha256=sha256(ROOT/'outputs/interruption_receipt.json'),
                action_accounting='32768 persisted on-policy actions in the resumed training lineage; an additional interrupted uncommitted tail of 0..256 physical calls is unknown')
        target=destination/target_name;digest=save_model(target,model,**metadata)
        loaded,exported=load_model(target,'cpu')
        equal=all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())
        if not equal:raise AssertionError('Export changed model tensors')
        outputs.append(dict(path=str(target.relative_to(ROOT)),bytes=target.stat().st_size,sha256=digest,
            source_checkpoint_sha256=sha256(source),parameters=sum(p.numel() for p in model.parameters()),
            tensor_equality=True,architecture=model.architecture(),role=role))
    (destination/'export_receipt.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(outputs,ensure_ascii=False))

if __name__=='__main__':main()
