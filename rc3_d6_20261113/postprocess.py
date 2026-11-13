"""Independent saved-output arithmetic and D6 report; no neural forward or search."""
import csv,gzip,json,math,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,HOME,D5,read,put,unarchive,sha
from .data import Pool
from rc3_conservative_refinement.runtime import SL_PATH,SL_TENSOR_SHA256,tensor_digest

class ReadOnly:
    def __init__(self):self.counts=Counter();self.end=read(HOME/'PROTOCOL.json')['deadline_unix_s']-300
    def check(self):
        if time.time()>self.end:raise TimeoutError('Original D6 closeout wall boundary')

def csvwrite(name,rows):
    if not rows:return
    with (HOME/name).open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)

def independent_credit(p,r):
    assert r['problem_identity']==p.identity and r['raw_route']==[0,*[a+1 for a in r['actions']],0]
    N=len(p.C);left=np.ones(N,dtype=bool);walk=0.;previous=0;prefix=[];q=[];steps=[]
    for t,a in enumerate(r['actions']):
        node=a+1;prefix.append(walk+.75*t);q.append(int(left.sum())/N)
        assert (left&p.C[:,a]).any()
        leg=float(p.D[previous,node]);assert math.isfinite(leg)
        steps.append(leg+.75);walk+=leg;left &= ~p.C[:,a];previous=node
    assert not left.any();exit_cost=float(p.D[previous,0]);rawJ=walk+exit_cost+.75*len(prefix)
    Hroute=r['route'];Hlegs=[float(p.D[a,b]) for a,b in zip(Hroute,Hroute[1:])];HJ=sum(Hlegs)+.75*(len(Hroute)-2)
    assert abs(rawJ-r['metrics']['raw_J'])<1e-7 and abs(HJ-r['metrics']['J'])<1e-7
    assert np.allclose(prefix,r['credit']['raw_prefix_cost_before'],atol=1e-7,rtol=0)
    assert q==r['credit']['q_before']
    steps[-1]+=exit_cost;steps[-1]+=HJ-rawJ
    assert abs(math.fsum(steps)-HJ)<1e-7
    return prefix,q,HJ

def main():
    started=time.time();ro=ReadOnly();pool=Pool(ro);refcache={};checks=0;updates={};models=[];rawrows=[];aggregates=[];paired=[];routechanges=[];creditrows=[]
    mf=read(D5/'REFERENCE_MANIFEST.json')
    def reference(tid):
        if tid not in refcache:
            from .common import gzread
            e=mf['records'][tid];assert sha(ROOT/e['path'])==e['sha256'];refcache[tid]=gzread(ROOT/e['path'])['metrics']['J']
        return refcache[tid]
    schedule=read(HOME/'COMMON_SCHEDULE.json')['batches']
    from . import evaluation
    all_reports={}
    for p in (HOME/'evaluations/development').glob('*.json'):
        rep=read(p);key=(p.stem.rsplit('_',1)[0],rep['endpoint']);all_reports[key]=rep
        for row in rep['rows']:
            ro.check();r=unarchive(row['record']);problem,evidence=pool.get(row['case_id'])
            assert r['problem_identity']==problem.identity and row['physical_key']==evidence['physical_key']
            for label,k in [('raw','raw_route'),('H','route')]:
                cost=sum(float(problem.D[a,b]) for a,b in zip(r[k],r[k][1:]));K=len(r[k])-2
                assert np.asarray(problem.C[:,np.array(r[k][1:-1])-1]).any(1).all()
                metricprefix='raw_' if label=='raw' else ''
                assert abs(cost-r['metrics'][metricprefix+'L'])<1e-7 and K==r['metrics'][metricprefix+'K']
                assert abs(cost+.75*K-r['metrics'][metricprefix+'J'])<1e-7
                assert r['raw_plan' if label=='raw' else 'plan']['validation']['valid_model_plan']
                checks+=4
            rawrows.append(dict(method=rep['method'],endpoint=rep['endpoint'],case_id=row['case_id'],physical_key=row['physical_key'],group=row['group'],width=row['width'],height=row['height'],valid=row['valid'],**r['metrics'],full_request_s=row['full_request_s'],raw_semantics=row['raw_semantics']))
        rebuilt=evaluation.summarize(rep['rows'])
        for group,s in rebuilt.items():
            for agg in ('request_mean','parent_equal_mean'):
                for metric,value in s[agg].items():assert abs(value-rep['summary'][group][agg][metric])<1e-8
                aggregates.append(dict(method=rep['method'],endpoint=rep['endpoint'],group=group,aggregation=agg,planned=s['planned'],valid=s['completed_valid'],parents=s['physical_parents'],**s[agg],p50_s=s['time_p50_s'],p95_s=s['time_p95_s']))
    for stream in ('A','B','SL'):
        path=HOME/'streams'/stream;updates[stream]=[];f=path/'UPDATES.jsonl.gz'
        if not f.exists():continue
        with gzip.open(f,'rt',encoding='utf-8') as reader:
            for line in reader:updates[stream].append(json.loads(line))
        prev=SL_TENSOR_SHA256
        for op,u in enumerate(updates[stream],1):
            assert u['cursor']==op and u['stream']==stream and u['before']==prev and math.isfinite(u['gradient_norm'])
            assert [s['task_id'] for s in u['loss_sources']]==schedule[op-1]
            assert len(u['loss_sources'])==8
            prev=u['after'];checks+=5
        if stream!='SL':
            records={}
            for line in (path/'REQUEST_POINTERS.jsonl').open(encoding='utf-8'):
                ro.check();ptr=json.loads(line);r=unarchive(ptr['record']);p,_=pool.get(r['task_id']);prefix,q,J=independent_credit(p,r);ref=reference(r['task_id'])
                values=[J-ref]*len(q) if stream=='A' else [J-v-qq*ref for v,qq in zip(prefix,q)]
                records[r['request_id']]=(r,values)
            used=set()
            for u in updates[stream]:
                for slot,s in enumerate(u['loss_sources']):
                    rid=s['episode_id'];assert rid not in used;used.add(rid);r,expected=records[rid]
                    assert r['stream']==stream and r['actor_tensor_sha256']==u['before']
                    assert np.array_equal(np.asarray(expected,dtype=np.float32),np.asarray(s['weights'],dtype=np.float32))
                    lp=torch.tensor(r['sample_log_prob'],dtype=torch.float32);ww=torch.tensor(expected,dtype=torch.float32)
                    loss=float(ww[0]*lp.sum()/8) if stream=='A' else float((ww*lp).sum()/8)
                    assert abs(loss-s['loss'])<=1e-4+1e-5*abs(loss),(stream,u['cursor'],loss,s['loss'])
                    creditrows.append(dict(stream=stream,update=u['cursor'],task_id=r['task_id'],size=f"{r['spec']['width']}x{r['spec']['height']}",
                        actions=len(expected),sample_J_H=r['metrics']['J'],reference_J_H=s['reference_J'],cost_difference=r['metrics']['J']-s['reference_J'],
                        H_correction=r['credit']['H_cost_correction'],coefficient_mean=float(ww.mean()),coefficient_std=float(ww.std(unbiased=False)),
                        coefficient_min=float(ww.min()),coefficient_max=float(ww.max()),negative_coefficient_fraction=float((ww<0).float().mean()),
                        loss=s['loss'],mixed_batch_gradient_norm=u['gradient_norm'],mixed_batch_clipped=u['clipped']))
                    checks+=5
            assert len(used)==len(updates[stream])*8
        base=torch.load(ROOT/SL_PATH,map_location='cpu',weights_only=False)['model']
        for f in sorted(path.glob('step*.actor.json')):
            entry=read(f);assert sha(ROOT/entry['path'])==entry['sha256'];p=torch.load(ROOT/entry['path'],map_location='cpu',weights_only=True)
            assert p['identity']==read(HOME/'SCIENTIFIC_IDENTITY.json')['identity'] and p['stream']==stream and not p['engineering_only']
            assert tensor_digest(p['model'])==entry['tensor_sha256']
            changed=[k for k in base if not torch.equal(base[k],p['model'][k])]
            assert set(changed)<=set(read(HOME/'PROTOCOL.json')['allowlist'])
            assert entry['endpoint']!=0 or not changed
            models.append(dict(**entry,changed_tensors=changed,production_default_promoted=False));checks+=4
    for endpoint in (128,256):
        if ('B',endpoint) not in all_reports:continue
        b=all_reports['B',endpoint]
        gate=read(HOME/f'QUALITY_GATE_{endpoint}.json')
        for control in ('FrozenSL','SL','A'):
            rep=all_reports[control,endpoint]
            for metric in ('J','raw_J','L','K'):
                gain=evaluation.gains(b['rows'],rep['rows'],metric)
                assert abs(gain['mean']-gate['comparisons'][control][metric]['mean'])<1e-12
                for parent,value in gain['parents'].items():paired.append(dict(endpoint=endpoint,control=control,metric=metric,physical_key=parent,relative_gain=value))
            for r in evaluation.route_changes(b,rep):routechanges.append(dict(endpoint=endpoint,control=control,**r))
    # New A remains a freshly collected control. Comparing saved parameter chains
    # here does not feed an old episode into any optimizer.
    old_prefix=[]
    with gzip.open(D5/'streams/2026111101/D5-PG/UPDATES.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            x=json.loads(line)
            if x['cursor']>256:break
            old_prefix.append(x)
    A_reproduced=all(a['after']==b['after'] and a['loss']==b['loss'] for a,b in zip(updates['A'],old_prefix)) if len(updates['A'])==len(old_prefix)==256 else None
    r=read(HOME/'RESOURCE_ACCOUNTING.json');complete=all(len(updates[s])==256 for s in updates) and len(rawrows)==792
    if complete:
        assert r['counts']['actor_optimizer_attempts']==768 and r['counts']['teacher_replays']==2048
        assert 4096<=r['counts']['train_requests']<=4128 and 792<=r['counts']['eval_requests']<=824
        assert 4888<=r['counts']['requests']<=4920
    csvwrite('COMPLETE_TASK_RESULTS.csv',rawrows);csvwrite('AGGREGATE_RESULTS.csv',aggregates);csvwrite('PARENT_PAIRED_GAINS.csv',paired);csvwrite('STOP_SET_ORDER_CHANGES.csv',routechanges);csvwrite('ONLINE_CREDIT_DIAGNOSTICS.csv',creditrows)
    put(HOME/'MODEL_INDEX.json',dict(models=models,production_default_promoted=False))
    put(HOME/'INDEPENDENT_NUMERICAL_AUDIT.json',dict(status='PASS',checks=checks,complete_scientific_grid=complete,actual_evaluation_rows=len(rawrows),
        actual_updates={s:len(u) for s,u in updates.items()},new_A_first256_matches_saved_D5_weights_and_losses=A_reproduced,
        off_policy_training_data_reused=False,new_forward_optimizer_H_native_navigation_teacher=0,post_input_IO=dict(ro.counts),wall_s=time.time()-started))
    make_report(all_reports,r,models)
    print(dict(status='PASS',checks=checks,complete=complete,wall_s=time.time()-started),flush=True)

def make_report(reports,resource,models):
    diag=read(HOME/'COST_DECOMPOSITION_VALIDATION.json');summary=['# D6完整路线对照\n']
    summary+=['本轮从准确SL256重新训练A（原D5完整轨迹损失）、B（因果剩余成本）、匹配SL，各256。旧D54096负结果与P2—P5 NOT_RUN不改。一个预定随机流seed2026111101，66个已暴露开发任务，SEL8+DEV3，每父6请求；仅开发机制筛选。\n',
        '[数学与因果性](MATHEMATICS.md) · [协议](PROTOCOL.json) · [数值复核](INDEPENDENT_NUMERICAL_AUDIT.json) · [模型](MODEL_INDEX.json) · [逐请求](COMPLETE_TASK_RESULTS.csv) · [资源](RESOURCE_ACCOUNTING.json)\n']
    summary+=['## 旧轨迹成本与信用统计\n',f"全部{diag['old_online_trajectories']}条旧轨迹通过分解，最大绝对误差{diag['max_abs_decomposition_error']:.3g}，预定容差1e-7。raw逐步增量含最后出口段；H校正只加入一次，最终等于保存J_H。未重跑H、教师或导航。delta=采样J_H−冻结同问题参考J_H，正值表示采样成本更高。\n",
        '|尺寸|轨迹|动作均值|J_raw均值|J_H均值|采样减参考均值/标准差|正/负差比例|绝对标量损失份额|\n|---|---:|---:|---:|---:|---:|---:|---:|']
    for size,d in diag['sizes'].items():
        m=d['metrics'];summary.append(f"|{size}|{d['trajectories']}|{m['actions']['mean']:.2f}|{m['raw_J']['mean']:.4f}|{m['J_H']['mean']:.4f}|{m['delta']['mean']:.4f}/{m['delta']['std']:.4f}|{100*d['positive_delta_fraction']:.2f}%/{100*d['negative_delta_fraction']:.2f}%|{100*d['absolute_loss_share']:.2f}%|")
    summary+=[f"\n原4096混合批次梯度范数均值{diag['gradient_norm']['mean']:.4f}，范数>1比例{100*diag['clipped_fraction']:.2f}%。这不是按尺寸分离的梯度。各尺寸损失是保存logpi重构的标量贡献，包含符号抵消，不是参数梯度贡献或方差改善证据。正负按原始差值严格判断。详细分位数、H校正及有符号/绝对损失见[COST_DECOMPOSITION_VALIDATION.json](COST_DECOMPOSITION_VALIDATION.json)和[D5_TRAJECTORY_CREDIT.csv](D5_TRAJECTORY_CREDIT.csv)。\n"]
    for endpoint in (128,256):
        if ('B',endpoint) not in reports:continue
        summary += [f'## 固定step{endpoint}完整任务结果\n','J=L+0.75K，L为导航长度，K为停靠次数；先父内均值再父等权。因每父6请求，请求等权绝对均值与父等权一致；父配对收益单独计算，不能用两个总体均值的比值替代。\n',
            '|方法|raw J/L/K|H后 J/L/K|有效|完整请求p50/p95秒|\n|---|---:|---:|---:|---:|']
        for label in ('FrozenSL','A','B','SL','Fixed3','Detour16'):
            s=reports[label,endpoint]['summary']['overall'];m=s['parent_equal_mean']
            summary.append(f"|{reports[label,endpoint]['method']}|{m['raw_J']:.4f}/{m['raw_L']:.4f}/{m['raw_K']:.4f}|{m['J']:.4f}/{m['L']:.4f}/{m['K']:.4f}|{s['completed_valid']}/{s['planned']}|{s['time_p50_s']:.4f}/{s['time_p95_s']:.4f}|")
        gate=read(HOME/f'QUALITY_GATE_{endpoint}.json');summary+=['\n|B对照|H-J父配对收益|raw-J收益|SEL8 H-J|DEV3 H-J|\n|---|---:|---:|---:|---:|']
        for c in ('FrozenSL','SL','A'):
            x=gate['comparisons'][c];gg=list(x['groups'].values());summary.append(f"|{c}|{100*x['J']['mean']:.4f}%|{100*x['raw_J']['mean']:.4f}%|{100*gg[0]['mean']:.4f}%|{100*gg[1]['mean']:.4f}%|")
        summary.append(f"\n该点门槛：**{gate['status']}**。128是固定描述点，256是唯一主终点；不挑中间模型。")
    primary=read(HOME/'QUALITY_GATE_256.json') if (HOME/'QUALITY_GATE_256.json').exists() else None
    summary += ['\n## 范围、时间与决定\n',
        'Fixed3/Detour16的raw/H两栏是同一原生final口径，不是神经无H消融。六方法在两终点均重新评价，未借用旧D5耗时；完整请求时间含本轮输入准备、特征、构造、H、导出验证及归档写入，随后日志尾部计入总墙钟。神经模型在网格开始时加载并常驻，加载费用计入作业总账，不在每请求p50/p95中；这不是含启动的冷服务计时。缓存冷热与执行顺序不同，不宣称公平部署加速。成功率是完整代理模型约束证书范围，不是实机IK/关节轨迹认证。\n',
        '各尺寸、SEL/DEV、raw/H分量与两聚合的全表见[AGGREGATE_RESULTS.csv](AGGREGATE_RESULTS.csv)，每父配对见[PARENT_PAIRED_GAINS.csv](PARENT_PAIRED_GAINS.csv)，停靠集合及访问顺序变化见[STOP_SET_ORDER_CHANGES.csv](STOP_SET_ORDER_CHANGES.csv)。路线变化和正负系数比例不替代任务收益。\n',
        f"实际资源：{resource['counts'].get('requests',0)}次完整请求，Actor尝试{resource['counts'].get('actor_optimizer_attempts',0)}、标签重放{resource['counts'].get('teacher_replays',0)}，Critic0。工程optimizer0。最终完整墙钟包含准备、统计、科学、审查与发布，以最终资源账为准；旧D5消费通过原账hash保留，不清零或覆写。\n",
        'B的状态依赖基线在数学上保持完整终端目标的score-function期望，不能据此宣称有限样本方差降低、训练更快或已实现独立泛化。没有读取原60、旧S3或确认/最终集合，没有新教师、场景或导航。\n']
    decision=primary['status'] if primary else 'TECHNICAL_INCOMPLETE'
    summary.append(f'主决定：**{decision}**。本次固定256预算结束；不自动增加学习率、宽度、次数、种子或恢复关闭支线。')
    (HOME/'EXPERIMENT_REPORT.md').write_text('\n'.join(summary)+'\n',encoding='utf-8')
    (HOME/'NEXT_STEP_DECISION.md').write_text(f'# D6研究决定\n\n固定256主终点：{decision}。完整数值见EXPERIMENT_REPORT.md和QUALITY_GATE_256.json。\n\n本轮预算结束，候选不自动扩大或续跑。仅记录本预定seed和已暴露开发任务的机制证据；不作方差、独立泛化、部署速度或真实机械臂认证的延伸。旧D5停止结论保持。\n',encoding='utf-8')

if __name__=='__main__':main()
