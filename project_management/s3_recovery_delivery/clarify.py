"""Tighten postmortem wording using saved-array evidence, without new navigation."""
import csv
import json
from collections import Counter,defaultdict
from rc3_s3_evaluator.common import ROOT, read, digest, write_new
from rc3_s3_recovery_20261108.authority import HOME


def main():
    old='该请求已完成输入保存与一次冷导航，尚未进行神经模型物化、贪心构造、native或H。'
    extra=('已保存F/A为233×800×2，233/233任务在全局候选上存在至少一项同臂F&A；因此不能称为“F&A自身没有覆盖”。'
           '失败栈定位于全父场景闭合端点连通候选过滤后的服务检查，早于4×3局部提取；不能据此断言局部窗口本身无解。'
           '失败时D/导航拓扑没有独立保存，现有证据不足以列出具体失联任务或归因到某条边；没有重算导航来补证据。')
    p=HOME/'EXPERIMENT_REPORT.md';s=p.read_text(encoding='utf8');s=s.replace(old,old+extra)
    s=s.replace('所有294个已观察case/seed/method组合路线与成本在重复及版本间一致，',
        '294个已观察case/seed/method组合在实际重复中路线和成本一致；其中实际跨持久化版本重叠的7个组合也一致，')
    p.write_text(s,encoding='utf8')
    (HOME/'PERSISTENCE_REPAIR_REPORT.md').write_text('''# 持久化最小修复与验收

**工程GO。** 原IO故障、Windows启动器兼容问题及新科学输入停止条件分别保留，不混为同一故障。

原 `HEAD.pending → HEAD.json` 替换访问拒绝由有界同事件重试处理：最多8次，只重试锚替换，不重复追加事件；检查pending文件身份/内容、旧HEAD和完整链哈希。独立错误收据不递归依赖失效Journal。OS单写入者租约及固定外层激活标记拒绝并发和改目录清账；旧180文件/原HEAD/pending从未改写。

父进程在Journal解析/收尾前保存独立退出钟，失败关闭后仍有FINAL/SEAL或独立故障收据。正常每请求计算方法、日志计量和计时边界保持，新增检查与重试属于计费IO。

17项零科学回归覆盖瞬态/持续替换失败、残留/被篡改pending、断链、孤立结果、父收尾/早期打开失败、重复恢复、预算不退款/不可清零、Windows真实venv子进程租约、缺失计时不填补及合成原统计一致性。最新套件PASS；工程suite累计82.03700110001955秒，科学函数入口计数均0。

首版修复启动遇到真实Windows venv redirector的父PID兼容问题，0个科学请求，完整9.802301499992609秒收费证据保留。source02使用实际持有的监督者OS租约而非直接PPID，独立真实子进程测试通过。source01与源码提交81290f43及原收据保持，source02及原v3/训练模型身份分开绑定。

实际source02续段新增1896预扣/1895成功/1全父连通服务前置检查失败。出现6份瞬态锚重试收据，对应6个不同事件，均成功，未重复追加；新链46023事件，HEAD齐、无新pending，错误与父FINAL正常闭合。后续停止发生在S3_042输入准备，早于模型/native/H，不是日志修复失效。

最终源码绑定：`767f13bd4619bef37501db997296982baf6b1565cb25ea8cb964c2e39d381bea`。依据：[测试](ENGINEERING_VERIFICATION_02.json)、[源码独立审查](INDEPENDENT_REVIEW_02.md)、[恢复决定](RECOVERY_REVISION_02.json)、[实际报告](EXPERIMENT_REPORT.md)、[资源审计](results/RECOVERY_AUDIT.json)。
''',encoding='utf8')
    with (HOME/'results/REQUEST_METRICS.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
    versions=Counter((r['kind'],r['persistence_revision']) for r in rows)
    bykey=defaultdict(set)
    for r in rows:bykey[(r['case_id'],r['seed'],r['method'])].add(r['persistence_revision'])
    write_new(HOME/'results/TIMING_REVISION_AUDIT.json',dict(
        scope='Saved completed request records only',
        by_kind_revision=[dict(kind=k[0],persistence_revision=k[1],n=v) for k,v in sorted(versions.items())],
        repeated_case_seed_method_combinations=len(bykey),actual_cross_version_overlaps=sum(len(v)>1 for v in bykey.values()),
        route_cost_consistency='PASS',same_complete_request_boundary=True,
        identical_persistence_overhead_claimed=False,
        missing_timing_filled=False,pooled_timing_scope='descriptive observed prefix only; no original deployment gate decision',
        original_timing_document='TIMING_COMPARABILITY.md',
        original_timing_document_sha256=digest(HOME/'TIMING_COMPARABILITY.md')))
    (HOME/'TIMING_COMPARABILITY_CLOSEOUT.md').write_text('''# 实际计时版本审计

旧b348持久化版本成功162请求（quality28/cold71/hot63）；source02成功1895请求（quality266/cold810/hot819）。首版修复source34fb只有0请求启动失败，无模型/路线计时样本。

CPU、线程、模型文件、七方法流程、固定随机顺序及完整请求计时边界保持。解析/来源/特征/编码/贪心/native/H/导出/验证/结果写哈希都在原完整请求范围内；请求终态Journal提交在阶段墙钟。热模式只复用导航，预热仍计入阶段，任务/编码/路线不免费缓存。

新锚身份检查、OS单写入者和真实锚重试是新增持久化IO，按实际所在请求/阶段计时，不声称修复前后IO开销相等。全部重复的294个case/seed/method组合路线成本稳定；实际同时有旧/新版本样本的7个组合一致。算法语义一致不等于计时分布严格同分布，当前样本版本与顺序混杂；分层结果必须保留，混合时间仅作描述。

原第163次cold失败仍缺失，新S3_042质量失败也无成本或时间值。当前cold881/6804、hot882/6804；全部原部署格不完整，不删格、不填值、不宣称总体部署通过。[分层机器表](results/TIMING_REVISION_AUDIT.json)与[各格分层时间](results/SCIENTIFIC_ANALYSIS.json)可复核。
''',encoding='utf8')
    print('Clarified endpoint-filter evidence and actual timing overlap; no new computations.')


if __name__=='__main__':main()
