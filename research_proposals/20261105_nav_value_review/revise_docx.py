"""Revise the supplied research proposal; never overwrite the original.

Run with the Codex bundled Python. No planner/policy/scene imports.
"""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import sys
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'RC3_研究模型与实验计划_20261105.docx'
TARGET=ROOT/'RC3_研究模型与实验计划_20261105_评审修订版.docx'
HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def text(p,value):
    props=deepcopy(p.runs[0]._r.rPr) if p.runs and p.runs[0]._r.rPr is not None else None
    p.clear();r=p.add_run(value)
    if props is not None:r._r.insert(0,props)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    original_hash=sha(SOURCE);d=Document(SOURCE);changes=[]
    initial_math=len(d.element.xpath('.//m:oMath'))
    def replace(prefix,value):
        matches=[p for p in d.paragraphs if p.text.startswith(prefix)]
        assert len(matches)==1,(prefix,len(matches))
        p=matches[0];assert not p._p.xpath('.//m:oMath'),prefix
        changes.append(dict(before=p.text,after=value));text(p,value)
    def annotate(prefix,value):
        matches=[p for p in d.paragraphs if p.text.startswith(prefix)]
        assert len(matches)==1,(prefix,len(matches))
        anchor=matches[0];e=OxmlElement('w:p');anchor._p.addnext(e)
        p=Paragraph(e,anchor._parent);p.style=anchor.style;p.add_run(value)
        changes.append(dict(after_paragraph=anchor.text,inserted=value,original_math_preserved=True))
    replace('导航关系表示与长期价值学习','导航关系表示与长期价值学习 评审修订版')
    replace('这个方向的目标是训练一个能快速构造完整任务路线的模型。','这个方向的目标是训练一个能快速构造完整任务路线的模型。先检验在当前完整后处理之后，长期动作选择是否仍有值得学习的改进机会；再通过有界的近似策略迭代检验网络能否利用该信号，以及强化学习是否提供独立增量。R0的教师收益可能全部来自额外搜索，不能提前称为可学性或RL收益。若只得到静态标签拟合、搜索提质或工程提速，应按实际贡献命名。')
    replace('RRNCO 使用真实道路关系并不意味着','RRNCO的真实道路关系编码可作设计参考；其训练采用多GPU，测试使用批处理和增广，不能换算成本项目800候选的CPU单请求延迟。GOAL借鉴异构mixed-attention结构，不沿用未经证明的无前缀BQ状态压缩；Gumbeldore的最好解监督模仿本身不等于本计划的价值策略迭代。[R1,R2,R5]')
    replace('SuperMimic 在连续控制中','SuperMimic在连续人形控制中使用可微世界模型与BPTT。当前离散覆盖更新、候选选择和H无法直接获得同类解析梯度。其作者项目中的失败率包含训练动作覆盖，硬件和数据条件也不同，不能作为本项目未见任务泛化或训练提速证据。适合迁移的原则是“已知转移精确计算，困难的长期后果交给学习”。[R9]')
    replace('在任务独立、一次服务覆盖该候选全部对应剩余任务、','在任务独立、每次服务完成该候选全部剩余可覆盖任务、固定B覆盖全任务，且B节点与合法前缀当前位置都属于同一基地往返可达导航关系时，g_B可补全任意合法前缀。对有向图须同时有0→v和v→0路径，并由一致最短路D保证节点间可达；仅写“从基地可达”不充分。每次从B开头扫描并跳过零增益节点，固定原ID并列规则，形成顺序一致的确定策略。理论根参照为J(H(B_effective))，不同于原生Fixed3，也不同于旧update128加H。')
    replace('每个前缀只评价一个有界候选集合，','每个前缀只评价一个有界唯一候选集合，始终包含参考动作。R0尚无新价值网络，使用参考、即时覆盖/距离规则与预定探索来源；R1再加入当前网络高分动作。重复候选去重，原ID稳定处理并列；所有实际补全及H调用扣预算。未查询动作标记未知，不能记为零收益或负标签。')
    replace('首试使用完整回报策略评估，','首试使用完整回报策略评估，避免同时引入TD自举误差。环境、g、邻域构图、并列规则与H全部固定且确定时，一次完整补全给出该参考策略下的精确标签，并非全局最优价值。随机邻域若不能固定，就必须把其随机性纳入估计和预算，不能继续称单次标签精确。Monte Carlo主要采样实例、前缀和探索动作。主要损失为总成本Huber回归，同前缀差值可作辅助项：')
    annotate('执行“冻结','首轮晋升、优化器状态、标签版本和未知动作审计日程必须在训练前固定。若参考未发生实际变化，应如实记录，不能描述成已验证有效的策略改进。')
    annotate('因为g(s)本身提供一个相同参考成本的候选。','具体地，在完整前缀状态上Y^g(s,a)=V^g(T(s,a))，且Y^g(s,g(s))=V^g(s)。条件是g固定、合法且顺序一致，H和并列规则确定。终局H改写尚未执行的前缀不破坏推导；保证相对g及统一H，不直接相对原生Fixed3。[R4]')
    replace('这是附加假设下的误差界。','这里ε以原始J成本单位计，若训练目标除以100，误差须先换回相同单位。由V^g(s′)≤Ŷ(s,μ(s))+ε≤Ŷ(s,g(s))+ε≤V^g(s)+2ε逐步累加可得该界。假设必须覆盖实际选中的动作、参考动作及到达的状态，包括原标签集未查询动作。RMSE、集成方差或95%误差分位数不能替代全域证书，不能据此声称对全部新场景非劣。')
    replace('R0首先统计查询集合内的真实动作后悔值、','R0主筛选固定使用全部预定32例，不从结果中选择“至少12例”。统一参照明确为g_B根状态的J(H(B_effective))。候选机会只表示查询集合内的改进空间，不是全动作最优。建议继续规则为32例平均相对收益≥0.5%、至少12例严格正收益、最大单例正相对收益占总正相对收益≤25%，并且全例可行。阈值为拟定研究筛选规则，不是已登记结果或统计显著性结论；任一不足或证据不全则停止/无定论，不自动扩预算。')
    replace('为使预算可执行，2048次查询必须沿','2048次动作终局查询只沿有界教师实际到达的前缀产生：在覆盖进度0、1/8、…、7/8首次达到时触发，重复前缀、跳过的阈值及终止状态不额外补位，每例最多8点、每点最多8个唯一合法动作；其他步骤按固定g_B。2304次为顶层终局评价总上限，提前预留256次用于32次原生Fixed3初始化、32次g_B根参照、32次教师终局以及最多160次额外校准/复核。Fixed3内部多配方及局部搜索另列内部计数；不能将顶层调用数称为相同算力。几何/数值验证若不运行H，另记证书；凡实际重跑H都扣预算。')
    replace('R1建议将4096次共享暖起点标签、','R1的16384次顶层终局评价封顶账分为：4096次共享暖起点标签、8192次两轮匹配策略评价、4096次验证和审计。任何原生基线规划、校准、补全或重复H评价均先从所属额度扣除；不得把初始化规划当成免费数据。缓存只在实例、完整有序前缀、动作、g/B版本、H及数值配置哈希全部一致时命中，真实调用与标签使用次数分列。固定参考组匹配新状态、回报查询预算和优化更新。')
    replace('暖起点标签由固定路线参考生成，','暖起点标签由固定路线参考生成，不计作策略改进轮次，但优化步数计入每配置2048上限：建议暖起点最多1024步，两个评价/改进轮次各512步。同表示两组共享暖起点权重，在首轮均重建相同Adam；第二轮均保留各自首轮状态，不进行单边重置。固定参考组始终评价g0；闭环组仅在预定开发均值改善≥0.5%、配对95%下界>0且全例可行时晋升新g，否则保留g0并明确“未晋升”。每配置每轮最多128前缀×8候选，4×2×128×8=8192次；可复用的首轮相同标签按真实调用和使用次数分报。')
    replace('4096次验证和审计额度中，','4096次验证/审计额度预留1024次检查网络实际选出而原标签未查询的动作，建议4配置×2轮×128个固定审计状态、每状态至多1个此类动作；没有未知动作则记跳过，不挑替代成功案例。其余最多3072次用于每轮固定开发路线、暖起点和旧参照、原生规则基线及其他校准/复核。审计状态预先隔离；若审计结果用于修补训练或选择模型，就改记开发证据，另保留未使用的确认数据。')
    replace('研究门槛建议定义为D相对','R1单种子进入R2的拟定筛选规则：主比较D对同结构固定参考组B的平均配对改善≥0.5%、95%区间下界>0；同时D对该表示共同暖起点以及旧update128加H的改善区间下界均>0，全部可行。只比退化更多的对照好不能通过。其余表示对比为支持性/探索性。R2才用三个匹配训练种子检查方向一致与分层不确定性；不能把三种子一致性当作单种子R1获准进入R2的先决条件。')
    replace('下一项有价值的工作是R0：','建议下一阶段先单独登记R0，以严格计账的完整补全检验H之后是否仍有有价值的动作改进机会；这不预先证明网络可学性、导航D的必要性、RL增量或部署优势。R0通过后才另行冻结最小网络与两轮近似策略迭代协议。本方案修订不执行R0，也不重启已收尾的同结构策略梯度支线。')
    replace('[R8] Li K,','[R8] Li K, Zhang T, Wang R, Wang Y, Han Y, Wang L. Deep Reinforcement Learning for Combinatorial Optimization: Covering Salesman Problems. IEEE Transactions on Cybernetics, 2022, 52(12):13142–13155. DOI:10.1109/TCYB.2021.3103811。另列2021预印本arXiv:2102.05875v1（五作者版本），不混用作者和卷期年。CSP_Attention仓库仅作待适配入口，未据README认定已完整复现正式论文。')
    replace('[R11] LMask:','[R11] Li T, Zou H, Wu J, Wen Z. LMask: Learn to Solve Constrained Routing Problems with Lazy Masking. ICLR, 2026. 原实验TSPTW/TSPDL；借鉴约束处理思路，不在首版替换RC3精确硬掩码或放松可行性。')
    # Remove an inherited locator to an unspecified attachment, not a verified page.
    for p in d.paragraphs:
        for r in p.runs:
            if '重点阅读附件第5、6–9、17页。' in r.text:
                r.text=r.text.replace('重点阅读附件第5、6–9、17页。','')
    # Clarify stage descriptions in the existing budget table without altering math.
    for table in d.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    if p.text=='完整H后仍有长期动作差异；有界教师具有可用增益':text(p,'检验完整H后动作机会与教师收益；不据此确认可学性或RL收益')
                    if p.text.startswith('最多32个百桌开发实例，每例8个前缀'):text(p,'预定32例，每例至多8前缀×8动作=2048次；根参照、教师终局、初始化与复核预留256；顶层终局调用合计≤2304')
    # Insert an actionable appendix before bibliography. All execution remains proposed.
    anchor=next(p for p in d.paragraphs if p.text=='参考文献与代码')
    h1=next(p.style for p in d.paragraphs if p.text.startswith('12 建议'))
    h2=next(p.style for p in d.paragraphs if p.text.startswith('9.1'))
    body=next(p.style for p in d.paragraphs if p.text.startswith('机制案例应'))
    def before(value,style=body):
        e=OxmlElement('w:p');anchor._p.addprevious(e);p=Paragraph(e,anchor._parent);p.style=style;p.add_run(value);return p
    before('13 R0执行前固定项',h1)
    before('13.1 实例与教师主参照',h2)
    before('拟定32例均为百桌新开发实例，regular/mixed/blocked按固定循环为11/11/10例。可采用预留地图种子8300000000+i、任务种子8400000000+i（i=0…31）；正式任务须核对历史未用并登记清单。本次不生成场景。所有预定病例进入主汇总，实例失败不静默剔除或补抽。原60保留测试不查询。')
    before('每例只运行一次原生Fixed3得到B；另从根执行确定g_B并统一H，作为主参照。记录原生Fixed3、g_B加H及有界教师三种成本，不把它们混称同一基线。每次动作选择来自同一实际前缀；后续继续g_B，最终统一H。原update128加H仅可作为明确计账的额外诊断，不占用未申报预算。')
    before('13.2 八动作队列与检查点',h2)
    before('队列先放参考动作1个；再按gain/(D_current,j+0.75)降序补入至4个唯一动作；再按当前导航增量升序补至6个；最后用独立固定seed的无放回合法动作顺序补至8个。每类遇重复就继续扫描；规则并列按原ID，合法动作不足8个则全部使用。拟定终局Y并列及严格正收益判定的J绝对容差为1e-7，近并列优先参考动作；门槛使用未四舍五入值。R0不引入新价值网络。')
    before('按已覆盖任务比例首次跨过0、1/8、…、7/8触发查询，一步跨过多个阈值只算一个前缀，任务已完成不再查询；每例至多8个不同检查点。教师在所查询动作中按真实Y最小值选择，并列优先参考动作再按ID；非检查点直接用g_B。保持参考动作可使精确查询点不增，但不把该性质推广到没有全域误差证书的神经策略。')
    before('13.3 计账与停止',h2)
    before('顶层预算表：动作Complete_g+H≤2048；原生Fixed3初始化≤32；g_B根参照H≤32；教师最终H≤32；其他校准/重复终局评价≤160，总计≤2304。缓存命中不增加实际调用，但必须增加标签使用计数。原生Fixed3内部构造、局部搜索、方案导出与标量验证分别记录；此上限不是FLOPs或所有内部代价函数求和次数的上限。')
    before('32例相对g_B加H的平均收益≥0.5%、至少12例正收益、最大单例正相对收益占比≤25%、所有最终方案有效，才记为R0拟定继续信号。正收益占比以max(q_i,0)/Σmax(q_i,0)计算；分母为0则不通过。预算耗尽、病例不全或信号不足均停止并保留无定论，不能改用成功子集。耗时上限须由已计账的小样本校准，在主要查询前固定；校准不得成为免费补全。')
    before('14 研究主张与门槛版本',h1)
    before('主要待验证主张仅两项：同结构、同预算下，更新参考策略的闭环优于固定参考价值学习；真实D关系相对不泄漏D的参数匹配覆盖表示提供增量。首先排除后处理更强、查询更多、初始化不同和控制组退化等解释。R0只负责检验教师机会；R1负责初筛；R2负责独立种子与冻结确认；R3负责分规模失效边界。')
    before('原G1–G4的严格信息不可辨识门槛保留未通过状态。若未来转为“显式关系对有限网络的归纳偏置/学习效率”假设，应在新任务明确改用受控实验路线，而不是声称原门槛已通过。旧支线仍收尾，新模型和所有R阶段均未启动。')
    before('首轮保留固定64维两层、真实总成本Huber主损失及完整前缀；默认不启用额外差值损失、MAML、PPO、可学习动力学、动作硬裁剪或部署搜索。新增组件只能在之后的独立受控任务中逐项检验。标签集、结构图、归一化尺度、优化器和晋升规则均应在正式协议版本中冻结。')
    assert len(d.element.xpath('.//m:oMath'))==initial_math,'Original OMML changed'
    d.core_properties.title='移动双臂任务规划研究模型与实验计划 评审修订版'
    d.core_properties.subject='覆盖导航关系 长期价值 近似策略迭代 研究设计'
    d.save(TARGET)
    assert sha(SOURCE)==original_hash,'Source document modified'
    record=dict(source=str(SOURCE),source_sha256=original_hash,output=str(TARGET),output_sha256=sha(TARGET),
        preserved_OMML_equations=initial_math,changes=changes,appended_sections=['13 R0执行前固定项','14 研究主张与门槛版本'],
        new_training=0,new_planning=0,new_scene_generation=0,status='review_revision_only_not_executed')
    (HERE/'revision_receipt.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in record.items() if k!='changes'},ensure_ascii=False))


if __name__=='__main__':main()
