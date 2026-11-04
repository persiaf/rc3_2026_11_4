from __future__ import annotations

import json
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

HERE = Path(__file__).resolve().parent
OUT = HERE / 'deliverables'
OUT.mkdir(exist_ok=True)
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.top_margin = sec.bottom_margin = Inches(.65)
sec.left_margin = sec.right_margin = Inches(.72)
sec.header_distance = sec.footer_distance = Inches(.25)
for x in list(sec._sectPr):
    if x.tag == qn('w:docGrid'):
        sec._sectPr.remove(x)

def font_style(style, size, bold=False):
    style.font.name = 'DejaVu Sans'
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0, 0, 0)
    rp = style.element.get_or_add_rPr()
    rf = rp.find(qn('w:rFonts'))
    if rf is None:
        rf = OxmlElement('w:rFonts'); rp.insert(0, rf)
    for attr in list(rf.attrib):
        if 'Theme' in attr:
            del rf.attrib[attr]
    rf.set(qn('w:ascii'), 'DejaVu Sans')
    rf.set(qn('w:hAnsi'), 'DejaVu Sans')
    rf.set(qn('w:eastAsia'), 'Droid Sans Fallback')
    for x in list(rp):
        if x.tag == qn('w:color'):
            x.attrib.pop(qn('w:themeColor'), None)
    pf = style.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    pf.line_spacing = Pt(size * 1.42)
    pf.space_before = Pt(0); pf.space_after = Pt(6)
    pp = style.element.get_or_add_pPr()
    snap = OxmlElement('w:snapToGrid'); snap.set(qn('w:val'), '0'); pp.append(snap)
    for x in list(pp):
        if x.tag == qn('w:pBdr'):
            pp.remove(x)

for s in doc.styles:
    if s.type == 1:
        font_style(s, 11)
font_style(doc.styles['Title'], 22, True)
font_style(doc.styles['Subtitle'], 10)
font_style(doc.styles['Heading 1'], 15, True)
font_style(doc.styles['Heading 2'], 12, True)
font_style(doc.styles['Caption'], 9.5)
for name in ['Heading 1', 'Heading 2']:
    doc.styles[name].paragraph_format.space_before = Pt(9)
    doc.styles[name].paragraph_format.space_after = Pt(6)
    doc.styles[name].paragraph_format.keep_with_next = True

def p(text='', bold=False, style=None):
    pp = doc.add_paragraph(style=style)
    pp.add_run(text).bold = bold
    return pp

def h(text, level=1, new=False):
    pp = doc.add_paragraph(text, style=f'Heading {level}')
    if new:
        pp.paragraph_format.page_break_before = True
    return pp

def table(headers, rows, widths, fs=10):
    t = doc.add_table(rows=1, cols=len(headers))
    t.autofit = False
    for i, w in enumerate(widths):
        t.columns[i].width = Inches(w)
    borders = OxmlElement('w:tblBorders')
    for edge in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        e = OxmlElement(f'w:{edge}')
        e.set(qn('w:val'), 'single'); e.set(qn('w:sz'), '4'); e.set(qn('w:color'), 'D9D9D9')
        borders.append(e)
    t._tbl.tblPr.append(borders)
    for rix, vals in enumerate([headers] + rows):
        cells = t.rows[0].cells if rix == 0 else t.add_row().cells
        for i, v in enumerate(vals):
            cell = cells[i]; cell.width = Inches(widths[i])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcpr = cell._tc.get_or_add_tcPr()
            mar = OxmlElement('w:tcMar')
            for edge in ['top', 'bottom', 'left', 'right']:
                e = OxmlElement(f'w:{edge}'); e.set(qn('w:w'), '80' if edge in ['top','bottom'] else '95'); e.set(qn('w:type'), 'dxa'); mar.append(e)
            tcpr.append(mar)
            if rix == 0:
                sh = OxmlElement('w:shd'); sh.set(qn('w:fill'), 'E5EDF5'); tcpr.append(sh)
            cp = cell.paragraphs[0]
            cp.paragraph_format.space_before = Pt(0); cp.paragraph_format.space_after = Pt(0)
            cp.paragraph_format.line_spacing = Pt(fs * 1.4)
            cp.paragraph_format.keep_with_next = False
            cp.alignment = WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER
            cr = cp.add_run(str(v)); cr.font.size = Pt(fs); cr.bold = rix == 0
        trpr = t.rows[rix]._tr.get_or_add_trPr()
        no_split = OxmlElement('w:cantSplit'); trpr.append(no_split)
        if rix == 0:
            repeat = OxmlElement('w:tblHeader'); trpr.append(repeat)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return t

def mr(text):
    r = OxmlElement('m:r'); tt = OxmlElement('m:t'); tt.text = text; r.append(tt)
    return r

def sub(base, lower):
    s = OxmlElement('m:sSub')
    e = OxmlElement('m:e'); e.append(mr(base))
    ss = OxmlElement('m:sub'); ss.append(mr(lower))
    s.append(e); s.append(ss); return s

def sup(base, upper):
    s = OxmlElement('m:sSup')
    e = OxmlElement('m:e'); e.append(mr(base))
    ss = OxmlElement('m:sup'); ss.append(mr(upper))
    s.append(e); s.append(ss); return s

def eq(*items):
    pp = doc.add_paragraph(); pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pp.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    mm = OxmlElement('m:oMath')
    for item in items:
        mm.append(mr(item) if isinstance(item, str) else item)
    pp._p.append(mm)
    return pp

footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
run = footer.add_run(); run.font.size = Pt(9)
fld = OxmlElement('w:fldSimple'); fld.set(qn('w:instr'), 'PAGE'); run._r.addnext(fld)

# Page 1
p('第四版规划研究验收与定稿建议', style='Title')
p('移动升降双臂机器人任务规划  2026年11月3日', style='Subtitle')
p('验收结论', True)
p('第四版已补齐外部可达关系接口、小规模离散精确参照、结构输入消融和审计修复。现有证据支持以 Fixed3 初解、Detour 局部修复和独立终验作为论文与部署主线。固定修复次数下存在学习质量收益；新结构模型的独立增益与同时间优势尚未达到采用条件。')
h('1 已完成的实际复核')
table(['复核对象', '本次结果', '解释边界'], [
    ['主实验保存方案', '855 / 855通过', '来自39个场景'],
    ['成本与场景方法汇总', '855与357项一致', '原时延统计重算'],
    ['小规模独立复验', '225 / 225通过', '另27场景45组合'],
    ['冻结版非Torch测试', '34 / 34通过', '原报告40项未全重跑'],
    ['文件完整性', '270 / 270匹配', '原上传源码未改'],
], [2.13, 1.77, 3.16])
p('855份方案由585次主比较、162次软预算和108次训练种子评测组成。小规模225份方案属于另一组实验。验证使用经审查且独立于规划器的原验证器；本次另写统计与精确参照复算脚本。训练日志和权重已核对，六组训练没有重新执行。')
h('百桌固定八次修复的结果', 2)
table(['方法', '相对Fixed3降本', '规划中位 ms', '规划P95 ms'], [
    ['Fixed3', '0', '93.61', '145.72'],
    ['Detour8', '0.651%', '274.67', '350.56'],
    ['冻结旧MLP8', '1.163%', '279.41', '411.21'],
    ['Control8', '1.152%', '441.36', '548.27'],
    ['Structured8', '1.418%', '427.12', '626.45'],
], [2.16, 1.68, 1.62, 1.60])
p('数据来自18个新百桌场景，每法3次。成本先逐场景计算相对下降再平均；时间先取场景内中位数，再跨场景汇总。规划时间包含初解、特征、选择器、修复与终验；公共导航和可达性预处理另计，不包含进程启动、模型加载及磁盘输出。', style='Caption')

# Page 2
h('2 学习收益与采用条件', new=True)
p('本轮应将“学习尚无价值”收敛为更准确的结论：固定8次修复时，学习选择器对规则选择器存在有限测试集质量收益；新增图结构的额外贡献和相同实际时间内的用途仍需区分。')
table(['配对比较', '平均降本', '95%场景区间'], [
    ['Structured 对 Detour', '0.7699%', '[0.4509%, 1.0842%]'],
    ['Structured 对 Control', '0.2651%', '[−0.1005%, 0.6223%]'],
    ['Structured 对旧MLP', '0.2550%', '[−0.0706%, 0.5802%]'],
    ['旧MLP 对 Detour', '0.5138%', '[0.1977%, 0.8319%]'],
], [2.55, 1.45, 3.06])
p('区间由20,000次场景配对bootstrap得到，不含训练种子不确定性，未作多重比较校正。旧MLP比较是审计追加的描述性结果。不能把0.77%全部归因于图输入，因为屏蔽图输入的Control本身也优于Detour约0.50%。')
p('旧MLP的总体规划中位数比为1.017，但逐场景耗时比中位数约1.088，P95也更高。只能说总体中位耗时接近，不能说每个场景只慢1.7%。旧权重未更新，结果变化来自测试分布与推理入口等条件，不能描述为又训练获得了提升。')
h('原版同时间预算与门槛', 2)
table(['400 ms软预算', '相对Fixed3降本', '实际中位 ms', '平均修复次数'], [
    ['Detour', '1.593%', '440.11', '23.37'],
    ['Control', '1.254%', '442.43', '10.48'],
    ['Structured', '1.488%', '444.86', '10.02'],
], [1.83, 1.80, 1.68, 1.75])
p('三法均54/54次超过400 ms。Structured对Detour为−0.1084%，区间[−0.4570%, 0.2026%]，没有同时间优势证据，也不能断言其显著更差。该截止允许正在执行的修复和强制终验超时。')
p('原采用条件要求相对Control及Detour均降本至少0.2%、区间下界大于0、总体规划中位时间比不超过1.25且全方案有效。Structured的Control比较未通过，时间比1.555也未通过，继续保留Fixed3／Detour合理。三个训练种子的正向趋势不能取消这些条件。')
h('冻结旧MLP的补充评测', 2)
supp_path = HERE / 'mlp_budget_supplement' / 'summary_for_report.json'
if supp_path.exists():
    supp = json.loads(supp_path.read_text())
    for paragraph in supp['paragraphs']:
        p(paragraph)
else:
    p('冻结后补充评测结果见本包独立记录。该结果在当前机器按同一软预算重新计时，与原Windows统计分开，不用于事后改变原主模型或采用门槛。')

# Page 3
h('3 小规模精确参照支撑规划方法的有效性', new=True)
p('原小规模实验包含27个独立场景，展开为45个场景与停靠罚项组合。精确动态规划重算得到44个最优状态、1个因任务数超限而unsupported；全部225份方案复验通过。当前默认精确上限是14个任务、128个候选，不是14张桌子。')
h('统一目标与精确模型', 2)
eq('J(P) = L(P) + λ |S(P)|')
p('L为避障导航折线路程，S为被访问的服务配置集合，主实验λ＝0.75等效米。该目标同时约束停靠选择和路线长度，尚不等于真实任务完成时间。若使用近似恒速和固定停靠准备时长，可用速度乘准备时长校准λ；升降变化和双臂操作时间另需模型与实测。')
p('对每个候选k，用任务位集合Cₖ表示任一机械臂可以服务的任务。令V(M,j)为覆盖集合M且最后停靠j的最小成本；由起点0到候选初始化，扩展时只保留新增覆盖的候选。')
eq('V(', sub('C','j'), ',j) = ', sub('D','0j'), ' + λ')
eq('V(M ∪ ', sub('C','k'), ', k) ← min {V(M ∪ ', sub('C','k'), ', k), V(M,j) + ', sub('D','jk'), ' + λ}')
p('所有任务覆盖后，再加末站返回起点的距离并取最小值。在非负最短路度量与非负停靠罚项下，无新增覆盖的停靠可被捷径替换且不增加成本，所以存在每次新增覆盖的最优路线。令n为任务数、m为候选数，时间和内存复杂度分别如下；最优性仅针对给定离散候选、F和D。')
eq('O(', sup('2','n'), sup('m','2'), '),   O(', sup('2','n'), 'm)')
h('去除空任务后的质量差距', 2)
table(['方法', '平均离散最优差距', '达到最优', '最坏差距'], [
    ['逐桌路线', '10.932%', '11 / 25', '57.432%'],
    ['固定停靠集合精确排序', '6.219%', '13 / 25', '57.432%'],
    ['Fixed3', '0.450%', '22 / 25', '8.564%'],
    ['Detour8', '0.102%', '24 / 25', '2.542%'],
], [2.70, 1.66, 1.34, 1.36])
p('统计限于λ＝0.75且取得精确证书的25个非空场景。它说明“仅把既定停靠点顺序排好”仍可能明显损失质量，覆盖选择与访问顺序联合处理有实际作用。这比仅与弱启发式比较更能支持研究内容二。不能把小场景0.102%的差距推广为百桌或400桌的最优误差。')
h('可靠性的条件性结论', 2)
eq(sub('P','out'), ' ∈ ', sub('F','model'), ' ,   J(', sub('P','out'), ') ≤ J(', sub('P','anchor'), ') + ε')
p('前提是存在通过终验的初解，ε为实现的数值比较容差。局部修复只接受覆盖完整且成本不增的候选，最后再次独立终验；失败则返回已经认证的初解。若初解也无法认证，接口明确报告失败和待服务任务，不宣称真实机器人不可行。这个保证不依赖网络预测准确。')

# Page 4
h('4 两项隔离维护改进', new=True)
h('外部用臂分配的保守误拒绝', 2)
p('原导出逻辑在规划表F允许左右臂时优先选左臂。如果审计表A只允许右臂，三条参考方案可能全部被拒绝；同一路线改用右臂即可验证通过。这是漏掉可行分配，不是放行不符合约束的方案，也不推翻原855份代理验证结果。')
p('补丁在导出分配阶段选择F与A共同允许的手臂；当前停靠点没有共同可用臂的任务，保留给后续已选停靠点。最终仍分别验证F和A，并实际执行调用者提供的service_checker。回调失败或异常仍拒绝，不能未经检查便假定另一臂可行。')
p('补丁在隔离副本中完成。9项原外部测试与11项新增回归共20项通过，原合成示例方案逐字段一致。新源码不属于原39场景冻结评测，合入后须记录新源码版本与哈希，保留原结果。')
p('分配使用A以后，论文应称“双表约束一致性终验”，不再把A称为算法未见过的盲测可达表。物理独立复核仍依赖外部checker与机器人资产。路线生成和规范化仍依据F，补丁只修复局部分配，不构成完备的可行性搜索。')
h('同一路线下缓存图嵌入', 2)
p('结构模型的图编码只依赖当前路线、窗口布局、任务候选语义、F、D及固定权重。同一环境内路线与窗口未改变时可复用图嵌入；预算统计、动作掩码、排序与STOP输出仍每步重新计算。任务插入、地图或机器人配置更新必须新建环境或使缓存失效。')
table(['18个原验证场景', '原版中位 ms', '缓存版中位 ms'], [
    ['完整规划', '627.38', '585.62'],
    ['选择器整体', '163.57', '121.77'],
    ['其中网络推理', '81.98', '42.85'],
], [3.02, 2.02, 2.02])
p('每场景每法3次交错运行。144次逐步全部头输出逐位一致，动作一致；108次完整规划对象一致并通过终验。总体规划中位下降6.66%，18场景中17个提速；另1场景有较大系统波动。可确认数值等价并显示提速潜力，不能据此保证稳定部署时延。')
p('这些是当前机器原验证集上的冻结后工程诊断，不与原Windows正式测试时延拼接，不重新定义采用门槛。即便速度继续改善，Structured相对Control的质量区间跨零这一结论也不会自动改变。')
h('维护文件', 2)
p('external_patch含修订external.py、统一差异文件和回归脚本；training含CachedGraphPolicy适配器、数值等价脚本及逐场景计时。原上传代码保持冻结。两个改进均不需要重新训练。')

# Page 5
h('5 三个研究内容的衔接与论证边界', new=True)
p('当前结果适合按“约束生成、联合规划、预算修复”组织三部分。硬件组合为问题提供特殊约束；论文创新应落在这些约束如何进入算法，以及它们怎样改变任务覆盖、路线质量和恢复能力上。')
table(['研究内容', '输入与输出', '已有证据与待补项'], [
    ['一 可达性与服务配置', '机器人和任务位姿 → 候选与双臂F', '外部接口已通；真实配置、IK见证和碰撞复核待闭环'],
    ['二 多停靠覆盖与路线', 'F及避障D → 初始完整方案', 'Fixed3、225方案与离散精确参照已完成'],
    ['三 预算内局部修复', '认证初解与预算 → 验证后的改善方案', 'Detour及冻结学习对照已完成；使用质量与时间共同评估'],
], [1.78, 2.26, 3.02])
h('第一部分向后两部分传递什么', 2)
eq(sub('F','ija'), ' = 1 ⇔ ', sub('Q','ija'), ' ≠ ∅')
p('Qᵢⱼₐ表示候选j下机械臂a满足任务i位姿、关节与碰撞等服务约束的构型集合。服务候选应明确底盘平面位置、航向、升降高度及服务模式。可达表来自已核验的运动学、关节限位、工具变换和碰撞检查，并保留IK关节见证与模型版本。第三部分仍可只在二维导航场景训练；机械臂约束通过F进入，不必把关节轨迹变成学习动作。')
h('第三部分应使用的研究表述', 2)
p('建议题名为“计算预算约束下的任务路线局部修复与学习引导”。核心技术是可验证初解、局部覆盖子问题精确修复、选择策略和终验回退。学习模型作为可替换的窗口选择器，规则选择器同样属于正式方法。第四版本身是收益监督学习，不能称为新增PPO或DDQN实验。')
p('若论文仍沿用“环境变化下在线重规划”，当前静态随机场景和局部搜索迭代不足以证明该结论。随机生成任务与运行途中出现新任务是不同实验。届时应在任务到达、候选失效或通道堵塞时锁定已执行前缀，以当前底盘状态为起点更新剩余任务、F、D及缓存，再比较重新全局规划与局部恢复；否则将在线表述收敛为当前已验证的预算修复。')
h('双臂与升降的准确范围', 2)
p('外部模式目前支持左右臂服务分配和离散服务高度，任务语义是单臂独立服务。它尚未证明双臂同步操作、负载均衡完成时间、升降过渡碰撞或完整机械臂轨迹可执行。可先按一臂工作、另一臂固定收拢的受控服务模式完成闭环，再讨论并行协作；不需要为保留“双臂”名称而虚构同步能力。')

# Page 6
h('6 真实配置接入与定稿工作', new=True)
p('AlohaMini1身份、固定提交引用和部分几何任务资料已经存在。当前缺口是与该版本一致、能够运行并独立复核的配置包，不是重新寻找机器人名称。不能把其他6R项目的参数，或尚未与URDF对齐的五关节范围直接作为真实限制。')
table(['配置项', '应交付的可执行内容'], [
    ['运动链与资产', '固定URDF及全部引用mesh，坐标系映射和文件哈希'],
    ['关节与工具', '有效零位、上下限、TCP、夹爪开口与接触规则'],
    ['升降定义', '参考零位、有效上下限；区分行程长度和绝对高度'],
    ['碰撞与服务模式', '碰撞几何、豁免对、安全间隙、非工作臂停放构型'],
    ['任务与复验', '抓取容差、IK关节见证、残差、碰撞日志和checker'],
], [1.72, 5.34])
p('先选小规模可检查场景覆盖左臂专属、右臂专属、双臂均可、不同高度及明确不可达案例。用固定资产生成F与审计证据，运行现有Fixed3／Detour，再通过独立IK和碰撞回调复核所选任务。涉及升降或接近动作时另验证过渡；接口成功和资产哈希一致不能替代运动学或物理检查。')
h('快速完成的顺序', 2)
p('第一步冻结v4作为正式实验结果，保留Fixed3快速模式和Detour质量模式。将本次外部补丁与图缓存作为独立维护版本，保留差异与回归证据。旧MLP保持冻结可选策略；是否提升为正式质量模式应依据同预算补充与尾部延迟，而不是增加训练轮数。')
p('第二步把实验图与表放入论文：第一部分展示候选服务区域、左右臂可达关系及验证误差；第二部分展示停靠选择、任务分配和小规模最优差距；第三部分展示同一场景修复前后路线、质量与耗时曲线、配对置信区间和超时比例。36／196桌分别报告，400桌仅3例应称压力规模可运行。')
p('第三步让MATLAB负责已有机器人运动学、工作空间、IK与结果可视化，Python负责prepare、plan、verify、evaluate和模型推理。交换数据以版本化任务、候选、F、D及见证为核心。取消重复写在多份脚本中的约束逻辑，保持一处生成约束和一处独立终验；不要为精简代码合并规划与验证实现。')
h('本轮可以收敛的结论', 2)
p('已经证明：在已测试场景的有限候选与代理模型内能稳定生成完整方案；小规模问题上覆盖联合规划优于只排序；局部修复有明确的相对认证初解不劣保证。尚不能证明：连续机器人全局最优、400 ms硬实时、双臂全过程可执行、新结构学习模型稳定优于强规则。优先完成机器人约束闭环与论文证据组织，不再把论文完成条件绑定为某个深度网络必须取胜。')

# Page 7
h('7 证据索引与复现说明', new=True)
p('本报告对应独立审查包rc3_v4_review_evidence.zip。解压后各脚本使用--source-root指向用户原交付包的解压根目录，--output-dir指向新目录。源码与结果分开存放，运行审查不会覆盖原冻结评测。')
table(['目录', '证据内容'], [
    ['experiments', '855份方案复验、357记录重建、配对CI、门槛复算和科学图'],
    ['exact', '45组合精确DP重算、225方案验证和质量差距'],
    ['training', '训练数据与六个权重核对、7项非Torch测试、图缓存诊断'],
    ['external', '外部接口、回调、20项非Torch测试和原反例'],
    ['external_patch', '隔离补丁、原新哈希、9＋11项回归和最小复现'],
    ['mlp_budget_supplement', '冻结旧MLP与Detour在当前机器的同预算补充'],
], [2.30, 4.76])
h('关键复现命令', 2)
p('主实验复验：python -B experiments/audit_v4.py --source-root 原包根目录 --output-dir 新审查目录')
p('小规模重算：python -B exact/audit_exact.py --source-root 原包根目录 --output-dir 新精确审查目录')
p('补丁回归：python -B external_patch/test_external_common_arm.py --source-root 原包根目录 --output-dir 新补丁结果目录')
p('图缓存诊断：python -B training/cache_diagnostic.py --source-root 原包根目录 --output-dir 新缓存诊断目录 --scenes 18 --repeats 3')
p('各目录内的README或审查说明提供依赖、适用范围与完整步骤。命令中的中文路径为占位说明，实际执行时替换为真实目录，并对含空格的路径加引号。')
h('主要证据来源', 2)
p('原交付：rc3_v4/results/原始重复方案、summary.json与decision.json；training/summary.json与六组日志及NPZ；small_results中的场景、方案、协议和精确状态；external.py、runtime.py、features.py及test模块。')
p('原机器人来源核查：rc3_v4/ROBOT_INPUT_STATUS.md及随包文档提取记录。固定提交引用已在本地文档中确认，远程URDF资产本轮未能独立获取；本报告不补造限位、TCP或碰撞配置。')
p('本次实际测试34项为external 9、audit 7、runtime 4、exact 7、features 3、legacy 4。原报告40项还包含Torch相关测试；当前环境未安装Torch，未声称重跑该6项。补丁20项包括重跑9项原external，不能与34项相加后称54项原单测。')
p('数值来源均可追溯到审查包JSON与CSV。原Windows性能、当前验证集图缓存诊断和当前机器旧MLP补充属于不同计时批次，应分别引用。科学图的置信区间以场景为重采样单位，不把方案次数、重复运行或训练种子当作新增独立环境样本。')

doc.core_properties.title = '第四版规划研究验收与定稿建议'
doc.core_properties.subject = '移动双臂机器人覆盖规划与学习模型验收'
doc.core_properties.author = ''
doc.core_properties.keywords = 'Fixed3, Detour, 可达性, 覆盖规划, 独立验证'
target = OUT / 'rc3_v4_acceptance_review.docx'
doc.save(target)
print(target)
