# Stage 3.5.2 冻结诊断协议

## 目的与范围

只诊断当前 Stage3.5.1 E1 的空间依赖、Gate/Residual 行为及折间差异；E0 是独立训练对照，E2 可选辅助。所有权重冻结，Eval/no_grad；不训练、不改架构、不改损失、不搜索超参数、不访问 Test loader。所有工具默认 dry-run，输出到不存在的新目录，部分结果不得自动恢复或覆盖。本任务不执行任何 Git 命令。

## 核查与复用

已核查 `model_ViLa_MIL_BiomedCLIP.py`、`cross_scale_modules.py`、`cross_scale_mapping.py`、`dataset_generic.py`、`core_utils.py`；Stage335/336/337、Stage340/342/343 工具；Stage3.3 历史稳定化、空间扰动/区分性记录；Stage3.4 互补性记录；Stage3.5.0/3.5.1 的来源映射和最新五折结果。

- Mapping 是正面积重叠关系的双向变长 CSR，不是固定 1:16，也不是逐行配对。Low Context 用有效父节点的反向 `parent_weight` 重新归一化求和；无有效父节点的 High 保留且 Context 为零。
- Router 在原始 High patch 特征进入原型 cross-attention 前作用。输入是归一化 High、归一化 Low Context、有效性和 High padding；没有直接输入 XY。Mapping 决定 Context 来源，空间可通过这些内容间接影响输出。
- Gate 为 sigmoid MLP，乘有效性。Residual 为 Gate×Context projection，逐 High patch 限制范数为 rho=0.10×原始 High 范数；High 主路径保留未归一化原特征。无可用 Context 的 High 残差为零。
- Global Low Prototype Conditioning 是独立可选模块，本阶段 E0/E1/E2 全部 OFF。
- Stage335 具备主要 Router 模式、严格加载和逐 Slide 输出；Stage336 提供 seed/配对指标；Stage337 提供 Context Mean 和 Low 特征空间统计。这些实际函数/模型方法被复用。
- Stage335 默认 split=test、会覆盖结果且多折仍使用同一个固定 checkpoint 参数；不能直接执行其旧 CLI。其 E1 构造器显式开启稳定化 Router，符合 Stage351 E1；包装器按来源映射提供每折 Checkpoint，先拒绝需要 key 清理的权重，加载严格。
- Stage340/342 的原 builder 也固定开启 Router，不适合直接加载 Clean High/Clean Dual；本包装器对 E0/E2 使用最小的关闭 Routing 构造器，保持严格加载，未改旧代码。
- Stage343 写死旧两折样本/纠错计数，并包含阈值拟合，不用于本阶段五折机制判断或诊断阈值选择。
- 旧 Stage336/337 的 Checkpoint 不等于本阶段 matched E1，历史数值仅作为背景，不能填充本阶段结果。

## 正式计划

默认五折，每折一次 E0 Normal、一次 E1 Normal、三次 E1 Context Shuffle（seed=352/353/354）、一次 E1 Context Mean、一次 E1 Residual Off，共 35 次折验证遍历、约 6776 次 Slide 前向。E2 默认不推理；`--include-e2` 增加五次正常遍历。不得挑选最优扰动种子。

| 标识 | 状态 | 同权重目的 |
|---|---|---|
| A0/B1 | E1 Normal | 正常映射基准 |
| A1 | E1 Context Shuffle | 去除局部 Context 内容与 High 的对应 |
| A2 | E1 Context Mean | 消除局部 Low Context 差别 |
| A3/B2 | E1 Residual Off | 同一 E1 权重下去除 Residual |
| B0 | E0 Normal | 独立训练 Clean High 对照 |

同一 Fold 的 E1 模型只加载一次，各模式仅设置已有诊断属性，参数不变。Normal 必须存在；E0 始终生成。只支持四种预先指定模式，不扩大 Route One/Mean 搜索。

### 扰动语义

复用真实 `_shuffle_parent_context()`：仅在本张 Slide 的 `high_parent_context_valid_mask` 有效行之间置换加权 Context，High/Slide/标签/坐标顺序、CSR、valid/padding 不变，无有效 Context 行不变。这是打乱聚合后的 High-row Context 内容，不是重建 Mapping 或置换 Low 坐标。生产 LowParentContext 的有效性已包括 High valid。

seed 由固定 base seed、Fold 和 Slide ID 的 SHA256 派生32位值；不依赖遍历顺序，记录逐 Slide seed。复用 CPU Generator 的实际 shuffle 实现。

复用真实 Context Mean：取本 Slide **有效且在前向 CSR 中有子节点的 Low parents 原始特征的非加权平均**，替换有有效 Context 的 High 行。它不是所有 High 聚合 Context 的平均，也不是跨 Slide 平均。无效行/元数据保留。

### Gate、Residual 口径

只读 hooks 捕获真实 route_score 输出、context_projection 输出和 Router 的输入/输出。Hooks 返回 None；不改变默认输出，不持久化完整特征矩阵。通过实际中间输出计算已有公式的标量统计，同时记录实际输出减输入的范数，以说明浮点消减误差。

- Gate 饱和阈值固定为 **<0.05 / >0.95**；主要统计只针对有效 mapped High。Unmapped 的零 Gate 不当作 Gate 饱和。无有效行时统计为 NaN，而非错误填0。
- 原始结构 unmapped 数与无可用有效 Context 数分别保存，避免混淆有父节点但父/High 无效的情况。
- Gate 的 Slide 内方差使用有效 High；空间分组保留全部合法 Low→High 边，一个 High 可属于多个组。每个 Low 区域内部按 `child_weight` 在保留子节点中重新归一化计算均值/方差；区域间方差给每个有效 Low group 相同权重。内部差异只汇总至少两个子节点的区域；区域不互斥，不将它解释为 ANOVA 独立分解或固定1:16。
- cap trigger 对比逐 patch **cap 前**残差范数与0.10×High范数。关闭 Residual 时 cap 前残差也置0，保留生产语义。
- 同时保存逐 patch 范数/ratio 分布和 WSI Frobenius 范数ratio；后者不是逐patch ratio均值。
- 固定20 bins 保存所有有效 mapped patches 的 Gate/ratio 直方图计数；统计使用全部 patch。仅 Normal 额外保存每 Slide 最多256个 High标量/坐标和最多256个 Low区域标量/坐标，以稳定seed取样，不保存特征、WSI、patch图片或case_id。
- Low空间统计复用 Stage337，固定最多256个采样配对；它的历史采样方式可能重复配对，属于描述性样本，不能宣称唯一/独立配对。
- Gate 不是病灶概率；Gate接近常数也不证明 Residual内容无条件化，需结合空间干预和 Low统计。

## 预测、类别与折间分析

AUC 正类为 **1 NonAdenocarcinoma**，临床腺癌为类0。预测保持概率argmax，无 threshold 搜索。Macro-F1沿用 sklearn macro/zero_division=0。

Margin同时记录 P1−P0、|P1−P0|及真实类别margin；配对结果保存平均概率绝对变化、signed margin平均变化、margin绝对变化、confidence margin变化。边界描述分层固定 **|P1−P0|≤0.10**，只作描述，不改变预测规则、不依据结果调整。

逐 Slide 对齐 ID和标签，输出both_correct/harmed/rescued/both_wrong、数量/比例、类别构成与E1 Normal Gate/Residual统计。B1→B2用于Residual直接作用，B0→B2用于独立训练模型间差异，B0→B1用于总差异。B0→B2不证明共适应是唯一原因，Checkpoint选中epoch也可能不同。

共享训练参数比较只读当前Checkpoint中的attention、cross-attention、norm、prototype与prompt ctx的L2/cosine，不保存权重值、不比较巨大的冻结预训练塔。参数坐标有非唯一性；不同epoch/训练过程的参数距离不能直接等同诊断语义或因果效应。

Fold4额外统计正负类概率排序配对credit（正确=1、并列=.5、错误=0）的变化，核对与AUC差一致；配合类别Confusion Matrix、class1 sensitivity/specificity及类0/1 recall解释排序与硬分类差异，不预设是校准问题。

五折指标按固定种子分别汇总；五折缺失不生成完整五折均值，其他种子不替补。单个病例及少量纠错仅作线索；不进行把相关五折当独立样本的显著性检验。

## 数据完整性、来源与中断

Checkpoint来自 Stage351 `common.py`白名单，不猜测文件名；预检核对来源、配置、schema、split和原结果。正常复评指标必须与Stage351选中epoch一致（atol=1e-7）才能生成job完成标记。实际模型前向使用Eval/no_grad；任何失败保留日志与lock，非零退出。

每个job仅在全部CSV写完后生成哈希完成标记；根完成标记记录整个manifest和job标记/参数差异文件hash。汇总排除未完成/无效job，不自动覆盖或续跑。元数据和预测/诊断CSV核对Slide覆盖、标签、模式、seed、权重来源；改变数据清单/Prompt/split或Checkpoint大小/mtime则报错。大权重/特征不全量hash，因此不能排除人为保留时间和大小替换内容。

所有默认工具只检查；明确 --execute 才推理、写统计报告或绘图。可以只做 Fold3/4预定子集，报告必须标记未覆盖完整五折，不能宣布五折机制结论。

## Q1–Q6 的证据等级和下一阶段

当前所有Stage352机制结果 **Pending**；Stage351支持“平均性能未稳定提升”，不能证明具体机制。

| 问题 | 判定证据 | 必须保留的解释边界 |
|---|---|---|
| Q1空间依赖 | Normal vs Shuffle三固定seed/Mean的逐Slide概率、margin、性能变化 | 小差异只能说明对这些局部扰动不敏感，不证明Low无信息 |
| Q2空间区分 | Gate内/间方差、饱和率、Low空间统计、干预 | 常数Gate也可投影不同Context；norm不含方向信息 |
| Q3残差作用 | B1 vs B2、实际范数/cap触发、harmed/rescued | 有特征变化不等于有效诊断贡献 |
| Q4共适应 | B0 vs B2、参数距离与选中epoch | 独立训练比较不能隔离唯一训练因果 |
| Q5折间差异 | Fold3受益与其他折退化分组；Fold4类别/排序/边界描述 | 关联不是因果，不按结果挑Slide或阈值 |
| Q6研究优先级 | 先列直接干预支持，再列相关线索，最后列Pending假说 | 不新增结构、不改超参数 |

不设人为“效果显著”阈值来自动判定成功；以连续量、变化病例和跨seed一致性报告。聚合报告会提供真实表格和判定边界，不自动断言失效机制。

进入Stage3.5.3的条件：默认35个job全部完成/核验、Normal复评对齐Stage351、五折三种扰动seed齐全、三种B对照与逐Slide/Gate/Residual统计完整，Q1–Q6明确区分实验支持/相关线索/证据不足。随后可研究Low/High互补性；本阶段不实现Stage3.6新结构。

## 历史限制

strict5目前为case-level受控划分，患者级独立性未确认；名义5x/20x实际约2.5x/10x；历史Checkpoint缺少当时代码/输入hash；验证折已参与开发；五折训练集重叠。全部限制延续，不改数据、划分和权重。
