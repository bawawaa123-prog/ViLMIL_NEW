# Stage 3.5.3 冻结研究协议

本阶段研究独立Low与Clean High的条件互补性。不是模型搜索：不增加网络、损失或融合，不调整分类阈值、Prompt、训练超参数或strict5。所有正式指标来自Validation；AUC代码正类=1 NonAdenocarcinoma，临床腺癌=0。预测规则沿用概率argmax。

## 来源核查与复用决定

| 来源 | 实际权重Fold | 核查结果 | 本阶段处理 |
|---|---|---|---|
| Stage351 E0 Clean High | 0–4 | 来源白名单、strict5、配置、完成日志、schema、选中epoch及Stage352复评全部可核验 | 权重不变，直接复用Stage352 E0 Validation预测 |
| Stage342 Independent Low | 0、1 | 独立Low loss，无Warm Start/Global；实际checkpoint含6个Router参数，routing/stabilization=True。seed1、lr1e-4、Adam、patience10、proto16与split有证据 | 保留历史背景，不能自动当作当前Clean协议；准备新五折 |
| Stage2 Low | 0–4 | 实际checkpoint无Router、proto16；严格split与选中Validation ACC的summary一致性逐折核查。旧配置未记录Routing/Global/Warm Start字段；缺少当时完整argv、Prompt/hash和训练源码版本 | 因追溯不足不纳入正式匹配。不是声称其训练有错误；旧Test评价本身不改变权重，也不是唯一拒绝理由 |
| Stage341 Dual-trained | 0、1 | scale_mode=dual，Routing开启，共同训练 | 不能当独立Low；不用于本阶段主对照 |

逐项实际字段、checkpoint身份、saved split、prototype shape、选中epoch一致性、Test列状态见 `reports/preparation_20261013/source_audit.json`。缺记录与确认不一致分开记录。Stage2文档曾写监控Validation loss；当前真实 `validate()` 向EarlyStopping传入val_error，等价选择首次最高Validation ACC。旧summary可检查是否与这一标准一致，但这不能补出当时的源码hash。

Stage342 Router作用于High分支，而Low loss取logits_low；不能据配置开关就声称Router影响了Low预测。历史模块初始化可能改变随机流，且缺少当时输入/代码hash；本阶段使用清晰的关闭Router协议。Stage350已修复Router初始化扰动共享随机流，历史模型不能据当前实现反推完全一致。

历史Stage342推理工具可以参考其严格ID配对和互补分组，但原builder固定开启Router，旧High源是Stage334 routed High，且包含alpha sweep。Stage343写死两折计数并拟合置信度规则，跨折迁移曾失败。本阶段复用Stage351来源/命令/metric validator、Stage352完整性/配对/class helper、Stage335 dataset/prompts与Stage336 seed；不执行旧CLI，不开展alpha/阈值搜索。

## 匹配Independent Low训练

`low_command()`逐项复用Stage351 E0训练argv，只变更scale_mode=low、实验名称与新输出位置。每折独立seed=1，strict5原始train/val/test不变；BiomedCLIP现有512维features、原Prompt CSV、proto16、hidden192；Adam base LR=1e-4、reg=1e-5、CE、80 epoch上限、patience10、原drop_out标志。Prompt LR沿用原自动值0.001，文本塔冻结；Routing/Stabilization/Global/Warm Start全部关闭。

训练调用现有main.py和core_utils，不修改它们。现有模型仍计算两尺度分支，但scale_mode=low仅以logits_low构造训练loss；这不同于Dual共同训练，未通过High标签loss训练Low模型。Checkpoint按首次严格最高Validation ACC选择，相等不覆盖；不用AUC、CE或Test选模型。每折开始前检查所有计划来源，不自动续训；只跳过验证通过的已完成本阶段Fold。中断保留锁和文件，需另用新输出版本或人工排查，不能覆盖部分结果。

训练manifest记录确切argv、CSV/Prompt/split与核心源码hash、依赖版本和特征路径。由于main.py的旧settings不记录reg、Prompt或全部flags，新来源以manifest argv作为必要证据，不能靠补猜settings。Checkpoint不全量hash，以路径/大小/mtime识别；特征内容不全量hash，局限明确保留。

## 预测与对齐

High沿用Stage352全量完整E0 CSV，源manifest、job/root完成标记、文件hash及Stage351三项指标均必须通过。复制原CSV字节避免二次序列化概率舍入；不重新训练或重新推理High。Low由对应Fold新训练checkpoint严格加载，eval/no_grad，确定seed，固定Validation loader；每折复评AUC/ACC/Macro-F1须与选中epoch一致，容差1e-7。

不能使用任意checkpoint模板、把Fold0权重用作Fold1、使用Dual branch代替独立Low或拼合旧Test结果。严格核对fold、arm、split、checkpoint、Slide完整覆盖、标签、重复、概率范围/和/argmax。根完成标记只在所有计划完成时生成，hash链接到逐Fold完成标记和两个CSV。完整五折均值仅在0–4各Fold齐全时生成；使用总体标准差ddof=0，不将Fold视作独立实验样本作显著性夸大。

## 互补性和固定条件

G1=High对Low对，G2=High对Low错，G3=High错Low对，G4=两者错。统计每组数量、比例、类别组成、各模型概率/置信度/margin及概率差；两模型分别输出类别0/1召回、TN/FP/FN/TP，AUC固定使用P1。

G3/全部样本反映纠错机会；G3/(G3+G4)反映High错误内的潜在纠错率；G2/(G1+G2)反映High正确内的干扰风险。分母为0返回NaN，不伪装为0。错误Jaccard=G4/(G2+G3+G4)。潜在G3−G2仅是数量差，不代表任意融合规则净收益。

Oracle ACC=(G1+G2+G3)/N，相对High上限增量=G3/N。**Oracle使用真实标签，仅是硬分类纠错上限，不能当作部署性能；不报告标签驱动的Oracle AUC。** 不训练验证集标签选择器，不尝试已失败的confidence switch，不搜索alpha。

Confidence=max(P0,P1)；signed margin=P1−P0；absolute margin=|P1−P0|；true-class margin=P(label)−P(other)。预先固定High absolute margin区间 `[0,.10]`、`(.10,.50]`、`(.50,.90]`、`(.90,1]`，相应confidence范围为[.50,.55]、(.55,.75]、(.75,.95]、(.95,1]。这些只用于描述，不能改预测规则。每折/类别/区间列明样本数、G3/G2、纠错和干扰率，观察方向是否重复，不挑选最优区间。

## 与Stage352的关联

只关联相同Fold、Slide、label的Normal E1 Gate/Residual/Low空间统计。High概率必须与Stage352 E0逐行完全相同。关联结果包含G3/G2与E1纠错/损害的重叠以及各组Gate、区域内外变化、残差、Low余弦/方差/dispersion描述量。

G3与E1救回相交仅是间接证据；E0/E1独立训练，不足以证明Low导致救回。Fold3的5例E1救回是否也是G3须等待新独立Low预测，当前Pending。小组差异不自动成为跨Fold规律，Attention/Gate不是病灶真值。

## 病例与组织形态复核

透明选择所有G3；G2对照在每Fold/真实类别内按原Slide ID字典序，选择 `min(可用G2,max(1,该类别G3数量))`，不按显著性、Gate或概率改变排序。分组跨类别无G2时不强行替补。复核模板保留Pending与空观察栏，不自动生成临床诊断。

现有WSI、patch PNG、H5坐标和Mapping可复用。历史可视化工具与大量attention/overlay产物存在，但多为Dual或外部数据集，不能认作当前独立H/L的注意力。Stage3.4目录本次限定核查没有发现可用于本比较的专家病理标注/独立H-L attention资产。

旧 `forward_with_attention()`最后固定用logits_low+logits_high，忽略scale_mode且缺少正常Mapping/Router路径，不能用于本阶段预测。新工具给**正常forward**的cross_attention_1添加只读Hook，Low选第一次调用、High选第二次调用，取得prototype-to-patch weights；先与本阶段独立概率核对。以跨原型等权平均定义注意力，不是分类贡献、GatedAttention prototype权重或病灶概率。CPU合成测试验证Hook不改变预测、shape与两次调用顺序。

默认复用Stage352有界Routing坐标散点，输出本地复核资源清单。显式 `--extract-attention --execute` 才对已选病例进行新推理，保存Top10坐标、最多1024等间距Patch的注意力散点、从已有Patch PNG制作拼图，以及Top High到全部Low parent的真实CSR对应。复用旧patch filename resolver，缺Patch则记Pending，不自动读取WSI重生成数据。可更改top_k=1..32作为导出量，不能据结果重选病例。模型关注模式与病灶面积、腺体结构等候选假设须专家确认；无专家复核则形态问题保持Pending。

## 标识、验收和研究边界

公开准备的病例表仅输出HMAC匿名case_token；随机salt与真实Slide映射保存在本地`private/`目录(mode700)，mapping/salt文件mode600。相同报告内同Slide token稳定；新报告使用新salt，复核始终读取该报告的mapping。不能将private或真实图像随公开报告分享。未经脱敏的训练/推理输出与图像目录mode700，不输出其他患者身份字段。

计算验收：Low五折匹配训练完成、对应权重Validation复评通过、High来源无变化、完整ID/标签/概率配对、各表/manifest齐全、无Invalid/Pending折。研究验收：基于真实表回答Q1–Q6并区分实验支持/间接线索/证据不足；组织形态无专家材料可明确Pending，但不能宣称已证明形态机制。

进入Stage3.6前应明确：G3是否跨折存在；G2是否更大；什么预定描述条件跨Fold重复；E1是否救回同一G3；关注组织结构是否有专家支持。此后才设计待独立验证的创新，不以本验证集同集拟合结果证明泛化。

限制延续：case-level strict5不等同已证实patient-level独立；Validation参与开发和早停；Fold训练集重叠；单seed；名义5x/20x实际约2.5x/10x；历史Test已被查看过；源hash不完整与大权重/特征未全量hash。不改变数据或划分。
