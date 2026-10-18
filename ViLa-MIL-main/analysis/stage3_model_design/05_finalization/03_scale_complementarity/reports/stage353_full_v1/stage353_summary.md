# Stage 3.5.3 Independent Low/High complementarity

完整五折配对计算已完成；病理形态证据须按复核状态解释。

Validation only；AUC正类=1 NonAdenocarcinoma；预测为argmax。H来自Stage351 E0/Stage352复评，L来自本阶段匹配独立训练。

- Fold 0: G1/G2/G3/G4=161/21/5/7; G3/High错误=0.416667; G2=21; Oracle ACC upper bound=0.963918。
- Fold 1: G1/G2/G3/G4=168/19/5/2; G3/High错误=0.714286; G2=19; Oracle ACC upper bound=0.989691。
- Fold 2: G1/G2/G3/G4=160/21/5/7; G3/High错误=0.416667; G2=21; Oracle ACC upper bound=0.963731。
- Fold 3: G1/G2/G3/G4=167/12/6/8; G3/High错误=0.428571; G2=12; Oracle ACC upper bound=0.958549。
- Fold 4: G1/G2/G3/G4=156/25/9/4; G3/High错误=0.692308; G2=25; Oracle ACC upper bound=0.979381。

**实验支持的五折描述：**
- Q1：L−H五折平均AUC/ACC/Macro-F1差值=-0.040900/-0.070221/-0.080848。
- Q2：五折G3数量=[5, 5, 5, 6, 9]；五折Slide观测合计G3=30，不能视为独立患者计数。
- Q3：五折G2数量=[21, 19, 21, 12, 25]；G3大于G2的Fold数=0/5。
- Q5：G3中被E1纠正的逐折数量=[2, 1, 1, 3, 1]；G2中被E1损害的逐折数量=[3, 1, 2, 0, 1]。重叠是相关线索，不是Low导致Routing收益的因果证据。

## Q1–Q6 与证据等级

- Q1 性能差异：independent_scale_metrics.csv 与 fivefold_scale_statistics.csv；后者只在五折齐全时生成均值/总体标准差。差值不作独立样本显著性检验。
- Q2 纠错数量：complementarity_summary.csv 的 G3 数量、全部样本比例及 High 错误内纠错率。实验支持仅限这些模型/验证集。
- Q3 机会与风险：同表比较 G3/G2；G3−G2 是潜在数量差，不是融合策略的可部署收益。Oracle 使用真实标签，仅提供硬分类上限，不报告标签驱动的 Oracle AUC。
- Q4 稳定条件：case_group_statistics.csv 与 conditional_complementarity.csv 按预定Margin区间、类别、Fold描述，比较各折方向与样本数。组织形态仍 Pending，除非补充独立专家复核；小组统计不能证明可迁移规则。
- Q5 Routing利用：routing_complementarity_association.csv 与 routing_group_statistics.csv 对齐G3/G2与E1纠错/损害，提供Gate/Residual/Low空间统计。病例重叠是间接证据，不足以证明Low导致Routing收益。
- Q6 创新假设：先确认跨折存在的G3条件与G2风险，再研究现有Routing未利用的组织上下文是否真实且可泛化；不预选模块，不拟合验证集选择器。尚未获得数据的方向为证据不足。

## 验收与限制

计算验收：五折Low训练来源/配置/Checkpoint复评通过，五折H/L严格对齐，完整性校验通过，统计表和病例复核清单齐全。研究验收：审阅Q1–Q6并按证据等级撰写结论；若无专家资料，形态问题明确Pending，不宣称病理真值。
Margin分层固定为[0,.10],(.10,.50],(.50,.90],(.90,1]；没有阈值调优/选择器/融合搜索。统计以Slide为单位，患者独立性尚未证明；strict5 case-level、验证集参与开发、五折训练集重叠、单训练seed、名义5x/20x实际约2.5x/10x。历史输入hash不足，大Checkpoint仅路径/大小/mtime校验。
公开准备表使用HMAC匿名case_token；private/仅本地使用，含真实ID映射和salt，禁止随公开报告分享。图像同样属于本地复核资料。
