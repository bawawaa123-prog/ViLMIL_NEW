"""Interpret authenticated existing Stage352 tables, without any model inference."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from stage353_common import *


def table(frame, columns):
    out=frame[columns].copy()
    for col in out.select_dtypes(include='number'):
        if col!='fold': out[col]=out[col].map(lambda v:f'{v:.6g}')
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in out.itertuples(index=False,name=None))


def interpretation(run, report):
    _,frames=load_stage352(run,report)
    names=['routing_intervention_metrics','routing_gate_statistics','routing_residual_statistics',
           'fold_comparison','correctness_group_statistics','class_metrics','parameter_differences',
           'fivefold_intervention_summary','ranking_changes']
    data={n:pd.read_csv(Path(report)/(n+'.csv')) for n in names}
    metrics=data['routing_intervention_metrics']
    normal_gate=data['routing_gate_statistics'].query("mode == 'normal'").groupby('fold').mean(numeric_only=True).reset_index()
    normal_res=data['routing_residual_statistics'].query("mode == 'normal'").groupby('fold').mean(numeric_only=True).reset_index()
    folds=data['fold_comparison']
    pairs=folds[folds.comparison=='B0_to_B1']
    off=folds[folds.comparison=='B1_to_B2']
    groups=data['correctness_group_statistics']
    classes=data['class_metrics']
    parameters=data['parameter_differences']
    lines=['# Stage 3.5.2 最终机制解释', '',
        '本报告仅整理既有35/35项Validation诊断；来源、完成标记、CSV哈希和E0/E1权重身份已核验。没有重新推理或训练。', '',
        f'结果目录：`{Path(report).resolve()}`。AUC正类为1（NonAdenocarcinoma），类0为腺癌；argmax分类；所有“病例”统计实际为Slide级，不等同独立患者。', '',
        '## Q1 局部空间Context的实际影响', '',
        '**实验支持：对本阶段定义的局部Context扰动，最终输出敏感性较低，但存在小幅连续概率和排序变化。**']
    for mode in ('context_shuffle','context_mean','residual_off'):
        part=metrics[metrics['mode']==mode]
        lines.append(f"- {mode}: 逐折/seed平均概率绝对变化范围 {part.mean_absolute_probability_change.min():.6f}–{part.mean_absolute_probability_change.max():.6f}；各任务预测类别变化合计 {int(part.changed_prediction_count.sum())}。Shuffle合计跨三个seed重复统计，不是不同病例数。")
    shuffle=metrics[metrics['mode']=='context_shuffle']
    mean=metrics[metrics['mode']=='context_mean']
    lines += ['','Shuffle/Mean在Fold 0、1、2、4均无类别变化；Fold3各Shuffle seed和Mean均改变2张Slide，每次损害1、纠正1，ACC不变。Fold2 Shuffle的AUC相对Normal改善0.002028–0.003102，Mean改善0.002983；这说明正常局部对应未稳定提供排序优势。', '',
        '**证据不足：**不能据此证明Low无信息、Router完全没使用Low，或空间映射本身错误。扰动针对聚合后有效High-row Context，保持High顺序和合法Mapping语义；只覆盖既定干预。', '',
        '## Q2 Patch级Gate区分性', '',
        '以下为每张Slide先统计，再在Fold内等权平均；饱和比例仅针对有效mapped High，阈值>0.95。', '',
        table(normal_gate,['fold','gate_mean','gate_variance','within_parent_gate_variance_mean','between_parent_gate_mean_variance','gate_near_one_fraction']), '',
        '**实验支持：**Fold0/4的Gate在Slide内近似常数；Fold1变化也较小。Fold3平均Gate约0.980867，>0.95比例的Slide均值约99.3586%，呈高值饱和。Fold2不达到0.95饱和阈值，但平均Gate较高。', '',
        '**间接证据：**局部Gate调制有限，可能限制Patch级选择能力；但常数Gate乘以不同投影Context仍可产生不同残差方向，Gate统计不是残差内容的完整描述。多父区域有重叠，区域内/间方差不能作独立ANOVA因果分解。', '',
        '**证据不足：**Gate不是病灶概率，也不能据其高低判定病理定位正确性。', '',
        '## Q3 Residual的直接贡献', '',
        '下表为有效mapped Patch residual/high范数比例的Slide均值，再对Fold等权平均；cap触发统计使用cap前残差。', '',
        table(normal_res,['fold','mapped_residual_ratio_mean','cap_trigger_fraction_mapped','no_usable_context_fraction']), '',
        '**实验支持：**实际平均残差比例约0.354%–2.954%，低于逐Patch的rho=10%上限；全部Normal Slide的cap触发比例为0。当前推理证据不支持“残差持续被cap截断”这一解释，但不能据此评价训练期cap作用。', '',
        table(off,['fold','changed_prediction_count','correct_to_wrong','wrong_to_correct','delta_auc','delta_accuracy','delta_macro_f1']), '',
        '关闭Residual共改变4张Slide：Fold0/1/2均0，Fold3改变3（损害2、纠正1），Fold4改变1（损害1）。相比Normal，五折平均AUC增加0.001028，ACC下降0.002067，Macro-F1下降0.002168。Residual对少量分类边界有净帮助，对排序的作用折间不一致。', '',
        '**证据不足：**特征范数较小不等于所有病例贡献为零；也不能由平均AUC改善推导应关闭Residual训练或改变rho。', '',
        '## Q4 独立训练差异与共适应', '',
        table(pairs,['fold','mean_absolute_probability_change','changed_prediction_count','correct_to_wrong','wrong_to_correct']), '',
        '**实验支持：**E0→E1的逐折平均概率绝对变化为0.038016–0.082923，明显大于同权重Residual Off的0.000954–0.009801。Fold0/1/2关闭Residual后仍低于E0，Fold3关闭后仍高于E0。这排除了“仅当前推理残差即可解释全部E0/E1差异”的简单解释。', '',
        '**间接证据：**共享训练参数存在差异，可作为训练轨迹/共适应线索。例如：', '']
    selected=parameters[parameters.parameter.isin(['learnable_image_center','prompt_learner.ctx','cross_attention_1.in_proj_weight'])]
    lines += [table(selected,['fold','parameter','relative_l2_to_e0','cosine_similarity']), '',
        '**证据不足：**不同Checkpoint由独立训练和不同选中epoch产生，参数空间存在非唯一性；参数距离不证明诊断信息变化或唯一共适应因果。不能把E0与E1收益直接归为Low提供正确信息。', '',
        '## Q5 Fold3、Fold4与类别差异', '',
        '**实验支持：**Fold3 E1相比E0纠正5、损害2，净增3张Slide，ACC提高3/193=0.015544。即使关闭Residual，E1仍相对E0纠正4、损害2，净增2；因此收益只有部分与当前Residual的直接分类作用一致。正确局部Context扰动并未降低Fold3 ACC，不能证明这5例由空间Low纠错。', '']
    fold3groups=groups[(groups.fold==3)&(groups.comparison=='B0_to_B1')]
    lines += [table(fold3groups,['group','n','class0_count','class1_count','near_boundary_count','gate_mean_mean','mapped_residual_ratio_mean_mean']), '',
        '上述边界数基于E0的固定|P1−P0|≤0.10规则；分组量小，不将Gate/Residual均值差异宣称为稳定规律。', '',
        'Fold4 E1相对E0纠正2、损害3，净少1张正确Slide；正负概率排序配对改善46、恶化28，净18/8448使AUC提高0.002131。因此排序改善与硬分类退化同时成立，无需先假定概率校准原因。', '',
        table(classes[(classes.fold.isin([0,3,4]))&(classes['mode']=='normal')],['fold','arm','tn','fp','fn','tp','recall_adenocarcinoma','recall_nonadenocarcinoma']), '',
        'Fold0损害8例均为类1，体现类别行为变化；该现象与独立训练过程相关，不自动说明Low内容错误。', '',
        '## Q6 下一阶段研究优先级', '',
        '1. **优先：独立Low/High匹配互补性。**当前尚无严格匹配的五折Low证据。确认G3（High错Low对）是否存在、是否跨折重复，同时量化G2干扰风险；这是Stage3.5.3的直接目标。',
        '2. **随后：互补病例与Routing纠错重叠。**用同Slide预测关联G3/G2与E1纠错/损害，区分独立Low优势与训练轨迹差异。',
        '3. **待验证：组织上下文与高倍率细节的条件差异。**使用透明病例选择和专家复核；没有形态标注时只能报告候选关注模式。',
        '4. **证据不足：具体新结构。**现在不选择新的Attention/Gating/Fusion、不搜索阈值、不重新训练Stage352模型。', '',
        '## 可完成范围与限制', '',
        'Stage352计算、完整性核验、图表和本机制解释已完成。仍未知的是独立Low信息是否在特定样本条件下可稳定补充High，以及现有Routing是否利用这种优势。',
        'strict5为case-level受控划分，patient-level独立性未确认；验证集参与早停/开发；五折训练集重叠、单seed，不能当作独立患者样本做显著性声明；名义5x/20x实际约2.5x/10x。历史记录缺少当时完整输入/源码hash；大Checkpoint仅路径/大小/mtime校验。', '']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,default=DEFAULT_352_RUN)
    p.add_argument('--report-dir',type=Path,default=DEFAULT_352_REPORT)
    p.add_argument('--output',type=Path,default=BASE.parent/'02_routing_diagnostics/stage352_final_interpretation.md')
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    print('Python:',matched.environment())
    body=interpretation(args.run_dir,args.report_dir)
    if args.output.exists(): raise FileExistsError(args.output)
    if not args.execute:
        print('Dry-run: verified existing tables; no document written.');return
    with args.output.open('x') as f:f.write(body)
    print('Interpretation:',args.output)


if __name__=='__main__':
    main()
