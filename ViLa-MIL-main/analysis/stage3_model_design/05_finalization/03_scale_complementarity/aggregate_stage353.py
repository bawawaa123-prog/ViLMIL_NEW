"""Integrity-checked complementarity report; default dry-run, no real inference."""
import argparse
from pathlib import Path
import secrets
import numpy as np
import pandas as pd
from stage353_common import *
from run_independent_inference import inspect_inference
from analyze_complementarity import pair_cases, summarize_cases, model_metrics, associate, select_review_cases


def fivefold_metrics(metrics):
    rows=[]
    for arm,part in metrics.groupby('arm'):
        if sorted(part.fold.tolist())!=list(range(5)): continue
        for key in ('val_auc','val_acc','val_macro_f1','recall_adenocarcinoma','recall_nonadenocarcinoma'):
            rows.append(dict(arm=arm,metric=key,n_folds=5,mean=float(part[key].mean()),
                std_population=float(part[key].std(ddof=0)),status='Fivefold'))
    return pd.DataFrame(rows,columns=['arm','metric','n_folds','mean','std_population','status'])


def write_report(output, state, frames, routing_frames, stage352_report, source_manifest):
    metrics, summaries, groups, strata, cases, associations, associated_groups = [],[],[],[],[],[],[]
    tables={name:pd.read_csv(Path(stage352_report)/name,dtype={'slide_id':str}) for name in
            ('routing_gate_statistics.csv','routing_residual_statistics.csv','low_context_statistics.csv')}
    for fold,(high,low) in sorted(frames.items()):
        c=pair_cases(high,low,fold)
        summary,g,s=summarize_cases(c)
        metrics.extend(model_metrics(high,low,fold)); summaries.append(summary)
        groups.append(g); strata.append(s); cases.append(c)
        selected=[]
        for name in tables:
            table=tables[name]
            selected.append(table[(table.fold==fold)&(table.arm=='E1')&(table['mode']=='normal')&(table.seed==0)])
        a,ag=associate(c,routing_frames[(fold,'E0','normal',0)],routing_frames[(fold,'E1','normal',0)],*selected)
        associations.append(a); associated_groups.append(ag)
    output=reserve(output)
    private=reserve(output/'private',private=True)
    salt=secrets.token_bytes(32)
    with (private/'id_salt.bin').open('xb') as f: f.write(salt)
    (private/'id_salt.bin').chmod(0o600)
    allcases=pd.concat(cases,ignore_index=True) if cases else pd.DataFrame(columns=['slide_id','fold','label','group'])
    mapping=allcases[['fold','slide_id']].drop_duplicates().copy()
    mapping['case_token']=mapping.slide_id.map(lambda s:case_token(salt,s))
    if mapping.groupby('case_token').slide_id.nunique().gt(1).any():
        raise ValueError('Pseudonym collision')
    mapping.to_csv(private/'slide_id_mapping.csv',index=False)
    (private/'slide_id_mapping.csv').chmod(0o600)
    def safe(frame):
        if 'slide_id' in frame:
            frame=frame.copy(); frame.insert(0,'case_token',frame.slide_id.map(lambda s:case_token(salt,s)))
            frame=frame.drop(columns='slide_id')
        return frame
    metricframe=pd.DataFrame(metrics)
    metricframe.to_csv(output/'independent_scale_metrics.csv',index=False)
    (fivefold_metrics(metricframe) if len(metricframe) else pd.DataFrame()).to_csv(output/'fivefold_scale_statistics.csv',index=False)
    differences=[]
    for fold,part in metricframe.groupby('fold') if len(metricframe) else []:
        h=part[part.arm=='H'].iloc[0]; l=part[part.arm=='L'].iloc[0]
        differences.append(dict(fold=int(fold),comparison='L_minus_H',
            **{key:float(l[key]-h[key]) for key in ('val_auc','val_acc','val_macro_f1','recall_adenocarcinoma','recall_nonadenocarcinoma')}))
    pd.DataFrame(differences).to_csv(output/'paired_scale_differences.csv',index=False)
    pd.DataFrame(summaries).to_csv(output/'complementarity_summary.csv',index=False)
    fivecomp=[]
    if len(summaries)==5 and sorted(r['fold'] for r in summaries)==list(range(5)):
        comp=pd.DataFrame(summaries)
        for key in ('G2_count','G3_count','G2_fraction','G3_fraction','high_error_rescue_rate','oracle_accuracy_upper_bound'):
            fivecomp.append(dict(metric=key,n_folds=5,mean=float(comp[key].mean()),std_population=float(comp[key].std(ddof=0))))
    pd.DataFrame(fivecomp,columns=['metric','n_folds','mean','std_population']).to_csv(output/'fivefold_complementarity_statistics.csv',index=False)
    safe(allcases).to_csv(output/'case_level_complementarity.csv',index=False)
    for name,items in [('case_group_statistics.csv',groups),('conditional_complementarity.csv',strata),
            ('routing_complementarity_association.csv',associations),('routing_group_statistics.csv',associated_groups)]:
        safe(pd.concat(items,ignore_index=True) if items else pd.DataFrame()).to_csv(output/name,index=False)
    review=select_review_cases(allcases) if len(allcases) else allcases
    safe(review).to_csv(output/'case_review_selection.csv',index=False)
    # Expert observations remain unfilled until an actual review; no generated pathology claims.
    checklist=safe(review)[['case_token','fold','group','label']].copy()
    for column in ('review_status','reviewer_code','candidate_low_structure','candidate_high_detail','same_region_comparison','expert_annotation_available','notes'):
        checklist[column]='Pending' if column=='review_status' else ''
    checklist.to_csv(output/'pathology_review_template.csv',index=False)
    state.to_csv(output/'experiment_status.csv',index=False)
    complete = sorted(frames)==list(range(5)) and state.status.eq('Complete').all()
    text=['# Stage 3.5.3 Independent Low/High complementarity', '',
        '完整五折配对计算已完成；病理形态证据须按复核状态解释。' if complete else 'Pending：尚无完整五折配对证据；不能报告五折结论。', '',
        'Validation only；AUC正类=1 NonAdenocarcinoma；预测为argmax。H来自Stage351 E0/Stage352复评，L来自本阶段匹配独立训练。', '']
    for row in summaries:
        text.append(f"- Fold {row['fold']}: G1/G2/G3/G4={row['G1_count']}/{row['G2_count']}/{row['G3_count']}/{row['G4_count']}; G3/High错误={row['high_error_rescue_rate']:.6f}; G2={row['G2_count']}; Oracle ACC upper bound={row['oracle_accuracy_upper_bound']:.6f}。")
    if complete:
        text += ['', '**实验支持的五折描述：**']
        d=pd.DataFrame(differences)
        text.append(f"- Q1：L−H五折平均AUC/ACC/Macro-F1差值={d.val_auc.mean():+.6f}/{d.val_acc.mean():+.6f}/{d.val_macro_f1.mean():+.6f}。")
        text.append(f"- Q2：五折G3数量={[r['G3_count'] for r in summaries]}；五折Slide观测合计G3={sum(r['G3_count'] for r in summaries)}，不能视为独立患者计数。")
        text.append(f"- Q3：五折G2数量={[r['G2_count'] for r in summaries]}；G3大于G2的Fold数={sum(r['G3_count']>r['G2_count'] for r in summaries)}/5。")
        agr=pd.concat(associated_groups,ignore_index=True)
        text.append(f"- Q5：G3中被E1纠正的逐折数量={agr[agr.group=='G3'].routing_rescued_count.tolist()}；G2中被E1损害的逐折数量={agr[agr.group=='G2'].routing_harmed_count.tolist()}。重叠是相关线索，不是Low导致Routing收益的因果证据。")
    text += ['', '## Q1–Q6 与证据等级', '',
        '- Q1 性能差异：independent_scale_metrics.csv 与 fivefold_scale_statistics.csv；后者只在五折齐全时生成均值/总体标准差。差值不作独立样本显著性检验。',
        '- Q2 纠错数量：complementarity_summary.csv 的 G3 数量、全部样本比例及 High 错误内纠错率。实验支持仅限这些模型/验证集。',
        '- Q3 机会与风险：同表比较 G3/G2；G3−G2 是潜在数量差，不是融合策略的可部署收益。Oracle 使用真实标签，仅提供硬分类上限，不报告标签驱动的 Oracle AUC。',
        '- Q4 稳定条件：case_group_statistics.csv 与 conditional_complementarity.csv 按预定Margin区间、类别、Fold描述，比较各折方向与样本数。组织形态仍 Pending，除非补充独立专家复核；小组统计不能证明可迁移规则。',
        '- Q5 Routing利用：routing_complementarity_association.csv 与 routing_group_statistics.csv 对齐G3/G2与E1纠错/损害，提供Gate/Residual/Low空间统计。病例重叠是间接证据，不足以证明Low导致Routing收益。',
        '- Q6 创新假设：先确认跨折存在的G3条件与G2风险，再研究现有Routing未利用的组织上下文是否真实且可泛化；不预选模块，不拟合验证集选择器。尚未获得数据的方向为证据不足。', '',
        '## 验收与限制', '',
        '计算验收：五折Low训练来源/配置/Checkpoint复评通过，五折H/L严格对齐，完整性校验通过，统计表和病例复核清单齐全。研究验收：审阅Q1–Q6并按证据等级撰写结论；若无专家资料，形态问题明确Pending，不宣称病理真值。',
        'Margin分层固定为[0,.10],(.10,.50],(.50,.90],(.90,1]；没有阈值调优/选择器/融合搜索。统计以Slide为单位，患者独立性尚未证明；strict5 case-level、验证集参与开发、五折训练集重叠、单训练seed、名义5x/20x实际约2.5x/10x。历史输入hash不足，大Checkpoint仅路径/大小/mtime校验。',
        '公开准备表使用HMAC匿名case_token；private/仅本地使用，含真实ID映射和salt，禁止随公开报告分享。图像同样属于本地复核资料。', '']
    with (output/'stage353_summary.md').open('x') as f: f.write('\n'.join(text))
    paths=list(output.glob('*.csv'))+[output/'stage353_summary.md',private/'slide_id_mapping.csv',private/'id_salt.bin']
    write_json(output/'experiment_manifest.json', dict(status='Complete fivefold calculations' if complete else 'Pending fivefold',
        morphology_status='Pending expert review',source_inference=source_manifest,
        margin_bins=list(MARGIN_BINS),review_selection='All G3; class/fold G2 first lexicographic IDs, min(available G2,max(1,G3 count))',
        output_sha256={str(p.relative_to(output)):matched.digest(p) for p in paths},
        analysis_code_sha256={str(p):matched.digest(p) for p in BASE.glob('*.py')}))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--require-complete',action='store_true')
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    print('Python:',matched.environment())
    manifest,state,frames=inspect_inference(args.run_dir)
    print(state.to_string(index=False))
    if state.status.eq('Invalid').any(): raise ValueError('Invalid inference jobs; no report written')
    if args.require_complete and (sorted(frames)!=list(range(5)) or not (args.run_dir/'completion.json').exists() or (args.run_dir/'.running.lock').exists()):
        raise ValueError('Pending complete fivefold inference; no report written')
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    if not args.execute:
        print('Dry-run only; no report created.'); return
    _,routing_frames=load_stage352(manifest['stage352_run'],manifest['stage352_report'])
    out=write_report(args.output_dir,state,frames,routing_frames,manifest['stage352_report'],
        dict(run_dir=str(args.run_dir.resolve()),manifest_sha256=matched.digest(args.run_dir/'experiment_manifest.json'),
             stage352_run=manifest['stage352_run'],stage352_report=manifest['stage352_report']))
    print('Report:',out)


if __name__=='__main__':
    main()
