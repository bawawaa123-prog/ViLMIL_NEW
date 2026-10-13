"""Validate completed jobs and summarize frozen interventions; missing jobs stay Pending."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import pandas as pd

from bridge import BASE, matched, checkpoint_identity, write_json, jobs, MODES, SEEDS
from run_diagnostics import job_dir, validate_predictions
from analysis_utils import paired_changes, class_metrics, ranking_changes, group_summary


def inspect_run(root):
    root = Path(root)
    manifest = json.loads((root/'experiment_manifest.json').read_text())
    if manifest['split']!='val':
        raise ValueError('Only Validation runs are allowed')
    for path,value in manifest['provenance']['sha256'].items():
        p = Path(path)
        # Refuse changed CSV/Prompt/splits. Code hashes are archived, not retroactively overwritten.
        if p==matched.CSV or p==matched.PROMPT or p.parent==matched.SPLITS:
            if matched.digest(p)!=value:
                raise ValueError(f'Input changed since run: {p}')
    for path,identity in manifest['checkpoint_identities'].items():
        if checkpoint_identity(path)!=identity:
            raise ValueError(f'Checkpoint changed since run: {path}')
    frames,records = {},[]
    expected_sources = {(r['arm'],r['fold']):r for r in manifest['sources']}
    expected_jobs = manifest['plan']
    keys = [(j['fold'],j['arm'],j['mode'],j['seed']) for j in expected_jobs]
    if len(keys)!=len(set(keys)):
        raise ValueError('Duplicate jobs in manifest')
    if (root/'completion.json').is_file():
        final = json.loads((root/'completion.json').read_text())
        if final.get('status')!='Complete' or final.get('n_jobs')!=len(expected_jobs) or final.get('manifest_sha256')!=matched.digest(root/'experiment_manifest.json'):
            raise ValueError('Root completion/manifest mismatch')
        expected_final = {str((job_dir(root,j)/'completion.json').relative_to(root)) for j in expected_jobs}
        expected_final |= {f'fold{f}/parameter_differences.csv' for f in manifest['folds']}
        if expected_final != set(final.get('sha256',{})):
            raise ValueError('Incomplete root artifact hashes')
        for name,checksum in final['sha256'].items():
            if matched.digest(root/name)!=checksum:
                raise ValueError(f'Final artifact modified: {name}')
    for job in expected_jobs:
        directory = job_dir(root,job)
        key = job['fold'],job['arm'],job['mode'],job['seed']
        state = dict(**job,status='Pending',issue='No complete job output',checkpoint=expected_sources[(job['arm'],job['fold'])]['checkpoint'])
        if not (directory/'completion.json').is_file():
            state['status'] = 'Partial' if directory.exists() else 'Pending'
            records.append(state)
            continue
        try:
            complete = json.loads((directory/'completion.json').read_text())
            if any(complete[k]!=job[k] for k in job) or complete['checkpoint']!=state['checkpoint'] or complete['split']!='val' or complete['status']!='Complete':
                raise ValueError('Job identity mismatch')
            if complete['checkpoint_identity']!=manifest['checkpoint_identities'][state['checkpoint']]:
                raise ValueError('Checkpoint identity mismatch')
            required = {'predictions.csv'}
            if job['arm']=='E1':
                required |= {'gate_statistics.csv','residual_statistics.csv','histograms.csv'}
                if job['mode']=='normal':
                    required |= {'patch_samples.csv','low_context_statistics.csv'}
            if not required <= set(complete['sha256']):
                raise ValueError('Missing expected hashed diagnostic files')
            for name,checksum in complete['sha256'].items():
                if matched.digest(directory/name)!=checksum:
                    raise ValueError(f'Diagnostic file modified: {name}')
            frame = pd.read_csv(directory/'predictions.csv',dtype={'slide_id':str})
            for name,value in {**job,'checkpoint':state['checkpoint'],'split':'val','status':'Complete'}.items():
                if name not in frame or not frame[name].eq(value).all():
                    raise ValueError(f'Prediction row provenance mismatch: {name}')
            actual = validate_predictions(frame,job['fold'])
            if len(frame)!=complete['n_slides']:
                raise ValueError('Validation size mismatch')
            for k,v in actual.items():
                if not np.isclose(v,complete['metrics'][k],rtol=0,atol=1e-12):
                    raise ValueError('Saved metrics and predictions disagree')
            if job['mode']=='normal':
                source = expected_sources[(job['arm'],job['fold'])]
                if any(not np.isclose(actual[k],source[k],rtol=0,atol=1e-7) for k in actual):
                    raise ValueError('Normal metrics differ from frozen Stage351 source')
            if job['arm']=='E1':
                for filename in ('gate_statistics.csv','residual_statistics.csv','low_context_statistics.csv'):
                    if filename not in complete['sha256']:
                        continue
                    diagnostic = pd.read_csv(directory/filename,dtype={'slide_id':str})
                    if diagnostic.slide_id.duplicated().any() or set(diagnostic.slide_id)!=set(frame.slide_id):
                        raise ValueError(f'Slide coverage mismatch: {filename}')
                    for name,value in {**job,'checkpoint':state['checkpoint'],'split':'val','status':'Complete'}.items():
                        if name not in diagnostic or not diagnostic[name].eq(value).all():
                            raise ValueError(f'Diagnostic row provenance mismatch: {filename}/{name}')
            state.update(status='Complete',issue='',**actual)
            frames[key] = frame
        except (ValueError,KeyError,OSError) as exc:
            state.update(status='Invalid',issue=str(exc))
        records.append(state)
    return manifest,pd.DataFrame(records),frames


def load_stats(root,job,filename):
    path = job_dir(root,job)/filename
    return pd.read_csv(path,dtype={'slide_id':str}) if path.is_file() else pd.DataFrame()


def table(frame):
    if frame.empty:
        return 'Pending: no completed paired data.'
    lines = ['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for _,row in frame.iterrows():
        lines.append('| '+' | '.join('Pending' if pd.isna(v) else f'{v:.6f}' if isinstance(v,float) else str(v).replace('|','/') for v in row)+' |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--require-complete',action='store_true')
    parser.add_argument('--execute',action='store_true',help='Write full statistics; default checks run state only')
    args = parser.parse_args()
    print('Python:',matched.environment())
    root = args.run_dir.resolve()
    manifest,state,frames = inspect_run(root)
    print(state[['fold','arm','mode','seed','status','issue']].to_string(index=False))
    if not args.execute:
        print('Dry-run: no statistical outputs written.')
        if state.status.isin(['Partial','Invalid']).any() or (args.require_complete and
            (not state.status.eq('Complete').all() or not (root/'completion.json').exists() or (root/'.running.lock').exists())):
            raise SystemExit(2)
        return
    output = (args.output_dir or BASE/'reports'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    output.mkdir(parents=True,exist_ok=False)
    state.to_csv(output/'experiment_status.csv',index=False)
    interventions,comparisons,changes,groups,rankings,classes,gates,residuals,low = [],[],[],[],[],[],[],[],[]
    for key,frame in frames.items():
        fold,arm,mode,seed = key
        info = dict(fold=fold,arm=arm,mode=mode,seed=seed,status='Complete',checkpoint=frame.checkpoint.iloc[0],split='val')
        classes.append({**info,**class_metrics(frame)})
        if arm!='E1':
            continue
        job = dict(fold=fold,arm=arm,mode=mode,seed=seed)
        gate = load_stats(root,job,'gate_statistics.csv')
        residual = load_stats(root,job,'residual_statistics.csv')
        gates.append(gate)
        residuals.append(residual)
        if mode=='normal':
            low.append(load_stats(root,job,'low_context_statistics.csv'))
        normal = frames.get((fold,'E1','normal',0))
        if normal is None:
            interventions.append({**info,'status':'Pending','issue':'Missing Normal baseline'})
            continue
        stats,delta = paired_changes(normal,frame)
        interventions.append({**info,**stats})
        delta.insert(0,'comparison',f'E1_normal_to_{mode}')
        for k,v in info.items():
            delta[k]=v
        changes.append(delta)
    for fold in manifest['folds']:
        normal_job = dict(fold=fold,arm='E1',mode='normal',seed=0)
        gate = load_stats(root,normal_job,'gate_statistics.csv') if (fold,'E1','normal',0) in frames else None
        residual = load_stats(root,normal_job,'residual_statistics.csv') if gate is not None else None
        pairs = [('B0_to_B1',('E0','normal'),('E1','normal')),
                 ('B1_to_B2',('E1','normal'),('E1','residual_off')),
                 ('B0_to_B2',('E0','normal'),('E1','residual_off'))]
        if any(j['arm']=='E2' for j in manifest['plan']):
            pairs += [('E0_to_E2',('E0','normal'),('E2','normal'))]
        for name,(aa,am),(ba,bm) in pairs:
            a,b = frames.get((fold,aa,am,0)),frames.get((fold,ba,bm,0))
            if a is None or b is None:
                comparisons.append(dict(fold=fold,comparison=name,status='Pending'))
                continue
            stats,delta = paired_changes(a,b)
            comparisons.append(dict(fold=fold,comparison=name,status='Complete',**stats))
            rankings.append(dict(fold=fold,comparison=name,status='Complete',**ranking_changes(a,b)))
            joined,summary = group_summary(delta,gate,residual)
            joined['comparison'],joined['fold'] = name,fold
            joined['before_checkpoint'],joined['after_checkpoint'] = a.checkpoint.iloc[0],b.checkpoint.iloc[0]
            joined['status']='Complete'
            changes.append(joined)
            summary['comparison'],summary['fold'] = name,fold
            summary['status']='Complete'
            groups.append(summary)
    datasets = {'routing_intervention_metrics.csv':pd.DataFrame(interventions),
        'routing_gate_statistics.csv':pd.concat(gates,ignore_index=True) if gates else pd.DataFrame(),
        'routing_residual_statistics.csv':pd.concat(residuals,ignore_index=True) if residuals else pd.DataFrame(),
        'prediction_changes.csv':pd.concat(changes,ignore_index=True) if changes else pd.DataFrame(),
        'fold_comparison.csv':pd.DataFrame(comparisons),'correctness_group_statistics.csv':pd.concat(groups,ignore_index=True) if groups else pd.DataFrame(),
        'ranking_changes.csv':pd.DataFrame(rankings),'class_metrics.csv':pd.DataFrame(classes),
        'low_context_statistics.csv':pd.concat(low,ignore_index=True) if low else pd.DataFrame()}
    drift = []
    for fold in manifest['folds']:
        file = root/f'fold{fold}/parameter_differences.csv'
        if file.is_file() and (fold,'E0','normal',0) in frames and (fold,'E1','normal',0) in frames:
            drift.append(pd.read_csv(file))
    datasets['parameter_differences.csv'] = pd.concat(drift,ignore_index=True) if drift else pd.DataFrame()
    summary_rows = []
    for (mode,seed),part in datasets['routing_intervention_metrics.csv'].groupby(['mode','seed']) if interventions else []:
        if 'delta_auc' not in part:
            continue
        part = part[part.status=='Complete']
        complete5 = set(part.fold)==set(range(5)) and len(part)==5
        for metric in ('delta_auc','delta_accuracy','delta_macro_f1','mean_absolute_probability_change','changed_prediction_count'):
            values = part[metric].dropna()
            summary_rows.append(dict(mode=mode,seed=seed,metric=metric,n_folds=len(values),
                mean=float(values.mean()) if complete5 else np.nan,std=float(values.std(ddof=0)) if complete5 else np.nan,
                status='Fivefold' if complete5 else 'Pending full fivefold'))
    datasets['fivefold_intervention_summary.csv']=pd.DataFrame(summary_rows)
    for filename,frame in datasets.items():
        frame.to_csv(output/filename,index=False)
    write_json(output/'experiment_manifest.json',{'source_run':str(root),'source_manifest_sha256':matched.digest(root/'experiment_manifest.json'),
        'analysis_code_sha256':{str(p):matched.digest(p) for p in BASE.glob('*.py')},'status':state.status.tolist(),
        'output_sha256':{p.name:matched.digest(p) for p in output.glob('*.csv')}})
    completed = state.status.eq('Complete').all() and (root/'completion.json').is_file() and not (root/'.running.lock').exists()
    required_plan = {(j['fold'],j['arm'],j['mode'],j['seed']) for j in jobs(range(5),MODES,SEEDS)}
    stage_ready = completed and required_plan <= set(frames)
    folds = manifest['folds']
    shuffle_jobs = [j for j in manifest['plan'] if j['mode']=='context_shuffle']
    available = [
        bool(shuffle_jobs) and all((f,'E1','normal',0) in frames and (f,'E1','context_mean',0) in frames for f in folds)
        and all((j['fold'],j['arm'],j['mode'],j['seed']) in frames for j in shuffle_jobs),
        all((f,'E1','normal',0) in frames for f in folds),
        all((f,'E1','normal',0) in frames and (f,'E1','residual_off',0) in frames for f in folds),
        all((f,'E0','normal',0) in frames and (f,'E1','normal',0) in frames and
            (f,'E1','residual_off',0) in frames and (root/f'fold{f}/parameter_differences.csv').is_file() for f in folds),
        all((f,'E0','normal',0) in frames and (f,'E1','normal',0) in frames for f in range(5)),
        stage_ready]
    questions = [
        ('Q1 空间依赖','仅当 Normal 与 Shuffle/Mean 的配对结果齐全才解释空间敏感性。小差异不证明 Low 无信息或 Router 未用 Low。'),
        ('Q2 Patch 区分性','结合有效 mapped Gate 方差、饱和比例、父节点内/间统计和 Low 空间统计；重叠父组不是互斥 ANOVA 分组。'),
        ('Q3 Residual 贡献','结合逐 patch 与 WSI 范数、cap 触发、B1_to_B2 预测变化。'),
        ('Q4 共适应','B0_to_B2 与共享训练参数差异是线索；独立训练及 Checkpoint epoch 不同，不能确定唯一因果。'),
        ('Q5 Fold 差异','查看 correctness_group_statistics、prediction_changes、class_metrics 与 ranking_changes；Fold3/4 不按结果挑选阈值。'),
        ('Q6 下一步','按配对敏感性和病例证据排序待验证瓶颈，不实现新结构、不按诊断结果改训练超参数。')]
    lines = ['# Stage 3.5.2 Routing diagnostics','',
        '完整计划已完成并核验。' if completed else 'Pending：存在未完成计划/缺少根完成标记，部分数据只作显式局部描述。','',
        '默认完整五折/三seed诊断齐全，可进入证据审阅。' if stage_ready else 'Stage3.5.2完整入口仍Pending：当前不是全部默认35个job的完整诊断。','',
        '只使用 Validation，AUC 正类=1 NonAdenocarcinoma；阈值保持概率 argmax。Gate 饱和阈值 0.05/0.95，边界描述 |P1-P0|≤0.10。','',
        '## 干预配对结果','',table(datasets['routing_intervention_metrics.csv']), '',
        '## 训练方案与同权重干预','',table(datasets['fold_comparison.csv']), '',
        '## 五折干预统计','',table(datasets['fivefold_intervention_summary.csv']), '',
        '## 完成状态','',table(state), '', '## 判定边界','']
    for (name,meaning),ready in zip(questions,available):
        lines += [f'- {name}：'+('当前请求范围有相应实验输出，需结合连续量审阅，不能自动确认为机制。' if ready else '必要模式/折的证据不足，Pending。')+meaning]
    lines += ['', '数据表保存具体证据，报告不自动将相关性判为因果。Gate 不是病灶概率。五折训练集重叠、单 seed，未进行独立样本显著性检验。', '',
        'case-level strict5 尚未证明 patient-level 独立；名义 5x/20x 实际约 2.5x/10x；验证折参与开发。Checkpoint 身份按路径/大小/mtime 核对，未全量 hash 大权重或特征。']
    (output/'stage352_summary.md').write_text('\n'.join(lines)+'\n')
    print('Report:',output)
    if state.status.isin(['Partial','Invalid']).any() or (args.require_complete and not completed):
        raise SystemExit(2)


if __name__=='__main__':
    main()
