"""Synthetic CSV -> integrity checks -> summary and bounded plots, all CPU."""
from pathlib import Path
import sys
import tempfile
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from bridge import MODES,matched,write_json,checkpoint_identity,jobs
from run_diagnostics import job_dir,validate_predictions
import aggregate_diagnostics as aggregate
import plot_diagnostics as plots


def synthetic_run(root,csv,splits):
    plan=jobs([0],MODES,[352])
    checkpoint0=root/'synthetic_e0.pt'; checkpoint0.write_bytes(b'synthetic-only')
    checkpoint1=root/'synthetic_e1.pt'; checkpoint1.write_bytes(b'synthetic-only')
    sources=[]
    for arm,path in [('E0',checkpoint0),('E1',checkpoint1)]:
        sources.append(dict(arm=arm,fold=0,checkpoint=str(path),val_auc=1.,val_acc=.75,val_macro_f1=(2/3+.8)/2))
    manifest=dict(split='val',folds=[0],plan=plan,sources=sources,provenance={'sha256':{}},
        checkpoint_identities={str(p):checkpoint_identity(p) for p in (checkpoint0,checkpoint1)})
    write_json(root/'experiment_manifest.json',manifest)
    for job in plan:
        directory=job_dir(root,job); directory.mkdir(parents=True)
        src=next(s for s in sources if s['arm']==job['arm'])
        meta={**job,'checkpoint':src['checkpoint'],'status':'Complete','split':'val'}
        frame=pd.DataFrame(dict(slide_id=['a','b','c','d'],label=[0,0,1,1],prediction=[0,1,1,1],
                            p_0=[.9,.4,.2,.1],p_1=[.1,.6,.8,.9]))
        if job['mode']=='residual_off':
            frame.loc[1,['prediction','p_0','p_1']]=[0,.6,.4]
        for k,v in meta.items(): frame[k]=v
        frame.to_csv(directory/'predictions.csv',index=False)
        if job['arm']=='E1':
            gate=pd.DataFrame([{**meta,'slide_id':sid,'gate_mean':.5,'gate_variance':.02,
                'gate_near_one_fraction':0.,'between_parent_gate_mean_variance':.01} for sid in ['a','b','c','d']])
            residual=pd.DataFrame([{**meta,'slide_id':sid,'mapped_residual_ratio_mean':.03,
                'aggregate_residual_high_ratio':.03,'no_usable_context_fraction':.1} for sid in ['a','b','c','d']])
            gate.to_csv(directory/'gate_statistics.csv',index=False)
            residual.to_csv(directory/'residual_statistics.csv',index=False)
            pd.DataFrame([{**meta,'slide_id':sid,'quantity':quantity,'bin_left':a,'bin_right':b,'count':1}
                         for sid in ['a','b','c','d'] for quantity,a,b in [('gate',0.,1.),('residual_ratio',0.,.1)]]).to_csv(directory/'histograms.csv',index=False)
            if job['mode']=='normal':
                pd.DataFrame([{**meta,'slide_id':sid,'patch_index':i,'x':i,'y':i,'gate':.5}
                              for sid in ['a','b','c','d'] for i in range(2)]).to_csv(directory/'patch_samples.csv',index=False)
                pd.DataFrame([{**meta,'slide_id':sid,'low_parent':i,'x':i,'y':i,'gate_mean':.5}
                              for sid in ['a','b','c','d'] for i in range(2)]).to_csv(directory/'region_samples.csv',index=False)
                pd.DataFrame([{**meta,'slide_id':sid,'within_slide_cosine_mean':.9}
                              for sid in ['a','b','c','d']]).to_csv(directory/'low_context_statistics.csv',index=False)
        metrics=validate_predictions(frame,0)
        write_json(directory/'completion.json',{**meta,'n_slides':4,'metrics':metrics,
            'checkpoint_identity':checkpoint_identity(src['checkpoint']),
            'sha256':{p.name:matched.digest(p) for p in directory.glob('*.csv')}})
    pd.DataFrame([dict(fold=0,parameter='synthetic.ctx',l2_difference=.1,status='Complete')]).to_csv(root/'fold0/parameter_differences.csv',index=False)
    files=[*(job_dir(root,j)/'completion.json' for j in plan),root/'fold0/parameter_differences.csv']
    write_json(root/'completion.json',{'status':'Complete','n_jobs':len(plan),
        'manifest_sha256':matched.digest(root/'experiment_manifest.json'),
        'sha256':{str(p.relative_to(root)):matched.digest(p) for p in files}})
    return plan


class PipelineTests(unittest.TestCase):
    def test_synthetic_summary_plots_and_corruption_rejection(self):
        matched.environment()
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); root=base/'run'; root.mkdir()
            csv=base/'labels.csv'; splits=base/'splits'; splits.mkdir()
            pd.DataFrame(dict(case_id=['a','b','c','d'],slide_id=['a','b','c','d'],
                label=['Adenocarcinoma','Adenocarcinoma','NonAdenocarcinoma','NonAdenocarcinoma'])).to_csv(csv,index=False)
            pd.DataFrame(dict(train=[None]*4,val=['a','b','c','d'],test=[None]*4)).to_csv(splits/'splits_0.csv',index=False)
            with patch.multiple(matched,CSV=csv,SPLITS=splits):
                plan=synthetic_run(root,csv,splits)
                _,state,frames=aggregate.inspect_run(root)
                self.assertTrue(state.status.eq('Complete').all())
                report=base/'report'
                with patch.object(sys,'argv',['aggregate','--run-dir',str(root),'--output-dir',str(report),'--execute','--require-complete']):
                    aggregate.main()
                changes=pd.read_csv(report/'fold_comparison.csv')
                self.assertEqual(len(changes),3)
                self.assertEqual(changes.loc[changes.comparison=='B1_to_B2','wrong_to_correct'].iloc[0],1)
                stats=pd.read_csv(report/'fivefold_intervention_summary.csv')
                self.assertTrue(stats['mean'].isna().all()) # one fold never becomes fivefold
                figure=base/'figures'
                with patch.object(sys,'argv',['plot','--run-dir',str(root),'--report-dir',str(report),'--output-dir',str(figure),'--execute']):
                    plots.main()
                self.assertGreaterEqual(len(list(figure.glob('*.png'))),5)
                self.assertTrue((figure/'figure_manifest.json').is_file())
                # Root completion metadata remains intact; per-job CSV tampering is reported Invalid.
                p=job_dir(root,plan[0])/'predictions.csv'
                p.write_text(p.read_text()+'\n')
                _,state,_=aggregate.inspect_run(root)
                self.assertEqual(state.iloc[0]['status'],'Invalid')
                # Rewriting an authenticated completion marker is rejected at root level.
                marker=job_dir(root,plan[0])/'completion.json'
                marker.write_text(marker.read_text()+'\n')
                with self.assertRaises(ValueError): aggregate.inspect_run(root)


if __name__=='__main__':
    unittest.main(verbosity=2)
