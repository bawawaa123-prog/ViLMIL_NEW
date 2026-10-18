"""CPU synthetic tests only; never construct a real pretrained model or infer WSI."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch import nn

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
import stage353_common as c
from analyze_complementarity import pair_cases,summarize_cases,associate,select_review_cases,margin_stratum
from aggregate_stage353 import write_report,fivefold_metrics
from prepare_case_visualizations import validate_report,attention_mean
import run_independent_inference as inference


def fixture():
    h=pd.DataFrame(dict(slide_id=['a','b','c','d'],label=[0,0,1,1],prediction=[0,0,0,0],
                       p_0=[.9,.7,.6,.8],p_1=[.1,.3,.4,.2]))
    l=pd.DataFrame(dict(slide_id=['a','b','c','d'],label=[0,0,1,1],prediction=[0,1,1,0],
                       p_0=[.8,.2,.1,.7],p_1=[.2,.8,.9,.3]))
    return h,l


class Stage353Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c.matched.environment();torch.set_num_threads(1)

    def test_pairing_groups_and_oracle(self):
        h,l=fixture();cases=pair_cases(h,l.sample(frac=1,random_state=1),0)
        self.assertEqual(cases.group.tolist(),list(c.GROUPS))
        summary,groups,strata=summarize_cases(cases)
        self.assertEqual([summary[g+'_count'] for g in c.GROUPS],[1,1,1,1])
        self.assertEqual(summary['high_error_rescue_rate'],.5)
        self.assertEqual(summary['oracle_accuracy_upper_bound'],.75)
        self.assertEqual(summary['oracle_accuracy_gain_over_high'],.25)
        self.assertEqual(summary['error_jaccard'],1/3)
        self.assertEqual(groups.n.sum(),4)
        self.assertEqual(strata[strata.label=='all'].n.sum(),4)
        for frame in (h,l):
            self.assertGreater(c.metrics_from_predictions(frame)['val_auc'],0)
        for bad in (l.iloc[:-1],pd.concat([l,l.iloc[:1]],ignore_index=True),l.assign(label=[1,0,1,1])):
            with self.assertRaises(ValueError):pair_cases(h,bad,0)

    def test_positive_class_and_argmax_direction(self):
        f=pd.DataFrame(dict(slide_id=['a','b'],label=[0,1],prediction=[0,1],p_0=[.8,.1],p_1=[.2,.9]))
        self.assertEqual(c.metrics_from_predictions(f)['val_auc'],1.)
        self.assertEqual(c.class_metrics(f)['recall_adenocarcinoma'],1.)
        with self.assertRaises(ValueError):c.metrics_from_predictions(f.assign(prediction=[1,0]))
        with self.assertRaises(ValueError):c.metrics_from_predictions(f.assign(p_1=[float('nan'),.9]))

    def test_margin_bins_and_empty_denominators(self):
        self.assertEqual(margin_stratum(pd.Series([0,.10,.10001,.50,.90,1])).astype(str).tolist(),
            ['0_to_0.10','0_to_0.10','0.10_to_0.50','0.10_to_0.50','0.50_to_0.90','0.90_to_1.00'])
        h,l=fixture();h['prediction']=h.label;h.p_0=[.9,.9,.1,.1];h.p_1=1-h.p_0
        summary,_,_=summarize_cases(pair_cases(h,h,0))
        self.assertTrue(np.isnan(summary['high_error_rescue_rate']))
        self.assertTrue(np.isnan(summary['error_jaccard']))

    def test_routing_association_and_strict_e0_equality(self):
        h,l=fixture();cases=pair_cases(h,l,3);e1=l.copy()
        stats=pd.DataFrame(dict(slide_id=['a','b','c','d'],gate_mean=[.5]*4))
        residual=pd.DataFrame(dict(slide_id=['a','b','c','d'],mapped_residual_ratio_mean=[.01]*4))
        lowctx=pd.DataFrame(dict(slide_id=['a','b','c','d'],low_feature_variance=[2.]*4))
        frame,groups=associate(cases,h,e1,stats,residual,lowctx)
        self.assertEqual(groups[groups.group=='G3'].routing_rescued_count.item(),1)
        self.assertEqual(groups[groups.group=='G2'].routing_harmed_count.item(),1)
        with self.assertRaises(ValueError):associate(cases,h,e1,stats.iloc[:-1],residual,lowctx)
        bad=cases.copy();bad.loc[0,'high_p1']+=.0001
        with self.assertRaises(ValueError):associate(bad,h,e1,stats,residual,lowctx)

    def test_transparent_review_selection(self):
        h,l=fixture();cases=pair_cases(h,l,0)
        extra=pd.concat([cases,cases.assign(slide_id=['e','f','g','h'])],ignore_index=True)
        chosen=select_review_cases(extra.sample(frac=1,random_state=3))
        self.assertEqual(set(chosen[chosen.group=='G3'].slide_id),{'c','g'})
        self.assertEqual(set(chosen[chosen.group=='G2'].slide_id),{'b'})
        self.assertTrue(set(chosen.group)<=set(['G2','G3']))

    def test_training_command_protocol_and_destination(self):
        argv=c.low_command(3,Path('/tmp/synthetic_training'))
        self.assertEqual(argv[argv.index('--scale_mode')+1],'low')
        self.assertEqual(argv[argv.index('--k_start')+1],'3')
        self.assertEqual(argv[argv.index('--k_end')+1],'3')
        self.assertIn('--skip_test_evaluation',argv)
        self.assertFalse(any(a in argv for a in ('--use_low_context_routing','--use_global_proto_context','--init_checkpoint_template')))
        old=c.matched.command('E0',3,Path('/tmp/synthetic_training'))
        changed=[a for a,b in zip(old,argv) if a!=b]
        self.assertEqual(changed,['high','stage351_matched_E0_fold3'])
        with self.assertRaises(ValueError):c.validate_folds([0,0])
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(c.inspect_low(root,0)['status'],'Pending')
            c.low_destination(root,0)[0].mkdir()
            self.assertEqual(c.inspect_low(root,0)['status'],'Invalid')

    def test_validation_identity_coverage_and_labels(self):
        h,_=fixture();h['fold']=0;h['arm']='H';h['split']='val';h['checkpoint']='mock'
        labels=pd.DataFrame(dict(slide_id=list('abcd'),label=['Adenocarcinoma']*2+['NonAdenocarcinoma']*2))
        with patch.object(c.matched,'split_members',return_value={'val':list('abcd')}),patch.object(c.pd,'read_csv',return_value=labels):
            c.validate_predictions(h,0,'H','mock')
            with self.assertRaises(ValueError):c.validate_predictions(h,1,'H','mock')
            with self.assertRaises(ValueError):c.validate_predictions(h.assign(label=[1,0,1,1]),0)
            with self.assertRaises(ValueError):c.validate_predictions(h.assign(split='test'),0)

    def test_low_training_completion_and_protocol_rejection(self):
        with tempfile.TemporaryDirectory() as root:
            parent,path=c.low_destination(root,0);path.mkdir(parents=True)
            hashes={'synthetic':'fixed'};code={'synthetic_code':'fixed'}
            c.write_json(parent/'run_manifest.json',dict(fold=0,argv=c.low_command(0,parent),inputs=hashes,core_code=code))
            cfg=dict(num_splits=5,k_start=0,k_end=0,task='task_adenocarcinoma',max_epochs=80,lr=1e-4,
                label_frac=1.,seed=1,model_type='ViLa_MIL_BiomedCLIP',mode='transformer',scale_mode='low',
                skip_test_evaluation=True,use_low_context_routing=False,use_routing_stabilization=False,
                use_global_proto_context=False,init_checkpoint_template=None,use_drop_out=True,
                weighted_sample=False,opt='adam',patience=10,split_dir=str(c.matched.SPLITS.relative_to(c.ROOT)),
                mapping_path=str(c.matched.MAPPING.relative_to(c.ROOT)))
            config=path/'experiment_synthetic.txt';config.write_text(repr(cfg))
            (parent/'train.log').write_text('synthetic CPU fixture\nend script\n')
            (path/'splits_0.csv').write_text('synthetic')
            pd.DataFrame(dict(fold=1,epoch=range(1,12),val_acc=.75,val_auc=.8,val_loss=.2,val_f1=.7)).to_csv(path/'epoch_details.csv',index=False)
            pd.DataFrame([dict(fold=1,val_acc=.75,val_auc=.8,test_auc=np.nan,test_acc=np.nan)]).to_csv(path/'fold_summary.csv',index=False)
            torch.save({'learnable_image_center':torch.zeros(16,1,512),'attention_V.0.weight':torch.zeros(192,512)},path/'s_0_checkpoint.pt')
            with patch.object(c,'input_hashes',return_value=hashes),patch.object(c,'code_hashes',return_value=code),patch.object(c.matched,'same_split',return_value=True):
                row=c.inspect_low(root,0,require_completion=False)
                self.assertEqual(row['status'],'Complete')
                self.assertEqual(row['selected_epoch'],1)
                self.assertEqual(c.inspect_low(root,0)['status'],'Invalid')
                c.write_json(parent/'completion.json',dict(result=row,manifest_sha256=c.matched.digest(parent/'run_manifest.json'),checkpoint_identity=c.routing.checkpoint_identity(row['checkpoint']),
                    sha256={str(p.relative_to(parent)):c.matched.digest(p) for p in [config,path/'epoch_details.csv',path/'fold_summary.csv']}))
                self.assertEqual(c.inspect_low(root,0)['status'],'Complete')
                cfg['init_checkpoint_template']='forbidden_warmstart';config.write_text(repr(cfg))
                self.assertEqual(c.inspect_low(root,0)['status'],'Invalid')

    def test_inference_completion_requires_source_coverage(self):
        h,l=fixture();h=h.assign(fold=0,arm='E0',split='val',checkpoint='H',status='Complete')
        l=l.assign(fold=0,arm='L',split='val',checkpoint='L',status='Complete')
        low=dict(arm='L',fold=0,status='Complete',checkpoint='L',**c.metrics_from_predictions(l))
        high=dict(arm='E0',fold=0,status='Complete',checkpoint='H',**c.metrics_from_predictions(h))
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);job=root/'fold0';job.mkdir()
            stage=root/'stage352';stage.mkdir();(stage/'experiment_manifest.json').write_text('{}')
            c.write_json(root/'experiment_manifest.json',dict(split='val',folds=[0],core_code={'mock':'code'},
                inputs={'0':{'mock':'inputs'}},stage352_run=str(stage),stage352_report=str(stage),
                stage352_manifest_sha256=c.matched.digest(stage/'experiment_manifest.json'),low_run='mock',low_sources={'0':low}))
            h.to_csv(job/'high_predictions.csv',index=False);l.to_csv(job/'low_predictions.csv',index=False)
            c.write_json(job/'completion.json',dict(fold=0,status='Complete',checkpoint_identity={'mock':'L'},
                sha256={p.name:c.matched.digest(p) for p in job.glob('*.csv')}))
            labels=pd.DataFrame(dict(slide_id=list('abcd'),label=['Adenocarcinoma']*2+['NonAdenocarcinoma']*2))
            orig_read=pd.read_csv
            def read(path,*args,**kwargs):
                return labels if Path(path)==c.matched.CSV else orig_read(path,*args,**kwargs)
            with patch.object(inference,'load_stage352',return_value=({}, {(0,'E0','normal',0):h})),patch.object(inference,'code_hashes',return_value={'mock':'code'}),patch.object(inference,'input_hashes',return_value={'mock':'inputs'}),patch.object(inference,'inspect_low',return_value=low),patch.object(c.matched,'collect',return_value=high),patch.object(c.routing,'checkpoint_identity',return_value={'mock':'L'}),patch.object(c.matched,'split_members',return_value={'val':list('abcd')}),patch.object(c.pd,'read_csv',side_effect=read):
                _,state,frames=inference.inspect_inference(root)
                self.assertEqual(state.status.tolist(),['Complete'])
                self.assertEqual(set(frames),{0})
                # Even a CSV with an updated checksum must be rejected if its fold is wrong.
                l.assign(fold=1).to_csv(job/'low_predictions.csv',index=False)
                marker=c.read_json(job/'completion.json');marker['sha256']['low_predictions.csv']=c.matched.digest(job/'low_predictions.csv')
                (job/'completion.json').write_text(__import__('json').dumps(marker))
                _,state,frames=inference.inspect_inference(root)
                self.assertEqual(state.status.tolist(),['Invalid'])
                self.assertEqual(frames,{})

    def test_synthetic_report_privacy_fivefold_and_integrity(self):
        h,l=fixture();h['checkpoint']='H_mock';l['checkpoint']='L_mock'
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);stats=root/'stats';stats.mkdir()
            for name,col in [('routing_gate_statistics.csv','gate_mean'),('routing_residual_statistics.csv','mapped_residual_ratio_mean'),('low_context_statistics.csv','low_feature_variance')]:
                pd.concat([pd.DataFrame(dict(fold=f,arm='E1',mode='normal',seed=0,slide_id=list('abcd'),**{col:[.1]*4})) for f in range(5)],ignore_index=True).to_csv(stats/name,index=False)
            frames={f:(h,l) for f in range(5)};routing={(f,a,'normal',0):h if a=='E0' else l for f in range(5) for a in ('E0','E1')}
            state=pd.DataFrame(dict(fold=list(range(5)),status='Complete'))
            out=write_report(root/'report',state,frames,routing,stats,{'synthetic':True})
            manifest,selected=validate_report(out)
            self.assertEqual(manifest['status'],'Complete fivefold calculations')
            self.assertEqual(manifest['morphology_status'],'Pending expert review')
            self.assertEqual(len(pd.read_csv(out/'fivefold_scale_statistics.csv')),10)
            self.assertNotIn('slide_id',pd.read_csv(out/'case_level_complementarity.csv'))
            self.assertEqual((out/'private').stat().st_mode&0o777,0o700)
            self.assertEqual((out/'private/slide_id_mapping.csv').stat().st_mode&0o777,0o600)
            self.assertTrue(all(s.startswith('C') for s in selected.case_token))
            with self.assertRaises(FileExistsError):write_report(out,state,frames,routing,stats,{'synthetic':True})
            with (out/'case_level_complementarity.csv').open('a') as f:f.write('corruption\n')
            with self.assertRaises(ValueError):validate_report(out)
            partial=write_report(root/'partial',state.iloc[:1],{0:(h,l)},routing,stats,{'synthetic':True})
            self.assertEqual(c.read_json(partial/'experiment_manifest.json')['status'],'Pending fivefold')
            self.assertEqual(len(pd.read_csv(partial/'fivefold_scale_statistics.csv')),0)

    def test_attention_actual_forward_matches_clean_prediction(self):
        import models.model_ViLa_MIL_BiomedCLIP as mm
        class TextStub(nn.Module):
            def __init__(self,model,**kwargs):
                super().__init__();self.model=model;self.register_buffer('features',torch.randn(4,512)*.02)
            def forward(self,*args,**kwargs):return self.features
        class PromptStub(nn.Module):
            def __init__(self,*args):
                super().__init__();self.ctx=nn.Parameter(torch.randn(16,32));self.register_buffer('tokenized_prompts',torch.ones(4,8,dtype=torch.long))
            def forward(self):return self.ctx
        cfg=SimpleNamespace(hidden_size=192,text_prompt=list('abcd'),prototype_number=16,scale_mode='low',
            use_low_context_routing=False,use_routing_stabilization=False,use_global_proto_context=False,finetune_text_encoder=False)
        with patch.object(mm,'create_model_from_pretrained',return_value=(nn.Identity(),None)),patch.object(mm,'get_tokenizer',return_value=object()),patch.object(mm,'_prepare_biomedclip_loading',return_value=('stub',None,True)),patch.object(mm,'BiomedCLIPTextEncoder',TextStub),patch.object(mm,'BiomedCLIPPromptLearner',PromptStub):
            model=mm.ViLa_MIL_BiomedCLIP(cfg,num_classes=2).eval()
        args=(torch.randn(1,3,512),torch.zeros(1,3,2),torch.randn(1,5,512),torch.zeros(1,5,2),torch.tensor([1]))
        captures=[]
        with torch.no_grad():
            reference=model(*args)
            handle=model.cross_attention_1.register_forward_hook(lambda m,x,out:captures.append(out[1]))
            actual=model(*args)
            handle.remove()
        self.assertTrue(torch.equal(reference[0],actual[0]))
        self.assertEqual(attention_mean(captures[0],3).shape,(3,))
        self.assertEqual(attention_mean(captures[1],5).shape,(5,))
        self.assertFalse(hasattr(model,'high_router'))


if __name__=='__main__':unittest.main()
