"""Synthetic CPU tests only; no real slide inference, downloads or GPU jobs."""
from pathlib import Path
import sys
import json
import io
import logging
import tempfile
from unittest.mock import patch
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
import torch
from torch import nn

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from bridge import ROOT, MODES, SEEDS, slide_seed, jobs, matched, write_json
from telemetry import RouterRecorder,parent_statistics
from analysis_utils import paired_changes,class_metrics,ranking_changes,align,group_summary
from run_diagnostics import job_dir,Tee
from aggregate_diagnostics import inspect_run
from models.cross_scale_modules import LowParentContext,HighRouter
from utils.cross_scale_mapping import build_cross_scale_mapping


def mapping_fixture():
    return build_cross_scale_mapping(np.array([[0,0],[2,0],[20,0]]),
        np.array([[0,0],[2,0],[4,0],[22,0],[50,0]]),low_patch_size=4,high_patch_size=2,
        low_downsample=1,high_downsample=1).to_dict()


def context_fixture(low,mapping):
    context=LowParentContext()(low,mapping,high_count=5)
    context['mapping_parent_ptr']=torch.as_tensor(mapping['parent_ptr'])
    context['mapping_child_indices']=torch.as_tensor(mapping['child_indices'])
    return context


def frame_fixture():
    return pd.DataFrame(dict(slide_id=['a','b','c','d'],label=[0,0,1,1],prediction=[0,1,1,1],
                             p_0=[.9,.4,.2,.1],p_1=[.1,.6,.8,.9]))


class DiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        matched.environment()
        torch.set_num_threads(1)

    def test_hook_preserves_output_and_state(self):
        torch.manual_seed(352)
        router=HighRouter(8,4,stabilize=True).eval()
        with torch.no_grad():
            router.context_projection.weight.copy_(torch.eye(8)*10)
            router.route_score[-1].bias.fill_(3.)
        low,high=torch.randn(3,8),torch.randn(5,8)
        mapping=mapping_fixture()
        context=context_fixture(low,mapping)
        state={k:v.clone() for k,v in router.state_dict().items()}
        with torch.no_grad():
            expected=router(high,context)
            recorder=RouterRecorder(router)
            recorder.prepare(mapping,np.arange(10).reshape(5,2),352,3)
            actual=router(high,context)
        self.assertTrue(torch.equal(expected,actual))
        for k,v in state.items(): self.assertTrue(torch.equal(v,router.state_dict()[k]))
        self.assertLessEqual(recorder.row['mapped_residual_ratio_max'],.100001)
        self.assertGreater(recorder.row['cap_trigger_fraction_mapped'],0)
        self.assertEqual(recorder.row['n_mapping_unmapped'],1)
        self.assertTrue(torch.equal(actual[4],high[4]))
        self.assertEqual(len(recorder.patches),3)
        self.assertEqual(sum(h['count'] for h in recorder.histograms if h['quantity']=='gate'),4)
        with torch.no_grad():
            recorder.prepare(mapping,np.arange(10).reshape(5,2),352,3)
            off=router(high,context,diagnostic_mode='residual_off')
        self.assertTrue(torch.equal(off,high))
        self.assertEqual(recorder.row['residual_frobenius_norm'],0)
        self.assertEqual(recorder.row['cap_trigger_fraction_all'],0)
        recorder.close()
        self.assertFalse(router._forward_hooks)
        self.assertFalse(router.route_score._forward_hooks)

    def test_logging_shutdown_after_log_context_closed(self):
        terminal,log = io.StringIO(),io.StringIO()
        tee = Tee(terminal,log)
        self.assertEqual(tee.write('during inference\n'),len('during inference\n'))
        self.assertEqual(log.getvalue(),'during inference\n')
        tee.close()
        self.assertFalse(terminal.closed)
        self.assertFalse(log.closed)
        log.close()  # main() restores sys streams then exits the log context
        handler = logging.StreamHandler(tee)
        handler.flush()
        tee.close()  # absl logging handler calls stream.close() at atexit
        tee.close()  # cleanup is idempotent
        handler.emit(logging.LogRecord('late',logging.INFO,'',0,'after inference',(),None))
        self.assertIn('after inference',terminal.getvalue())
        self.assertFalse(terminal.closed)

    def test_all_invalid_context_is_not_gate_saturation(self):
        mapping=mapping_fixture()
        mapping['low_valid_mask']=np.zeros(3,dtype=bool)
        low,high=torch.randn(3,8),torch.randn(5,8)
        context=context_fixture(low,mapping)
        router=HighRouter(8,4,stabilize=True).eval()
        recorder=RouterRecorder(router)
        recorder.prepare(mapping,np.zeros((5,2)),3)
        with torch.no_grad(): actual=router(high,context)
        self.assertTrue(torch.equal(actual,high))
        self.assertEqual(recorder.row['n_effective_mapped'],0)
        self.assertTrue(np.isnan(recorder.row['gate_near_zero_fraction']))
        self.assertEqual(sum(h['count'] for h in recorder.histograms),0)
        recorder.close()

    def test_multi_parent_regions_use_all_valid_weighted_edges(self):
        mapping=mapping_fixture()
        self.assertEqual(np.diff(mapping['high_parent_ptr'])[1],2)
        stats,regions=parent_statistics(np.array([.1,.5,.9,.2,0]),np.array([1,1,1,1,0],dtype=bool),mapping)
        self.assertEqual(stats['low_parent_groups'],3)
        self.assertEqual(stats['multipatch_parent_groups'],2)
        self.assertAlmostEqual(regions[0]['gate_mean'],.3)
        self.assertAlmostEqual(regions[1]['gate_mean'],.7)
        self.assertAlmostEqual(stats['within_parent_gate_variance_mean'],.04)
        self.assertGreater(stats['between_parent_gate_mean_variance'],0)

    def test_region_ids_keep_original_low_rows_after_filtering(self):
        mapping=mapping_fixture()
        mapping['low_valid_mask'][0]=False
        _,regions=parent_statistics(np.array([.1,.5,.9,.2,0]),np.array([1,1,1,1,0],dtype=bool),mapping)
        self.assertEqual([r['low_parent'] for r in regions],[1,2])

    def test_real_context_interventions_preserve_mapping_and_high_order(self):
        from models.model_ViLa_MIL_BiomedCLIP import ViLa_MIL_BiomedCLIP
        model=ViLa_MIL_BiomedCLIP.__new__(ViLa_MIL_BiomedCLIP)
        nn.Module.__init__(model)
        model.low_parent_context=LowParentContext()
        model.routing_diagnostic_mode='normal'
        model.routing_diagnostic_seed=352
        low,high=torch.randn(3,8),torch.randn(5,8)
        mapping=mapping_fixture()
        normal=model.build_mapping_context(low,high,mapping)
        high_before=high.clone()
        shuffled=model._shuffle_parent_context(normal,high)
        repeated=model._shuffle_parent_context(normal,high)
        self.assertTrue(torch.equal(shuffled['high_parent_context'],repeated['high_parent_context']))
        valid=normal['high_parent_context_valid_mask']
        self.assertTrue(torch.equal(shuffled['high_parent_context'][~valid],normal['high_parent_context'][~valid]))
        for key in normal:
            if key!='high_parent_context': self.assertTrue(torch.equal(normal[key],shuffled[key]))
        model.routing_diagnostic_mode='context_mean'
        mean=model.build_mapping_context(low,high,mapping)
        torch.testing.assert_close(mean['high_parent_context'][valid],low.mean(0).expand(int(valid.sum()),8))
        self.assertTrue(torch.equal(mean['high_parent_context'][~valid],normal['high_parent_context'][~valid]))
        self.assertTrue(torch.equal(high,high_before))

    def test_full_production_forward_hook_parity_with_stub_text(self):
        import models.model_ViLa_MIL_BiomedCLIP as mm
        class TextStub(nn.Module):
            def __init__(self,model,**kwargs):
                super().__init__(); self.model=model
                self.register_buffer('text_features',torch.randn(4,512)*.02)
            def forward(self,*args,**kwargs): return self.text_features
        class PromptStub(nn.Module):
            def __init__(self,*args):
                super().__init__(); self.ctx=nn.Parameter(torch.randn(16,32))
                self.register_buffer('tokenized_prompts',torch.ones(4,8,dtype=torch.long))
            def forward(self): return self.ctx
        cfg=SimpleNamespace(hidden_size=192,text_prompt=['a','b','c','d'],prototype_number=16,
            scale_mode='high',use_low_context_routing=True,use_routing_stabilization=True,
            use_global_proto_context=False,routing_residual_ratio=.10,finetune_text_encoder=False)
        with patch.object(mm,'create_model_from_pretrained',return_value=(nn.Identity(),None)), \
             patch.object(mm,'get_tokenizer',return_value=object()), \
             patch.object(mm,'_prepare_biomedclip_loading',return_value=('stub',None,True)), \
             patch.object(mm,'BiomedCLIPTextEncoder',TextStub),patch.object(mm,'BiomedCLIPPromptLearner',PromptStub):
            model=mm.ViLa_MIL_BiomedCLIP(cfg,num_classes=2).eval()
        low,high=torch.randn(1,3,512),torch.randn(1,5,512)
        mapping=mapping_fixture(); coords=torch.zeros(5,2); label=torch.tensor([1])
        with torch.no_grad():
            expected=model(low,torch.zeros(3,2),high,coords,label,mapping=mapping)
            recorder=RouterRecorder(model.high_router); recorder.prepare(mapping,coords.numpy(),1)
            actual=model(low,torch.zeros(3,2),high,coords,label,mapping=mapping)
            for a,b in zip(expected,actual): self.assertTrue(torch.equal(a,b))
            for mode in MODES:
                model.routing_diagnostic_mode=mode; model.routing_diagnostic_seed=352
                recorder.prepare(mapping,coords.numpy(),1)
                result=model(low,torch.zeros(3,2),high,coords,label,mapping=mapping)
                self.assertTrue(torch.isfinite(result[0]).all())
        recorder.close()

    def test_class_direction_pairing_and_margins(self):
        a=frame_fixture(); b=a.copy()
        b.loc[1,['prediction','p_0','p_1']]=[0,.6,.4]
        stats,changes=paired_changes(a,b.sample(frac=1,random_state=0))
        self.assertEqual(stats['wrong_to_correct'],1)
        self.assertEqual(stats['correct_to_wrong'],0)
        self.assertEqual(stats['changed_prediction_count'],1)
        self.assertAlmostEqual(stats['mean_absolute_probability_change'],.05)
        self.assertAlmostEqual(stats['mean_absolute_margin_change'],.1)
        cls=class_metrics(a)
        self.assertEqual(cls['auc_positive_class'],1)
        self.assertEqual(cls['sensitivity_class1'],1)
        self.assertEqual(cls['specificity_class1'],.5)
        mismatch=b.copy(); mismatch.loc[0,'label']=1
        with self.assertRaises(ValueError): paired_changes(a,mismatch)
        duplicate=pd.concat([a,a.iloc[:1]])
        with self.assertRaises(ValueError): align(a,duplicate)
        joined,groups=group_summary(changes)
        self.assertEqual(groups.n.sum(),4)
        self.assertEqual(groups.loc[groups.group=='rescued','n'].iloc[0],1)

    def test_rank_pair_credit_matches_auc_change(self):
        a=frame_fixture(); b=a.copy()
        b.loc[2,['p_0','p_1','prediction']]=[.7,.3,0]
        ranking=ranking_changes(a,b)
        stats,_=paired_changes(a,b)
        self.assertAlmostEqual(ranking['delta_pair_auc'],stats['delta_auc'])

    def test_seed_identity_independent_of_slide_order(self):
        self.assertEqual(slide_seed(352,3,'abc'),slide_seed(352,3,'abc'))
        self.assertNotEqual(slide_seed(352,3,'abc'),slide_seed(353,3,'abc'))
        plan=jobs(range(5),MODES,SEEDS)
        self.assertEqual(len(plan),35)
        self.assertEqual(sum(j['arm']=='E0' for j in plan),5)
        self.assertEqual(sum(j['mode']=='context_shuffle' for j in plan),15)

    def test_partial_and_tampered_outputs_never_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); job=dict(fold=0,arm='E0',mode='normal',seed=0)
            source=dict(arm='E0',fold=0,checkpoint='/synthetic_checkpoint',val_auc=1.,val_acc=.75,val_macro_f1=.7333333333)
            manifest=dict(split='val',provenance={'sha256':{}},checkpoint_identities={},sources=[source],plan=[job])
            write_json(root/'experiment_manifest.json',manifest)
            _,state,frames=inspect_run(root)
            self.assertEqual(state.iloc[0]['status'],'Pending')
            directory=job_dir(root,job); directory.mkdir(parents=True)
            _,state,_=inspect_run(root)
            self.assertEqual(state.iloc[0]['status'],'Partial')
            write_json(directory/'completion.json',{**job,'checkpoint':source['checkpoint'],'split':'val',
                'status':'Complete','checkpoint_identity':{},'sha256':{'predictions.csv':'wrong'}})
            _,state,_=inspect_run(root)
            self.assertEqual(state.iloc[0]['status'],'Invalid')

    def test_dryrun_never_constructs_model_or_creates_output(self):
        import run_diagnostics as runner
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'newrun'
            args=SimpleNamespace(folds=[0],modes=['normal'],shuffle_seeds=[352],include_e2=False,
                                 output_dir=out,patch_samples=3,execute=False)
            source=dict(arm='E0',fold=0,checkpoint='/synthetic_e0')
            routed=dict(arm='E1',fold=0,checkpoint='/synthetic_e1')
            with patch.object(runner,'parse_args',return_value=args),patch.object(runner,'audit',return_value=([source,routed],[])), \
                 patch.object(runner,'build_model') as model:
                runner.main()
                model.assert_not_called()
                self.assertFalse(out.exists())
            out.mkdir()
            with patch.object(runner,'parse_args',return_value=args),patch.object(runner,'audit',return_value=([source,routed],[])):
                with self.assertRaises(FileExistsError): runner.main()


if __name__=='__main__':
    unittest.main(verbosity=2)
