"""CPU-only regression tests: selection, provenance, overwrite guards and RNG."""
import ast
import copy
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

import common
from common import ARMS, ROOT, environment
from aggregate_fivefold import paired, statistics
from evaluate_validation import metrics_from_predictions
from run_experiment import plan, launch

sys.path.insert(0, str(ROOT))


class MatchedTests(unittest.TestCase):
    def test_paired_missing_never_produces_fivefold_mean(self):
        rows = [dict(arm=arm, fold=fold, status='Complete' if arm != 'E2' else 'Pending',
                     **{k: .8 + .01 * fold + (.02 if arm == 'E1' else 0)
                        if arm != 'E2' else np.nan for k in common.METRICS})
                for arm in ARMS for fold in range(5)]
        frame = pd.DataFrame(rows)
        summaries = statistics(frame)
        self.assertTrue(summaries[summaries.arm == 'E2']['mean'].isna().all())
        diff, stats = paired(frame)
        valid = stats[stats.comparison == 'E1-E0']
        np.testing.assert_allclose(valid.mean_delta, .02)
        self.assertTrue(valid.improvement_folds.eq(5).all())
        self.assertTrue(diff[diff.comparison == 'E2-E0'].delta_val_auc.isna().all())
        self.assertTrue(stats[stats.comparison == 'E1-E2'].mean_delta.isna().all())

    def test_validation_metric_conventions(self):
        frame = pd.DataFrame(dict(label=[0, 0, 1, 1], prediction=[0, 1, 1, 1],
                                  p_0=[.9, .4, .2, .1], p_1=[.1, .6, .8, .9]))
        values = metrics_from_predictions(frame)
        self.assertEqual(values['val_auc'], 1.0)
        self.assertEqual(values['val_acc'], .75)
        self.assertAlmostEqual(values['val_macro_f1'], (2/3 + .8)/2)
        with self.assertRaises(ValueError):
            metrics_from_predictions(frame.assign(p_0=frame.p_1, p_1=frame.p_0))

    def test_first_acc_maximum_wins_not_loss_or_auc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            splits = root / 'splits'
            splits.mkdir()
            source = root / '00_clean_high_baseline/clean_high/source_s1'
            source.mkdir(parents=True)
            split = pd.DataFrame({'train': ['a'], 'val': ['b'], 'test': ['c']})
            split.to_csv(splits / 'splits_0.csv', index=False)
            split.to_csv(source / 'splits_0.csv', index=False)
            cfg = dict(num_splits=5, task='task_adenocarcinoma', max_epochs=80, lr=1e-4,
                label_frac=1.0, seed=1, model_type='ViLa_MIL_BiomedCLIP', mode='transformer',
                scale_mode='high', skip_test_evaluation=True, use_low_context_routing=False,
                use_routing_stabilization=False, use_global_proto_context=False,
                init_checkpoint_template=None, use_drop_out=True, weighted_sample=False,
                opt='adam', patience=10, split_dir='splits', mapping_path='mapping', k_start=0, k_end=1)
            (source / 'experiment_fixture.txt').write_text(repr(cfg))
            (source / 's_0_checkpoint.pt').write_bytes(b'fixture')
            epochs = pd.DataFrame([dict(fold=1, epoch=i, val_acc=.8 if i == 1 else .9,
                val_auc=.92 if i == 2 else .99, val_loss=1/i, val_f1=.88 if i == 2 else .95)
                for i in range(1, 13)])
            epochs.to_csv(source / 'epoch_details.csv', index=False)
            final = pd.DataFrame([dict(fold=1, val_acc=.9, val_auc=.92,
                                      test_auc=np.nan, test_acc=np.nan, test_f1=np.nan)])
            final.to_csv(source / 'fold_summary.csv', index=False)
            log = root / 'train.log'
            log.write_text('finished!\nend script\n')
            with patch.multiple(common, ROOT=root, SPLITS=splits, MAPPING=root/'mapping'):
                result = common.inspect_result(source, log, 'E0', 0, check_schema=False)
                self.assertEqual(result['selected_epoch'], 2)
                self.assertEqual(result['val_macro_f1'], .88)
                epochs.drop(columns='val_f1').to_csv(source / 'epoch_details.csv', index=False)
                self.assertEqual(common.inspect_result(source, log, 'E0', 0, check_schema=False)['status'], 'MissingMetric')
                epochs.to_csv(source / 'epoch_details.csv', index=False)
                final['test_f1'] = .99
                final.to_csv(source / 'fold_summary.csv', index=False)
                with self.assertRaises(ValueError):
                    common.inspect_result(source, log, 'E0', 0, check_schema=False)
                final['test_f1'] = np.nan
                final.to_csv(source / 'fold_summary.csv', index=False)
                epochs.iloc[:-1].to_csv(source / 'epoch_details.csv', index=False)
                with self.assertRaises(ValueError):
                    common.inspect_result(source, log, 'E0', 0, check_schema=False)

    def test_partial_jobs_abort_before_any_training(self):
        def fake(arm, fold):
            return dict(status='Invalid/Partial' if fold == 3 else 'Complete', fold=fold)
        with patch('run_experiment.collect', side_effect=fake), patch('run_experiment.launch') as launch:
            with self.assertRaises(RuntimeError):
                plan('E0')
            launch.assert_not_called()

    def test_ambiguous_sources_are_not_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a', Path(tmp)/'b'
            a.mkdir()
            b.mkdir()
            with patch('common.destinations', return_value=[(a, a/'log'), (b, b/'log')]):
                row = common.collect('E0', 2)
                self.assertEqual(row['status'], 'Invalid/Partial')
                self.assertIn('Ambiguous', row['issue'])

    def test_reserved_output_is_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            parent = base / 'runs/E2/fold0'
            parent.mkdir(parents=True)
            with patch('common.BASE', base), patch('common.destinations', return_value=[(parent/'result', parent/'train.log')]):
                self.assertEqual(common.collect('E2', 0)['status'], 'Invalid/Partial')

    def test_nonzero_training_preserves_log_and_blocks_relaunch(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / 'fold0'
            destination = parent / 'run_s1'
            process = SimpleNamespace(stdout=iter(['failure fixture\n']), wait=lambda: 7)
            with patch('run_experiment.destinations', return_value=[(destination, parent/'train.log')]), \
                 patch('run_experiment.provenance', return_value={'fixture': True}), \
                 patch('run_experiment.subprocess.Popen', return_value=process) as popen:
                with self.assertRaises(RuntimeError):
                    launch('E2', 0)
                self.assertTrue((parent/'.running.lock').exists())
                self.assertEqual((parent/'train.log').read_text(), 'failure fixture\n')
                with self.assertRaises(FileExistsError):
                    launch('E2', 0)
                self.assertEqual(popen.call_count, 1)

    def test_production_constructor_shared_initialization_rng(self):
        # Exercise the real model constructor and real HighRouter. Replace only
        # external pretrained loading/text wrappers to avoid downloads or GPU use.
        import models.model_ViLa_MIL_BiomedCLIP as model_module
        class TextStub(nn.Module):
            def __init__(self, model, **kwargs):
                super().__init__()
                self.model = model
        class PromptStub(nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.context = nn.Parameter(torch.randn(4, 16, 32))
        def pretrained(*args, **kwargs):
            return nn.Linear(32, 32), None
        models, rngs = {}, {}
        torch.set_num_threads(1)
        with patch.object(model_module, 'create_model_from_pretrained', side_effect=pretrained), \
             patch.object(model_module, 'get_tokenizer', return_value=object()), \
             patch.object(model_module, '_prepare_biomedclip_loading', return_value=('stub', None, True)), \
             patch.object(model_module, 'BiomedCLIPTextEncoder', TextStub), \
             patch.object(model_module, 'BiomedCLIPPromptLearner', PromptStub):
            for arm, (scale, routing) in ARMS.items():
                torch.manual_seed(1)
                cfg = SimpleNamespace(hidden_size=192, text_prompt=['a','b','c','d'], prototype_number=16,
                    scale_mode=scale, use_low_context_routing=routing, use_routing_stabilization=routing,
                    use_global_proto_context=False, routing_residual_ratio=.10, finetune_text_encoder=False)
                model = model_module.ViLa_MIL_BiomedCLIP(config=cfg, num_classes=2)
                models[arm] = copy.deepcopy(model.state_dict())
                rngs[arm] = torch.random.get_rng_state().clone()
        for arm in ('E1', 'E2'):
            shared = {k: v for k, v in models[arm].items() if not k.startswith('high_router.')}
            self.assertEqual(set(shared), set(models['E0']))
            for key, value in shared.items():
                self.assertTrue(torch.equal(value, models['E0'][key]), key)
            self.assertTrue(torch.equal(rngs[arm], rngs['E0']))
        self.assertTrue(torch.count_nonzero(models['E1']['high_router.context_projection.weight']) == 0)

    def test_fold_commands_are_inclusive_and_frozen(self):
        tree = ast.parse((ROOT/'main.py').read_text())
        options = {arg.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'
                   for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
        for arm in ARMS:
            for fold in range(5):
                cmd = common.command(arm, fold, Path('/tmp/unexecuted'))
                self.assertEqual(cmd[cmd.index('--k_start')+1], str(fold))
                self.assertEqual(cmd[cmd.index('--k_end')+1], str(fold))
                self.assertIn('--skip_test_evaluation', cmd)
                self.assertNotIn('--init_checkpoint_template', cmd)
                self.assertEqual('--use_low_context_routing' in cmd, arm == 'E1')
                self.assertNotIn('--use_global_proto_context', cmd)
                self.assertTrue({a for a in cmd if a.startswith('--')} <= options)


if __name__ == '__main__':
    environment()
    unittest.main(verbosity=2)
