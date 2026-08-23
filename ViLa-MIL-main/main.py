from __future__ import print_function
import argparse
import os
from utils.file_utils import save_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_MIL_Dataset
import torch
import pandas as pd
import numpy as np
import time

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, help='data directory')
parser.add_argument('--data_folder_s', type=str, default=None, help='dir under data directory' )
parser.add_argument('--data_folder_l', type=str, default=None, help='dir under data directory' )
parser.add_argument('--max_epochs', type=int, default=80, help='maximum number of epochs to train (default: 80)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None)
parser.add_argument('--log_data', action='store_true', default=True, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--patience', type=int, default=15, help='early stopping patience (fixed: 15)')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--model_type', type=str, choices=['ViLa_MIL', 'ViLa_MIL_BiomedCLIP'], default='ViLa_MIL_BiomedCLIP', help='type of model')
parser.add_argument('--mode', type=str, choices=['transformer'], default='transformer')
parser.add_argument('--scale_mode', type=str, choices=['dual', 'low', 'high'], default='dual',
                    help='BiomedCLIP scale ablation mode; dual preserves original behavior')
parser.add_argument('--use_low_context_routing', action='store_true', default=False,
                    help='Enable Stage 3.3.2 soft low-conditioned high routing')
parser.add_argument('--mapping_path', type=str, default=None,
                    help='Stage 3.1 per-slide mapping directory or template path')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--reg', type=float, default=1e-5, help='weight decay (default: 1e-5)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'focal'], default='ce')
parser.add_argument('--task', type=str)
parser.add_argument("--text_prompt", type=str, default=None)
parser.add_argument("--text_prompt_path", type=str, default=None)
parser.add_argument("--prototype_number", type=int, default=16)
parser.add_argument(
    '--finetune_text_encoder',
    action='store_true',
    default=False,
    help='(BiomedCLIP) finetune text encoder parameters (default: False / frozen)'
)

# (BiomedCLIP) Optimizer parameter groups & finetune scope
parser.add_argument(
    '--prompt_lr',
    type=float,
    default=None,
    help='(BiomedCLIP) learning rate for prompt_learner. If omitted, auto-derive from --lr (typically 1e-4~1e-3).'
)
parser.add_argument(
    '--text_lr',
    type=float,
    default=None,
    help='(BiomedCLIP) learning rate for text encoder trainable parameters. If omitted, default to min(--lr, 1e-5).'
)
parser.add_argument(
    '--text_finetune_mode',
    type=str,
    choices=['proj', 'last', 'full'],
    default='proj',
    help='(BiomedCLIP) when --finetune_text_encoder is set: finetune only projection (proj), projection + last N layers (last), or full text tower (full).'
)
parser.add_argument(
    '--text_unfreeze_last_n',
    type=int,
    default=2,
    help='(BiomedCLIP) used when --text_finetune_mode=last: unfreeze last N transformer layers (default: 2).'
)

args = parser.parse_args()
if args.use_low_context_routing and not args.mapping_path:
    parser.error('--use_low_context_routing requires --mapping_path')
# Robustly load text prompts so the model receives a flat list[str].
# Expected by ViLa_MIL: first N are low-res prompts, next N are high-res prompts.
if args.text_prompt_path:
    try:
        df_tp = pd.read_csv(args.text_prompt_path)
        cols = [c.strip().lower() for c in df_tp.columns]

        def to_str_list(series):
            return series.astype(str).fillna("").tolist()

        low_prompts = []
        high_prompts = []

        if 'low_resolution_description' in cols and 'high_resolution_description' in cols:
            low_idx = cols.index('low_resolution_description')
            high_idx = cols.index('high_resolution_description')
            low_prompts = to_str_list(df_tp.iloc[:, low_idx])
            high_prompts = to_str_list(df_tp.iloc[:, high_idx])
        elif len(df_tp.columns) >= 2:
            # Fallback: assume last two columns are low/high descriptions
            low_prompts = to_str_list(df_tp.iloc[:, -2])
            high_prompts = to_str_list(df_tp.iloc[:, -1])
        elif len(df_tp.columns) == 1:
            # Single column: treat it as low prompts only
            low_prompts = to_str_list(df_tp.iloc[:, 0])
            high_prompts = []
        else:
            low_prompts = []
            high_prompts = []

        # Compose in expected order: [low... , high...]
        args.text_prompt = list(map(str, low_prompts)) + list(map(str, high_prompts))
    except Exception:
        # Last-resort fallback compatible with older simple CSVs (no header)
        arr = pd.read_csv(args.text_prompt_path, header=None).values
        # Flatten and stringify
        args.text_prompt = [str(x) for x in arr.reshape(-1).tolist()]

# 强制将最大 epoch 限制为 80（如果传入更大值则截断）
if args.max_epochs > 80:
    print(f"[Info] 检测到 max_epochs={args.max_epochs}，已强制限制为 80")
args.max_epochs = min(args.max_epochs, 80)

# 固定早停耐心为 10（忽略命令行传入的其它值）
args.patience = 10

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'label_frac': args.label_frac,
            'seed': args.seed,
            'model_type': args.model_type,
            'mode': args.mode,
            'scale_mode': args.scale_mode,
            'use_low_context_routing': args.use_low_context_routing,
            'mapping_path': args.mapping_path,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt,
            'patience': args.patience}

print('\nLoad Dataset')

if args.task == 'task_tcga_rcc_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/TCGA_RCC_subtyping.csv',
                                  mode = args.mode,
                                  data_dir_s = os.path.join(args.data_root_dir, args.data_folder_s),
                                  data_dir_l = os.path.join(args.data_root_dir, args.data_folder_l),
                                  shuffle = False,
                                  print_info = True,
                                  label_dict = {'CCRCC':0, 'PRCC':1, 'CRCC':2},
                                  patient_strat= False,
                                  ignore=[], mapping_path=args.mapping_path,
                                  return_mapping=args.use_low_context_routing)
                                  
elif args.task == 'task_tcga_lung_subtyping':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/TCGA_Lung_subtyping.csv',
                                  mode = args.mode,
                                  data_dir_s = os.path.join(args.data_root_dir, args.data_folder_s),
                                  data_dir_l = os.path.join(args.data_root_dir, args.data_folder_l),
                                  shuffle = False,
                                  print_info = True,
                                  label_dict = {'LUAD':0, 'LUSC':1},
                                  patient_strat= False,
                                  ignore=[], mapping_path=args.mapping_path,
                                  return_mapping=args.use_low_context_routing)

elif args.task == 'task_adenocarcinoma':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/all_data.csv',
                                  mode = args.mode,
                                  data_dir_s = os.path.join(args.data_root_dir, args.data_folder_s),
                                  data_dir_l = os.path.join(args.data_root_dir, args.data_folder_l),
                                  shuffle = False,
                                  print_info = True,
                                  label_dict = {'Adenocarcinoma':0, 'NonAdenocarcinoma':1},
                                  patient_strat= False,
                                  ignore=[], mapping_path=args.mapping_path,
                                  return_mapping=args.use_low_context_routing)

    # Sanity check for text prompts
    if isinstance(args.text_prompt, list):
        # Ensure flat list of strings
        args.text_prompt = [str(x) for x in args.text_prompt]
        expected = args.n_classes * 2
        print(f"Text prompts loaded: {len(args.text_prompt)} items (expected {expected} = 2 x n_classes)")
        if len(args.text_prompt) != expected:
            print("[Warning] The number of text prompts does not match 2 x n_classes.\n"
                  "          Ensure your CSV has both low_resolution_description and high_resolution_description columns.")
    else:
        print("[Warning] args.text_prompt is not a list. Please check --text_prompt_path parsing.")

else:
    raise NotImplementedError

if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    # If split_dir already starts with 'splits/', use it as is
    if args.split_dir.startswith('splits/') or args.split_dir.startswith('splits\\'):
        args.split_dir = args.split_dir
    else:
        args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))


def main(args):
    total_start_time = time.time() # 总计时器开始
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        # Treat k_end as an inclusive fold index.
        end = args.k_end + 1

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    all_test_f1 = []
    all_epoch_details = []  # 存储所有折的epoch详情
    
    # folds are 0-based indices into split CSVs: splits_{i}.csv
    folds = np.arange(start, end)
    if len(folds) == 0:
        raise ValueError(
            f"Empty fold range computed from --k_start={args.k_start} --k_end={args.k_end} with --k={args.k}. "
            "Note: --k_end is inclusive. Example single-fold run: --k_start 0 --k_end 0"
        )
    
    print(f"\n{'='*100}")
    print(f"🎯 开始 {len(folds)} 折交叉验证训练")
    print(f"{'='*100}")
    
    for i in folds:
        fold_start_time = time.time() # 每折计时器开始
        print(f"\n{'🔥'*50}")
        print(f"🚀 开始训练 FOLD {i+1}/{len(folds)}")
        print(f"{'🔥'*50}")
        
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, csv_path='{}/splits_{}.csv'.format(args.split_dir, i)) 
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc, _, test_f1, epoch_details = train(datasets, i, args)

        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_f1.append(test_f1)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        all_epoch_details.extend(epoch_details)  # 添加该折的epoch详情
        
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)
        
        fold_end_time = time.time() # 每折计时器结束
        fold_duration = fold_end_time - fold_start_time
        
        print(f"\n✅ FOLD {i+1} 完成! (用时: {fold_duration/60:.2f} 分钟)")
        print(f"   Final Test AUC: {test_auc:.4f}")
        print(f"   Final Test ACC: {test_acc:.4f}")
        print(f"   Final Test F1:  {test_f1:.4f}")

    # 保存epoch详情到CSV
    if all_epoch_details:
        epoch_df = pd.DataFrame(all_epoch_details)
        epoch_csv_path = os.path.join(args.results_dir, 'epoch_details.csv')
        epoch_df.to_csv(epoch_csv_path, index=False)
        print(f"\n📁 已保存epoch详情到: {epoch_csv_path}")

    # 保存折汇总数据
    fold_summary_data = []
    for i, fold in enumerate(folds):
        fold_summary_data.append({
            'fold': fold + 1,
            'test_auc': all_test_auc[i],
            'test_acc': all_test_acc[i],
            'test_f1': all_test_f1[i],
            'val_auc': all_val_auc[i],
            'val_acc': all_val_acc[i]
        })
    
    fold_summary_df = pd.DataFrame(fold_summary_data)
    fold_summary_csv = os.path.join(args.results_dir, 'fold_summary.csv')
    fold_summary_df.to_csv(fold_summary_csv, index=False)
    print(f"📁 已保存折汇总到: {fold_summary_csv}")

    # 详细总结报告
    print(f"\n{'='*100}")
    print(f"📊 {len(folds)} 折交叉验证 - 详细总结报告")
    print(f"{'='*100}")
    
    # 每折结果
    print(f"\n📋 各折详细结果:")
    print(f"{'Fold':<6}{'Test_AUC':<10}{'Test_ACC':<10}{'Test_F1':<10}{'Val_AUC':<10}")
    print(f"{'-'*50}")
    for i, fold in enumerate(folds):
        print(f"{fold+1:<6}{all_test_auc[i]:<10.4f}{all_test_acc[i]:<10.4f}{all_test_f1[i]:<10.4f}{all_val_auc[i]:<10.4f}")
    
    # 统计总结
    print(f"\n📈 统计总结:")
    print(f"Test AUC:  Mean={np.mean(all_test_auc):.4f}, Std={np.std(all_test_auc):.4f}")
    print(f"Test ACC:  Mean={np.mean(all_test_acc):.4f}, Std={np.std(all_test_acc):.4f}")
    print(f"Test F1:   Mean={np.mean(all_test_f1):.4f}, Std={np.std(all_test_f1):.4f}")
    print(f"Val AUC:   Mean={np.mean(all_val_auc):.4f}, Std={np.std(all_val_auc):.4f}")

    total_end_time = time.time() # 总计时器结束
    total_duration = total_end_time - total_start_time
    print(f"\n{'='*100}")
    print(f"🎉 训练全部完成! 总用时: {total_duration/3600:.2f} 小时 ({total_duration/60:.2f} 分钟)")
    print(f"{'='*100}")

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 'test_acc': all_test_acc, 'test_f1': all_test_f1})
    result_df = pd.DataFrame({'metric': ['mean', 'std'],
                              'test_auc': [np.mean(all_test_auc), np.std(all_test_auc)],
                              'test_f1': [np.mean(all_test_f1), np.std(all_test_f1)],
                              'test_acc': [np.mean(all_test_acc), np.std(all_test_acc)],
                              'val_auc': [np.mean(all_val_auc), np.std(all_val_auc)],
                              })

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
        result_name = 'result_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
        result_name = 'result.csv'

    result_df.to_csv(os.path.join(args.results_dir, result_name), index=False)
    final_df.to_csv(os.path.join(args.results_dir, save_name))


if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")


