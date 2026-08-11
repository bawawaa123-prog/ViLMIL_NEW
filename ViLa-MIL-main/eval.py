from __future__ import print_function

import numpy as np
import argparse
import torch
import os
import pandas as pd
from utils.utils import *
from math import floor
from datasets.dataset_generic import Generic_MIL_Dataset
from utils.eval_utils import *
import time
from tqdm import tqdm

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evaluation settings 
parser = argparse.ArgumentParser(description='Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None, help='data directory')
parser.add_argument('--data_folder_s', type=str, default=None, help='dir under data directory' )
parser.add_argument('--data_folder_l', type=str, default=None, help='dir under data directory' )
parser.add_argument('--results_dir', type=str, default='./results')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['ViLa_MIL', 'ViLa_MIL_BiomedCLIP'], default='ViLa_MIL')
parser.add_argument('--mode', type=str, choices=['transformer'], default='transformer')
parser.add_argument('--drop_out', action='store_true', default=False, help='whether model uses dropout')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str)
parser.add_argument('--csv_path', type=str, default='dataset_csv/all_data.csv',
                    help='dataset csv file to evaluate')
parser.add_argument("--text_prompt", type=str, default=None)
parser.add_argument("--text_prompt_path", type=str, default=None)
parser.add_argument("--prototype_number", type=int, default=16, help='number of prototypes (default: 16)')

args = parser.parse_args()

# 更稳健地加载文本提示
if args.text_prompt_path:
    try:
        df_tp = pd.read_csv(args.text_prompt_path)
        cols = [c.strip().lower() for c in df_tp.columns]
        low_prompts, high_prompts = [], []
        if 'low_resolution_description' in cols and 'high_resolution_description' in cols:
            low_idx, high_idx = cols.index('low_resolution_description'), cols.index('high_resolution_description')
            low_prompts = df_tp.iloc[:, low_idx].astype(str).fillna("").tolist()
            high_prompts = df_tp.iloc[:, high_idx].astype(str).fillna("").tolist()
        args.text_prompt = list(map(str, low_prompts)) + list(map(str, high_prompts))
    except Exception:
        arr = pd.read_csv(args.text_prompt_path, header=None).values
        args.text_prompt = [str(x) for x in arr.reshape(-1).tolist()]

args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'mode': args.mode,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(os.path.join(args.save_dir, 'eval_experiment_{}.txt'.format(args.save_exp_code)), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)

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
                                  ignore=[])

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
                                  ignore=[])

elif args.task == 'task_adenocarcinoma':
    args.n_classes = 2
    # Allow external validation by switching CSV via --csv_path.
    dataset = Generic_MIL_Dataset(
        csv_path=args.csv_path,
        mode=args.mode,
        data_dir_s=os.path.join(args.data_root_dir, args.data_folder_s),
        data_dir_l=os.path.join(args.data_root_dir, args.data_folder_l),
        shuffle=False,
        print_info=True,
        label_dict={'Adenocarcinoma': 0, 'NonAdenocarcinoma': 1},
        patient_strat=False,
        ignore=[]
    )

else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

# 自动检测实际可用的折数（只处理存在的模型和split文件）
available_folds = []
for i in range(start, end):
    ckpt_path = os.path.join(args.models_dir, f's_{i}_checkpoint.pt')
    split_path = os.path.join(args.splits_dir, f'splits_{i}.csv')
    if os.path.isfile(ckpt_path) and os.path.isfile(split_path):
        available_folds.append(i)
    else:
        missing = []
        if not os.path.isfile(ckpt_path):
            missing.append('ckpt')
        if not os.path.isfile(split_path):
            missing.append('split')
        print(f"[Warn] fold {i} skipped (missing {','.join(missing)}): {ckpt_path}, {split_path}")

if args.fold != -1:
    # 如果指定了fold，只处理该fold（前提是文件存在）
    if args.fold in available_folds:
        available_folds = [args.fold]
    else:
        print(f"指定的fold {args.fold} 不存在有效的模型或split文件，程序退出。")
        exit(1)

if not available_folds:
    print("没有检测到任何可用的模型权重和split文件，程序退出。")
    exit(1)

ckpt_paths = [os.path.join(args.models_dir, f's_{fold}_checkpoint.pt') for fold in available_folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    total_start_time = time.time()

    all_results = []
    all_auc = []
    all_acc = []
    all_f1 = []
    all_true = []
    all_pred = []
    timing_records = []

    print(f"\n共检测到 {len(available_folds)} 个有效折：{available_folds}")
    for ckpt_idx, current_fold in enumerate(tqdm(available_folds, desc="Overall Progress", ncols=80)):
        fold_start_time = time.time()
        print(f"\n>>> Processing Fold {current_fold}...")

        ckpt_path = ckpt_paths[ckpt_idx]
        split_path = os.path.join(args.splits_dir, f'splits_{current_fold}.csv')
        if os.path.isfile(ckpt_path):
            print(f"✔️ 成功检测到权重文件: {ckpt_path}")
        else:
            print(f"❌ 未找到权重文件: {ckpt_path}，跳过该折。")
            continue
        if not os.path.isfile(split_path):
            print(f"❌ 未找到split文件: {split_path}，跳过该折。")
            continue

        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            datasets = dataset.return_splits(from_id=False, csv_path=split_path)
            split_dataset = datasets[datasets_id[args.split]]

        # ========== 评估时加WSI进度条 ==========
        # 这里假设eval函数返回的df是每个WSI一行
        try:
            wsi_ids = split_dataset.slide_data['slide_id'].tolist() if hasattr(split_dataset, 'slide_data') else None
            wsi_bar = tqdm(total=len(split_dataset), desc=f"Fold {current_fold} WSI", ncols=80)
            # 修改eval_utils.eval函数，让其支持传入进度条对象（如果你有源码，可以在eval内部每处理一个WSI就wsi_bar.update(1)）
            # 如果不能改eval内部，可以在外部模拟进度条
            eval_start_time = time.time()
            model, patient_results, test_error, auc, test_f1, df, _ = eval(args.mode, split_dataset, args, ckpt_path)
            eval_end_time = time.time()
            wsi_bar.update(len(split_dataset))
            wsi_bar.close()
            print(f"✅ 权重加载成功: {ckpt_path}")
        except Exception as e:
            print(f"❌ 加载权重或评估时出错: {ckpt_path}\n错误信息: {e}")
            continue

        # 记录详细时间
        fold_end_time = time.time()
        fold_duration = fold_end_time - fold_start_time
        eval_duration = eval_end_time - eval_start_time
        avg_wsi_time = eval_duration / len(df) if len(df) > 0 else 0
        timing_records.append({
            'fold': current_fold,
            'duration_seconds': fold_duration,
            'eval_seconds': eval_duration,
            'num_wsi': len(df),
            'avg_wsi_seconds': avg_wsi_time
        })

        all_results.append(patient_results)
        all_auc.append(auc)
        all_acc.append(1-test_error)
        all_f1.append(test_f1)
        all_true += list(df['Y'])
        all_pred += list(df['Y_hat'])

        df.to_csv(os.path.join(args.save_dir, f'fold_{current_fold}.csv'), index=False)

        # ========== 记录误判WSI ==========
        # 优先使用label（字符串标签），否则用Y（数字标签）
        misclassified = df[df['Y'] != df['Y_hat']]
        if 'slide_id' not in misclassified.columns and 'case_id' in misclassified.columns:
            misclassified = misclassified.rename(columns={'case_id': 'slide_id'})
        # 优先保存label字段
        save_cols = ['slide_id']
        if 'label' in misclassified.columns:
            save_cols.append('label')
        elif 'Y' in misclassified.columns:
            save_cols.append('Y')
        if 'Y_hat' in misclassified.columns:
            save_cols.append('Y_hat')
        if 'prob' in misclassified.columns:
            save_cols.append('prob')
        misclassified[save_cols].to_csv(
            os.path.join(args.save_dir, f'fold_{current_fold}_misclassified.csv'), index=False)
        print(f"<<< Fold {current_fold} processed in {fold_duration:.2f} seconds. "
              f"AUC: {auc:.4f}, Acc: {1-test_error:.4f}, "
              f"Avg WSI Time: {avg_wsi_time:.2f}s, Misclassified: {len(misclassified)}")

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    print(f"\n{'='*40}")
    print(f"Total evaluation finished in {total_duration:.2f} seconds ({total_duration/60:.2f} minutes).")
    print(f"{'='*40}")

    timing_df = pd.DataFrame(timing_records)
    timing_df['total_duration_seconds'] = total_duration
    timing_log_path = os.path.join(args.save_dir, 'timing_details.csv')
    timing_df.to_csv(timing_log_path, index=False)
    print(f"🕒 Timing details saved to: {timing_log_path}")

    if len(all_auc) == 0:
        raise ValueError(
            "No folds were successfully evaluated. "
            "This usually means the selected split file does not match the external dataset, "
            "or all slide_ids were filtered out."
        )

    final_df = pd.DataFrame({'folds': available_folds, 'test_auc': all_auc,'test_acc': all_acc, 'test_f1': all_f1})
    result_df = pd.DataFrame({'metric': ['mean', 'var'],
                              'test_auc': [np.mean(all_auc), np.std(all_auc)],
                              'test_acc': [np.mean(all_acc), np.std(all_acc)],
                              'test_f1': [np.mean(all_f1), np.std(all_f1)],
                              })

    if len(available_folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(available_folds[0], available_folds[-1])
        result_name = 'result_partial_{}_{}.csv'.format(available_folds[0], available_folds[-1])
    else:
        save_name = 'summary.csv'
        result_name = 'result.csv'

    result_df.to_csv(os.path.join(args.save_dir, result_name), index=False)
    final_df.to_csv(os.path.join(args.save_dir, save_name))
