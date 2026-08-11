import pdb
import os
import pandas as pd
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset
import argparse
import numpy as np
from utils.eval_utils import *
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.utils import *
import h5py
from utils.eval_utils import initiate_model as eval_initiate_model
from utils.core_utils import Accuracy_Logger
import pickle
import random

def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_text_prompts(text_prompt_path):
    """
    从CSV文件中加载文本提示
    """
    try:
        df = pd.read_csv(text_prompt_path)
        # 假设CSV文件有'prompt'列或者取第一列
        if 'prompt' in df.columns:
            prompts = df['prompt'].tolist()
        elif 'text' in df.columns:
            prompts = df['text'].tolist()
        else:
            # 取第一列
            prompts = df.iloc[:, 0].tolist()
        return prompts
    except Exception as e:
        print(f"Warning: Could not load text prompts from {text_prompt_path}: {e}")
        # 返回默认的腺癌相关文本提示
        return [
            "a histopathology image of adenocarcinoma",
            "a histopathology image of non-adenocarcinoma"
        ]

def eval_enhanced(dataset, args, ckpt_path, dataset_type="test"):
    model = eval_initiate_model(args, ckpt_path)
    
    print(f'Init Model for {dataset_type} evaluation')    
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_MIL)
    
    Y_hat = []
    Y = []
    Y_prob = []
    results_dict = {}
    
    # 用于存储详细预测结果
    slide_ids = []
    predictions = []
    true_labels = []
    predicted_probs = []
    
    with torch.no_grad():
        for batch_idx, (data, label, slide_id) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            
            logits, Y_prob_batch, Y_hat_batch, _, results_dict_batch = model(data, label=label, slide_id=slide_id)
            
            Y.append(label.item())
            Y_hat.append(Y_hat_batch.item())
            Y_prob.append(Y_prob_batch.cpu().numpy())
            
            # 保存详细信息
            slide_ids.append(slide_id[0])
            predictions.append(Y_hat_batch.item())
            true_labels.append(label.item())
            predicted_probs.append(Y_prob_batch.cpu().numpy()[0])
            
            acc_logger.log(Y_hat_batch, label)
    
    del dataset
    
    Y = np.array(Y)
    Y_hat = np.array(Y_hat)
    Y_prob = np.array(Y_prob)
    
    # 计算性能指标
    auc_value = roc_auc_score(Y, Y_prob[:, 1])
    test_error = 1 - acc_logger.get_summary()
    f1_value = f1_score(Y, Y_hat, average='weighted')
    
    print(f'{dataset_type}_error: ', test_error)
    print(f'{dataset_type}_auc: ', auc_value)
    print(f'{dataset_type}_f1: ', f1_value)
    
    # 增强功能：详细分类统计
    cm = confusion_matrix(Y, Y_hat)
    
    # 假设标签映射：0=Adenocarcinoma, 1=NonAdenocarcinoma
    label_dict = {'Adenocarcinoma': 0, 'NonAdenocarcinoma': 1}
    reverse_label_dict = {v: k for k, v in label_dict.items()}
    
    # 统计每个类别的正确识别数
    adenocarcinoma_correct = np.sum((Y == 0) & (Y_hat == 0))
    non_adenocarcinoma_correct = np.sum((Y == 1) & (Y_hat == 1))
    
    # 统计每个类别的总数
    adenocarcinoma_total = np.sum(Y == 0)
    non_adenocarcinoma_total = np.sum(Y == 1)
    
    # 计算准确率
    adenocarcinoma_acc = adenocarcinoma_correct / adenocarcinoma_total if adenocarcinoma_total > 0 else 0
    non_adenocarcinoma_acc = non_adenocarcinoma_correct / non_adenocarcinoma_total if non_adenocarcinoma_total > 0 else 0
    
    # 找出错误识别的样本
    error_indices = np.where(Y != Y_hat)[0]
    error_slide_ids = [slide_ids[i] for i in error_indices]
    error_true_labels = [reverse_label_dict[Y[i]] for i in error_indices]
    error_pred_labels = [reverse_label_dict[Y_hat[i]] for i in error_indices]
    error_probs = [predicted_probs[i] for i in error_indices]
    
    # 创建详细结果字典
    detailed_results = {
        'error': test_error,
        'auc': auc_value,
        'f1': f1_value,
        'accuracy': 1 - test_error,
        'confusion_matrix': cm.tolist(),
        'adenocarcinoma_correct': int(adenocarcinoma_correct),
        'adenocarcinoma_total': int(adenocarcinoma_total),
        'adenocarcinoma_accuracy': float(adenocarcinoma_acc),
        'non_adenocarcinoma_correct': int(non_adenocarcinoma_correct),
        'non_adenocarcinoma_total': int(non_adenocarcinoma_total),
        'non_adenocarcinoma_accuracy': float(non_adenocarcinoma_acc),
        'error_slide_ids': error_slide_ids,
        'error_details': [
            {
                'slide_id': error_slide_ids[i],
                'true_label': error_true_labels[i],
                'predicted_label': error_pred_labels[i],
                'prediction_probability': float(error_probs[i])
            } for i in range(len(error_slide_ids))
        ]
    }
    
    # 创建详细的fold结果DataFrame
    fold_df = pd.DataFrame({
        'slide_id': slide_ids,
        'true_label': [reverse_label_dict[label] for label in true_labels],
        'predicted_label': [reverse_label_dict[pred] for pred in predictions],
        'prediction_probability': predicted_probs,
        'correct': [t == p for t, p in zip(true_labels, predictions)]
    })
    
    return test_error, auc_value, f1_value, detailed_results, fold_df

def check_overfitting(train_metrics, val_metrics, test_metrics, threshold=0.1):
    """
    检查是否过拟合
    threshold: 如果训练集指标明显高于测试集指标超过此阈值，认为过拟合
    """
    overfitting_signs = []
    
    # 检查AUC
    if train_metrics['auc'] - test_metrics['auc'] > threshold:
        overfitting_signs.append(f"AUC: Train({train_metrics['auc']:.4f}) >> Test({test_metrics['auc']:.4f})")
    
    # 检查准确率
    if train_metrics['accuracy'] - test_metrics['accuracy'] > threshold:
        overfitting_signs.append(f"Accuracy: Train({train_metrics['accuracy']:.4f}) >> Test({test_metrics['accuracy']:.4f})")
    
    # 检查F1
    if train_metrics['f1'] - test_metrics['f1'] > threshold:
        overfitting_signs.append(f"F1: Train({train_metrics['f1']:.4f}) >> Test({test_metrics['f1']:.4f})")
    
    return len(overfitting_signs) > 0, overfitting_signs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced ViLa-MIL Evaluation Script with Overfitting Detection')
    parser.add_argument('--drop_out', action='store_true', default=False, 
                        help='enable dropout')
    parser.add_argument('--k', type=int, default=10, help='number of folds')
    parser.add_argument('--k_start', type=int, default=-1)
    parser.add_argument('--k_end', type=int, default=-1)
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--micro_average', action='store_true', default=False)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--save_dir', type=str, default='./eval_results', 
                        help='results directory (default: ./eval_results)')
    parser.add_argument('--models_dir', type=str, default=None, 
                        help='models directory')
    parser.add_argument('--models_exp_code', type=str, default=None, 
                        help='experiment code')
    parser.add_argument('--save_exp_code', type=str, default=None, 
                        help='experiment code')
    parser.add_argument('--model_type', type=str, default='ViLa_MIL', help='type of model')
    parser.add_argument('--model_size', type=str, default=None, help='size of model')
    parser.add_argument('--results_dir', type=str, default='./results', 
                        help='results directory')
    parser.add_argument('--data_root_dir', type=str, default=None, 
                        help='data directory')
    parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping', 'task_adenocarcinoma'])
    parser.add_argument('--slide_ext', type=str, default= '.svs')
    parser.add_argument('--splits_dir', type=str, default=None, 
                        help='splits directory, if using custom splits other than what matches the task')
    parser.add_argument('--mode', type=str, choices=['path', 'omic', 'pathomic', 'cluster', 'coattn', 'transformer'], 
                        default='path', help='which modalities to use')
    parser.add_argument('--data_folder_s', type=str, default=None, 
                        help='directory under data directory')  
    parser.add_argument('--data_folder_l', type=str, default=None, 
                        help='directory under data directory')
    parser.add_argument('--text_prompt_path', type=str, default='text_prompt.csv', 
                        help='text prompts file path')
    parser.add_argument('--text_prompt', type=str, default=None, 
                        help='text prompt string')
    parser.add_argument('--prototype_number', type=int, default=16, help='number of prototypes')
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment')
    parser.add_argument('--overfitting_threshold', type=float, default=0.1, 
                        help='threshold for overfitting detection (default: 0.1)')

    args = parser.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载文本提示
    if args.text_prompt is None and args.text_prompt_path:
        text_prompts = load_text_prompts(args.text_prompt_path)
        args.text_prompt = text_prompts
        print(f"Loaded text prompts: {text_prompts}")
    elif args.text_prompt is None:
        # 使用默认的腺癌相关文本提示
        args.text_prompt = [
            "a histopathology image of adenocarcinoma",
            "a histopathology image of non-adenocarcinoma"
        ]
        print(f"Using default text prompts: {args.text_prompt}")

    args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
    args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

    os.makedirs(args.save_dir, exist_ok = True)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    if args.fold == -1:
        folds = range(start, end)
    else:
        folds = range(args.fold, args.fold+1)
    ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
    datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

    if args.task == 'task_adenocarcinoma':
        args.n_classes=2
        dataset = Generic_MIL_Dataset(
            csv_path='datasss.csv',
            data_dir_s=args.data_folder_s,
            data_dir_l=args.data_folder_l,
            shuffle=False,
            seed=args.seed,
            print_info=True,
            label_dict={'Adenocarcinoma': 0, 'NonAdenocarcinoma': 1},
            patient_strat=False,
            ignore=[],
            mode=args.mode
        )
    else:
        raise NotImplementedError

    if args.splits_dir is None:
        args.splits_dir = args.models_dir
    
    assert os.path.isdir(args.models_dir)
    assert os.path.isdir(args.splits_dir)
    
    print('\nLoad Dataset')

    # 存储所有数据集的指标
    all_train_metrics = []
    all_val_metrics = []
    all_test_metrics = []
    
    # 存储所有fold的详细结果
    all_fold_details = {}
    all_fold_dataframes = {}
    overall_error_summary = []
    overfitting_analysis = []

    for i, (ckpt_path, fold) in enumerate(zip(ckpt_paths, folds)):
        print(f"\n{'='*50}")
        print(f"🔍 Evaluating Fold {fold}")
        print(f"{'='*50}")
        
        seed_torch(args.seed)
        
        # 修复：正确获取分割的数据集
        split_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.splits_dir, fold))
        
        # 检查返回的数据结构
        if isinstance(split_dataset, dict):
            train_dataset = split_dataset['train']
            val_dataset = split_dataset['val']
            test_dataset = split_dataset['test']
        elif isinstance(split_dataset, (list, tuple)):
            # 如果返回的是元组或列表，通常顺序是 [train, val, test]
            train_dataset = split_dataset[0]
            val_dataset = split_dataset[1]
            test_dataset = split_dataset[2]
        else:
            raise ValueError(f"Unexpected split_dataset type: {type(split_dataset)}")
        
        # 评估训练集
        print("\n📊 Training Set Evaluation:")
        train_error, train_auc, train_f1, train_detailed, _ = eval_enhanced(
            train_dataset, args, ckpt_path, "train")
        
        # 评估验证集
        print("\n📊 Validation Set Evaluation:")
        val_error, val_auc, val_f1, val_detailed, _ = eval_enhanced(
            val_dataset, args, ckpt_path, "val")
        
        # 评估测试集
        print("\n📊 Test Set Evaluation:")
        test_error, test_auc, test_f1, test_detailed, fold_df = eval_enhanced(
            test_dataset, args, ckpt_path, "test")
        
        # 存储指标
        train_metrics = {'auc': train_auc, 'accuracy': 1-train_error, 'f1': train_f1}
        val_metrics = {'auc': val_auc, 'accuracy': 1-val_error, 'f1': val_f1}
        test_metrics = {'auc': test_auc, 'accuracy': 1-test_error, 'f1': test_f1}
        
        all_train_metrics.append(train_metrics)
        all_val_metrics.append(val_metrics)
        all_test_metrics.append(test_metrics)
        
        # 过拟合检测
        is_overfitting, overfitting_signs = check_overfitting(
            train_metrics, val_metrics, test_metrics, args.overfitting_threshold)
        
        overfitting_info = {
            'fold': fold,
            'is_overfitting': is_overfitting,
            'overfitting_signs': overfitting_signs,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'test_auc': test_auc,
            'train_acc': 1-train_error,
            'val_acc': 1-val_error,
            'test_acc': 1-test_error,
            'train_f1': train_f1,
            'val_f1': val_f1,
            'test_f1': test_f1
        }
        overfitting_analysis.append(overfitting_info)
        
        # 保存每个fold的详细结果
        all_fold_details[f'fold_{fold}'] = test_detailed
        all_fold_dataframes[f'fold_{fold}'] = fold_df
        
        # 添加到总体错误汇总
        for error in test_detailed['error_details']:
            error['fold'] = fold
            overall_error_summary.append(error)
        
        # 保存fold详细CSV
        fold_df.to_csv(os.path.join(args.save_dir, f'fold_{fold}_detailed.csv'), index=False)
        
        # 打印结果对比
        print(f"\n📈 Fold {fold} Results Summary:")
        print(f"  📚 Train: AUC={train_auc:.4f}, Acc={1-train_error:.4f}, F1={train_f1:.4f}")
        print(f"  📝 Val:   AUC={val_auc:.4f}, Acc={1-val_error:.4f}, F1={val_f1:.4f}")
        print(f"  🎯 Test:  AUC={test_auc:.4f}, Acc={1-test_error:.4f}, F1={test_f1:.4f}")
        
        if is_overfitting:
            print(f"  ⚠️  OVERFITTING DETECTED:")
            for sign in overfitting_signs:
                print(f"     - {sign}")
        else:
            print(f"  ✅ No significant overfitting detected")
        
        print(f"  Adenocarcinoma Correct: {test_detailed['adenocarcinoma_correct']}/{test_detailed['adenocarcinoma_total']} ({test_detailed['adenocarcinoma_accuracy']:.4f})")
        print(f"  NonAdenocarcinoma Correct: {test_detailed['non_adenocarcinoma_correct']}/{test_detailed['non_adenocarcinoma_total']} ({test_detailed['non_adenocarcinoma_accuracy']:.4f})")
        print(f"  Errors: {len(test_detailed['error_slide_ids'])} slides")

    # 创建增强的汇总表（包含train/val/test对比）
    enhanced_summary = []
    for i, fold in enumerate(folds):
        fold_details = all_fold_details[f'fold_{fold}']
        overfitting_info = overfitting_analysis[i]
        enhanced_summary.append({
            'fold': fold,
            'train_auc': overfitting_info['train_auc'],
            'val_auc': overfitting_info['val_auc'],
            'test_auc': overfitting_info['test_auc'],
            'train_acc': overfitting_info['train_acc'],
            'val_acc': overfitting_info['val_acc'],
            'test_acc': overfitting_info['test_acc'],
            'train_f1': overfitting_info['train_f1'],
            'val_f1': overfitting_info['val_f1'],
            'test_f1': overfitting_info['test_f1'],
            'is_overfitting': overfitting_info['is_overfitting'],
            'adenocarcinoma_correct': fold_details['adenocarcinoma_correct'],
            'adenocarcinoma_total': fold_details['adenocarcinoma_total'],
            'adenocarcinoma_accuracy': fold_details['adenocarcinoma_accuracy'],
            'non_adenocarcinoma_correct': fold_details['non_adenocarcinoma_correct'],
            'non_adenocarcinoma_total': fold_details['non_adenocarcinoma_total'],
            'non_adenocarcinoma_accuracy': fold_details['non_adenocarcinoma_accuracy'],
            'total_errors': len(fold_details['error_slide_ids'])
        })

    enhanced_summary_df = pd.DataFrame(enhanced_summary)
    enhanced_summary_df.to_csv(os.path.join(args.save_dir, 'enhanced_summary_with_overfitting.csv'), index=False)

    # 保存过拟合分析
    overfitting_df = pd.DataFrame(overfitting_analysis)
    overfitting_df.to_csv(os.path.join(args.save_dir, 'overfitting_analysis.csv'), index=False)

    # 创建错误分析汇总
    error_summary_df = pd.DataFrame(overall_error_summary)
    if not error_summary_df.empty:
        error_summary_df.to_csv(os.path.join(args.save_dir, 'error_analysis.csv'), index=False)

    # 计算最终统计
    final_results = {
        'train_auc_mean': np.mean([m['auc'] for m in all_train_metrics]),
        'train_auc_std': np.std([m['auc'] for m in all_train_metrics]),
        'val_auc_mean': np.mean([m['auc'] for m in all_val_metrics]),
        'val_auc_std': np.std([m['auc'] for m in all_val_metrics]),
        'test_auc_mean': np.mean([m['auc'] for m in all_test_metrics]),
        'test_auc_std': np.std([m['auc'] for m in all_test_metrics]),
        'train_acc_mean': np.mean([m['accuracy'] for m in all_train_metrics]),
        'train_acc_std': np.std([m['accuracy'] for m in all_train_metrics]),
        'val_acc_mean': np.mean([m['accuracy'] for m in all_val_metrics]),
        'val_acc_std': np.std([m['accuracy'] for m in all_val_metrics]),
        'test_acc_mean': np.mean([m['accuracy'] for m in all_test_metrics]),
        'test_acc_std': np.std([m['accuracy'] for m in all_test_metrics]),
        'train_f1_mean': np.mean([m['f1'] for m in all_train_metrics]),
        'train_f1_std': np.std([m['f1'] for m in all_train_metrics]),
        'val_f1_mean': np.mean([m['f1'] for m in all_val_metrics]),
        'val_f1_std': np.std([m['f1'] for m in all_val_metrics]),
        'test_f1_mean': np.mean([m['f1'] for m in all_test_metrics]),
        'test_f1_std': np.std([m['f1'] for m in all_test_metrics]),
        'total_adenocarcinoma_correct': sum([details['adenocarcinoma_correct'] for details in all_fold_details.values()]),
        'total_adenocarcinoma_samples': sum([details['adenocarcinoma_total'] for details in all_fold_details.values()]),
        'total_non_adenocarcinoma_correct': sum([details['non_adenocarcinoma_correct'] for details in all_fold_details.values()]),
        'total_non_adenocarcinoma_samples': sum([details['non_adenocarcinoma_total'] for details in all_fold_details.values()]),
        'total_errors': len(overall_error_summary),
        'overfitting_folds': sum([1 for analysis in overfitting_analysis if analysis['is_overfitting']]),
        'total_folds': len(overfitting_analysis)
    }

    # 保存最终结果
    final_results_df = pd.DataFrame([final_results])
    final_results_df.to_csv(os.path.join(args.save_dir, 'final_enhanced_results_with_overfitting.csv'), index=False)

    # 保存详细的fold信息为JSON（便于查看）
    import json
    with open(os.path.join(args.save_dir, 'detailed_fold_results_with_overfitting.json'), 'w') as f:
        json.dump({'fold_details': all_fold_details, 'overfitting_analysis': overfitting_analysis}, f, indent=2)

    print(f"\n{'='*60}")
    print("🎉 Enhanced Evaluation with Overfitting Detection Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {args.save_dir}")
    print("\n📁 Generated files:")
    print("- enhanced_summary_with_overfitting.csv: 每折的详细统计(包含train/val/test对比)")
    print("- overfitting_analysis.csv: 过拟合检测结果")
    print("- error_analysis.csv: 错误识别的详细信息")
    print("- final_enhanced_results_with_overfitting.csv: 最终汇总结果")
    print("- fold_X_detailed.csv: 每折的详细预测结果")
    print("- detailed_fold_results_with_overfitting.json: 完整的fold详细信息")
    
    print(f"\n📊 Final Results Summary:")
    print(f"🚆 Train: AUC={final_results['train_auc_mean']:.4f}±{final_results['train_auc_std']:.4f}, Acc={final_results['train_acc_mean']:.4f}±{final_results['train_acc_std']:.4f}, F1={final_results['train_f1_mean']:.4f}±{final_results['train_f1_std']:.4f}")
    print(f"📝 Val:   AUC={final_results['val_auc_mean']:.4f}±{final_results['val_auc_std']:.4f}, Acc={final_results['val_acc_mean']:.4f}±{final_results['val_acc_std']:.4f}, F1={final_results['val_f1_mean']:.4f}±{final_results['val_f1_std']:.4f}")
    print(f"🎯 Test:  AUC={final_results['test_auc_mean']:.4f}±{final_results['test_auc_std']:.4f}, Acc={final_results['test_acc_mean']:.4f}±{final_results['test_acc_std']:.4f}, F1={final_results['test_f1_mean']:.4f}±{final_results['test_f1_std']:.4f}")
    print(f"🔍 Overfitting Detection: {final_results['overfitting_folds']}/{final_results['total_folds']} folds show overfitting signs")
    print(f"✅ Total Adenocarcinoma Correct: {final_results['total_adenocarcinoma_correct']}/{final_results['total_adenocarcinoma_samples']}")
    print(f"✅ Total NonAdenocarcinoma Correct: {final_results['total_non_adenocarcinoma_correct']}/{final_results['total_non_adenocarcinoma_samples']}")
    print(f"❌ Total Errors: {final_results['total_errors']}")