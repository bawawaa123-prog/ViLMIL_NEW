import os
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import h5py
import numpy as np
import clip
from sklearn.metrics.pairwise import cosine_similarity

# 导入项目模块
from utils.eval_utils import initiate_model
from utils.utils import *

def load_h5_features(path):
    """从h5文件中加载特征"""
    with h5py.File(path, 'r') as f:
        for key in ['features', 'feats', 'feature', 'patch_features']:
            if key in f:
                return np.array(f[key])
        # 如果找不到常见key，使用第一个数据集
        keys = list(f.keys())
        if keys:
            return np.array(f[keys[0]])
    raise RuntimeError(f"无法在 {path} 中找到特征数据")

class ModifiedTextFeatureExtractor:
    def __init__(self, model, clip_model, device):
        self.model = model
        self.clip_model = clip_model
        self.device = device
        
        # 构建医学词汇表用于解码
        self.build_medical_vocabulary()
    
    def build_medical_vocabulary(self):
        """构建医学相关的词汇表，用于文本特征解码"""
        self.medical_vocab = [
            # 基础组织学术语
            "normal tissue", "abnormal tissue", "healthy cells", "cancer cells",
            "tumor", "malignant", "benign", "neoplasm", "carcinoma", "adenocarcinoma",
            
            # 细胞形态
            "large cells", "small cells", "round cells", "elongated cells",
            "dense nuclei", "sparse nuclei", "irregular nuclei", "regular nuclei",
            "high cellularity", "low cellularity", "cellular atypia", "pleomorphism",
            
            # 组织结构
            "glandular pattern", "solid pattern", "infiltrative pattern",
            "well differentiated", "poorly differentiated", "moderately differentiated",
            "stromal invasion", "basement membrane", "epithelial cells", "stromal cells",
            
            # 病理特征
            "inflammation", "necrosis", "fibrosis", "hemorrhage", "calcification",
            "mitotic activity", "apoptosis", "angiogenesis", "lymphocytic infiltration",
            
            # 诊断相关
            "adenocarcinoma features", "non-adenocarcinoma features",
            "primary tumor", "metastatic lesion", "pre-cancerous changes",
            "dysplasia", "hyperplasia", "atrophy", "hypertrophy",
            
            # 分级和分期
            "grade 1", "grade 2", "grade 3", "well-formed glands", "poorly formed glands",
            "surface involvement", "deep invasion", "vascular invasion", "neural invasion",
            
            # 免疫和分子特征
            "immune infiltration", "T cell infiltration", "B cell infiltration",
            "macrophage infiltration", "neutrophil infiltration", "eosinophil infiltration",
            
            # 组织学亚型
            "acinar adenocarcinoma", "papillary adenocarcinoma", "solid adenocarcinoma",
            "mucinous adenocarcinoma", "signet ring cell carcinoma", "clear cell carcinoma"
        ]
        
        # 编码医学词汇
        vocab_tokens = clip.tokenize(self.medical_vocab).to(self.device)
        with torch.no_grad():
            self.vocab_features = self.clip_model.encode_text(vocab_tokens)
            self.vocab_features = F.normalize(self.vocab_features, dim=-1)
    
    def extract_modified_text_features(self, slide_features_5x, slide_features_20x):
        """
        提取被图像特征修正后的文本特征向量
        """
        h5x = torch.tensor(slide_features_5x, dtype=torch.float32, device=self.device)
        h20x = torch.tensor(slide_features_20x, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            # 获取模型的中间输出
            if hasattr(self.model, 'forward_with_intermediate'):
                # 如果模型有专门的中间输出方法
                outputs = self.model.forward_with_intermediate(h5x, h20x)
            else:
                # 修改模型前向传播以获取中间特征
                outputs = self.forward_with_intermediate(h5x, h20x)
            
            return outputs
    
    def forward_with_intermediate(self, h5x, h20x):
        """
        修改后的前向传播，返回中间的文本特征
        """
        # 获取原始文本提示特征
        prompts = self.model.prompt_learner()  # [n_classes, n_tokens, dim]
        original_text_features = self.model.text_encoder(prompts)
        original_text_features = F.normalize(original_text_features, dim=-1)
        
        # 处理图像特征 - 5x (全局上下文)
        A_5x, h_5x = self.model.attention_net(h5x)
        A_5x = torch.transpose(A_5x, 1, 0)
        A_5x = F.softmax(A_5x, dim=1)
        M_5x = torch.mm(A_5x, h_5x)  # [n_classes, dim]
        
        # 处理图像特征 - 20x (局部细节)
        A_20x, h_20x = self.model.attention_net(h20x)
        A_20x = torch.transpose(A_20x, 1, 0)
        A_20x = F.softmax(A_20x, dim=1)
        M_20x = torch.mm(A_20x, h_20x)  # [n_classes, dim]
        
        # 多尺度视觉特征融合
        M_fused = 0.6 * M_5x + 0.4 * M_20x  # 可调权重
        
        # 将视觉特征投影到文本空间
        visual_projected = self.model.visual_projection(M_fused)
        visual_projected = F.normalize(visual_projected, dim=-1)
        
        # 视觉-文本特征交互（这里是关键！）
        # 使用注意力机制让视觉特征"修正"文本特征
        interaction_weights = torch.softmax(
            torch.matmul(visual_projected, original_text_features.T), dim=-1
        )  # [n_classes, n_classes]
        
        # 修正后的文本特征
        modified_text_features = torch.matmul(interaction_weights, original_text_features)
        modified_text_features = F.normalize(modified_text_features, dim=-1)
        
        # 最终的分类logits
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * torch.matmul(visual_projected, modified_text_features.T)
        
        return {
            'original_text_features': original_text_features,
            'modified_text_features': modified_text_features,
            'visual_features_5x': M_5x,
            'visual_features_20x': M_20x,
            'visual_projected': visual_projected,
            'attention_5x': A_5x,
            'attention_20x': A_20x,
            'interaction_weights': interaction_weights,
            'logits': logits
        }
    
    def decode_text_features(self, text_features, top_k=10):
        """
        将文本特征向量解码为最相近的文字描述
        """
        results = []
        
        for i, feat in enumerate(text_features):
            # 计算与词汇表的相似度
            feat_norm = F.normalize(feat.unsqueeze(0), dim=-1)
            similarities = torch.matmul(feat_norm, self.vocab_features.T).squeeze(0)
            
            # 获取top-k最相似的词汇
            top_k_indices = torch.topk(similarities, k=min(top_k, len(self.medical_vocab)))[1]
            top_k_scores = similarities[top_k_indices]
            
            # 构建文本描述
            descriptions = []
            for idx, score in zip(top_k_indices, top_k_scores):
                descriptions.append({
                    'text': self.medical_vocab[idx.item()],
                    'similarity': score.item()
                })
            
            results.append({
                'class_id': i,
                'top_descriptions': descriptions
            })
        
        return results
    
    def generate_semantic_summary(self, original_desc, modified_desc):
        """
        生成语义变化的总结
        """
        summary = []
        
        for i, (orig, mod) in enumerate(zip(original_desc, modified_desc)):
            class_name = "Adenocarcinoma" if i == 0 else "NonAdenocarcinoma"
            
            # 获取变化最大的描述
            orig_texts = set([d['text'] for d in orig['top_descriptions'][:5]])
            mod_texts = set([d['text'] for d in mod['top_descriptions'][:5]])
            
            new_concepts = mod_texts - orig_texts
            lost_concepts = orig_texts - mod_texts
            
            summary.append({
                'class': class_name,
                'original_top3': [d['text'] for d in orig['top_descriptions'][:3]],
                'modified_top3': [d['text'] for d in mod['top_descriptions'][:3]],
                'new_concepts': list(new_concepts),
                'lost_concepts': list(lost_concepts),
                'semantic_shift_score': len(new_concepts) / max(len(orig_texts), 1)
            })
        
        return summary

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 处理文本提示参数 - 修复CSV解析问题
    if args.text_prompt is None:
        args.text_prompt = 'text_prompt/adenocarcinoma_dual_scale_prompt.csv'
    
    print(f"加载文本提示: {args.text_prompt}")
    try:
        if os.path.exists(args.text_prompt):
            text_prompt_df = pd.read_csv(args.text_prompt)
            # 正确解析双尺度文本提示CSV格式
            if 'class_name' in text_prompt_df.columns:
                # 这是标准的双尺度提示格式
                adenocarcinoma_row = text_prompt_df[text_prompt_df['class_name'] == 'Adenocarcinoma']
                non_adenocarcinoma_row = text_prompt_df[text_prompt_df['class_name'] == 'NonAdenocarcinoma']
                
                if not adenocarcinoma_row.empty and not non_adenocarcinoma_row.empty:
                    # 使用低分辨率描述作为主要文本提示
                    args.text_prompt = [
                        adenocarcinoma_row['low_resolution_description'].iloc[0],
                        non_adenocarcinoma_row['low_resolution_description'].iloc[0]
                    ]
                else:
                    raise ValueError("CSV格式错误：缺少类别行")
            else:
                # 简单格式：一行一个提示
                args.text_prompt = text_prompt_df.iloc[:, 0].tolist()
        else:
            print(f"文本提示文件不存在，使用默认提示")
            args.text_prompt = [
                "a histopathology image of adenocarcinoma",
                "a histopathology image of non-adenocarcinoma"
            ]
    except Exception as e:
        print(f"加载文本提示失败: {e}，使用默认提示")
        args.text_prompt = [
            "a histopathology image of adenocarcinoma", 
            "a histopathology image of non-adenocarcinoma"
        ]
    
    print(f"文本提示内容: {args.text_prompt}")
    
    # 加载模型
    print("加载ViLa-MIL模型...")
    model = initiate_model(args, args.ckpt_path)
    model.to(device).eval()
    
    # 加载CLIP模型用于文本解码
    print("加载CLIP模型...")
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()
    
    # 初始化特征提取器
    extractor = ModifiedTextFeatureExtractor(model, clip_model, device)
    
    # 读取数据列表
    df = pd.read_csv(args.dataset_csv)
    
    all_results = []
    
    print(f"开始处理 {len(df)} 个样本...")
    
    for idx, row in df.iterrows():
        slide_id = row['slide_id']
        print(f"处理样本 {idx+1}/{len(df)}: {slide_id}")
        
        try:
            # 加载多尺度特征
            features_5x_path = os.path.join(args.features_5x_dir, f"{slide_id}.h5")
            features_20x_path = os.path.join(args.features_20x_dir, f"{slide_id}.h5")
            
            if not os.path.exists(features_5x_path) or not os.path.exists(features_20x_path):
                print(f"跳过 {slide_id}: 特征文件不存在")
                continue
            
            features_5x = load_h5_features(features_5x_path)
            features_20x = load_h5_features(features_20x_path)
            
            # 提取修正后的文本特征
            outputs = extractor.extract_modified_text_features(features_5x, features_20x)
            
            # 解码原始和修正后的文本特征
            original_decoded = extractor.decode_text_features(
                outputs['original_text_features'], top_k=15
            )
            modified_decoded = extractor.decode_text_features(
                outputs['modified_text_features'], top_k=15
            )
            
            # 生成语义变化总结
            semantic_summary = extractor.generate_semantic_summary(
                original_decoded, modified_decoded
            )
            
            # 保存结果
            result = {
                'slide_id': slide_id,
                'label': row['label'],
                'original_text_features': original_decoded,
                'modified_text_features': modified_decoded,
                'semantic_summary': semantic_summary,
                'interaction_weights': outputs['interaction_weights'].cpu().numpy().tolist()
            }
            
            all_results.append(result)
            
            # 打印部分结果
            print(f"  {slide_id} - 修正后的主要描述:")
            for j, summary in enumerate(semantic_summary):
                print(f"    {summary['class']}: {', '.join(summary['modified_top3'])}")
            
        except Exception as e:
            print(f"处理 {slide_id} 时出错: {e}")
            continue
    
    # 保存所有结果
    import json
    output_json = os.path.join(args.output_dir, 'modified_text_features.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # 生成CSV摘要
    csv_results = []
    for result in all_results:
        for summary in result['semantic_summary']:
            csv_results.append({
                'slide_id': result['slide_id'],
                'label': result['label'],
                'class': summary['class'],
                'modified_top1': summary['modified_top3'][0] if summary['modified_top3'] else '',
                'modified_top2': summary['modified_top3'][1] if len(summary['modified_top3']) > 1 else '',
                'modified_top3': summary['modified_top3'][2] if len(summary['modified_top3']) > 2 else '',
                'semantic_shift_score': summary['semantic_shift_score'],
                'new_concepts_count': len(summary['new_concepts'])
            })
    
    csv_df = pd.DataFrame(csv_results)
    csv_output = os.path.join(args.output_dir, 'modified_text_features_summary.csv')
    csv_df.to_csv(csv_output, index=False)
    
    print(f"\n✅ 处理完成!")
    print(f"📄 详细结果保存至: {output_json}")
    print(f"📊 CSV摘要保存至: {csv_output}")
    
    # 生成统计报告
    print(f"\n📈 统计报告:")
    print(f"总处理样本数: {len(all_results)}")
    if all_results:
        avg_shift = np.mean([s['semantic_shift_score'] for r in all_results for s in r['semantic_summary']])
        print(f"平均语义偏移分数: {avg_shift:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', required=True, help='模型检查点路径')
    parser.add_argument('--dataset_csv', required=True, help='数据集CSV文件')
    parser.add_argument('--features_5x_dir', required=True, help='5x特征目录')
    parser.add_argument('--features_20x_dir', required=True, help='20x特征目录')
    parser.add_argument('--output_dir', default='./text_feature_analysis', help='输出目录')
    
    # 模型相关参数（完整版本，包含所有可能需要的参数）
    parser.add_argument('--task', default='task_adenocarcinoma', help='任务类型')
    parser.add_argument('--model_type', default='ViLa_MIL', help='模型类型')
    parser.add_argument('--drop_out', type=float, default=0.25, help='dropout rate')
    parser.add_argument('--n_classes', type=int, default=2, help='类别数')
    parser.add_argument('--model_size', default=None, help='模型规模参数')
    parser.add_argument('--text_prompt', default=None, help='文本提示CSV文件路径')
    parser.add_argument('--prototype_number', type=int, default=10, help='原型数量')  # 新增
    parser.add_argument('--k_sample', type=int, default=8, help='k_sample参数')
    parser.add_argument('--instance_loss', default='svm', help='实例损失类型')
    parser.add_argument('--bag_loss', default='ce', help='包损失类型')
    parser.add_argument('--B', type=int, default=8, help='B参数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--reg', type=float, default=10e-5, help='正则化参数')
    parser.add_argument('--seed', type=int, default=1, help='随机种子')
    parser.add_argument('--log_data', action='store_true', help='是否记录数据')
    parser.add_argument('--weighted_sample', action='store_true', help='是否使用加权采样')
    parser.add_argument('--opt', default='adam', help='优化器类型')
    parser.add_argument('--bag_weight', type=float, default=0.7, help='包权重')
    parser.add_argument('--inst_rate', type=float, default=0.3, help='实例比率')
    parser.add_argument('--no_inst_cluster', action='store_true', help='不使用实例聚类')
    parser.add_argument('--testing', action='store_true', help='测试模式')
    parser.add_argument('--early_stopping', action='store_true', help='早停')
    parser.add_argument('--max_epochs', type=int, default=200, help='最大轮数')
    parser.add_argument('--results_dir', default='results', help='结果目录')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    main(args)