# coding=utf-8
"""
ViLa-MIL with BiomedCLIP
使用BiomedCLIP替换原始CLIP的图像和文本编码器
"""

from __future__ import absolute_import, division, print_function
import logging
import warnings
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# BiomedCLIP依赖(不再需要tokenize函数,使用tokenizer对象)
from open_clip import create_model_from_pretrained, get_tokenizer

from .model_utils import MultiheadAttention

logger = logging.getLogger(__name__)


class BiomedCLIPTextEncoder(nn.Module):
    """
    BiomedCLIP文本编码器封装
    使用PubMedBERT作为backbone
    """
    def __init__(self, biomedclip_model):
        super().__init__()
        self.model = biomedclip_model
        
    def forward(self, text_tokens):
        """
        前向传播函数，用于处理输入的文本token并提取文本特征
        参数:
            text_tokens: tokenized text [batch, seq_len]
                - 批量处理文本的token序列
                - batch: 批次大小
                - seq_len: 序列长度
        返回:
            text_features: [batch, 512]
        """
        with torch.no_grad():  # 文本编码器冻结
            text_features = self.model.encode_text(text_tokens)
        return text_features


class BiomedCLIPPromptLearner(nn.Module):
    """
    可学习的提示词模块(适配BiomedCLIP)
    """
    def __init__(self, classnames, biomedclip_model, tokenizer, n_ctx=16):
        super().__init__()
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.tokenizer = tokenizer
        
        # 获取文本嵌入维度(BiomedCLIP使用text.transformer结构)
        # BiomedCLIP的文本模型是CustomTextCLIP,通过text属性访问
        text_model = biomedclip_model.text if hasattr(biomedclip_model, 'text') else biomedclip_model
        
        # 获取embedding层(兼容不同结构)
        if hasattr(text_model, 'transformer'):
            # HuggingFace BERT结构
            token_embedding_layer = text_model.transformer.embeddings.word_embeddings
            ctx_dim = token_embedding_layer.embedding_dim
        elif hasattr(text_model, 'token_embedding'):
            # 原始CLIP结构
            with torch.no_grad():
                dummy_tokens = tokenizer(["test"]).to(next(biomedclip_model.parameters()).device)
                dummy_emb = text_model.token_embedding(dummy_tokens)
                ctx_dim = dummy_emb.shape[-1]
        else:
            # 默认使用512(BiomedCLIP标准维度)
            ctx_dim = 512
        
        # 可学习的上下文向量
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=torch.float32)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
        # 预处理类别名称
        classnames = [name.replace("_", " ") for name in classnames]
        self.classnames = classnames
        
        # Tokenize类别名称(使用tokenizer对象)
        prompts = [f"a histopathology image of {name}" for name in classnames]
        self.tokenized_prompts = tokenizer(prompts)
        
        # 获取token嵌入(使用embedding层)
        with torch.no_grad():
            device = next(biomedclip_model.parameters()).device
            token_ids = self.tokenized_prompts.to(device)
            
            # 根据模型结构选择embedding方法
            if hasattr(text_model, 'transformer'):
                # HuggingFace BERT
                embedding = text_model.transformer.embeddings.word_embeddings(token_ids)
            elif hasattr(text_model, 'token_embedding'):
                # 原始CLIP
                embedding = text_model.token_embedding(token_ids)
            else:
                raise AttributeError("Cannot find token embedding layer in BiomedCLIP model")
        
        self.register_buffer("token_prefix", embedding[:, :1, :])  # [SOS]
        self.register_buffer("token_suffix", embedding[:, 1:, :])  # class name + [EOS]
    
    def forward(self):
        """
        生成可学习的提示词嵌入
        返回: prompts [n_cls, seq_len, dim]
        """
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prefix = self.token_prefix
        suffix = self.token_suffix
        
        # 拼接: [SOS] + ctx + class_name + [EOS]
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        
        return prompts


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """截断正态分布初始化"""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b]", stacklevel=2)
    
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class ViLa_MIL_BiomedCLIP(nn.Module):
    """
    ViLa-MIL模型(BiomedCLIP版本)
    
    主要变化:
    1. 图像特征: 512维 (BiomedCLIP ViT-B/16) vs 1024维 (CLIP RN50)
    2. 文本编码器: PubMedBERT vs OpenAI CLIP Transformer
    3. 医学领域预训练优势
    """
    def __init__(self, config, num_classes=2, 
                 model_path='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'):
        super().__init__()
        self.loss_ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        
        # 特征维度适配(BiomedCLIP输出512维)
        self.L = 512  # BiomedCLIP图像特征维度
        self.D = config.hidden_size  # 隐藏层维度(保持原设计)
        self.K = 1
        
        # 注意力模块
        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)
        
        # 加载BiomedCLIP
        print(f"🔬 Loading BiomedCLIP from: {model_path}")
        biomedclip_model, _ = create_model_from_pretrained(model_path)
        tokenizer = get_tokenizer(model_path)
        
        # 文本编码器(冻结)
        self.text_encoder = BiomedCLIPTextEncoder(biomedclip_model)
        self.tokenizer = tokenizer
        
        # 可学习提示词
        self.prompt_learner = BiomedCLIPPromptLearner(
            config.text_prompt, 
            biomedclip_model,
            tokenizer
        )
        
        # LayerNorm(适配512维)
        self.norm = nn.LayerNorm(self.L)
        
        # 交叉注意力
        self.cross_attention_1 = MultiheadAttention(embed_dim=self.L, num_heads=1)
        self.cross_attention_2 = MultiheadAttention(embed_dim=self.L, num_heads=1)
        
        # 可学习的图像原型
        self.learnable_image_center = nn.Parameter(
            torch.Tensor(config.prototype_number, 1, self.L)
        )
        trunc_normal_(self.learnable_image_center, std=0.02)
        
        # 冻结BiomedCLIP图像编码器(特征已预提取)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    def forward(self, x_s, coord_s, x_l, coords_l, label):
        """
        前向传播
        
        参数:
            x_s: 低分辨率patch特征 [1, N_s, 512]
            coord_s: 低分辨率坐标
            x_l: 高分辨率patch特征 [1, N_l, 512]
            coords_l: 高分辨率坐标
            label: 标签 [1]
        
        返回:
            Y_prob: 预测概率 [1, num_classes]
            Y_hat: 预测类别 [1, 1]
            loss: 交叉熵损失
        """
        # 生成文本特征
        prompts = self.prompt_learner()
        # 注意: BiomedCLIP需要tokenized text而非embedding
        # 这里我们直接使用预先tokenized的prompts
        tokenized_prompts = self.prompt_learner.tokenized_prompts.to(x_s.device)
        text_features = self.text_encoder(tokenized_prompts)  # [2*num_classes, 512]
        
        # ========== 低分辨率分支 ==========
        M = x_s.float()
        compents, _ = self.cross_attention_1(self.learnable_image_center, M, M)
        compents = self.norm(compents + self.learnable_image_center)
        
        H = compents.squeeze().float()
        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        image_features_low = torch.mm(A, H)
        
        # ========== 高分辨率分支 ==========
        M_high = x_l.float()
        compents_high, _ = self.cross_attention_1(self.learnable_image_center, M_high, M_high)
        compents_high = self.norm(compents_high + self.learnable_image_center)
        
        H_high = compents_high.squeeze().float()
        A_V_high = self.attention_V(H_high)
        A_U_high = self.attention_U(H_high)
        A_high = self.attention_weights(A_V_high * A_U_high)
        A_high = torch.transpose(A_high, 1, 0)
        A_high = F.softmax(A_high, dim=1)
        image_features_high = torch.mm(A_high, H_high)
        
        # ========== 文本-图像对齐 ==========
        text_features_low = text_features[:self.num_classes]
        image_context = torch.cat((compents.squeeze(), M.squeeze(0)), dim=0)
        text_context_features, _ = self.cross_attention_2(
            text_features_low.unsqueeze(1), image_context, image_context
        )
        text_features_low = text_context_features.squeeze() + text_features_low
        
        text_features_high = text_features[self.num_classes:]
        image_context_high = torch.cat((compents_high.squeeze(), M_high.squeeze(0)), dim=0)
        text_context_features_high, _ = self.cross_attention_2(
            text_features_high.unsqueeze(1), image_context_high, image_context_high
        )
        text_features_high = text_context_features_high.squeeze() + text_features_high
        
        # ========== 分类 ==========
        logits_low = image_features_low @ text_features_low.T
        logits_high = image_features_high @ text_features_high.T
        logits = logits_low + logits_high
        
        loss = self.loss_ce(logits, label)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]
        
        return Y_prob, Y_hat, loss
    
    def forward_with_attention(self, x_s, coord_s, x_l, coords_l, label):
        """前向传播并返回注意力权重(用于热力图)"""
        tokenized_prompts = self.prompt_learner.tokenized_prompts.to(x_s.device)
        text_features = self.text_encoder(tokenized_prompts)
        
        M = x_s.float()
        compents, cross_attn_weights_s = self.cross_attention_1(
            self.learnable_image_center, M, M
        )
        compents = self.norm(compents + self.learnable_image_center)
        
        M_high = x_l.float()
        compents_high, cross_attn_weights_l = self.cross_attention_1(
            self.learnable_image_center, M_high, M_high
        )
        compents_high = self.norm(compents_high + self.learnable_image_center)
        
        H = compents.squeeze().float()
        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        image_features_low = torch.mm(A, H)
        
        H_high = compents_high.squeeze().float()
        A_V_high = self.attention_V(H_high)
        A_U_high = self.attention_U(H_high)
        A_high = self.attention_weights(A_V_high * A_U_high)
        A_high = torch.transpose(A_high, 1, 0)
        A_high = F.softmax(A_high, dim=1)
        image_features_high = torch.mm(A_high, H_high)
        
        text_features_low = text_features[:self.num_classes]
        image_context = torch.cat((compents.squeeze(), M.squeeze(0)), dim=0)
        text_context_features, _ = self.cross_attention_2(
            text_features_low.unsqueeze(1), image_context, image_context
        )
        text_features_low = text_context_features.squeeze() + text_features_low
        
        text_features_high = text_features[self.num_classes:]
        image_context_high = torch.cat((compents_high.squeeze(), M_high.squeeze(0)), dim=0)
        text_context_features_high, _ = self.cross_attention_2(
            text_features_high.unsqueeze(1), image_context_high, image_context_high
        )
        text_features_high = text_context_features_high.squeeze() + text_features_high
        
        logits_low = image_features_low @ text_features_low.T
        logits_high = image_features_high @ text_features_high.T
        logits = logits_low + logits_high
        
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]
        
        # 注意力权重
        patch_attention_s = cross_attn_weights_s.mean(dim=1).squeeze(0).mean(dim=0)
        patch_attention_l = cross_attn_weights_l.mean(dim=1).squeeze(0).mean(dim=0)
        
        return logits, Y_prob, Y_hat, patch_attention_s, patch_attention_l
