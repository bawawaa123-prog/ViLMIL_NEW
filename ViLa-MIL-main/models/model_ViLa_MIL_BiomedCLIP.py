# coding=utf-8
"""
ViLa-MIL with BiomedCLIP
使用BiomedCLIP替换原始CLIP的图像和文本编码器
"""

from __future__ import absolute_import, division, print_function
import logging
import warnings
import math
import inspect
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F

DEFAULT_BIOMEDCLIP_REPO = 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
DEFAULT_BIOMEDCLIP_MODEL = f'hf-hub:{DEFAULT_BIOMEDCLIP_REPO}'
DEFAULT_TEXT_REPO = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'


def _candidate_cache_dirs(explicit_cache_dir=None):
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        explicit_cache_dir,
        os.environ.get('HUGGINGFACE_HUB_CACHE'),
        repo_root / 'hf_cache',
        repo_root.parent / 'hf_cache',
        repo_root / 'model_cache',
    ]

    seen = set()
    resolved = []
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate).expanduser()
        candidate_str = str(candidate_path)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        resolved.append(candidate_path)
    return resolved


def _resolve_snapshot_dir(cache_dir, repo_id):
    snapshots_dir = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}" / 'snapshots'
    if not snapshots_dir.is_dir():
        return None
    snapshot_dirs = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshot_dirs:
        return None
    return snapshot_dirs[-1]


def _bootstrap_hf_environment():
    for candidate_cache_dir in _candidate_cache_dirs():
        if not candidate_cache_dir.exists():
            continue

        clip_snapshot = _resolve_snapshot_dir(candidate_cache_dir, DEFAULT_BIOMEDCLIP_REPO)
        text_snapshot = _resolve_snapshot_dir(candidate_cache_dir, DEFAULT_TEXT_REPO)
        if not clip_snapshot:
            continue

        os.environ.setdefault('HF_HOME', str(candidate_cache_dir))
        os.environ.setdefault('HUGGINGFACE_HUB_CACHE', str(candidate_cache_dir))
        if text_snapshot:
            os.environ.setdefault('HF_HUB_OFFLINE', '1')
            os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        return str(candidate_cache_dir)

    return None


def _prepare_biomedclip_loading(model_path, cache_dir=None):
    for candidate_cache_dir in _candidate_cache_dirs(cache_dir):
        if not candidate_cache_dir.exists():
            continue

        clip_snapshot = _resolve_snapshot_dir(candidate_cache_dir, DEFAULT_BIOMEDCLIP_REPO)
        text_snapshot = _resolve_snapshot_dir(candidate_cache_dir, DEFAULT_TEXT_REPO)
        if not clip_snapshot:
            continue

        os.environ.setdefault('HF_HOME', str(candidate_cache_dir))
        os.environ.setdefault('HUGGINGFACE_HUB_CACHE', str(candidate_cache_dir))

        offline_enabled = text_snapshot is not None
        if offline_enabled:
            os.environ.setdefault('HF_HUB_OFFLINE', '1')
            os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

        resolved_model_path = model_path
        if model_path == DEFAULT_BIOMEDCLIP_MODEL:
            resolved_model_path = f'local-dir:{clip_snapshot}'

        return resolved_model_path, str(candidate_cache_dir), offline_enabled

    return model_path, cache_dir, False


_BOOTSTRAP_CACHE_DIR = _bootstrap_hf_environment()


# BiomedCLIP依赖(不再需要tokenize函数,使用tokenizer对象)
from open_clip import create_model_from_pretrained, get_tokenizer

from .model_utils import MultiheadAttention
from .cross_scale_modules import HighRouter, LowParentContext, MaskedGatedAttentionPool

logger = logging.getLogger(__name__)


class BiomedCLIPTextEncoder(nn.Module):
    """
    BiomedCLIP文本编码器封装
    使用PubMedBERT作为backbone
    """
    def __init__(self, biomedclip_model, *, n_ctx: int = 16, finetune: bool = False):
        super().__init__()
        self.model = biomedclip_model
        self.n_ctx = int(n_ctx)
        self.finetune = bool(finetune)
        self._warned_embed_fallback = False
        # Debug/observability: track whether we ever had to fall back.
        self.fallback_count = 0
        self.fallback_last_error = None
        
    def forward(self, text_tokens, prompt_embeddings: torch.Tensor | None = None, eos_indices: torch.Tensor | None = None):
        """
        前向传播函数，用于处理输入的文本token并提取文本特征
        参数:
            text_tokens: tokenized text [batch, seq_len]
                - 批量处理文本的token序列
                - batch: 批次大小
                - seq_len: 序列长度
            prompt_embeddings: learned prompt embeddings [batch, seq_len, dim] (optional)
            eos_indices: indices of end token in the *prompt_embeddings* sequence [batch] (optional)
        返回:
            text_features: [batch, 512]
        """
        # Prefer prompt-tuning path (uses learnable ctx) when provided.
        if prompt_embeddings is not None:
            try:
                text_model = self.model.text if hasattr(self.model, 'text') else self.model

                # Case A: CLIP-style text tower (token_embedding/positional_embedding/transformer/ln_final/text_projection)
                if all(hasattr(text_model, a) for a in ['positional_embedding', 'transformer', 'ln_final', 'text_projection']):
                    x = prompt_embeddings
                    x = x + text_model.positional_embedding.to(x.dtype)
                    x = x.permute(1, 0, 2)  # (seq, batch, dim)
                    x = text_model.transformer(x)
                    x = x.permute(1, 0, 2)  # (batch, seq, dim)
                    x = text_model.ln_final(x)

                    if eos_indices is None:
                        # Fallback: use last non-pad token position (token!=0) then shift by n_ctx.
                        attn_mask = (text_tokens != 0)
                        eos_indices = attn_mask.long().sum(dim=1) - 1
                        eos_indices = torch.clamp(eos_indices + self.n_ctx, min=0, max=x.shape[1] - 1)

                    x = x[torch.arange(x.shape[0], device=x.device), eos_indices]
                    proj = text_model.text_projection
                    if isinstance(proj, (torch.Tensor, nn.Parameter)):
                        text_features = x @ proj
                    elif isinstance(proj, nn.Module):
                        text_features = proj(x)
                    else:
                        raise TypeError(f'Unsupported text_projection type: {type(proj)}')
                    return text_features

                # Case B: HF-style transformer (supports inputs_embeds)
                transformer = getattr(text_model, 'transformer', None)
                if transformer is not None and hasattr(transformer, 'forward'):
                    sig = None
                    try:
                        sig = inspect.signature(transformer.forward)
                    except Exception:
                        sig = None

                    if sig is not None and 'inputs_embeds' in sig.parameters:
                        token_mask = (text_tokens != 0)
                        # Build a new attention_mask aligned to prompt_embeddings
                        # prompt = [prefix] + [ctx] + [suffix_trunc]
                        bsz, seq_len = text_tokens.shape
                        if prompt_embeddings.shape[1] != seq_len:
                            # Best-effort: align to prompt_embeddings length
                            seq_len = prompt_embeddings.shape[1]
                        prefix_mask = token_mask[:, :1]
                        ctx_mask = torch.ones((bsz, self.n_ctx), device=text_tokens.device, dtype=token_mask.dtype)
                        suffix_keep = max(int(text_tokens.shape[1]) - 1 - self.n_ctx, 0)
                        suffix_mask = token_mask[:, 1:1 + suffix_keep]
                        attn_mask = torch.cat([prefix_mask, ctx_mask, suffix_mask], dim=1)
                        attn_mask = attn_mask[:, :prompt_embeddings.shape[1]]

                        out = transformer(inputs_embeds=prompt_embeddings, attention_mask=attn_mask, return_dict=True)
                        hidden = getattr(out, 'last_hidden_state', None)
                        if hidden is None and isinstance(out, (tuple, list)) and len(out) > 0:
                            hidden = out[0]
                        if hidden is None:
                            raise RuntimeError('HF transformer output missing last_hidden_state')

                        # Prefer CLS token representation for HF models.
                        pooled = hidden[:, 0]
                        if hasattr(text_model, 'ln_final'):
                            pooled = text_model.ln_final(pooled)
                        proj = None
                        for attr in ['proj', 'text_projection']:
                            if hasattr(text_model, attr):
                                proj = getattr(text_model, attr)
                                break
                        if proj is None and hasattr(self.model, 'text_projection'):
                            proj = getattr(self.model, 'text_projection')
                        if proj is not None:
                            if isinstance(proj, (torch.Tensor, nn.Parameter)):
                                pooled = pooled @ proj
                            elif isinstance(proj, nn.Module):
                                pooled = proj(pooled)
                            else:
                                raise TypeError(f'Unsupported projection type: {type(proj)}')
                        return pooled

                # If we get here, we can't run embeddings through text tower; fall back.
                raise RuntimeError('Unsupported BiomedCLIP text tower for prompt embeddings')

            except Exception as e:
                self.fallback_count += 1
                self.fallback_last_error = str(e)
                if not self._warned_embed_fallback:
                    msg = (
                        "[BiomedCLIPTextEncoder] prompt-embedding path failed, "
                        "falling back to encode_text(tokens). "
                        f"Error: {e}"
                    )
                    # Ensure it's visible in training logs even if logging isn't configured.
                    print(msg)
                    logger.warning(msg)
                    self._warned_embed_fallback = True

        # Fallback: token-id path (no prompt-learning). If finetune=True, allow grads.
        if self.finetune:
            return self.model.encode_text(text_tokens)
        with torch.no_grad():
            return self.model.encode_text(text_tokens)


class BiomedCLIPPromptLearner(nn.Module):
    """
    可学习的提示词模块(适配BiomedCLIP)
    """
    def __init__(self, classnames, biomedclip_model, tokenizer, n_ctx=16):
        super().__init__()
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.tokenizer = tokenizer
        self._pad_id = 0
        
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

        # NOTE: Keep prompt length constant. We insert n_ctx tokens after the first token,
        # so we must drop n_ctx tokens from the tail (usually padding) to preserve seq_len.
        seq_len = embedding.shape[1]
        suffix_keep = max(seq_len - 1 - n_ctx, 0)
        
        self.register_buffer("token_prefix", embedding[:, :1, :])  # [SOS]
        self.register_buffer("token_suffix", embedding[:, 1:1 + suffix_keep, :])  # class name + [EOS] (+ truncated padding)

        # Track end-token indices in the *prompt_embeddings* sequence for CLIP-style pooling.
        # Use last non-pad token position in original tokens, then shift by n_ctx.
        with torch.no_grad():
            pad_id = self._pad_id
            non_pad = (token_ids != pad_id)
            last_idx = non_pad.long().sum(dim=1) - 1
            eos_idx = torch.clamp(last_idx + n_ctx, min=0, max=seq_len - 1)
        self.register_buffer("eos_indices", eos_idx)
    
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
        self.scale_mode = str(getattr(config, 'scale_mode', 'dual'))
        if self.scale_mode not in {'dual', 'low', 'high'}:
            raise ValueError(f"Unsupported scale_mode={self.scale_mode!r}")

        # Optional offline/local override for environments without HF access.
        # Example:
        #   export BIOMEDCLIP_MODEL_PATH=/path/to/local/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
        #   # or a mirror/hf-hub id if needed
        env_model_path = os.environ.get('BIOMEDCLIP_MODEL_PATH', '').strip()
        if env_model_path:
            model_path = env_model_path
        
        # 特征维度适配(BiomedCLIP输出512维)
        self.L = 512  # BiomedCLIP图像特征维度
        self.D = config.hidden_size  # 隐藏层维度(保持原设计)
        self.K = 1
        
        # 注意力模块
        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)
        # Parameter-free wrapper: legacy attention_* modules remain root-owned.
        self.gated_attention_pool = MaskedGatedAttentionPool()
        self.low_parent_context = LowParentContext()
        self.use_low_context_routing = bool(getattr(config, 'use_low_context_routing', False))
        self.routing_scale = float(getattr(config, 'routing_scale', 1.0))
        if self.routing_scale < 0:
            raise ValueError('routing_scale must be non-negative')
        if self.use_low_context_routing:
            # Do not let newly introduced router initialization perturb the
            # shared E0/E1 initialization stream under the same seed.
            rng_state = torch.random.get_rng_state()
            self.use_routing_stabilization = bool(getattr(config, 'use_routing_stabilization', False))
            self.routing_residual_ratio = float(getattr(config, 'routing_residual_ratio', 0.10))
            self.high_router = HighRouter(
                feature_dim=self.L,
                stabilize=self.use_routing_stabilization,
                residual_ratio=self.routing_residual_ratio,
            )
            torch.random.set_rng_state(rng_state)
            self.last_routing_diagnostics = None
            # Set only by the diagnostics runner; normal training/inference
            # leaves this as ``normal`` and follows Stage 3.3.4 exactly.
            self.routing_diagnostic_mode = str(getattr(config, 'routing_diagnostic_mode', 'normal'))
            self.routing_diagnostic_seed = int(getattr(config, 'routing_diagnostic_seed', 0))
        self._last_mapping_context = None
        
        # 加载BiomedCLIP
        resolved_model_path, resolved_cache_dir, offline_enabled = _prepare_biomedclip_loading(model_path)
        print(f"🔬 Loading BiomedCLIP from: {resolved_model_path}")
        if resolved_cache_dir:
            print(f"📦 Using HuggingFace cache: {resolved_cache_dir}")
        if offline_enabled:
            print('📴 Offline cache mode enabled')
        try:
            biomedclip_model, _ = create_model_from_pretrained(
                resolved_model_path,
                cache_dir=resolved_cache_dir,
            )
            tokenizer = get_tokenizer(resolved_model_path)
        except Exception as e:
            offline = os.environ.get('HF_HUB_OFFLINE', '0') == '1'
            msg = (
                "[Error] Failed to load BiomedCLIP from HuggingFace Hub. "
                "This is usually caused by transient network/proxy/SSL issues or missing local cache.\n"
                f"- model_path: {resolved_model_path}\n"
                f"- cache_dir: {resolved_cache_dir}\n"
                f"- HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', '0')} (offline={offline})\n"
                "Fix options:\n"
                "1) Ensure the model is fully downloaded into cache (run a one-time warmup download).\n"
                "2) If you already downloaded it, re-run with offline cache only: export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n"
                "3) If you use a proxy, make sure HTTPS proxy is stable and supports TLS properly.\n"
                f"Original error: {e}"
            )
            print(msg)
            raise
        
        # 文本编码器：可选微调
        finetune_text = bool(getattr(config, 'finetune_text_encoder', False))
        self.text_encoder = BiomedCLIPTextEncoder(biomedclip_model, n_ctx=16, finetune=finetune_text)
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
        
        # 文本编码器微调开关：如果关闭则冻结，否则保持可训练
        # Notes:
        # - We NEVER need to finetune BiomedCLIP visual tower here because we consume pre-extracted image features.
        # - For stability, default finetune scope is projection-only (unless user explicitly requests more).
        self._configure_biomedclip_finetune(config)

    def _configure_biomedclip_finetune(self, config):
        finetune_text = bool(getattr(config, 'finetune_text_encoder', False))
        mode = str(getattr(config, 'text_finetune_mode', 'proj'))
        last_n = int(getattr(config, 'text_unfreeze_last_n', 2))

        # Freeze everything in the BiomedCLIP wrapper by default.
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        # Always freeze visual tower (not used in forward).
        text_clip = self.text_encoder.model
        if hasattr(text_clip, 'visual'):
            for p in text_clip.visual.parameters():
                p.requires_grad = False

        if not finetune_text:
            return

        # Unfreeze text tower according to mode.
        text_model = text_clip.text if hasattr(text_clip, 'text') else text_clip

        def _unfreeze_module(m: nn.Module | None):
            if m is None:
                return
            for p in m.parameters():
                p.requires_grad = True

        def _unfreeze_obj(obj):
            if obj is None:
                return
            if isinstance(obj, nn.Module):
                _unfreeze_module(obj)
            elif isinstance(obj, (torch.Tensor, nn.Parameter)):
                obj.requires_grad = True

        # Always prefer enabling grads in text_encoder token-id fallback path.
        self.text_encoder.finetune = True

        # 1) Projection-only: proj/text_projection/ln_final where available.
        if mode == 'proj':
            for attr in ['proj', 'text_projection', 'ln_final']:
                if hasattr(text_model, attr):
                    _unfreeze_obj(getattr(text_model, attr))
            # Some open_clip wrappers keep projection on the parent model.
            for attr in ['text_projection']:
                if hasattr(text_clip, attr):
                    _unfreeze_obj(getattr(text_clip, attr))
            return

        # 2) Last-N layers: unfreeze proj + last N encoder layers.
        if mode == 'last':
            # Unfreeze proj first
            for attr in ['proj', 'text_projection', 'ln_final']:
                if hasattr(text_model, attr):
                    _unfreeze_obj(getattr(text_model, attr))

            transformer = getattr(text_model, 'transformer', None)
            # HF: BertModel has encoder.layer
            encoder = getattr(transformer, 'encoder', None) if transformer is not None else None
            layers = getattr(encoder, 'layer', None) if encoder is not None else None
            if layers is not None and hasattr(layers, '__len__'):
                n_layers = len(layers)
                n = max(0, min(int(last_n), n_layers))
                for i in range(n_layers - n, n_layers):
                    _unfreeze_module(layers[i])
            else:
                # Best-effort fallback: if we can't locate encoder layers, fall back to full.
                mode = 'full'

        # 3) Full: unfreeze all text-model parameters (still keep visual frozen).
        if mode == 'full':
            _unfreeze_module(text_model)
            # Also unfreeze projections on parent if present
            for attr in ['text_projection']:
                if hasattr(text_clip, attr):
                    _unfreeze_obj(getattr(text_clip, attr))
    
    def build_mapping_context(self, x_s, x_l, mapping):
        """Build per-high low-parent context without changing prediction."""
        if mapping is None:
            self._last_mapping_context = None
            return None
        low_features = x_s.squeeze(0) if x_s.ndim == 3 and x_s.shape[0] == 1 else x_s
        high_features = x_l.squeeze(0) if x_l.ndim == 3 and x_l.shape[0] == 1 else x_l
        context = self.low_parent_context(low_features.float(), mapping, high_count=high_features.shape[0])
        # Preserve the low-to-high CSR relation for per-parent route statistics.
        context['mapping_parent_ptr'] = torch.as_tensor(mapping['parent_ptr'], device=low_features.device, dtype=torch.long)
        context['mapping_child_indices'] = torch.as_tensor(mapping['child_indices'], device=low_features.device, dtype=torch.long)
        if getattr(self, 'routing_diagnostic_mode', 'normal') == 'context_mean':
            # Inference-only ablation: replace each mapped high row's local
            # parent context with this slide's mean valid low-parent feature.
            low_valid = torch.as_tensor(
                mapping['low_valid_mask'], device=low_features.device, dtype=torch.bool
            )
            parent_ptr = context['mapping_parent_ptr']
            has_child = (parent_ptr[1:] - parent_ptr[:-1]) > 0
            valid_low_parent = low_valid & has_child
            if bool(valid_low_parent.any()):
                mean_context = low_features.float()[valid_low_parent].mean(dim=0)
                mapped_high = context['high_parent_context_valid_mask'].to(
                    low_features.device, dtype=torch.bool
                )
                mean_parent_context = context['high_parent_context'].clone()
                mean_parent_context[mapped_high] = mean_context.to(mean_parent_context.dtype)
                context['high_parent_context'] = mean_parent_context
        self._last_mapping_context = context
        return context

    def _shuffle_parent_context(self, context, high_features):
        """Shuffle mapped low contexts within a slide for context_shuffle."""
        valid = context['high_parent_context_valid_mask'].to(high_features.device, dtype=torch.bool)
        if int(valid.sum()) > 1:
            indices = torch.where(valid)[0]
            generator = torch.Generator(device='cpu').manual_seed(self.routing_diagnostic_seed)
            perm = torch.randperm(indices.numel(), generator=generator).to(indices.device)
            shuffled = context['high_parent_context'].clone()
            shuffled[indices] = shuffled[indices[perm]]
            context = dict(context)
            context['high_parent_context'] = shuffled
        return context

    def forward(self, x_s, coord_s, x_l, coords_l, label, mapping=None):
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
        # Stage 3.3.1 plumbing only: context is constructed but not used in logits.
        mapping_context = self.build_mapping_context(x_s, x_l, mapping)
        # 生成文本特征（Prompt-learning + 可选文本微调）
        tokenized_prompts = self.prompt_learner.tokenized_prompts.to(x_s.device)
        prompt_embeddings = self.prompt_learner().to(x_s.device)
        eos_indices = getattr(self.prompt_learner, 'eos_indices', None)
        if eos_indices is not None:
            eos_indices = eos_indices.to(x_s.device)
        text_features = self.text_encoder(tokenized_prompts, prompt_embeddings=prompt_embeddings, eos_indices=eos_indices)  # [2*num_classes, 512]
        
        # ========== 低分辨率分支 ==========
        M = x_s.float()
        compents, _ = self.cross_attention_1(self.learnable_image_center, M, M)
        compents = self.norm(compents + self.learnable_image_center)
        
        H = compents.squeeze().float()
        image_features_low, _ = self.gated_attention_pool(
            H, self.attention_V, self.attention_U, self.attention_weights
        )
        
        # ========== 高分辨率分支 ==========
        M_high = x_l.float()
        if self.use_low_context_routing and mapping_context is not None:
            high_rows = M_high.squeeze(0)
            diagnostic_mode = str(getattr(self, 'routing_diagnostic_mode', 'normal'))
            if diagnostic_mode == 'context_shuffle':
                mapping_context = self._shuffle_parent_context(mapping_context, high_rows)
            routed_rows = self.high_router(
                high_rows,
                mapping_context,
                diagnostic_mode=(
                    'normal' if diagnostic_mode in {'context_shuffle', 'context_mean'}
                    else diagnostic_mode
                ),
            )
            if diagnostic_mode in {'context_shuffle', 'context_mean'}:
                self.high_router.last_diagnostics['diagnostic_mode'] = diagnostic_mode
            self.last_routing_diagnostics = dict(self.high_router.last_diagnostics)
            if self.routing_scale != 1.0:
                residual = routed_rows - high_rows
                routed_rows = high_rows + self.routing_scale * residual
                self.last_routing_diagnostics['routing_residual_norm'] = float(
                    (self.routing_scale * residual).detach().norm().cpu()
                )
                self.last_routing_diagnostics['routed_high_change_norm'] = self.last_routing_diagnostics['routing_residual_norm']
            M_high = routed_rows.unsqueeze(0)
        compents_high, _ = self.cross_attention_1(self.learnable_image_center, M_high, M_high)
        compents_high = self.norm(compents_high + self.learnable_image_center)
        
        H_high = compents_high.squeeze().float()
        image_features_high, _ = self.gated_attention_pool(
            H_high, self.attention_V, self.attention_U, self.attention_weights
        )
        
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
        # Scale ablation: retain the original dual sum exactly, while allowing
        # controlled single-scale training without changing feature files.
        if self.scale_mode == 'low':
            logits = logits_low
        elif self.scale_mode == 'high':
            logits = logits_high
        else:
            logits = logits_low + logits_high
        
        loss = self.loss_ce(logits, label)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]
        
        return Y_prob, Y_hat, loss
    
    def forward_with_attention(self, x_s, coord_s, x_l, coords_l, label):
        """前向传播并返回注意力权重(用于热力图)"""
        tokenized_prompts = self.prompt_learner.tokenized_prompts.to(x_s.device)
        prompt_embeddings = self.prompt_learner().to(x_s.device)
        eos_indices = getattr(self.prompt_learner, 'eos_indices', None)
        if eos_indices is not None:
            eos_indices = eos_indices.to(x_s.device)
        text_features = self.text_encoder(tokenized_prompts, prompt_embeddings=prompt_embeddings, eos_indices=eos_indices)
        
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
        image_features_low, A = self.gated_attention_pool(
            H, self.attention_V, self.attention_U, self.attention_weights
        )
        
        H_high = compents_high.squeeze().float()
        image_features_high, A_high = self.gated_attention_pool(
            H_high, self.attention_V, self.attention_U, self.attention_weights
        )
        
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
