import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from logzero import logger


class NeuralKVLayer(nn.Module):
    """Per-layer neural KV cache with batched einsum MLPs.

    Supports two modes controlled by ``group_shared``:
    - group_shared=True  (default): one MLP per GQA group (4 MLPs for 28 heads).
      Heads within the same group share weights, reducing params by num_group.
    - group_shared=False: one MLP per query head (28 MLPs). More capacity but
      ~7x more parameters.

    Weights are stored as (N, out, in) where N = num_heads_kv or num_heads.
    """

    def __init__(self, num_heads, num_heads_kv, dim_head, hidden_mult=2,
                 group_shared=True):
        super().__init__()
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.num_group = num_heads // num_heads_kv  # 7
        self.dim_head = dim_head
        self.group_shared = group_shared

        N = num_heads_kv if group_shared else num_heads
        D = dim_head       # 128
        M = dim_head * hidden_mult  # 256

        self.w1 = nn.Parameter(torch.randn(N, M, D) * (2.0 / D) ** 0.5)
        self.b1 = nn.Parameter(torch.zeros(N, M))
        self.w2 = nn.Parameter(torch.randn(N, M, M) * (2.0 / M) ** 0.5)
        self.b2 = nn.Parameter(torch.zeros(N, M))
        self.w3 = nn.Parameter(torch.randn(N, D, M) * (2.0 / M) ** 0.5)
        self.b3 = nn.Parameter(torch.zeros(N, D))

    def forward(self, q):
        """
        q: (batch, num_heads, len_q, dim_head)  e.g. (B, 28, T, 128)
        returns:
            attn_out: (batch, num_heads, len_q, dim_head)  e.g. (B, 28, T, 128)
        """
        B, H, T, D = q.shape

        if self.group_shared:
            G = self.num_heads_kv   # 4
            Hg = self.num_group     # 7
            # (B, 28, T, D) -> (B, 4, 7*T, D)
            x = q.view(B, G, Hg, T, D).reshape(B, G, Hg * T, D)
            x = torch.einsum('bgti,goi->bgto', x, self.w1) + self.b1[None, :, None, :]
            x = F.gelu(x)
            x = torch.einsum('bgti,goi->bgto', x, self.w2) + self.b2[None, :, None, :]
            x = F.gelu(x)
            x = torch.einsum('bgti,goi->bgto', x, self.w3) + self.b3[None, :, None, :]
            # (B, 4, 7*T, D) -> (B, 28, T, D)
            x = x.view(B, G, Hg, T, D).reshape(B, H, T, D)
        else:
            # Per-head: N = num_heads, einsum dim label 'h' for heads
            x = torch.einsum('bhti,hoi->bhto', q, self.w1) + self.b1[None, :, None, :]
            x = F.gelu(x)
            x = torch.einsum('bhti,hoi->bhto', x, self.w2) + self.b2[None, :, None, :]
            x = F.gelu(x)
            x = torch.einsum('bhti,hoi->bhto', x, self.w3) + self.b3[None, :, None, :]

        return x


def _apply_rope_to_q(q, position_bias, total_kv_len=None):
    """Apply RoPE to Q tensor, matching position_bias.forward() convention.

    Position assignment:
      - If total_kv_len is provided and >= len_q:
            Q gets positions [total_kv_len - len_q, total_kv_len)
            (Q is placed at the END of the KV sequence, like real inference)
      - Otherwise:
            Q gets positions [0, len_q)

    Args:
        q: (batch, num_heads, len_q, dim_head) or (num_heads, len_q, dim_head)
        position_bias: RotaryEmbeddingESM module
        total_kv_len: total number of KV tokens (video tokens). When provided,
                      Q is positioned after K, consistent with position_bias.forward().

    Returns:
        q with RoPE applied, same shape and dtype as input.
    """
    squeeze = False
    if q.dim() == 3:
        q = q.unsqueeze(0)
        squeeze = True
    dtype = q.dtype
    len_q = q.size(-2)
    # right = end position for Q. Q gets positions [right - len_q, right).
    if total_kv_len is not None and total_kv_len >= len_q:
        right = total_kv_len
    else:
        right = len_q
    max_len = max(len_q, right)
    position_bias._seq_len_cached = -1
    cos, sin = position_bias._update_cos_sin_tables_len(max_len, q.device, dim=4)
    q_rope = position_bias.apply_rotary_pos_emb(q, len_q, right, cos, sin)
    q_rope = q_rope.to(dtype)
    if squeeze:
        q_rope = q_rope.squeeze(0)
    return q_rope


class NeuralKVCache(nn.Module):
    """Collection of per-layer NeuralKV modules."""

    def __init__(self, num_layers, num_heads, num_heads_kv, dim_head,
                 hidden_mult=2, group_shared=True):
        super().__init__()
        self.num_layers = num_layers
        self.dim_head = dim_head
        self.layers = nn.ModuleList([
            NeuralKVLayer(num_heads, num_heads_kv, dim_head, hidden_mult,
                          group_shared=group_shared)
            for _ in range(num_layers)
        ])
        self.position_bias = None  # set after training
        self.total_kv_len = None   # set after training (num video tokens)

    def forward(self, layer_idx, q):
        """Apply RoPE to Q (if position_bias is set), then run the MLP.

        Q gets positions [total_kv_len - len_q, total_kv_len), matching
        the position_bias.forward() convention used for GT computation.

        Args:
            q: (batch, num_heads, len_q, dim_head)
        Returns:
            attn_out: (batch, num_heads, len_q, dim_head)
        """
        if self.position_bias is not None:
            q = _apply_rope_to_q(q, self.position_bias, self.total_kv_len)
        return self.layers[layer_idx](q)


def collect_all_kv_from_context_manager(ctx_mgr):
    """Extract all K, V from a ContextManager's global_blocks (CPU) + init + local.

    Returns:
        all_k: (num_heads_kv, total_tokens, dim_head) on CPU
        all_v: (num_heads_kv, total_tokens, dim_head) on CPU
    """
    k_list = []
    v_list = []

    u = 0  # batch_size is always 1

    if ctx_mgr.init_k.size(-2) > 0:
        k_list.append(ctx_mgr.init_k[u].cpu())
        v_list.append(ctx_mgr.init_v[u].cpu())

    for block in ctx_mgr.global_blocks[u]:
        block_kv = block.cpu_data
        k_list.append(block_kv[0].cpu())
        v_list.append(block_kv[1].cpu())

    if ctx_mgr.local_k.size(-2) > 0:
        k_list.append(ctx_mgr.local_k[u].cpu())
        v_list.append(ctx_mgr.local_v[u].cpu())

    all_k = torch.cat(k_list, dim=1)
    all_v = torch.cat(v_list, dim=1)

    return all_k, all_v


def collect_real_q_from_model(qa_model, video_tensor, device='cuda'):
    """Re-run prefill with hooks to collect real Q from each layer.

    Returns:
        all_layer_q: list of (num_heads, total_tokens, dim_head) on CPU, one per layer
    """
    all_layer_q = [[] for _ in range(len(qa_model.language_model.model.layers))]
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            hidden_states = kwargs.get('hidden_states', None) if kwargs else None
            if hidden_states is None:
                hidden_states = args[0]
            h_q = module.q_proj(hidden_states)
            batch_size, seq_len = h_q.shape[0], h_q.shape[1]
            num_heads = module.num_heads
            dim_head = module.head_dim
            h_q = h_q.view(batch_size, seq_len, num_heads, dim_head).permute(0, 2, 1, 3)
            # (batch, num_heads, seq_len, dim_head) -> take batch=0
            all_layer_q[layer_idx].append(h_q[0].cpu())
            return output
        return hook_fn

    for i, layer in enumerate(qa_model.language_model.model.layers):
        h = layer.self_attn.register_forward_hook(make_hook(i), with_kwargs=True)
        hooks.append(h)

    # Re-encode: init_prompt + video
    qa_model.clear_cache()
    qa_model.encode_init_prompt()
    qa_model.encode_video(video_tensor)

    for h in hooks:
        h.remove()

    # Concatenate all chunks per layer
    result = []
    for layer_idx in range(len(all_layer_q)):
        q_cat = torch.cat(all_layer_q[layer_idx], dim=1)  # (num_heads, total_tokens, dim_head)
        result.append(q_cat)

    return result


def compute_attention_output(q, k, v, num_heads, num_heads_kv, return_lse=False,
                             position_bias=None):
    """Compute standard attention output: softmax(Q @ K^T / sqrt(d)) @ V

    If position_bias (RoPE) is provided, apply RoPE to Q and K before computing
    attention. Uses position_bias.forward() convention:
      - K positions: [0, len_k)
      - Q positions: [len_k - len_q, len_k)   (when len_q <= len_k)
      - When len_q > len_k: pad K to len_q, then K at [0, len_k), Q at [0, len_q)

    This matches _compute_gt_video_attention in rekv_attention.py.

    Args:
        q: (num_heads, len_q, dim_head)
        k: (num_heads_kv, total_kv, dim_head)
        v: (num_heads_kv, total_kv, dim_head)
        return_lse: if True, also return log-sum-exp of logits
        position_bias: RoPE module (optional)

    Returns:
        attn_out: (num_heads, len_q, dim_head)
        lse (optional): (num_heads, len_q)  -- log(sum(exp(logits)))
    """
    dim_head = q.size(-1)

    if num_heads != num_heads_kv:
        num_group = num_heads // num_heads_kv
        k = k.unsqueeze(1).expand(-1, num_group, -1, -1)
        k = k.reshape(num_heads, -1, dim_head)
        v = v.unsqueeze(1).expand(-1, num_group, -1, -1)
        v = v.reshape(num_heads, -1, dim_head)

    # Apply RoPE if provided — use position_bias.forward() convention
    if position_bias is not None:
        len_q = q.size(-2)
        len_k = k.size(-2)
        q_4d = q.unsqueeze(0)   # (1, H, len_q, D)
        k_4d = k.unsqueeze(0)   # (1, H, len_k, D)
        position_bias._seq_len_cached = -1
        if len_q > len_k:
            # Pad K so position_bias.forward doesn't crash.
            # K: [0, len_k), Q: [0, len_q) (both start from 0)
            k_padded = F.pad(k_4d, (0, 0, 0, len_q - len_k))
            q_rope, k_rope = position_bias(q_4d, k_padded)
            k = k_rope[0, :, :len_k, :].to(k.dtype)
        else:
            # Standard: K at [0, len_k), Q at [len_k - len_q, len_k)
            q_rope, k_rope = position_bias(q_4d, k_4d)
            k = k_rope[0].to(k.dtype)
        q = q_rope[0].to(q.dtype)

    scale = 1.0 / math.sqrt(dim_head)
    logits = torch.matmul(q, k.transpose(-1, -2)) * scale  # (num_heads, len_q, total_kv)

    if return_lse:
        lse = torch.logsumexp(logits, dim=-1)  # (num_heads, len_q)

    attn_weights = F.softmax(logits, dim=-1)
    attn_out = torch.matmul(attn_weights, v)

    if return_lse:
        return attn_out, lse
    return attn_out


def visualize_qkv_pca(all_layer_q, all_layer_k, all_layer_v,
                      num_heads, num_heads_kv, dim_head,
                      save_dir='results/neural_kv_eval',
                      n_sample=500):
    """Visualize Q, K, V with separate PCA projections and L2 norm distributions."""
    os.makedirs(save_dir, exist_ok=True)
    num_layers = len(all_layer_q)
    num_group = num_heads // num_heads_kv

    layer_indices = sorted(set([
        0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1
    ]))
    n_layers_plot = len(layer_indices)

    total_q_tokens = all_layer_q[0].size(1)
    total_kv_tokens = all_layer_k[0].size(1)
    n_q_sample = min(n_sample, total_q_tokens)
    n_kv_sample = min(n_sample, total_kv_tokens)
    # Use the smaller count so we can combine Q, K, V in joint plots
    n_sample = min(n_q_sample, n_kv_sample)
    q_sample_idx = torch.randperm(total_q_tokens)[:n_sample]
    kv_sample_idx = torch.randperm(total_kv_tokens)[:n_sample]

    # ===== Plot 1: Separate PCA for Q, K, V (head-averaged) =====
    fig, axes = plt.subplots(3, n_layers_plot, figsize=(5 * n_layers_plot, 14))

    for col, layer_idx in enumerate(layer_indices):
        q = all_layer_q[layer_idx][:, q_sample_idx, :].mean(dim=0).numpy()
        k = all_layer_k[layer_idx][:, kv_sample_idx, :].mean(dim=0).numpy()
        v = all_layer_v[layer_idx][:, kv_sample_idx, :].mean(dim=0).numpy()

        for row, (data, name, color) in enumerate([
            (q, 'Q', '#e74c3c'), (k, 'K', '#2ecc71'), (v, 'V', '#3498db')
        ]):
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(data)
            var_ratio = pca.explained_variance_ratio_
            axes[row, col].scatter(data_2d[:, 0], data_2d[:, 1], s=3, alpha=0.5, c=color)
            axes[row, col].set_title(
                f'Layer {layer_idx} - {name}\n'
                f'PC1: {var_ratio[0]:.1%}, PC2: {var_ratio[1]:.1%}',
                fontsize=9
            )
            axes[row, col].set_xlabel('PC1')
            axes[row, col].set_ylabel('PC2')

    plt.suptitle('PCA: Q / K / V separately (head-averaged)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pca_qkv_separate.png'), dpi=150)
    plt.close()
    logger.info(f"[NeuralKV Eval] Saved pca_qkv_separate.png")

    # ===== Plot 2: Joint PCA (Q+K+V together) to see relative positions =====
    fig, axes = plt.subplots(1, n_layers_plot, figsize=(6 * n_layers_plot, 5))
    if n_layers_plot == 1:
        axes = [axes]

    for col, layer_idx in enumerate(layer_indices):
        q = all_layer_q[layer_idx][:, q_sample_idx, :].mean(dim=0).numpy()
        k = all_layer_k[layer_idx][:, kv_sample_idx, :].mean(dim=0).numpy()
        v = all_layer_v[layer_idx][:, kv_sample_idx, :].mean(dim=0).numpy()

        combined = np.concatenate([q, k, v], axis=0)
        pca = PCA(n_components=2)
        embedded = pca.fit_transform(combined)
        var_ratio = pca.explained_variance_ratio_

        axes[col].scatter(embedded[:n_sample, 0], embedded[:n_sample, 1],
                          s=3, alpha=0.5, c='#e74c3c', label='Q')
        axes[col].scatter(embedded[n_sample:2*n_sample, 0], embedded[n_sample:2*n_sample, 1],
                          s=3, alpha=0.5, c='#2ecc71', label='K')
        axes[col].scatter(embedded[2*n_sample:, 0], embedded[2*n_sample:, 1],
                          s=3, alpha=0.5, c='#3498db', label='V')
        axes[col].set_title(f'Layer {layer_idx}\nPC1: {var_ratio[0]:.1%}, PC2: {var_ratio[1]:.1%}', fontsize=9)
        axes[col].legend(markerscale=3, fontsize=8)
        axes[col].set_xlabel('PC1')
        axes[col].set_ylabel('PC2')

    plt.suptitle('PCA: Q vs K vs V jointly (head-averaged)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pca_qkv_joint.png'), dpi=150)
    plt.close()
    logger.info(f"[NeuralKV Eval] Saved pca_qkv_joint.png")

    # ===== Plot 3: Per-head PCA for middle layer =====
    mid_layer = num_layers // 2
    n_heads_plot = min(8, num_heads)
    head_indices = [int(i * num_heads / n_heads_plot) for i in range(n_heads_plot)]

    fig, axes = plt.subplots(2, (n_heads_plot + 1) // 2, figsize=(5 * ((n_heads_plot + 1) // 2), 10))
    axes_flat = axes.flatten()

    for plot_idx, h_idx in enumerate(head_indices):
        kv_h_idx = h_idx // num_group

        q_h = all_layer_q[mid_layer][h_idx, q_sample_idx, :].numpy()
        k_h = all_layer_k[mid_layer][kv_h_idx, kv_sample_idx, :].numpy()
        v_h = all_layer_v[mid_layer][kv_h_idx, kv_sample_idx, :].numpy()

        combined = np.concatenate([q_h, k_h, v_h], axis=0)
        pca = PCA(n_components=2)
        embedded = pca.fit_transform(combined)

        axes_flat[plot_idx].scatter(embedded[:n_sample, 0], embedded[:n_sample, 1],
                                    s=3, alpha=0.5, c='#e74c3c', label='Q')
        axes_flat[plot_idx].scatter(embedded[n_sample:2*n_sample, 0], embedded[n_sample:2*n_sample, 1],
                                    s=3, alpha=0.5, c='#2ecc71', label='K')
        axes_flat[plot_idx].scatter(embedded[2*n_sample:, 0], embedded[2*n_sample:, 1],
                                    s=3, alpha=0.5, c='#3498db', label='V')
        axes_flat[plot_idx].set_title(f'Q-Head {h_idx} / KV-Head {kv_h_idx}', fontsize=9)
        axes_flat[plot_idx].legend(markerscale=3, fontsize=7)

    for i in range(len(head_indices), len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.suptitle(f'Layer {mid_layer}: PCA per head (Q vs K vs V)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pca_qkv_per_head.png'), dpi=150)
    plt.close()
    logger.info(f"[NeuralKV Eval] Saved pca_qkv_per_head.png")

    # ===== Plot 4: L2 norm distributions =====
    # 4a: Per-layer mean norm (bar chart)
    q_norms_per_layer = []
    k_norms_per_layer = []
    v_norms_per_layer = []

    for layer_idx in range(num_layers):
        q_norm = all_layer_q[layer_idx].float().norm(dim=-1).mean().item()
        k_norm = all_layer_k[layer_idx].float().norm(dim=-1).mean().item()
        v_norm = all_layer_v[layer_idx].float().norm(dim=-1).mean().item()
        q_norms_per_layer.append(q_norm)
        k_norms_per_layer.append(k_norm)
        v_norms_per_layer.append(v_norm)

    x = np.arange(num_layers)
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(14, num_layers * 0.5), 5))
    ax.bar(x - width, q_norms_per_layer, width, label='Q', color='#e74c3c', alpha=0.8)
    ax.bar(x, k_norms_per_layer, width, label='K', color='#2ecc71', alpha=0.8)
    ax.bar(x + width, v_norms_per_layer, width, label='V', color='#3498db', alpha=0.8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean L2 Norm')
    ax.set_title('Per-Layer Mean L2 Norm of Q, K, V')
    ax.legend()
    ax.set_xticks(x[::2])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'norm_per_layer.png'), dpi=150)
    plt.close()
    logger.info(f"[NeuralKV Eval] Saved norm_per_layer.png")

    # 4b: Norm histograms for selected layers
    fig, axes = plt.subplots(3, n_layers_plot, figsize=(5 * n_layers_plot, 12))

    for col, layer_idx in enumerate(layer_indices):
        q_n = all_layer_q[layer_idx][:, q_sample_idx, :].float().norm(dim=-1).flatten().numpy()
        k_n = all_layer_k[layer_idx][:, kv_sample_idx, :].float().norm(dim=-1).flatten().numpy()
        v_n = all_layer_v[layer_idx][:, kv_sample_idx, :].float().norm(dim=-1).flatten().numpy()

        for row, (data, name, color) in enumerate([
            (q_n, 'Q', '#e74c3c'), (k_n, 'K', '#2ecc71'), (v_n, 'V', '#3498db')
        ]):
            axes[row, col].hist(data, bins=50, color=color, edgecolor='white', alpha=0.8)
            axes[row, col].axvline(x=np.mean(data), color='black', linestyle='--',
                                   label=f'mean={np.mean(data):.2f}')
            axes[row, col].axvline(x=np.std(data) + np.mean(data), color='gray', linestyle=':',
                                   label=f'std={np.std(data):.2f}')
            axes[row, col].set_title(f'Layer {layer_idx} - {name} norm', fontsize=9)
            axes[row, col].set_xlabel('L2 Norm')
            axes[row, col].set_ylabel('Count')
            axes[row, col].legend(fontsize=7)

    plt.suptitle('L2 Norm Distributions of Q, K, V', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'norm_histograms.png'), dpi=150)
    plt.close()
    logger.info(f"[NeuralKV Eval] Saved norm_histograms.png")

    # ===== Plot 5: t-SNE Q vs K vs V (head-averaged) =====
    fig, axes = plt.subplots(1, n_layers_plot, figsize=(6 * n_layers_plot, 5))
    if n_layers_plot == 1:
        axes = [axes]

    for col, layer_idx in enumerate(layer_indices):
        q = all_layer_q[layer_idx][:, q_sample_idx, :].mean(dim=0).numpy()
        k = all_layer_k[layer_idx][:, kv_sample_idx, :].mean(dim=0).numpy()
        v = all_layer_v[layer_idx][:, kv_sample_idx, :].mean(dim=0).numpy()

        combined = np.concatenate([q, k, v], axis=0)
        tsne = TSNE(n_components=2, perplexity=min(30, n_sample - 1),
                     random_state=42, max_iter=1000)
        embedded = tsne.fit_transform(combined)

        axes[col].scatter(embedded[:n_sample, 0], embedded[:n_sample, 1],
                          s=3, alpha=0.5, c='#e74c3c', label='Q')
        axes[col].scatter(embedded[n_sample:2*n_sample, 0], embedded[n_sample:2*n_sample, 1],
                          s=3, alpha=0.5, c='#2ecc71', label='K')
        axes[col].scatter(embedded[2*n_sample:, 0], embedded[2*n_sample:, 1],
                          s=3, alpha=0.5, c='#3498db', label='V')
        axes[col].set_title(f'Layer {layer_idx} (head-avg)')
        axes[col].legend(markerscale=3, fontsize=8)
        axes[col].set_xticks([])
        axes[col].set_yticks([])

    plt.suptitle('t-SNE: Q vs K vs V (head-averaged)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tsne_qkv_head_avg.png'), dpi=150)
    plt.close()
    logger.info(f"[NeuralKV Eval] Saved tsne_qkv_head_avg.png")


COMMON_QUESTIONS_MC = [
    ("What is happening in the video?", ["Something happens", "Nothing happens", "A person walks", "An object moves"]),
    ("What color is the main object in the video?", ["Red", "Blue", "Green", "Yellow"]),
    ("How many people are in the video?", ["One", "Two", "Three", "Four"]),
    ("Where does the video take place?", ["Indoor", "Outdoor", "Kitchen", "Office"]),
    ("What action is the person performing?", ["Walking", "Running", "Sitting", "Standing"]),
    ("What is the main topic of this video?", ["Sports", "Cooking", "Travel", "Education"]),
    ("What happens at the end of the video?", ["The person leaves", "The scene changes", "Nothing happens", "The person speaks"]),
    ("What is the person wearing?", ["A hat", "A jacket", "A dress", "A uniform"]),
    ("What is the background music like?", ["Cheerful", "Sad", "Dramatic", "No music"]),
    ("What time of day is it in the video?", ["Morning", "Afternoon", "Evening", "Night"]),
    ("What is the relationship between the people?", ["Friends", "Family", "Colleagues", "Strangers"]),
    ("What emotion does the main character show?", ["Happy", "Sad", "Angry", "Neutral"]),
]

COMMON_QUESTIONS_OPEN = [
    "What is the main content of this video?",
    "Describe what happens in the video.",
    "What are the key events in this video?",
    "Who are the main characters in the video?",
    "What is the setting of this video?",
    "Summarize the video in a few sentences.",
    "What is the mood or tone of this video?",
    "What objects can you see in the video?",
    "What is the purpose of this video?",
    "Describe the beginning of the video.",
    "What can you tell about the location?",
    "What sounds can be heard in the video?",
]


def _format_mc_prompt(question, choices, choice_letters="ABCDEFGHIJ"):
    formatted_choices = "\n".join([
        f"({choice_letters[i]}) {c}" for i, c in enumerate(choices)
    ])
    return f"Question: {question}\nOptions:\n{formatted_choices}\nOnly give the best option."


def synthesize_decode_q(qa_model, all_layer_k, all_layer_v,
                        num_heads, num_heads_kv, dim_head,
                        batch_token_size=512, device='cuda',
                        position_bias=None):
    """Synthesize decode-like Q and their GT attention outputs for training.

    Runs common question templates through the model to collect Q vectors
    from each layer, then computes GT attention outputs using real K, V.
    If position_bias is provided, RoPE is applied for GT computation.

    Returns:
        synth_q: list of (num_heads, n_tokens, dim_head) per layer on CPU
        synth_targets: list of (num_heads, n_tokens, dim_head) per layer on CPU
    """
    prompts = []
    for question, choices in COMMON_QUESTIONS_MC:
        formatted = _format_mc_prompt(question, choices)
        prompts.append(qa_model.get_prompt(formatted, mc=True))
    for question in COMMON_QUESTIONS_OPEN:
        prompts.append(qa_model.get_prompt(question, mc=False))

    logger.info(f"[NeuralKV Synth] Collecting Q from {len(prompts)} question templates...")
    synth_q = collect_decode_q(qa_model, prompts, device=device)

    n_synth_tokens = synth_q[0].size(1) if synth_q[0].dim() == 3 else 0
    logger.info(f"[NeuralKV Synth] Collected {n_synth_tokens} synthetic Q tokens per layer")

    if n_synth_tokens == 0:
        return None, None

    # Compute GT attention outputs for synthesized Q (with RoPE if available)
    num_layers = len(all_layer_k)
    synth_targets = []
    for layer_idx in range(num_layers):
        q_real = synth_q[layer_idx]  # (num_heads, n_tokens, dim_head)
        k = all_layer_k[layer_idx].clone()
        v = all_layer_v[layer_idx].clone()

        target_chunks = []
        for st in range(0, n_synth_tokens, batch_token_size):
            ed = min(st + batch_token_size, n_synth_tokens)
            q_chunk = q_real[:, st:ed, :].to(device).float()
            k_gpu = k.to(device).float()
            v_gpu = v.to(device).float()

            with torch.no_grad():
                target_chunk = compute_attention_output(
                    q_chunk, k_gpu, v_gpu, num_heads, num_heads_kv,
                    position_bias=position_bias
                )
            target_chunks.append(target_chunk.cpu())

        target = torch.cat(target_chunks, dim=1)
        synth_targets.append(target)

        if layer_idx == 0:
            logger.info(f"[NeuralKV Synth] Layer 0 synth target shape: {target.shape}")

    return synth_q, synth_targets


def collect_decode_q(qa_model, questions, device='cuda'):
    """Collect Q from question-answering (decode) stage via hooks.

    Runs tokenization + forward pass for each question text through the language model
    to collect the Q vectors that would be seen during decode.
    Uses use_cache=True with the existing kv_cache (required by patched attention).

    Args:
        qa_model: the ReKV model after encode_video
        questions: list of question strings
        device: device

    Returns:
        all_layer_q: list of (num_heads, total_q_tokens, dim_head) on CPU, one per layer
    """
    num_layers = len(qa_model.language_model.model.layers)
    all_layer_q = [[] for _ in range(num_layers)]
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            hidden_states = kwargs.get('hidden_states', None) if kwargs else None
            if hidden_states is None:
                hidden_states = args[0]
            h_q = module.q_proj(hidden_states)
            batch_size, seq_len = h_q.shape[0], h_q.shape[1]
            num_heads = module.num_heads
            dim_head = module.head_dim
            h_q = h_q.view(batch_size, seq_len, num_heads, dim_head).permute(0, 2, 1, 3)
            all_layer_q[layer_idx].append(h_q[0].cpu())
            return output
        return hook_fn

    for i, layer in enumerate(qa_model.language_model.model.layers):
        h = layer.self_attn.register_forward_hook(make_hook(i), with_kwargs=True)
        hooks.append(h)

    # Set retrieval mode (required for question encoding in ReKV)
    for layer_kv in qa_model.kv_cache:
        layer_kv.set_retrieval()

    for q_text in questions:
        input_ids = qa_model.processor.tokenizer(q_text).input_ids
        input_ids = torch.as_tensor([input_ids], device=device)
        with torch.no_grad():
            qa_model.language_model(
                input_ids=input_ids,
                use_cache=True,
                past_key_values=qa_model.kv_cache,
            )

    for layer_kv in qa_model.kv_cache:
        layer_kv.reset_retrieval()

    for h in hooks:
        h.remove()

    result = []
    for layer_idx in range(num_layers):
        if all_layer_q[layer_idx]:
            q_cat = torch.cat(all_layer_q[layer_idx], dim=1)
        else:
            q_cat = torch.zeros(0)
        result.append(q_cat)

    return result


def visualize_train_vs_decode_q(all_layer_q, all_layer_k, decode_q,
                                 num_heads, num_heads_kv,
                                 save_dir='results/neural_kv_eval',
                                 n_sample=500, synth_q=None):
    """Visualize distribution difference between training Q/K, synth Q, and decode Q using t-SNE.

    Training data = prefill Q (video tokens) + K (used as Q during training)
    Synth data = Q from common question templates (used in Phase C training)
    Decode data = Q from actual test questions seen during inference
    """
    os.makedirs(save_dir, exist_ok=True)
    num_layers = len(all_layer_q)
    num_group = num_heads // num_heads_kv
    has_synth = synth_q is not None and synth_q[0].dim() == 3 and synth_q[0].size(1) > 0

    layer_indices = sorted(set([
        0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1
    ]))
    n_layers_plot = len(layer_indices)

    total_q_tokens = all_layer_q[0].size(1)
    total_k_tokens = all_layer_k[0].size(1)
    n_q_sample = min(n_sample, total_q_tokens)
    n_k_sample = min(n_sample, total_k_tokens)
    q_sample_idx = torch.randperm(total_q_tokens)[:n_q_sample]
    k_sample_idx = torch.randperm(total_k_tokens)[:n_k_sample]

    # ===== Plot 1: t-SNE comparing train Q, train K, synth Q, and decode Q =====
    fig, axes = plt.subplots(1, n_layers_plot, figsize=(6 * n_layers_plot, 5))
    if n_layers_plot == 1:
        axes = [axes]

    for col, layer_idx in enumerate(layer_indices):
        # Train Q (prefill Q or synth Q, head-averaged)
        train_q = all_layer_q[layer_idx][:, q_sample_idx, :].mean(dim=0).numpy()

        # Train K expanded to num_heads (as used in training), head-averaged
        k_raw = all_layer_k[layer_idx][:, k_sample_idx, :]
        if num_heads != num_heads_kv:
            k_exp = k_raw.unsqueeze(1).expand(-1, num_group, -1, -1).reshape(num_heads, -1, k_raw.size(-1))
        else:
            k_exp = k_raw
        train_k = k_exp.mean(dim=0).numpy()

        # Synth Q (head-averaged)
        if has_synth:
            n_synth = synth_q[layer_idx].size(1)
            n_synth_sample = min(n_sample, n_synth)
            synth_sample_idx = torch.randperm(n_synth)[:n_synth_sample]
            syn_q = synth_q[layer_idx][:, synth_sample_idx, :].mean(dim=0).numpy()
        else:
            syn_q = np.zeros((0, train_q.shape[1]))
            n_synth_sample = 0

        # Decode Q (head-averaged)
        if decode_q[layer_idx].dim() == 3:
            n_dec = decode_q[layer_idx].size(1)
            n_dec_sample = min(n_sample, n_dec)
            dec_sample_idx = torch.randperm(n_dec)[:n_dec_sample]
            dec_q = decode_q[layer_idx][:, dec_sample_idx, :].mean(dim=0).numpy()
        else:
            dec_q = np.zeros((1, train_q.shape[1]))
            n_dec_sample = 1

        parts = [train_q, train_k]
        if n_synth_sample > 0:
            parts.append(syn_q)
        parts.append(dec_q)
        combined = np.concatenate(parts, axis=0)

        min_n = min(n_q_sample, n_k_sample, n_dec_sample)
        if n_synth_sample > 0:
            min_n = min(min_n, n_synth_sample)
        perp = max(2, min(30, min_n - 1))
        tsne = TSNE(n_components=2, perplexity=perp,
                     random_state=42, max_iter=1000)
        embedded = tsne.fit_transform(combined)

        n1 = n_q_sample
        n2 = n_k_sample
        offset = 0
        axes[col].scatter(embedded[offset:offset+n1, 0], embedded[offset:offset+n1, 1],
                          s=5, alpha=0.4, c='#e74c3c', label=f'Train Q (n={n1})')
        offset += n1
        axes[col].scatter(embedded[offset:offset+n2, 0], embedded[offset:offset+n2, 1],
                          s=5, alpha=0.4, c='#2ecc71', label=f'Train K-as-Q (n={n2})')
        offset += n2
        if n_synth_sample > 0:
            axes[col].scatter(embedded[offset:offset+n_synth_sample, 0], embedded[offset:offset+n_synth_sample, 1],
                              s=15, alpha=0.9, c='#f39c12', marker='D', label=f'Synth Q (n={n_synth_sample})')
            offset += n_synth_sample
        axes[col].scatter(embedded[offset:, 0], embedded[offset:, 1],
                          s=15, alpha=0.9, c='#9b59b6', marker='x', label=f'Decode Q (n={n_dec_sample})')
        axes[col].set_title(f'Layer {layer_idx}', fontsize=11)
        axes[col].legend(markerscale=2, fontsize=6, loc='best')
        axes[col].set_xticks([])
        axes[col].set_yticks([])

    plt.suptitle('t-SNE: Training Q/K vs Synth Q vs Decode Q', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tsne_train_vs_decode_q.png'), dpi=150)
    plt.close()
    logger.info(f"[NeuralKV Eval] Saved tsne_train_vs_decode_q.png")

    # ===== Plot 2: L2 norm comparison =====
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    train_q_norms = []
    train_k_norms = []
    synth_q_norms = []
    decode_q_norms = []
    for layer_idx in range(num_layers):
        train_q_norms.append(all_layer_q[layer_idx].float().norm(dim=-1).mean().item())
        train_k_norms.append(all_layer_k[layer_idx].float().norm(dim=-1).mean().item())
        if has_synth:
            synth_q_norms.append(synth_q[layer_idx].float().norm(dim=-1).mean().item())
        else:
            synth_q_norms.append(0.0)
        if decode_q[layer_idx].dim() == 3 and decode_q[layer_idx].size(1) > 0:
            decode_q_norms.append(decode_q[layer_idx].float().norm(dim=-1).mean().item())
        else:
            decode_q_norms.append(0.0)

    x = np.arange(num_layers)
    n_bars = 4 if has_synth else 3
    width = 0.8 / n_bars
    axes[0].bar(x - 1.5*width, train_q_norms, width, label='Train Q (prefill)', color='#e74c3c', alpha=0.8)
    axes[0].bar(x - 0.5*width, train_k_norms, width, label='Train K-as-Q', color='#2ecc71', alpha=0.8)
    if has_synth:
        axes[0].bar(x + 0.5*width, synth_q_norms, width, label='Synth Q', color='#f39c12', alpha=0.8)
        axes[0].bar(x + 1.5*width, decode_q_norms, width, label='Decode Q', color='#9b59b6', alpha=0.8)
    else:
        axes[0].bar(x + 0.5*width, decode_q_norms, width, label='Decode Q', color='#9b59b6', alpha=0.8)
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Mean L2 Norm')
    axes[0].set_title('Q/K Norm: Train vs Synth vs Decode')
    axes[0].legend(fontsize=7)
    axes[0].set_xticks(x[::2])

    # Norm ratio: decode / train, synth / train
    ratios_q = [d / max(t, 1e-8) for d, t in zip(decode_q_norms, train_q_norms)]
    ratios_k = [d / max(t, 1e-8) for d, t in zip(decode_q_norms, train_k_norms)]
    axes[1].plot(range(num_layers), ratios_q, 'o-', color='#e74c3c', label='Decode/Train_Q', markersize=4)
    axes[1].plot(range(num_layers), ratios_k, 's-', color='#2ecc71', label='Decode/Train_K', markersize=4)
    if has_synth:
        ratios_synth = [s / max(t, 1e-8) for s, t in zip(synth_q_norms, train_q_norms)]
        axes[1].plot(range(num_layers), ratios_synth, 'D-', color='#f39c12', label='Synth/Train_Q', markersize=4)
    axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Norm Ratio')
    axes[1].set_title('Norm Ratio vs Train Q')
    axes[1].legend(fontsize=7)
    axes[1].set_xticks(x[::2])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'norm_train_vs_decode.png'), dpi=150)
    plt.close()
    logger.info(f"[NeuralKV Eval] Saved norm_train_vs_decode.png")

    # ===== Plot 3: Per-layer norm histograms for selected layers =====
    n_rows = 4 if has_synth else 3
    fig, axes = plt.subplots(n_rows, n_layers_plot, figsize=(5 * n_layers_plot, 4 * n_rows))

    for col, layer_idx in enumerate(layer_indices):
        tq_n = all_layer_q[layer_idx][:, q_sample_idx, :].float().norm(dim=-1).flatten().numpy()
        tk_n = all_layer_k[layer_idx][:, k_sample_idx, :].float().norm(dim=-1).flatten().numpy()

        hist_data = [
            (tq_n, 'Train Q', '#e74c3c'),
            (tk_n, 'Train K', '#2ecc71'),
        ]
        if has_synth:
            sq_n = synth_q[layer_idx].float().norm(dim=-1).flatten().numpy()
            hist_data.append((sq_n, 'Synth Q', '#f39c12'))

        if decode_q[layer_idx].dim() == 3 and decode_q[layer_idx].size(1) > 0:
            dq_n = decode_q[layer_idx].float().norm(dim=-1).flatten().numpy()
        else:
            dq_n = np.array([0.0])
        hist_data.append((dq_n, 'Decode Q', '#9b59b6'))

        for row, (data, name, color) in enumerate(hist_data):
            axes[row, col].hist(data, bins=50, color=color, edgecolor='white', alpha=0.8)
            axes[row, col].axvline(x=np.nanmean(data), color='black', linestyle='--',
                                   label=f'mean={np.nanmean(data):.2f}')
            axes[row, col].set_title(f'Layer {layer_idx} - {name} norm', fontsize=9)
            axes[row, col].set_xlabel('L2 Norm')
            axes[row, col].set_ylabel('Count')
            axes[row, col].legend(fontsize=7)

    plt.suptitle('L2 Norm: Train Q/K vs Synth Q vs Decode Q', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'norm_hist_train_vs_decode.png'), dpi=150)
    plt.close()
    logger.info(f"[NeuralKV Eval] Saved norm_hist_train_vs_decode.png")


def evaluate_neural_kv(neural_kv, all_layer_q, all_layer_k, all_layer_v,
                       num_heads, num_heads_kv, dim_head,
                       batch_token_size=512, device='cuda',
                       save_dir='results/neural_kv_eval',
                       decode_q=None, synth_q=None,
                       position_bias=None):
    """Evaluate neural_kv using real Q from prefill (or synth Q if real Q is None).

    For each layer, sample Q tokens, compute:
      - GT: softmax(Q @ K^T / sqrt(d)) @ V  (with RoPE if position_bias provided)
      - Pred: neural_kv(Q)
    Then compute metrics and visualizations.

    Args:
        all_layer_q: list of (num_heads, n_tokens, dim_head) per layer on CPU, or None.
                     If None and synth_q is available, synth_q is used for evaluation.
        decode_q: optional list of (num_heads, n_tokens, dim_head) per layer on CPU.
                  If provided, t-SNE will compare training Q/K vs decode Q distribution.
        synth_q: optional list of (num_heads, n_tokens, dim_head) per layer on CPU.
                 If provided, shown in t-SNE as a separate color.
                 Also used as fallback eval Q when all_layer_q is None.
        position_bias: RoPE module for GT computation (optional).
    """
    os.makedirs(save_dir, exist_ok=True)

    # Determine which Q to use for evaluation
    eval_q = all_layer_q if all_layer_q is not None else synth_q
    if eval_q is None:
        logger.warning("[NeuralKV Eval] No Q available for evaluation (both all_layer_q and synth_q are None). Skipping.")
        return {
            'per_layer_mse': [], 'per_layer_cosine': [], 'per_layer_r2': [],
            'per_layer_gt_v_norm': [], 'per_layer_pred_v_norm': [],
            'mean_mse': float('nan'), 'mean_cosine': float('nan'),
            'mean_r2': float('nan'),
        }

    q_source = "real Q" if all_layer_q is not None else "synth Q"
    num_layers = len(eval_q)
    total_tokens = eval_q[0].size(1)

    # Sample a subset of tokens for evaluation (avoid OOM on full 87k tokens)
    n_eval = min(2048, total_tokens)
    eval_indices = torch.randperm(total_tokens)[:n_eval]

    per_layer_mse = []
    per_layer_cosine = []
    per_layer_r2 = []
    per_layer_gt_v_norm = []
    per_layer_pred_v_norm = []

    logger.info(f"[NeuralKV Eval] Evaluating on {n_eval} {q_source} tokens across {num_layers} layers...")

    all_gt_flat = []
    all_pred_flat = []

    for layer_idx in range(num_layers):
        q_real = eval_q[layer_idx][:, eval_indices, :].to(device).float()  # (num_heads, n_eval, dim_head)
        k = all_layer_k[layer_idx].clone().to(device).float()
        v = all_layer_v[layer_idx].clone().to(device).float()

        # GT attention output in chunks (with RoPE if available)
        gt_chunks = []
        for st in range(0, n_eval, batch_token_size):
            ed = min(st + batch_token_size, n_eval)
            with torch.no_grad():
                gt_chunk = compute_attention_output(
                    q_real[:, st:ed, :], k, v, num_heads, num_heads_kv,
                    position_bias=position_bias
                )
            gt_chunks.append(gt_chunk)
        gt = torch.cat(gt_chunks, dim=1)  # (num_heads, n_eval, dim_head)

        # Neural KV prediction
        with torch.no_grad():
            pred_chunks = []
            for st in range(0, n_eval, batch_token_size):
                ed = min(st + batch_token_size, n_eval)
                q_input = q_real[:, st:ed, :].unsqueeze(0)  # (1, num_heads, chunk, dim_head)
                pred_attn_chunk = neural_kv(layer_idx, q_input)
                pred_chunks.append(pred_attn_chunk.squeeze(0))
            pred = torch.cat(pred_chunks, dim=1)  # (num_heads, n_eval, dim_head)

        # --- Metrics ---
        mse = F.mse_loss(pred, gt).item()
        per_layer_mse.append(mse)

        gt_flat = gt.reshape(-1, dim_head)
        pred_flat = pred.reshape(-1, dim_head)
        cosine = F.cosine_similarity(pred_flat, gt_flat, dim=-1).mean().item()
        per_layer_cosine.append(cosine)

        ss_res = ((gt_flat - pred_flat) ** 2).sum().item()
        ss_tot = ((gt_flat - gt_flat.mean(dim=0, keepdim=True)) ** 2).sum().item()
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
        per_layer_r2.append(r2)

        # V norm
        gt_v_norm = gt.norm(dim=-1).mean().item()
        pred_v_norm = pred.norm(dim=-1).mean().item()
        per_layer_gt_v_norm.append(gt_v_norm)
        per_layer_pred_v_norm.append(pred_v_norm)

        if layer_idx in [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]:
            all_gt_flat.append((layer_idx, gt_flat.cpu()))
            all_pred_flat.append((layer_idx, pred_flat.cpu()))

    # --- Print summary ---
    logger.info(f"[NeuralKV Eval] ========== Evaluation Results ==========")
    logger.info(f"[NeuralKV Eval] {'Layer':>5} | {'MSE':>12} | {'Cosine':>8} | {'R²':>8} | {'GT_V_norm':>10} | {'Pred_V_norm':>11} | {'V_diff%':>8}")
    logger.info(f"[NeuralKV Eval] {'-'*5}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*11}-+-{'-'*8}")
    for i in range(num_layers):
        gt_n = per_layer_gt_v_norm[i]
        pred_n = per_layer_pred_v_norm[i]
        diff_pct = abs(pred_n - gt_n) / max(gt_n, 1e-8) * 100
        logger.info(f"[NeuralKV Eval] {i:5d} | {per_layer_mse[i]:12.6f} | {per_layer_cosine[i]:8.4f} | "
                     f"{per_layer_r2[i]:8.4f} | {gt_n:10.4f} | {pred_n:11.4f} | {diff_pct:7.2f}%")
    logger.info(f"[NeuralKV Eval] {'-'*5}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*11}-+-{'-'*8}")
    logger.info(f"[NeuralKV Eval] {'Mean':>5} | {np.mean(per_layer_mse):12.6f} | {np.mean(per_layer_cosine):8.4f} | "
                 f"{np.mean(per_layer_r2):8.4f} | {np.mean(per_layer_gt_v_norm):10.4f} | {np.mean(per_layer_pred_v_norm):11.4f} |")

    # --- Visualization ---

    # 1. Per-layer metrics bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar(range(num_layers), per_layer_mse, color='steelblue')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Per-Layer Attn Output MSE')

    axes[1].bar(range(num_layers), per_layer_cosine, color='darkorange')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].set_title('Per-Layer Cosine Similarity (Real Q)')
    axes[1].set_ylim([min(0, min(per_layer_cosine) - 0.05), 1.05])

    axes[2].bar(range(num_layers), per_layer_r2, color='forestgreen')
    axes[2].set_xlabel('Layer')
    axes[2].set_ylabel('R² Score')
    axes[2].set_title('Per-Layer R² Score (Real Q)')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_layer_metrics.png'), dpi=150)
    plt.close()
    logger.info(f"[NeuralKV Eval] Saved per_layer_metrics.png")

    # 2. Scatter plots: GT vs Pred for selected layers
    n_scatter_layers = len(all_gt_flat)
    fig, axes = plt.subplots(1, n_scatter_layers, figsize=(5 * n_scatter_layers, 5))
    if n_scatter_layers == 1:
        axes = [axes]

    for idx, ((layer_idx, gt_f), (_, pred_f)) in enumerate(zip(all_gt_flat, all_pred_flat)):
        # subsample for scatter
        n_pts = min(5000, gt_f.shape[0])
        sample_idx = torch.randperm(gt_f.shape[0])[:n_pts]
        # pick first dimension for visualization
        gt_vals = gt_f[sample_idx, 0].numpy()
        pred_vals = pred_f[sample_idx, 0].numpy()

        axes[idx].scatter(gt_vals, pred_vals, alpha=0.1, s=1, c='steelblue')
        lim_min = min(gt_vals.min(), pred_vals.min())
        lim_max = max(gt_vals.max(), pred_vals.max())
        axes[idx].plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1)
        axes[idx].set_xlabel('GT')
        axes[idx].set_ylabel('Pred')
        axes[idx].set_title(f'Layer {layer_idx} (dim=0)')
        axes[idx].set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scatter_gt_vs_pred.png'), dpi=150)
    plt.close()
    logger.info(f"[NeuralKV Eval] Saved scatter_gt_vs_pred.png")

    # 3. Histogram of per-token cosine similarity for a few layers
    fig, axes = plt.subplots(1, n_scatter_layers, figsize=(5 * n_scatter_layers, 4))
    if n_scatter_layers == 1:
        axes = [axes]

    for idx, ((layer_idx, gt_f), (_, pred_f)) in enumerate(zip(all_gt_flat, all_pred_flat)):
        cos_per_token = F.cosine_similarity(pred_f, gt_f, dim=-1).numpy()
        axes[idx].hist(cos_per_token, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        axes[idx].axvline(x=np.mean(cos_per_token), color='red', linestyle='--',
                          label=f'mean={np.mean(cos_per_token):.3f}')
        axes[idx].set_xlabel('Cosine Similarity')
        axes[idx].set_ylabel('Count')
        axes[idx].set_title(f'Layer {layer_idx}')
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cosine_histogram.png'), dpi=150)
    plt.close()
    logger.info(f"[NeuralKV Eval] Saved cosine_histogram.png")

    # 4. Heatmap: GT vs Pred for one layer, a few heads, a few tokens
    sample_layer = num_layers // 2
    q_real = eval_q[sample_layer][:, eval_indices[:64], :].to(device).float()
    k = all_layer_k[sample_layer].clone().to(device).float()
    v = all_layer_v[sample_layer].clone().to(device).float()

    with torch.no_grad():
        gt_sample = compute_attention_output(q_real, k, v, num_heads, num_heads_kv,
                                            position_bias=position_bias)  # (num_heads, 64, dim_head)
        pred_sample = neural_kv(sample_layer, q_real.unsqueeze(0))
        pred_sample = pred_sample.squeeze(0)  # (num_heads, 64, dim_head)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for h_idx in range(min(4, num_heads)):
        gt_h = gt_sample[h_idx, :32, :32].cpu().numpy()
        pred_h = pred_sample[h_idx, :32, :32].cpu().numpy()

        vmin = min(gt_h.min(), pred_h.min())
        vmax = max(gt_h.max(), pred_h.max())

        axes[0, h_idx].imshow(gt_h, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[0, h_idx].set_title(f'GT Head {h_idx}')
        axes[0, h_idx].set_xlabel('dim')
        axes[0, h_idx].set_ylabel('token')

        axes[1, h_idx].imshow(pred_h, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1, h_idx].set_title(f'Pred Head {h_idx}')
        axes[1, h_idx].set_xlabel('dim')
        axes[1, h_idx].set_ylabel('token')

    plt.suptitle(f'Layer {sample_layer}: GT vs Pred (32 tokens x 32 dims)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'heatmap_gt_vs_pred.png'), dpi=150)
    plt.close()
    logger.info(f"[NeuralKV Eval] Saved heatmap_gt_vs_pred.png")

    # 5. GT V norm vs Pred V norm bar chart
    x = np.arange(num_layers)
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(14, num_layers * 0.5), 5))
    ax.bar(x - width / 2, per_layer_gt_v_norm, width, label='GT V norm', color='#2ecc71', alpha=0.8)
    ax.bar(x + width / 2, per_layer_pred_v_norm, width, label='Pred V norm', color='#e74c3c', alpha=0.8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean L2 Norm')
    ax.set_title('Per-Layer Attention Output (V) Norm: GT vs Pred')
    ax.legend()
    ax.set_xticks(x[::2])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'v_norm_gt_vs_pred.png'), dpi=150)
    plt.close()
    logger.info(f"[NeuralKV Eval] Saved v_norm_gt_vs_pred.png")

    # 6. PCA, norm, t-SNE visualization of Q, K, V
    logger.info(f"[NeuralKV Eval] Running PCA/norm/t-SNE visualization...")
    visualize_qkv_pca(
        eval_q, all_layer_k, all_layer_v,
        num_heads, num_heads_kv, dim_head,
        save_dir=save_dir,
    )

    # 7. Train Q/K vs Synth Q vs Decode Q distribution comparison (t-SNE)
    if decode_q is not None:
        logger.info(f"[NeuralKV Eval] Visualizing train vs decode Q distribution...")
        visualize_train_vs_decode_q(
            eval_q, all_layer_k, decode_q,
            num_heads, num_heads_kv,
            save_dir=save_dir,
            synth_q=synth_q,
        )

    logger.info(f"[NeuralKV Eval] All visualizations saved to {save_dir}")

    return {
        'per_layer_mse': per_layer_mse,
        'per_layer_cosine': per_layer_cosine,
        'per_layer_r2': per_layer_r2,
        'per_layer_gt_v_norm': per_layer_gt_v_norm,
        'per_layer_pred_v_norm': per_layer_pred_v_norm,
        'mean_mse': float(np.mean(per_layer_mse)),
        'mean_cosine': float(np.mean(per_layer_cosine)),
        'mean_r2': float(np.mean(per_layer_r2)),
    }


def collect_all_kv(qa_model):
    """Extract all K, V from all layers of the model's KV cache.

    Returns:
        all_layer_k: list of (num_heads_kv, total_tokens, dim_head) on CPU
        all_layer_v: list of (num_heads_kv, total_tokens, dim_head) on CPU
    """
    kv_cache = qa_model.kv_cache
    num_layers = len(kv_cache)

    all_layer_k = []
    all_layer_v = []
    for layer_idx in range(num_layers):
        ctx = kv_cache[layer_idx]
        k, v = collect_all_kv_from_context_manager(ctx)
        all_layer_k.append(k)
        all_layer_v.append(v)
        if layer_idx == 0:
            logger.info(f"[NeuralKV] Layer 0: K shape={k.shape}, V shape={v.shape}")

    return all_layer_k, all_layer_v


def train_neural_kv(qa_model, all_layer_q=None, num_epochs=100, lr=1e-3,
                    batch_token_size=512, device='cuda',
                    synth_q=None, synth_targets=None,
                    all_layer_k=None, all_layer_v=None,
                    position_bias=None, synth_only=False,
                    group_shared=True):
    """Train video-specific neural KV cache after video prefill.

    Training data consists of up to THREE sources:
    1. K-as-Q: use each token's K (expanded to num_heads) as query
    2. Real Q: use the actual Q from prefill (collected via hooks)
    3. Synth Q: synthesized decode-like Q from common question templates

    When synth_only=True, only Synth Q (source 3) is used for training.
    K-as-Q and Real Q are skipped entirely.

    GT targets are computed with RoPE applied to Q and K (if position_bias is provided),
    so that the MLP learns to predict the RoPE-aware attention output.
    RoPE is also applied to Q before feeding to the MLP, so input and target are consistent.

    Args:
        qa_model: the ReKV model after encode_video
        all_layer_q: list of real Q per layer, each (num_heads, total_tokens, dim_head) on CPU.
        synth_q: list of synthesized Q per layer on CPU (from synthesize_decode_q)
        synth_targets: list of GT attention outputs for synth Q on CPU
        all_layer_k: pre-collected K per layer (if None, collected from qa_model)
        all_layer_v: pre-collected V per layer (if None, collected from qa_model)
        num_epochs: training epochs
        lr: learning rate
        batch_token_size: number of query tokens per mini-batch
        device: training device
        position_bias: RoPE module for computing GT with RoPE
        synth_only: if True, only train with synthetic Q (skip K-as-Q and real Q)
        group_shared: if True, use one MLP per GQA group; otherwise one MLP per head
    """
    kv_cache = qa_model.kv_cache
    num_layers = len(kv_cache)

    ctx0 = kv_cache[0]
    num_heads = ctx0.num_heads
    num_heads_kv = ctx0.num_heads_kv
    dim_head = ctx0.dim_head

    logger.info(f"[NeuralKV] num_layers={num_layers}, num_heads={num_heads}, "
                f"num_heads_kv={num_heads_kv}, dim_head={dim_head}")

    # Step 1: Collect all KV from each layer (or use pre-collected)
    if all_layer_k is None or all_layer_v is None:
        logger.info("[NeuralKV] Collecting KV from all layers...")
        all_layer_k, all_layer_v = collect_all_kv(qa_model)
    else:
        logger.info("[NeuralKV] Using pre-collected KV.")

    total_tokens = all_layer_k[0].size(1)
    use_k_as_q = not synth_only
    use_real_q = (all_layer_q is not None) and (not synth_only)
    use_synth = synth_q is not None and synth_targets is not None
    logger.info(f"[NeuralKV] Total tokens per layer: {total_tokens}, "
                f"synth_only: {synth_only}, use_k_as_q: {use_k_as_q}, "
                f"use_real_q: {use_real_q}, use_synth: {use_synth}")
    if use_synth:
        total_synth_tokens = synth_q[0].size(1)
        logger.info(f"[NeuralKV] Synth Q tokens per layer: {total_synth_tokens}")

    with torch.inference_mode(False):

        # Step 2: Create neural KV model
        neural_kv = NeuralKVCache(
            num_layers=num_layers,
            num_heads=num_heads,
            num_heads_kv=num_heads_kv,
            dim_head=dim_head,
            group_shared=group_shared,
        ).to(device).float()

        param_count = sum(p.numel() for p in neural_kv.parameters())
        logger.info(f"[NeuralKV] Model parameters: {param_count:,}")

        optimizer = torch.optim.Adam(neural_kv.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # Step 3: Precompute ground-truth attention outputs (with RoPE if available)
        use_rope = position_bias is not None
        if use_rope:
            logger.info("[NeuralKV] RoPE enabled for GT target computation.")
        else:
            logger.info("[NeuralKV] No RoPE — GT targets computed without positional encoding.")

        # 3a: K-as-Q targets
        all_layer_k_targets = []
        if use_k_as_q:
            logger.info("[NeuralKV] Precomputing K-as-Q ground-truth...")
            for layer_idx in range(num_layers):
                k = all_layer_k[layer_idx].clone()
                v = all_layer_v[layer_idx].clone()

                if num_heads != num_heads_kv:
                    num_group = num_heads // num_heads_kv
                    q_from_k = k.unsqueeze(1).expand(-1, num_group, -1, -1)
                    q_from_k = q_from_k.reshape(num_heads, total_tokens, dim_head)
                else:
                    q_from_k = k

                target_chunks = []
                for st in range(0, total_tokens, batch_token_size):
                    ed = min(st + batch_token_size, total_tokens)
                    q_chunk = q_from_k[:, st:ed, :].to(device).float()
                    k_gpu = k.to(device).float()
                    v_gpu = v.to(device).float()

                    with torch.no_grad():
                        target_chunk = compute_attention_output(
                            q_chunk, k_gpu, v_gpu, num_heads, num_heads_kv,
                            position_bias=position_bias
                        )
                    target_chunks.append(target_chunk.cpu())

                target = torch.cat(target_chunks, dim=1)
                all_layer_k_targets.append(target)

                if layer_idx == 0:
                    logger.info(f"[NeuralKV] Layer 0 K-as-Q target shape: {target.shape}")
        else:
            logger.info("[NeuralKV] Skipping K-as-Q ground-truth (synth_only mode).")

        # 3b: Real Q targets (if available)
        all_layer_q_targets = []
        if use_real_q:
            logger.info("[NeuralKV] Precomputing real-Q ground-truth...")
            total_q_tokens = all_layer_q[0].size(1)
            for layer_idx in range(num_layers):
                q_real = all_layer_q[layer_idx]  # (num_heads, total_q_tokens, dim_head)
                k = all_layer_k[layer_idx].clone()
                v = all_layer_v[layer_idx].clone()

                target_chunks = []
                for st in range(0, total_q_tokens, batch_token_size):
                    ed = min(st + batch_token_size, total_q_tokens)
                    q_chunk = q_real[:, st:ed, :].to(device).float()
                    k_gpu = k.to(device).float()
                    v_gpu = v.to(device).float()

                    with torch.no_grad():
                        target_chunk = compute_attention_output(
                            q_chunk, k_gpu, v_gpu, num_heads, num_heads_kv,
                            position_bias=position_bias
                        )
                    target_chunks.append(target_chunk.cpu())

                target = torch.cat(target_chunks, dim=1)
                all_layer_q_targets.append(target)

                if layer_idx == 0:
                    logger.info(f"[NeuralKV] Layer 0 real-Q target shape: {target.shape}")

        # Step 4: Pre-move all data to GPU to avoid repeated CPU→GPU transfers
        logger.info("[NeuralKV] Moving training data to GPU...")

        # K-as-Q: precompute expanded Q from K for all layers, stack into (num_layers, H, T, D)
        if use_k_as_q:
            all_k_q_gpu = []
            all_k_target_gpu = []
            for layer_idx in range(num_layers):
                k_l = all_layer_k[layer_idx]
                if num_heads != num_heads_kv:
                    num_group = num_heads // num_heads_kv
                    q_from_k = k_l.unsqueeze(1).expand(-1, num_group, -1, -1)
                    q_from_k = q_from_k.reshape(num_heads, total_tokens, dim_head)
                else:
                    q_from_k = k_l
                all_k_q_gpu.append(q_from_k.to(device).float())
                all_k_target_gpu.append(all_layer_k_targets[layer_idx].to(device).float())
            # Stack: (num_layers, H, T, D)
            k_q_stacked = torch.stack(all_k_q_gpu, dim=0)
            k_tgt_stacked = torch.stack(all_k_target_gpu, dim=0)
            del all_k_q_gpu, all_k_target_gpu, all_layer_k_targets
        else:
            k_q_stacked = k_tgt_stacked = None

        # Real Q
        if use_real_q:
            rq_stacked = torch.stack([q.to(device).float() for q in all_layer_q], dim=0)
            rt_stacked = torch.stack([t.to(device).float() for t in all_layer_q_targets], dim=0)
            del all_layer_q_targets
        else:
            rq_stacked = rt_stacked = None

        # Synth Q
        if use_synth:
            sq_stacked = torch.stack([q.to(device).float() for q in synth_q], dim=0)
            st_stacked = torch.stack([t.to(device).float() for t in synth_targets], dim=0)
            if synth_only:
                n_synth_repeats = 1  # no oversampling needed when synth is the only source
            else:
                n_synth_repeats = max(1, total_tokens // max(total_synth_tokens, 1))
                n_synth_repeats = min(n_synth_repeats, 5)
        else:
            sq_stacked = st_stacked = None
            n_synth_repeats = 0

        def _train_batch(q_all_layers, tgt_all_layers, batch_indices,
                         kv_len_for_rope=None):
            """Train one batch across all layers. q/tgt: (num_layers, H, T, D) on GPU.
            RoPE is applied to Q before feeding to the MLP.
            kv_len_for_rope: total KV length for position assignment. Q gets
                positions [kv_len - batch_size, kv_len), matching inference."""
            q_batch = q_all_layers[:, :, batch_indices, :]   # (L, H, B, D)
            t_batch = tgt_all_layers[:, :, batch_indices, :] # (L, H, B, D)
            # Apply RoPE to Q if available
            if use_rope:
                q_batch_rope = torch.stack([
                    _apply_rope_to_q(q_batch[li], position_bias, kv_len_for_rope)
                    for li in range(num_layers)
                ], dim=0)
            else:
                q_batch_rope = q_batch
            layer_loss = 0.0
            for li in range(num_layers):
                pred = neural_kv.layers[li](q_batch_rope[li].unsqueeze(0)).squeeze(0)
                layer_loss = layer_loss + F.mse_loss(pred, t_batch[li])
            layer_loss = layer_loss / num_layers
            optimizer.zero_grad()
            layer_loss.backward()
            optimizer.step()
            return layer_loss.item()

        # Step 5: Training loop
        logger.info(f"[NeuralKV] Training {num_layers} layers for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0

            # --- Phase A: K-as-Q ---
            # K-as-Q: Q comes from K, both have total_tokens length.
            # position_bias.forward gives K:[0,T), Q:[0,T) since len_q==len_k.
            # So kv_len_for_rope = total_tokens.
            if use_k_as_q:
                perm_k = torch.randperm(total_tokens, device=device)
                for st in range(0, total_tokens, batch_token_size):
                    ed = min(st + batch_token_size, total_tokens)
                    total_loss += _train_batch(k_q_stacked, k_tgt_stacked,
                                               perm_k[st:ed], total_tokens)
                    num_batches += 1

            # --- Phase B: Real Q ---
            # Real Q from prefill also has total_tokens (== total_kv) length.
            if use_real_q:
                perm_q = torch.randperm(total_q_tokens, device=device)
                for st in range(0, total_q_tokens, batch_token_size):
                    ed = min(st + batch_token_size, total_q_tokens)
                    total_loss += _train_batch(rq_stacked, rt_stacked,
                                               perm_q[st:ed], total_tokens)
                    num_batches += 1

            # --- Phase C: Synth Q ---
            # Synth Q may have more tokens than K (len_q > len_k).
            # GT uses padding: K:[0,len_k), Q:[0,len_q). So kv_len=None
            # (falls back to [0, len_q) in _apply_rope_to_q).
            if use_synth:
                for _rep in range(n_synth_repeats):
                    perm_s = torch.randperm(total_synth_tokens, device=device)
                    for st in range(0, total_synth_tokens, batch_token_size):
                        ed = min(st + batch_token_size, total_synth_tokens)
                        total_loss += _train_batch(sq_stacked, st_stacked,
                                                   perm_s[st:ed], None)
                        num_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(num_batches, 1)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"[NeuralKV] Epoch {epoch+1}/{num_epochs}, loss={avg_loss:.6f}, "
                            f"lr={scheduler.get_last_lr()[0]:.6f}")

        # Free GPU training data
        del k_q_stacked, k_tgt_stacked, rq_stacked, rt_stacked, sq_stacked, st_stacked

    # Store position_bias and total_kv_len for inference-time RoPE on Q
    neural_kv.position_bias = position_bias
    neural_kv.total_kv_len = total_tokens

    logger.info("[NeuralKV] Training complete.")
    return neural_kv, all_layer_k, all_layer_v
