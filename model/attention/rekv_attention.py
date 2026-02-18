import copy
import math
import torch
import torch.nn.functional as F
from typing import Optional

from .kv_cache_manager import ContextManager
from .dot_production_attention import get_multi_stage_dot_production_attention


# Global list to accumulate per-step error metrics during Neural KV decoding
_NEURAL_KV_ERROR_LOG = []

# Global list to accumulate video attention weight in original (non-NeuralKV) path
_ORIGINAL_W_VIDEO_LOG = []

# Global list to accumulate 3-part attention scores (system/video/question)
_ATTN_SCORE_3PART_LOG = []


def get_neural_kv_error_log():
    return _NEURAL_KV_ERROR_LOG


def clear_neural_kv_error_log():
    _NEURAL_KV_ERROR_LOG.clear()


def get_original_w_video_log():
    return _ORIGINAL_W_VIDEO_LOG


def clear_original_w_video_log():
    _ORIGINAL_W_VIDEO_LOG.clear()


def get_attn_score_3part_log():
    return _ATTN_SCORE_3PART_LOG


def clear_attn_score_3part_log():
    _ATTN_SCORE_3PART_LOG.clear()


def _compute_gt_video_attention(h_q, all_k, all_v, num_heads, num_heads_kv,
                                position_bias=None):
    """Compute ground-truth attention output and LSE of Q attending to ALL video KV.

    All computation in float32 to avoid fp16 overflow on layers with large Q/K norms.
    If position_bias is provided, RoPE is applied to Q and K for consistency with text LSE.

    Args:
        h_q: (batch, num_heads, len_q, dim_head)
        all_k: (num_heads_kv, total_video_tokens, dim_head) on CPU
        all_v: (num_heads_kv, total_video_tokens, dim_head) on CPU
        position_bias: RoPE module (optional)

    Returns:
        gt_video_attn: (batch, num_heads, len_q, dim_head)
        gt_lse: (batch, num_heads, len_q)
    """
    device = h_q.device
    dim_head = h_q.size(-1)

    k = all_k.to(device).float()
    v = all_v.to(device).float()

    if num_heads != num_heads_kv:
        num_group = num_heads // num_heads_kv
        k = k.unsqueeze(1).expand(-1, num_group, -1, -1).reshape(num_heads, -1, dim_head)
        v = v.unsqueeze(1).expand(-1, num_group, -1, -1).reshape(num_heads, -1, dim_head)

    q = h_q[0].float()  # (num_heads, len_q, dim_head)

    # Apply RoPE if provided
    if position_bias is not None:
        # q: (num_heads, len_q, dim_head) -> (1, num_heads, len_q, dim_head)
        # k: (num_heads, total_kv, dim_head) -> (1, num_heads, total_kv, dim_head)
        q_4d = q.unsqueeze(0)
        k_4d = k.unsqueeze(0)
        position_bias._seq_len_cached = -1  # Force cache rebuild for correct dim
        q_rope, k_rope = position_bias(q_4d, k_4d)
        q = q_rope[0].float()
        k = k_rope[0].float()

    scale = 1.0 / math.sqrt(dim_head)
    logits = torch.matmul(q, k.transpose(-1, -2)) * scale  # (num_heads, len_q, T)
    lse = torch.logsumexp(logits, dim=-1)  # (num_heads, len_q)
    attn_weights = F.softmax(logits, dim=-1)
    gt = torch.matmul(attn_weights, v)  # (num_heads, len_q, dim_head)

    return gt.unsqueeze(0), lse.unsqueeze(0)  # (1, num_heads, len_q, dim_head), (1, num_heads, len_q)


def _compute_video_lse(h_q, all_k, num_heads, num_heads_kv, position_bias=None):
    """Compute only the LSE of Q attending to ALL video KV (no V needed).

    All computation in float32 to avoid fp16 overflow.
    If position_bias is provided, RoPE is applied to Q and K for consistency with text LSE.

    Args:
        h_q: (batch, num_heads, len_q, dim_head)
        all_k: (num_heads_kv, total_video_tokens, dim_head) on CPU
        position_bias: RoPE module (optional)

    Returns:
        video_lse: (batch, num_heads, len_q) in float32
    """
    device = h_q.device
    dim_head = h_q.size(-1)

    k = all_k.to(device).float()

    if num_heads != num_heads_kv:
        num_group = num_heads // num_heads_kv
        k = k.unsqueeze(1).expand(-1, num_group, -1, -1).reshape(num_heads, -1, dim_head)

    q = h_q[0].float()  # (num_heads, len_q, dim_head)

    # Apply RoPE if provided
    if position_bias is not None:
        q_4d = q.unsqueeze(0)
        k_4d = k.unsqueeze(0)
        position_bias._seq_len_cached = -1  # Force cache rebuild for correct dim
        q_rope, k_rope = position_bias(q_4d, k_4d)
        q = q_rope[0].float()
        k = k_rope[0].float()

    scale = 1.0 / math.sqrt(dim_head)
    logits = torch.matmul(q, k.transpose(-1, -2)) * scale  # (num_heads, len_q, T)
    lse = torch.logsumexp(logits, dim=-1)  # (num_heads, len_q)

    return lse.unsqueeze(0)  # (1, num_heads, len_q)


def _lse_fusion(o1, lse1, o2, lse2):
    """Exact softmax fusion using log-sum-exp.

    All computation in float32 to avoid overflow/nan.

    Given two attention outputs and their LSEs:
        o1 = softmax(s1) @ V1,  lse1 = log(sum(exp(s1)))
        o2 = softmax(s2) @ V2,  lse2 = log(sum(exp(s2)))

    The exact joint attention output is:
        o = w1 * o1 + w2 * o2
    where w1 = exp(lse1) / (exp(lse1) + exp(lse2)) = sigmoid(lse1 - lse2)

    Args:
        o1: (batch, num_heads, len_q, dim_head) - neural KV output
        lse1: (batch, num_heads, len_q) - video LSE
        o2: (batch, num_heads, len_q, dim_head) - text attention output
        lse2: (batch, num_heads, len_q) - text attention LSE
    Returns:
        fused: (batch, num_heads, len_q, dim_head)
    """
    orig_dtype = o1.dtype
    w1 = torch.sigmoid((lse1.float() - lse2.float())).unsqueeze(-1)  # (batch, num_heads, len_q, 1)
    w2 = 1.0 - w1
    fused = w1 * o1.float() + w2 * o2.float()
    return fused.to(orig_dtype)


def _compute_3part_attn_scores(
    h_q, h_k, h_v, position_bias, n_init, n_local,
    num_heads, num_heads_kv, dim_head, len_q, len_k,
    past_k_len, layer_idx
):
    """Compute attention weight distribution over 3 parts: system prompt, video, question.

    After retrieval, h_k = [init(n_init), video(past_k_len - n_init), question(len_q)].
    We compute full attention weights (with RoPE + causal mask) and sum them per segment.

    All computation in float32.
    """
    batch_size = h_q.size(0)
    n_video = past_k_len - n_init
    n_question = len_q

    # Apply RoPE
    h_q_rope, h_k_rope = position_bias(h_q, h_k)

    # Expand KV heads for GQA
    if num_heads != num_heads_kv:
        num_group = num_heads // num_heads_kv
        h_k_rope = h_k_rope[:, :, None, :, :].expand(
            batch_size, num_heads_kv, num_group, len_k, dim_head
        ).reshape(batch_size, num_heads, len_k, dim_head)

    scale = 1.0 / math.sqrt(dim_head)
    logits = torch.matmul(h_q_rope.float(), h_k_rope.float().transpose(-1, -2)) * scale

    # Causal mask
    if len_k > len_q:
        dist = torch.arange(len_q, device=h_q.device)[:, None] - \
               torch.arange(len_k, device=h_q.device)[None, :] + (len_k - len_q)
        mask = (dist >= 0)
        logits = logits.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    else:
        causal_mask = torch.triu(
            torch.ones(len_q, len_k, device=h_q.device, dtype=torch.bool), diagonal=1
        )
        logits = logits.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    attn_weights = F.softmax(logits, dim=-1)  # (B, H, Lq, Lk)

    # Partition: [0, n_init) = system, [n_init, n_init+n_video) = video, [n_init+n_video, end) = question
    w_system = attn_weights[:, :, :, :n_init].sum(dim=-1).mean().item()
    w_video = attn_weights[:, :, :, n_init:n_init + n_video].sum(dim=-1).mean().item()
    w_question = attn_weights[:, :, :, n_init + n_video:].sum(dim=-1).mean().item()

    _ATTN_SCORE_3PART_LOG.append({
        'layer_idx': layer_idx,
        'w_system': w_system,
        'w_video': w_video,
        'w_question': w_question,
        'n_system': n_init,
        'n_video': n_video,
        'n_question': n_question,
    })


def _compute_text_attention_with_lse(h_q, h_k, h_v, position_bias, n_local, Attn, num_heads, dim_head):
    """Compute text-only attention output AND its LSE.

    Logits/softmax computed in float32 to avoid overflow.

    Args:
        h_q: (batch, num_heads, len_q, dim_head)
        h_k: (batch, num_heads_kv, len_k, dim_head)
        h_v: (batch, num_heads_kv, len_k, dim_head)
    Returns:
        text_out: (batch, num_heads, len_q, dim_head) in original dtype
        text_lse: (batch, num_heads, len_q) in float32
    """
    batch_size = h_q.size(0)
    num_heads_kv = h_k.size(1)
    len_q = h_q.size(2)
    len_k = h_k.size(2)

    # Apply RoPE (in original dtype)
    h_q_rope, h_k_rope = position_bias(h_q, h_k)

    # Expand KV heads for GQA
    if num_heads != num_heads_kv:
        num_group = num_heads // num_heads_kv
        h_k_rope = h_k_rope[:, :, None, :, :].expand(
            batch_size, num_heads_kv, num_group, len_k, dim_head
        ).reshape(batch_size, num_heads, len_k, dim_head)
        h_v_exp = h_v[:, :, None, :, :].expand(
            batch_size, num_heads_kv, num_group, len_k, dim_head
        ).reshape(batch_size, num_heads, len_k, dim_head)
    else:
        h_v_exp = h_v

    scale = 1.0 / math.sqrt(dim_head)
    # Compute logits in float32
    logits = torch.matmul(h_q_rope.float(), h_k_rope.float().transpose(-1, -2)) * scale

    # Causal / sliding window mask
    if len_k > len_q:
        dist = torch.arange(len_q, device=h_q.device)[:, None] - \
               torch.arange(len_k, device=h_q.device)[None, :] + (len_k - len_q)
        mask = (dist >= 0) & (dist < n_local)
        logits = logits.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    else:
        causal_mask = torch.triu(
            torch.ones(len_q, len_k, device=h_q.device, dtype=torch.bool), diagonal=1
        )
        logits = logits.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    lse = torch.logsumexp(logits, dim=-1)  # (B, H, Lq) float32
    attn_weights = F.softmax(logits, dim=-1)  # float32
    text_out = torch.matmul(attn_weights, h_v_exp.float())  # (B, H, Lq, D) float32
    text_out = text_out.to(h_q.dtype)

    return text_out, lse  # text_out in original dtype, lse in float32


def rekv_attention_forward(
    n_local, n_init, topk, chunk_size,
    block_size, max_cached_block,
    exc_block_size, fattn,
    async_global_stream=True,
    pin_memory=False,
    *args, **kwargs
):
    Attn, _ = get_multi_stage_dot_production_attention(fattn)
    def forward(self, query : torch.Tensor,
                    key_value : torch.Tensor,
                    position_bias : Optional[torch.Tensor],
                    use_cache: bool,
                    past_key_value,
                    project_q, project_k, project_v, attention_out, 
                    dim_head, num_heads, num_heads_kv,
    ):

        """ 1. Project QKV """
        batch_size = query.size(0)
        len_q = query.size(1)
        len_k = key_value.size(1)

        assert use_cache

        h_q = project_q(query)             # (batch, len_q, num_heads * dim_head)
        h_k = project_k(key_value)         # (batch, len_k, num_heads * dim_head)
        h_v = project_v(key_value)         # (batch, len_k, num_heads * dim_head)

        h_q = h_q.view(batch_size, len_q, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()      # (batch, num_heads, len_q, dim_head)
        h_k = h_k.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)
        h_v = h_v.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)

        if position_bias._cos_cached is not None and position_bias._cos_cached.device != h_q.device:
            position_bias = copy.deepcopy(position_bias)
            if position_bias.inv_freq.device != h_q.device:
                position_bias.inv_freq = position_bias.inv_freq.to(h_q.device)
            if position_bias._cos_cached is not None:
                position_bias._cos_cached = position_bias._cos_cached.to(h_q.device)
            if position_bias._sin_cached is not None:
                position_bias._sin_cached = position_bias._sin_cached.to(h_q.device)

        if past_key_value is None:
            past_key_value = ContextManager(
                position_bias,
                n_init, n_local, 
                block_size, max_cached_block, topk, chunk_size, exc_block_size,
                fattn,
                async_global_stream,
                pin_memory,
            )

        local_q, local_k, local_v = h_q, h_k, h_v
        global_q, global_k, global_v = h_q, h_k, h_v

        # NOTE: Question-answering, fall back to sliding-window attention (infinite_lm)
        if type(past_key_value) is not ContextManager or past_key_value.to_retrieve:
            if type(past_key_value) is ContextManager:  # retrieval
                # ===== Check for Neural KV mode =====
                neural_kv_cache = getattr(past_key_value, '_neural_kv_cache', None)
                all_video_k = getattr(past_key_value, '_all_video_k', None)
                all_video_v = getattr(past_key_value, '_all_video_v', None)
                layer_idx = getattr(past_key_value, '_layer_idx', None)

                use_neural_kv = (neural_kv_cache is not None and layer_idx is not None)

                if use_neural_kv:
                    # ===== Neural KV path: predicted V + real GT LSE for fusion =====
                    # Neural KV predicts: attn_out_video (V only)
                    # Real GT LSE is computed on-the-fly from actual video K
                    # Text attention computes: (attn_out_text, lse_text)
                    # Exact fusion: w1 = sigmoid(gt_video_lse - text_lse)
                    #   output = w1 * neural_video_out + (1 - w1) * text_out

                    # 1. Neural KV prediction (V only)
                    neural_video_out = neural_kv_cache(layer_idx, h_q.float())
                    neural_video_out = neural_video_out.to(h_q.dtype)

                    # 2. Compute real GT video LSE from actual video K (float32, with RoPE)
                    with torch.no_grad():
                        video_lse = _compute_video_lse(
                            h_q, all_video_k, num_heads, num_heads_kv,
                            position_bias=position_bias
                        )

                    # 3. GT attention for error tracking (with RoPE)
                    if all_video_k is not None and all_video_v is not None:
                        with torch.no_grad():
                            gt_video_out, gt_lse = _compute_gt_video_attention(
                                h_q, all_video_k, all_video_v, num_heads, num_heads_kv,
                                position_bias=position_bias
                            )
                            mse = F.mse_loss(neural_video_out.float(), gt_video_out.float()).item()
                            cos = F.cosine_similarity(
                                neural_video_out.float().reshape(-1, dim_head),
                                gt_video_out.float().reshape(-1, dim_head),
                                dim=-1
                            ).mean().item()
                            _NEURAL_KV_ERROR_LOG.append({
                                'layer_idx': layer_idx,
                                'step': 'retrieval',
                                'len_q': len_q,
                                'mse': mse,
                                'cosine': cos,
                            })

                    # 4. Text-only attention with LSE (lse in float32)
                    if h_k.size(-2) > 0:
                        text_out, text_lse = _compute_text_attention_with_lse(
                            h_q, h_k, h_v, position_bias, n_local, Attn,
                            num_heads, dim_head
                        )
                    else:
                        text_out = torch.zeros_like(neural_video_out)
                        text_lse = torch.full(
                            (batch_size, num_heads, len_q), float('-inf'),
                            device=h_q.device, dtype=torch.float32
                        )

                    # 5. Exact LSE-based fusion using real video LSE (all float32 internally)
                    score = _lse_fusion(neural_video_out, video_lse, text_out, text_lse)

                    # Log video/text weight ratio
                    with torch.no_grad():
                        w_video = torch.sigmoid(video_lse.float() - text_lse.float())
                        w_video_mean = w_video.mean().item()
                        if _NEURAL_KV_ERROR_LOG and _NEURAL_KV_ERROR_LOG[-1]['layer_idx'] == layer_idx:
                            _NEURAL_KV_ERROR_LOG[-1]['gt_w_video'] = w_video_mean

                    score = score.permute(0, 2, 1, 3)  # (batch, len_q, num_heads, dim_head)
                    score = score.reshape(batch_size, len_q, num_heads * dim_head)
                    score = attention_out(score)

                    current_key_value = (h_k, h_v, neural_kv_cache, all_video_k, all_video_v, layer_idx)
                    return score, current_key_value

                else:
                    # ===== Original retrieval path =====
                    if past_key_value.retrieved_block_indices is None:
                        past_k, past_v = past_key_value.get_retrieved_kv(global_q)
                    else:
                        past_k, past_v = past_key_value.get_retrieved_kv()
                    updata_kv_cache = False

                    # Log GT w_video if all_video_k is attached (for comparison)
                    _orig_all_video_k = getattr(past_key_value, '_all_video_k', None)
                    _orig_layer_idx = getattr(past_key_value, '_layer_idx', None)
                    if _orig_all_video_k is not None and _orig_layer_idx is not None:
                        with torch.no_grad():
                            _vid_lse = _compute_video_lse(
                                h_q, _orig_all_video_k, num_heads, num_heads_kv,
                                position_bias=position_bias
                            )
                            _txt_lse = _compute_text_attention_with_lse(
                                h_q, h_k, h_v, position_bias, n_local, Attn,
                                num_heads, dim_head
                            )[1]
                            _w_vid = torch.sigmoid(_vid_lse.float() - _txt_lse.float()).mean().item()
                            _ORIGINAL_W_VIDEO_LOG.append({
                                'layer_idx': _orig_layer_idx,
                                'step': 'retrieval',
                                'w_video': _w_vid,
                            })

            else:
                # sliding-window attention (prompt prefill + decode)
                # Check if this is a Neural KV extended tuple
                if isinstance(past_key_value, tuple) and len(past_key_value) == 6:
                    past_k, past_v, neural_kv_cache, all_video_k, all_video_v, layer_idx = past_key_value

                    # Neural KV: predicted V only
                    neural_video_out = neural_kv_cache(layer_idx, h_q.float())
                    neural_video_out = neural_video_out.to(h_q.dtype)

                    # Compute real GT video LSE from actual video K (float32, with RoPE)
                    with torch.no_grad():
                        video_lse = _compute_video_lse(
                            h_q, all_video_k, num_heads, num_heads_kv,
                            position_bias=position_bias
                        )

                    # GT attention for error tracking (with RoPE)
                    if all_video_k is not None and all_video_v is not None:
                        with torch.no_grad():
                            gt_video_out, gt_lse = _compute_gt_video_attention(
                                h_q, all_video_k, all_video_v, num_heads, num_heads_kv,
                                position_bias=position_bias
                            )
                            mse = F.mse_loss(neural_video_out.float(), gt_video_out.float()).item()
                            cos = F.cosine_similarity(
                                neural_video_out.float().reshape(-1, dim_head),
                                gt_video_out.float().reshape(-1, dim_head),
                                dim=-1
                            ).mean().item()
                            _NEURAL_KV_ERROR_LOG.append({
                                'layer_idx': layer_idx,
                                'step': 'decode',
                                'len_q': len_q,
                                'mse': mse,
                                'cosine': cos,
                            })

                    # Text-only attention with LSE
                    h_k_text = torch.cat([past_k, h_k], dim=-2)
                    h_v_text = torch.cat([past_v, h_v], dim=-2)
                    len_k_text = h_k_text.size(-2)

                    # Sliding window cache update
                    if len_k_text <= n_local:
                        h_k_cache = h_k_text
                        h_v_cache = h_v_text
                    else:
                        h_k_cache = h_k_text[:, :, max(0, h_k_text.size(-2) - n_local):, :]
                        h_v_cache = h_v_text[:, :, max(0, h_v_text.size(-2) - n_local):, :]

                    # Text attention with LSE
                    text_out, text_lse = _compute_text_attention_with_lse(
                        h_q, h_k_text, h_v_text, position_bias, n_local, Attn,
                        num_heads, dim_head
                    )

                    # Exact LSE-based fusion using real video LSE
                    score = _lse_fusion(neural_video_out, video_lse, text_out, text_lse)

                    # Log video/text weight ratio
                    with torch.no_grad():
                        w_video = torch.sigmoid(video_lse.float() - text_lse.float())
                        w_video_mean = w_video.mean().item()
                        if _NEURAL_KV_ERROR_LOG and _NEURAL_KV_ERROR_LOG[-1]['layer_idx'] == layer_idx:
                            _NEURAL_KV_ERROR_LOG[-1]['gt_w_video'] = w_video_mean

                    score = score.permute(0, 2, 1, 3)  # (batch, len_q, num_heads, dim_head)
                    score = score.reshape(batch_size, len_q, num_heads * dim_head)
                    score = attention_out(score)

                    current_key_value = (h_k_cache, h_v_cache, neural_kv_cache, all_video_k, all_video_v, layer_idx)
                    return score, current_key_value

                else:
                    # Standard sliding-window (original path)
                    past_k = past_key_value[0]
                    past_v = past_key_value[1]
                    updata_kv_cache = True

            """ 2. Update KV w/ past KV cache """
            h_k = torch.cat([past_k, h_k], dim=-2)
            h_v = torch.cat([past_v, h_v], dim=-2)
            len_k += past_k.shape[2]

            """ 3. Update KV cache """
            if updata_kv_cache:
                if len_k <= n_local + n_init:
                    h_k_cache = h_k
                    h_v_cache = h_v
                else:
                    h_k_cache = torch.cat([h_k[:,:, :n_init, :], h_k[:, :, max(0, h_k.size(-2) - n_local):, :]], dim=2)
                    h_v_cache = torch.cat([h_v[:,:, :n_init, :], h_v[:, :, max(0, h_k.size(-2) - n_local):, :]], dim=2)
                current_key_value = (h_k_cache, h_v_cache)
            else:
                current_key_value = (past_k, past_v)

            """ 4. Get local QKV and apply RoPE to local QK """
            h_q_, h_k_, h_v_ = h_q, h_k, h_v
            if len_q + n_local < h_k_.size(-2):
                h_k_ = h_k_[:, :, h_k_.size(-2) - len_q - n_local:, :]
                h_v_ = h_v_[:, :, h_v_.size(-2) - len_q - n_local:, :]

            local_h_q, local_h_k = position_bias(h_q_, h_k_)
            local_h_v = h_v_

            """ 5. Get init QKV and apply RoPE to init Q (Infinite-LM assigns the same position_ids to initial tokens) """
            if len_k > n_local:
                init_h_q = position_bias.apply_rotary_pos_emb_one_angle(
                    h_q, n_local
                )
                init_h_k = h_k
                init_h_v = h_v
                init_h_k = init_h_k[:, :, :n_init, :].contiguous()
                init_h_v = init_h_v[:, :, :n_init, :].contiguous()

            else:
                init_h_q = h_q
                init_h_k = torch.empty(
                    (batch_size, num_heads_kv, 0, dim_head),
                    device=h_k.device,
                    dtype=h_k.dtype
                )
                init_h_v = torch.empty(
                    (batch_size, num_heads_kv, 0, dim_head),
                    device=h_v.device,
                    dtype=h_v.dtype
                )

            """ 6. Sliding Window Attention """
            attn = Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
            attn.append(local_h_q, local_h_k, local_h_v, sliding_window=n_local)
            attn.append(init_h_q, init_h_k, init_h_v, end=True, sliding_window=(len_k - len_q, n_local), complement_sliding_window=True)
            score, _ = attn.get_result()

            # --- Log 3-part attention scores (system / video / question) ---
            _log_layer_idx = getattr(past_key_value, '_layer_idx', None) if type(past_key_value) is ContextManager else None
            if _log_layer_idx is not None and not updata_kv_cache:
                with torch.no_grad():
                    _compute_3part_attn_scores(
                        h_q, h_k, h_v, position_bias, n_init, n_local,
                        num_heads, num_heads_kv, dim_head, len_q, len_k,
                        past_k.shape[2], _log_layer_idx
                    )

            score = score.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3) # (batch, len_q, num_heads, dim_head)
            score = score.reshape(batch_size, len_q, num_heads * dim_head) # (batch, len_q, num_heads * dim_head)
            score = attention_out(score)

            return score, current_key_value

        # NOTE: Encode video, managed by the KVCacheManager
        else:
            o = past_key_value.append(
                local_q, local_k, local_v,
                global_q, global_k, global_v,
            )
            o = o.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3)
            o = o.reshape(batch_size, len_q, dim_head * num_heads)
            o = attention_out(o)

            return o, past_key_value

    return forward
