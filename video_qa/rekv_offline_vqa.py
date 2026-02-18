import os
import json
import torch
import numpy as np
from logzero import logger

from video_qa.base import BaseVQA, work
from model.neural_kv import (
    train_neural_kv, evaluate_neural_kv, collect_real_q_from_model,
    collect_decode_q, synthesize_decode_q, collect_all_kv
)
from model.attention.rekv_attention import (
    get_original_w_video_log, clear_original_w_video_log,
    get_attn_score_3part_log, clear_attn_score_3part_log,
)


class ReKVOfflineVQA(BaseVQA):
    def video_open_qa(self, question, max_new_tokens=1024, retrieved_indices=None):
        input_text = {
            "question": question,
            "prompt": self.qa_model.get_prompt(question)
        }

        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=max_new_tokens, retrieved_indices=retrieved_indices)

        return {
            'pred_answer': pred_answer.replace('\n', ''),
        }

    def video_close_qa(self, question, candidates, correct_choice, retrieved_indices=None):
        input_text = self.format_mcqa_prompt(question, candidates)
        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=16, retrieved_indices=retrieved_indices)
        pred_letter = self.extract_characters_regex(pred_answer)
        return {
            'pred_answer': pred_answer.replace('\n', ''),
            'pred_choice': pred_letter,
            'acc': float(pred_letter == correct_choice),
        }

    def no_video_open_qa(self, question, max_new_tokens=1024):
        input_text = {
            "question": question,
            "prompt": self.qa_model.get_prompt(question)
        }
        pred_answer = self.qa_model.question_answering_no_video(input_text, max_new_tokens=max_new_tokens)
        return {
            'pred_answer': pred_answer.replace('\n', ''),
        }

    def no_video_close_qa(self, question, candidates, correct_choice):
        input_text = self.format_mcqa_prompt(question, candidates)
        pred_answer = self.qa_model.question_answering_no_video(input_text, max_new_tokens=16)
        pred_letter = self.extract_characters_regex(pred_answer)
        return {
            'pred_answer': pred_answer.replace('\n', ''),
            'pred_choice': pred_letter,
            'acc': float(pred_letter == correct_choice),
        }

    def video_open_qa_neural_kv(self, question, neural_kv, all_layer_k, all_layer_v, max_new_tokens=1024):
        input_text = {
            "question": question,
            "prompt": self.qa_model.get_prompt(question)
        }
        pred_answer, error_log = self.qa_model.question_answering_with_neural_kv(
            input_text, neural_kv, all_layer_k, all_layer_v, max_new_tokens=max_new_tokens
        )
        return {
            'pred_answer': pred_answer.replace('\n', ''),
            'error_log': error_log,
        }

    def video_close_qa_neural_kv(self, question, candidates, correct_choice, neural_kv, all_layer_k, all_layer_v):
        input_text = self.format_mcqa_prompt(question, candidates)
        pred_answer, error_log = self.qa_model.question_answering_with_neural_kv(
            input_text, neural_kv, all_layer_k, all_layer_v, max_new_tokens=16
        )
        pred_letter = self.extract_characters_regex(pred_answer)
        return {
            'pred_answer': pred_answer.replace('\n', ''),
            'pred_choice': pred_letter,
            'acc': float(pred_letter == correct_choice),
            'error_log': error_log,
        }

    def _summarize_error_log(self, error_log, save_dir=None):
        """Summarize and optionally save Neural KV error metrics."""
        if not error_log:
            return

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Group by layer
        layer_mse = {}
        layer_cos = {}
        layer_gt_w_video = {}
        for entry in error_log:
            lid = entry['layer_idx']
            if lid not in layer_mse:
                layer_mse[lid] = []
                layer_cos[lid] = []
                layer_gt_w_video[lid] = []
            layer_mse[lid].append(entry['mse'])
            layer_cos[lid].append(entry['cosine'])
            if 'gt_w_video' in entry and not np.isnan(entry['gt_w_video']):
                layer_gt_w_video[lid].append(entry['gt_w_video'])

        layers = sorted(layer_mse.keys())
        mean_mse_per_layer = [np.nanmean(layer_mse[l]) for l in layers]
        mean_cos_per_layer = [np.nanmean(layer_cos[l]) for l in layers]
        mean_gt_w_video_per_layer = [np.nanmean(layer_gt_w_video[l]) if layer_gt_w_video[l] else 0.0 for l in layers]

        overall_mse = np.nanmean([e['mse'] for e in error_log])
        overall_cos = np.nanmean([e['cosine'] for e in error_log])

        logger.info(f"[NeuralKV Decode] Overall MSE={overall_mse:.6f}, Cosine={overall_cos:.4f}")
        logger.info(f"[NeuralKV Decode] Per-layer metrics (mean over steps):")
        logger.info(f"  {'Layer':>5} | {'MSE':>10} | {'Cosine':>8} | {'w_video':>8}")
        logger.info(f"  {'-'*5}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
        for l, m, c, gw in zip(layers, mean_mse_per_layer, mean_cos_per_layer,
                                mean_gt_w_video_per_layer):
            logger.info(f"  {l:5d} | {m:10.6f} | {c:8.4f} | {gw:8.4f}")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            axes[0].bar(layers, mean_mse_per_layer, color='steelblue')
            axes[0].set_xlabel('Layer')
            axes[0].set_ylabel('MSE')
            axes[0].set_title('Per-Layer Attn MSE')

            axes[1].bar(layers, mean_cos_per_layer, color='darkorange')
            axes[1].set_xlabel('Layer')
            axes[1].set_ylabel('Cosine Similarity')
            axes[1].set_title('Per-Layer Cosine')
            axes[1].set_ylim([min(0, min(mean_cos_per_layer) - 0.05), 1.05])

            axes[2].bar(layers, mean_gt_w_video_per_layer, color='forestgreen')
            axes[2].set_xlabel('Layer')
            axes[2].set_ylabel('Video Weight (sigmoid)')
            axes[2].set_title('Video Attention Weight (real LSE)')
            axes[2].set_ylim([0, 1.05])
            axes[2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'neural_kv_decode_error.png'), dpi=150)
            plt.close()

            with open(os.path.join(save_dir, 'neural_kv_decode_error.json'), 'w') as f:
                json.dump({
                    'overall_mse': float(overall_mse),
                    'overall_cosine': float(overall_cos),
                    'per_layer_mse': {str(l): float(m) for l, m in zip(layers, mean_mse_per_layer)},
                    'per_layer_cosine': {str(l): float(c) for l, c in zip(layers, mean_cos_per_layer)},
                    'per_layer_gt_w_video': {str(l): float(w) for l, w in zip(layers, mean_gt_w_video_per_layer)},
                }, f, indent=2)

            logger.info(f"[NeuralKV Decode] Error metrics saved to {save_dir}")

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        # load and preprocess video frames for QA
        video_path = video_sample['video_path']
        video = self.load_video(video_path)
        if not isinstance(video, torch.Tensor):
            video_tensor = torch.from_numpy(video)
        else:
            video_tensor = video

        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()
        self.qa_model.encode_video(video_tensor)

        # Get shared RoPE module for consistent positional encoding
        position_bias = getattr(self.qa_model.language_model.model, 'position_bias', None)
        if position_bias is not None:
            logger.info("[NeuralKV] RoPE module found — will be used for GT computation and LSE fusion.")

        # Collect real Q via re-running prefill with hooks (before training)
        if self.synth_only:
            logger.info("[NeuralKV] synth_only mode — skipping real Q collection.")
            all_layer_q = None
        else:
            logger.info("[NeuralKV] Collecting real Q from prefill...")
            all_layer_q = collect_real_q_from_model(self.qa_model, video_tensor, device=self.qa_model.device)

            # Re-encode video (collect_real_q re-ran prefill)
            self.qa_model.clear_cache()
            self.qa_model.encode_init_prompt()
            self.qa_model.encode_video(video_tensor)

        # Collect K/V before synthesizing Q
        logger.info("[NeuralKV] Collecting KV from all layers...")
        all_layer_k, all_layer_v = collect_all_kv(self.qa_model)
        ctx0 = self.qa_model.kv_cache[0]

        # Synthesize decode-like Q from common question templates
        logger.info("[NeuralKV] Synthesizing decode-like Q for training...")
        synth_q, synth_targets = synthesize_decode_q(
            self.qa_model, all_layer_k, all_layer_v,
            num_heads=ctx0.num_heads,
            num_heads_kv=ctx0.num_heads_kv,
            dim_head=ctx0.dim_head,
            batch_token_size=512,
            device=self.qa_model.device,
            position_bias=position_bias,
        )

        # Re-encode video (synth Q collection may have corrupted KV cache)
        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()
        self.qa_model.encode_video(video_tensor)

        # Train video-specific neural KV cache
        neural_kv, all_layer_k, all_layer_v = train_neural_kv(
            self.qa_model,
            all_layer_q=all_layer_q,
            num_epochs=100,
            lr=1e-3,
            batch_token_size=512,
            device=self.qa_model.device,
            synth_q=synth_q,
            synth_targets=synth_targets,
            all_layer_k=all_layer_k,
            all_layer_v=all_layer_v,
            position_bias=position_bias,
            synth_only=self.synth_only,
            group_shared=self.group_shared,
        )

        # Collect decode Q from actual question texts for distribution viz
        question_texts = [s['question'] for s in video_sample['conversations']]
        if question_texts:
            logger.info(f"[NeuralKV] Collecting decode Q from {len(question_texts)} questions...")
            decode_q = collect_decode_q(self.qa_model, question_texts, device=self.qa_model.device)
        else:
            decode_q = None

        # Re-encode video (hooks may have corrupted KV cache)
        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()
        self.qa_model.encode_video(video_tensor)

        # Evaluate neural KV with real Q (with RoPE for GT)
        ctx0 = self.qa_model.kv_cache[0]
        eval_results = evaluate_neural_kv(
            neural_kv, all_layer_q, all_layer_k, all_layer_v,
            num_heads=ctx0.num_heads,
            num_heads_kv=ctx0.num_heads_kv,
            dim_head=ctx0.dim_head,
            batch_token_size=512,
            device=self.qa_model.device,
            save_dir=f'results/neural_kv_eval/{video_sample["video_id"]}',
            decode_q=decode_q,
            synth_q=synth_q,
            position_bias=position_bias,
        )
        logger.info(f"[NeuralKV Eval] Mean MSE={eval_results['mean_mse']:.6f}, "
                     f"Mean Cosine={eval_results['mean_cosine']:.4f}, "
                     f"Mean R²={eval_results['mean_r2']:.4f}")

        # Re-encode video for QA (since hooks re-ran prefill)
        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()
        self.qa_model.encode_video(video_tensor)

        save_dir = f'results/neural_kv_eval/{video_sample["video_id"]}'

        for sample in video_sample['conversations']:
            logger.debug(f'sample: {sample}')
            question = sample['question']
            answer = sample['answer']

            # ===== Original QA first (clean KV cache state) =====
            # Attach all_video_k + layer_idx for w_video logging (no _neural_kv_cache → original path)
            for i, layer_kv in enumerate(self.qa_model.kv_cache):
                layer_kv._all_video_k = all_layer_k[i]
                layer_kv._layer_idx = i
            clear_original_w_video_log()
            clear_attn_score_3part_log()

            if 'choices' in sample:  # CloseQA
                choices = sample['choices']
                if answer is None:
                    answer = choices[0]
                correct_choice = self.choice_letters[choices.index(answer)]

                qa_results = self.video_close_qa(question, choices, correct_choice)
            else:  # OpenQA
                choices = None
                correct_choice = None
                qa_results = self.video_open_qa(question)

            # Log original w_video
            orig_w_log = get_original_w_video_log()
            if orig_w_log:
                layer_w = {}
                for entry in orig_w_log:
                    lid = entry['layer_idx']
                    if lid not in layer_w:
                        layer_w[lid] = []
                    layer_w[lid].append(entry['w_video'])
                layers_sorted = sorted(layer_w.keys())
                logger.info(f"[Original] w_video per layer (retrieval step):")
                logger.info(f"  {'Layer':>5} | {'w_video':>8}")
                logger.info(f"  {'-'*5}-+-{'-'*8}")
                for l in layers_sorted:
                    logger.info(f"  {l:5d} | {np.mean(layer_w[l]):8.4f}")
                logger.info(f"  {'Mean':>5} | {np.mean([np.mean(layer_w[l]) for l in layers_sorted]):8.4f}")

            # Log 3-part attention scores (system / video / question)
            attn_3part_log = get_attn_score_3part_log()
            if attn_3part_log:
                layer_3part = {}
                for entry in attn_3part_log:
                    lid = entry['layer_idx']
                    if lid not in layer_3part:
                        layer_3part[lid] = {'w_system': [], 'w_video': [], 'w_question': []}
                    layer_3part[lid]['w_system'].append(entry['w_system'])
                    layer_3part[lid]['w_video'].append(entry['w_video'])
                    layer_3part[lid]['w_question'].append(entry['w_question'])
                layers_sorted = sorted(layer_3part.keys())
                n_sys = attn_3part_log[0]['n_system']
                n_vid = attn_3part_log[0]['n_video']
                n_q = attn_3part_log[0]['n_question']
                logger.info(f"[Original] 3-part attention scores (system={n_sys}tok, video={n_vid}tok, question={n_q}tok):")
                logger.info(f"  {'Layer':>5} | {'System':>8} | {'Video':>8} | {'Question':>8}")
                logger.info(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
                for l in layers_sorted:
                    ws = np.mean(layer_3part[l]['w_system'])
                    wv = np.mean(layer_3part[l]['w_video'])
                    wq = np.mean(layer_3part[l]['w_question'])
                    logger.info(f"  {l:5d} | {ws:8.4f} | {wv:8.4f} | {wq:8.4f}")
                mean_ws = np.mean([np.mean(layer_3part[l]['w_system']) for l in layers_sorted])
                mean_wv = np.mean([np.mean(layer_3part[l]['w_video']) for l in layers_sorted])
                mean_wq = np.mean([np.mean(layer_3part[l]['w_question']) for l in layers_sorted])
                logger.info(f"  {'Mean':>5} | {mean_ws:8.4f} | {mean_wv:8.4f} | {mean_wq:8.4f}")

            # Clean up attached attributes from original QA
            for layer_kv in self.qa_model.kv_cache:
                for attr in ('_all_video_k', '_layer_idx'):
                    if hasattr(layer_kv, attr):
                        delattr(layer_kv, attr)

            # ===== No-Video baseline QA =====
            if choices is not None:
                novid_results = self.no_video_close_qa(question, choices, correct_choice)
            else:
                novid_results = self.no_video_open_qa(question)

            # ===== Re-encode video before Neural KV QA =====
            self.qa_model.clear_cache()
            self.qa_model.encode_init_prompt()
            self.qa_model.encode_video(video_tensor)

            # ===== Neural KV QA =====
            if choices is not None:  # CloseQA
                nkv_results = self.video_close_qa_neural_kv(
                    question, choices, correct_choice, neural_kv, all_layer_k, all_layer_v
                )
                self._summarize_error_log(nkv_results.get('error_log', []), save_dir=save_dir)

                logger.info(f"[QA Result] Q: {question[:100]}")
                logger.info(f"  Correct answer: {answer} (choice={correct_choice})")
                logger.info(f"  NoVideo:  {novid_results['pred_answer']} (choice={novid_results['pred_choice']}, acc={novid_results['acc']:.0f})")
                logger.info(f"  Original: {qa_results['pred_answer']} (choice={qa_results['pred_choice']}, acc={qa_results['acc']:.0f})")
                logger.info(f"  NeuralKV: {nkv_results['pred_answer']} (choice={nkv_results['pred_choice']}, acc={nkv_results['acc']:.0f})")

                self.record[(self.retrieve_size, self.chunk_size)].append({
                    'video_id': video_sample['video_id'],
                    'question': question,
                    'choices': choices,
                    'answer': answer,
                    'correct_choice': correct_choice,
                    'novid_pred_answer': novid_results['pred_answer'],
                    'novid_pred_choice': novid_results['pred_choice'],
                    'novid_qa_acc': novid_results['acc'] * 100,
                    'pred_answer': qa_results['pred_answer'],
                    'pred_choice': qa_results['pred_choice'],
                    'qa_acc': qa_results['acc'] * 100,
                    'nkv_pred_answer': nkv_results['pred_answer'],
                    'nkv_pred_choice': nkv_results['pred_choice'],
                    'nkv_qa_acc': nkv_results['acc'] * 100,
                })
            else:  # OpenQA
                nkv_results = self.video_open_qa_neural_kv(
                    question, neural_kv, all_layer_k, all_layer_v
                )
                self._summarize_error_log(nkv_results.get('error_log', []), save_dir=save_dir)

                logger.info(f"[QA Result] Q: {question[:100]}")
                logger.info(f"  Correct answer: {answer}")
                logger.info(f"  NoVideo:  {novid_results['pred_answer'][:100]}")
                logger.info(f"  Original: {qa_results['pred_answer'][:100]}")
                logger.info(f"  NeuralKV: {nkv_results['pred_answer'][:100]}")

                self.record[(self.retrieve_size, self.chunk_size)].append({
                    'video_id': video_sample['video_id'],
                    'question': question,
                    'answer': answer,
                    'novid_pred_answer': novid_results['pred_answer'],
                    'pred_answer': qa_results['pred_answer'],
                    'nkv_pred_answer': nkv_results['pred_answer'],
                })

            # Re-encode video for next question
            self.qa_model.clear_cache()
            self.qa_model.encode_init_prompt()
            self.qa_model.encode_video(video_tensor)

            if 'question_type' in sample:
                self.record[(self.retrieve_size, self.chunk_size)][-1]['task'] = sample['question_type']

        # Summary for this video
        records = self.record[(self.retrieve_size, self.chunk_size)]
        video_records = [r for r in records if r.get('video_id') == video_sample['video_id']]
        if video_records and 'qa_acc' in video_records[0]:
            n_q = len(video_records)
            novid_acc = np.mean([r['novid_qa_acc'] for r in video_records])
            orig_acc = np.mean([r['qa_acc'] for r in video_records])
            nkv_acc = np.mean([r['nkv_qa_acc'] for r in video_records])
            logger.info(f"[Video Summary] {video_sample['video_id']}: "
                        f"{n_q} questions, NoVideo acc={novid_acc:.1f}%, "
                        f"Original acc={orig_acc:.1f}%, NeuralKV acc={nkv_acc:.1f}%")


if __name__ == "__main__":
    work(ReKVOfflineVQA)
