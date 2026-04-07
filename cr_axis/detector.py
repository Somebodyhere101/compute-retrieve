"""Compute/Retrieve mode detector for transformer hidden states."""

import torch
import numpy as np


class CRDetector:
    """Separates compute from retrieve mode via a mean-difference direction in hidden states.

    det = CRDetector(model, tokenizer)
    det.calibrate(compute_prompts, retrieve_prompts)
    det.detect("What is 347 + 289?")   # positive = compute
    det.detect("The capital of France is")  # negative = retrieve
    """

    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tok = tokenizer
        self.device = device or next(model.parameters()).device
        self.axis = None
        self.layer = None
        self.layers = _find_layers(model)
        self.n_layers = len(self.layers)

    @torch.no_grad()
    def _get_hidden(self, prompt, layer_idx):
        captured = {}
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured['h'] = h[0, -1].float().cpu().numpy()

        handle = self.layers[layer_idx].register_forward_hook(hook)
        try:
            text = _format_prompt(prompt, self.tok)
            ids = self.tok(text, return_tensors='pt').to(self.device)
            self.model(ids['input_ids'])
        finally:
            handle.remove()

        if 'h' not in captured:
            raise RuntimeError(f"Hook at layer {layer_idx} never fired")
        return captured['h']

    def calibrate(self, compute_prompts, retrieve_prompts, test_layers=None):
        """Mean-difference axis, best layer by AUC. Returns (auc, layer_idx)."""
        if len(compute_prompts) < 2 or len(retrieve_prompts) < 2:
            raise ValueError(f"Need >=2 prompts per class, got {len(compute_prompts)}/{len(retrieve_prompts)}")

        if test_layers is None:
            test_layers = sorted(set([
                max(1, self.n_layers // 4),
                self.n_layers // 2,
                int(self.n_layers * 0.75),
                self.n_layers - 2,
            ]))

        from sklearn.metrics import roc_auc_score
        best_auc = 0

        for li in test_layers:
            hc = np.array([self._get_hidden(p, li) for p in compute_prompts])
            hr = np.array([self._get_hidden(p, li) for p in retrieve_prompts])

            ax = hc.mean(0) - hr.mean(0)
            n = np.linalg.norm(ax)
            if n < 1e-10:
                continue
            ax /= n

            pc, pr = hc @ ax, hr @ ax
            try:
                auc = roc_auc_score(
                    np.concatenate([np.ones(len(pc)), np.zeros(len(pr))]),
                    np.concatenate([pc, pr]))
            except ValueError:
                continue

            if auc > best_auc:
                best_auc, self.axis, self.layer = auc, ax, li

        if self.axis is None:
            raise ValueError("Calibration failed on all layers")
        return best_auc, self.layer

    def detect(self, prompt):
        """Returns projection: positive=compute, negative=retrieve."""
        if self.axis is None:
            raise RuntimeError("calibrate() or load() first")
        return float(self._get_hidden(prompt, self.layer) @ self.axis)

    def detect_batch(self, prompts):
        return [self.detect(p) for p in prompts]

    def save(self, path):
        if self.axis is None:
            raise RuntimeError("nothing to save")
        np.savez(path, axis=self.axis, layer=np.array(self.layer))

    def load(self, path):
        d = np.load(path)
        self.axis, self.layer = d['axis'], int(d['layer'])


def _find_layers(model):
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        return model.model.language_model.layers  # gemma
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers  # llama, qwen, mistral
    if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return model.gpt_neox.layers  # pythia
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h  # gpt2
    raise ValueError(f"Unknown architecture: {type(model).__name__}")


def _format_prompt(prompt, tokenizer):
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return prompt
