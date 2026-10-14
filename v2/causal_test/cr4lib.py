"""Model wrapper for paper 4: the published CR detector, additive steering at one layer,
greedy generation with a per-layer logit lens, and the outcome parsers."""

from __future__ import annotations

import math
import os
import re
import sys

import numpy as np
import torch
import transformers

HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.path.join(HERE, "..", "..")  # repository root, which holds cr_axis/ (version-1 code, commit ce0c61f)
sys.path.insert(0, UP)
from cr_axis.detector import CRDetector, _find_layers, _format_prompt  # noqa: E402

MODELS = {
    "1.7b": "Qwen/Qwen3-1.7B",
    "q25-7b": "Qwen/Qwen2.5-7B-Instruct",
    "8b": "Qwen/Qwen3-8B",
    "14b": "Qwen/Qwen3-14B",
    "4b-2507": "Qwen/Qwen3-4B-Instruct-2507",
}


def cal_prompts():
    d = os.path.join(UP, "data", "calibration")
    rd = lambda f: [l.strip() for l in open(os.path.join(d, f), encoding="utf-8") if l.strip()]
    return rd("compute.txt"), rd("retrieve.txt")


def unit(v):
    return v / np.linalg.norm(v)


def calibrate(hf, tok, dev) -> dict:
    """The published CRDetector calibration, plus the gap and the within-class length axis."""
    comp, retr = cal_prompts()
    det = CRDetector(hf, tok, device=dev)
    auc, layer = det.calibrate(comp, retr)
    hc = np.array([det._get_hidden(p, layer) for p in comp])
    hr = np.array([det._get_hidden(p, layer) for p in retr])
    axis = det.axis.astype(np.float64)
    gap = float((hc @ axis).mean() - (hr @ axis).mean())
    n_tok = lambda p: len(tok(_format_prompt(p, tok))["input_ids"])

    def long_minus_short(H, ps):
        order = np.argsort([n_tok(p) for p in ps], kind="stable")
        h = len(order) // 2
        return H[order[h:]].mean(0) - H[order[:h]].mean(0)

    len_axis = unit((long_minus_short(hc, comp) + long_minus_short(hr, retr)) / 2)
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(hc)), np.zeros(len(hr))]
    return dict(auc=float(auc), layer=int(layer), gap=float(gap), axis=axis, len_axis=len_axis,
                cos_len=float(axis @ len_axis),
                len_axis_class_auc=float(roc_auc_score(y, np.r_[hc @ len_axis, hr @ len_axis])),
                resid_norm_median=float(np.median(np.linalg.norm(np.r_[hc, hr], axis=1))),
                proj_c=(hc @ axis).tolist(), proj_r=(hr @ axis).tolist(), det=det)


class Model:
    def __init__(self, name: str, device: str = "cuda", gpu_layers: int | None = None):
        """gpu_layers: keep only the first N blocks on the GPU and stream the rest from CPU
        memory (paper 3's loader), for models that do not fit next to the batch."""
        self.name, self.dev = name, device
        self.tok = transformers.AutoTokenizer.from_pretrained(name)
        if gpu_layers is None:
            self.hf = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device).eval()
        else:
            from stream import _stream_from_cpu
            self.hf = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).eval()
            mm = self.hf.model
            for mod in (mm.embed_tokens, mm.norm, mm.rotary_emb, self.hf.lm_head):
                mod.to(device)
            for i, layer in enumerate(mm.layers):
                if i < gpu_layers:
                    layer.to(device)
                else:
                    _stream_from_cpu(layer, device)
        for p in self.hf.parameters():
            p.requires_grad_(False)
        self.layers = _find_layers(self.hf)
        self.L = len(self.layers)
        self.d = self.hf.config.hidden_size
        self.norm = self.hf.model.norm
        self.qwen3 = "Qwen3" in name and "2507" not in name
        self.steer = None  # dict(layer, vecs [B, d], start)
        self.capture = None  # list of L tensors [B, d]: last-position output of every layer
        self.eos = {self.tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
        for i, layer in enumerate(self.layers):
            layer.register_forward_hook(self._hook(i))

    def _hook(self, i):
        def f(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            s = self.steer
            if s is not None and i == s["layer"]:
                T = h.shape[1]
                if T > 1:
                    m = (torch.arange(T, device=h.device) >= s["start"]).to(h.dtype)[None, :, None]
                    h = h + m * s["vecs"][:, None, :]
                else:
                    h = h + s["vecs"][:, None, :]
            if self.capture is not None:
                self.capture[i] = h[:, -1].detach()
            if isinstance(out, tuple):
                return (h,) + tuple(out[1:])
            return h
        return f

    # ---------- prompts ----------
    def chat_text(self, user: str) -> str:
        kw = {"enable_thinking": False} if self.qwen3 else {}
        return self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                            add_generation_prompt=True, **kw)

    def start_index(self, user: str, full_ids: list[int]) -> int:
        """Detector reading position (last token of the published _format_prompt text)."""
        det_ids = self.tok(_format_prompt(user, self.tok))["input_ids"]
        assert full_ids[: len(det_ids)] == det_ids, "detector text is not a token prefix"
        return len(det_ids) - 1

    # ---------- calibration ----------
    def calibrate(self) -> dict:
        out = calibrate(self.hf, self.tok, self.dev)
        self.det = out.pop("det")
        return out

    # ---------- generation ----------
    @torch.no_grad()
    def generate(self, text: str, user: str, vecs: torch.Tensor | None, layer: int, n_new: int,
                 lens: bool) -> dict:
        """Greedy continuation of `text` for every row of `vecs` ([B, d]; row = one condition).
        All rows share the prompt, so there is no padding. With lens=True, returns per step and
        row the logit-lens top-1 token and JSD-to-final (bits) at every layer, and the final
        log-probabilities at the first step."""
        ids = self.tok(text)["input_ids"]
        start = self.start_index(user, ids)
        B = vecs.shape[0]
        x = torch.tensor([ids] * B, device=self.dev)
        self.steer = dict(layer=layer, vecs=vecs.to(self.dev, torch.bfloat16), start=start)
        cache = transformers.DynamicCache()
        out_tok, top1s, jsds, lp0 = [], [], [], None
        T = x.shape[1]
        try:
            for step in range(n_new):
                self.capture = [None] * self.L if lens else None
                pos = torch.arange(T - x.shape[1], T, device=self.dev)[None].expand(B, -1) if step else \
                    torch.arange(T, device=self.dev)[None].expand(B, -1)
                o = self.hf(input_ids=x, past_key_values=cache, use_cache=True, position_ids=pos)
                logits = o.logits[:, -1].float()
                nxt = logits.argmax(-1)
                if step == 0:
                    lp0 = torch.log_softmax(logits, -1)
                if lens:
                    t1, js = self._lens(logits)
                    top1s.append(t1)
                    jsds.append(js)
                out_tok.append(nxt)
                x = nxt[:, None]
                T += 1
        finally:
            self.steer, self.capture = None, None
        toks = torch.stack(out_tok, 1).cpu()  # [B, n_new]
        res = dict(tokens=toks, lp0=lp0, start=start, n_prompt=len(ids))
        if lens:
            res["top1"] = torch.stack(top1s, 1).cpu()  # [B, n_new, L]
            res["jsd"] = torch.stack(jsds, 1).cpu()
        return res

    @torch.no_grad()
    def tf_kl(self, text: str, user: str, vecs: torch.Tensor, layer: int, cont: list[int],
              ans_idx: list[int]) -> tuple[list[float], list[float]]:
        """Teacher-forced on text + cont (row 0's own continuation): KL(row 0 || row b) summed over
        every position from the reading position on, and over the positions predicting the
        answer digits cont[ans_idx]."""
        ids = self.tok(text)["input_ids"]
        start = self.start_index(user, ids)
        full = ids + cont
        x = torch.tensor([full] * vecs.shape[0], device=self.dev)
        self.steer = dict(layer=layer, vecs=vecs.to(self.dev, torch.bfloat16), start=start)
        try:
            h = self.hf.model(input_ids=x).last_hidden_state
        finally:
            self.steer = None
        kls = []
        for t in range(start, len(full) - 1):
            lp = torch.log_softmax(self.hf.lm_head(h[:, t]).float(), -1)
            kls.append((lp[0].exp() * (lp[0] - lp)).sum(-1).clamp_min(0))
        K = torch.stack(kls, 1).cpu()
        cols = [len(ids) - 1 + i - start for i in ans_idx]
        return K.sum(1).tolist(), (K[:, cols].sum(1) if cols else torch.zeros(K.shape[0])).tolist()

    def _lens(self, final_logits):
        lp_final = torch.log_softmax(final_logits, -1)
        p_final = lp_final.exp()
        t1, js = [], []
        for l in range(self.L):
            lg = self.hf.lm_head(self.norm(self.capture[l][:, None]))[:, 0].float()
            t1.append(lg.argmax(-1))
            lp = torch.log_softmax(lg, -1)
            p = lp.exp()
            m = 0.5 * (p + p_final)
            lm = torch.log(m.clamp_min(1e-30))
            j = 0.5 * ((p * (lp - lm)).sum(-1) + (p_final * (lp_final - lm)).sum(-1)) / math.log(2)
            js.append(j.clamp_min(0))
        return torch.stack(t1, -1), torch.stack(js, -1)


# ---------- outcome parsers ----------
def settle_top1(top1_row: torch.Tensor) -> int:
    """Smallest layer l such that the lens top-1 equals the final top-1 at every layer >= l."""
    L = top1_row.shape[0]
    final = top1_row[-1]
    l = L - 1
    while l > 0 and top1_row[l - 1] == final:
        l -= 1
    return l


def settle_jsd(jsd_row: torch.Tensor, thr: float = 0.5) -> int:
    """Chen et al. (2026): first layer (1-indexed) at which the running minimum of JSD to the
    final distribution falls to <= thr bits."""
    runmin = math.inf
    for l, v in enumerate(jsd_row.tolist()):
        runmin = min(runmin, v)
        if runmin <= thr:
            return l + 1
    return len(jsd_row)


def answer_span(tok, toks: list[int]) -> tuple[int | None, list[int]]:
    """First run of digit tokens (commas between digits allowed), read as an integer.
    Returns (value, indices of the digit tokens)."""
    pieces = [tok.decode([t]) for t in toks]
    i, idx = 0, []
    while i < len(pieces) and pieces[i].strip() == "":
        i += 1
    while i < len(pieces):
        p = pieces[i]
        if p.isascii() and p.isdigit():
            idx.append(i)
        elif not (p == "," and idx and i + 1 < len(pieces) and pieces[i + 1].isascii() and pieces[i + 1].isdigit()):
            break
        i += 1
    if not idx:
        return None, []
    return int("".join(pieces[j] for j in idx)), idx


NUM = re.compile(r"\d[\d,]*\d|\d")


def shown_working(text: str, a: int, b: int) -> tuple[int, int | None]:
    nums = [int(n.replace(",", "")) for n in NUM.findall(text)]
    if not nums:
        return 0, None
    final = nums[-1]
    return int(any(n not in (a, b, final) for n in nums)), final


def error_type(ans: int | None, a: int, b: int, correct: int) -> dict:
    if ans is None or ans == correct:
        return dict(OR=0, OP=0, DC=0, other=0)
    OR = any(ans == a2 * b for a2 in range(max(1, a - 2), a + 3) if a2 != a) or \
        any(ans == a * b2 for b2 in range(max(1, b - 2), b + 3) if b2 != b)
    OP = ans == a + b
    sa, sc = str(ans), str(correct)
    DC = len(sa) == len(sc) and sum(x != y for x, y in zip(sa, sc)) == 1
    return dict(OR=int(OR), OP=int(OP), DC=int(DC), other=int(not (OR or OP or DC)))
