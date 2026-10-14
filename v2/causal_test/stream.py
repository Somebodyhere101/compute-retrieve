"""Stream a transformer block's weights from pinned CPU memory per forward pass (used for Qwen3-14B)."""

import torch

PIN_BUDGET = 6 * 2**30
_PINNED = 0


def _stream_from_cpu(layer: torch.nn.Module, device: str) -> None:
    """Hold a block's weights in pinned CPU memory and copy them to the GPU only while
    the block runs. Registered before the Controller's hooks, so it runs first."""
    def pin(t):
        # Windows caps page-locked memory and a failed pin poisons the CUDA context, so
        # pin only within a fixed budget and leave the rest in ordinary memory.
        global _PINNED
        nbytes = t.numel() * t.element_size()
        if _PINNED + nbytes > PIN_BUDGET:
            return t
        _PINNED += nbytes
        return t.pin_memory()

    cpu = {name: pin(p.data) for name, p in layer.named_parameters()}
    params = dict(layer.named_parameters())
    for name, p in params.items():
        p.data = cpu[name]

    def pre(module, args):
        for name, p in params.items():
            p.data = cpu[name].to(device, non_blocking=True)

    def post(module, args, output):
        for name, p in params.items():
            p.data = cpu[name]

    layer.register_forward_pre_hook(pre)
    layer.register_forward_hook(post)
