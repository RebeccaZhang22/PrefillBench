"""Latent exit: run the prefill only up to decoder block L and stop there (no final norm, lm_head or sampling).

Applied in-process to the loaded model, right before CUDA graphs are captured (ModelRunner.init_cuda_graphs), so
prefill / decode graphs are captured on the cut model. The installed SGLang files are untouched.

  model.model.end_layer = L + 1      the decoder loop is range(start_layer, end_layer): blocks 0..L
  model.model.norm  -> _LatentOutput returns the block-L residual stream (hidden + residual), not normalized
  model.logits_processor -> _LatentHead  keeps each request's last-token latent [#seq, hidden]; no lm_head

Two ways to finish a forward:
  sampler=False  (one_batch)  the latent itself is returned as next_token_logits and ModelRunner.sample is a no-op
  sampler=True   (server)     the scheduler needs a token per request, so the head hands the sampler an all-zero
                              [#seq, vocab] tensor (greedy -> token 0); the latent is still computed (gathered
                              on the GPU, not attached to the output so the scheduler never copies it). No lm_head
                              GEMM either way.

install(layer, sampler) patches ModelRunner.init_cuda_graphs; call it at import time of the entry script so that
spawned SGLang subprocesses (which re-import the main module) get it too.
"""
import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.model_runner import ModelRunner


class _LatentOutput(torch.nn.Module):
    """Stands in for the final RMSNorm. Same call convention: norm(x) -> x, norm(x, residual) -> (out, residual)."""

    def forward(self, x, residual=None, *args, **kwargs):
        if residual is None:
            return x
        x = x + residual
        return x, x


class _LatentHead(torch.nn.Module):
    """Stands in for the LogitsProcessor: no lm_head; keeps each request's last-token latent (all rows in decode,
    which only runs during CUDA graph capture)."""

    def __init__(self, vocab_size=None):
        super().__init__()
        self.vocab_size = vocab_size  # set: hand the sampler zero logits of this width

    def forward(self, input_ids, hidden_states, lm_head, forward_batch, *args, **kwargs):
        if forward_batch.forward_mode.is_extend() and forward_batch.extend_seq_lens is not None:
            hidden_states = hidden_states[torch.cumsum(forward_batch.extend_seq_lens, 0) - 1]
        if self.vocab_size is None:
            return LogitsProcessorOutput(next_token_logits=hidden_states)
        logits = torch.zeros(hidden_states.shape[0], self.vocab_size, device=hidden_states.device,
                             dtype=torch.float32)
        return LogitsProcessorOutput(next_token_logits=logits)


def apply_latent_exit(runner, layer, sampler=False):
    """Cut the loaded model after decoder block `layer` (0-based). Idempotent."""
    if getattr(runner, "_gpubench_latent", None) == layer:
        return
    model = runner.model
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "end_layer") or not hasattr(model, "logits_processor"):
        raise RuntimeError(f"latent mode supports Qwen2/Qwen3-style models, not {type(model).__name__}")
    if inner.start_layer != 0 or not inner.pp_group.is_last_rank:
        raise RuntimeError("latent mode does not support pipeline parallelism")
    if not 0 <= layer < inner.end_layer:
        raise RuntimeError(f"latent layer {layer} out of range: the model has {inner.end_layer} decoder blocks")
    inner.end_layer = layer + 1
    inner.norm = _LatentOutput()
    model.logits_processor = _LatentHead(runner.model_config.vocab_size if sampler else None)
    if not sampler:
        runner.sample = lambda *args, **kwargs: None
    runner._gpubench_latent = layer


_official_init_cuda_graphs = ModelRunner.init_cuda_graphs
_installed = None


def install(layer_fn, sampler=False):
    """layer_fn() -> latent layer (< 0 = full model), evaluated in the process that builds the model."""
    global _installed
    if _installed is not None:
        return

    def _init_cuda_graphs_latent(self, *args, **kwargs):
        layer = layer_fn()
        if layer >= 0:
            apply_latent_exit(self, layer, sampler)
            print(f"[latent_exit] {type(self.model).__name__}: decoder blocks 0..{layer}, no final norm / lm_head",
                  flush=True)
        return _official_init_cuda_graphs(self, *args, **kwargs)

    ModelRunner.init_cuda_graphs = _init_cuda_graphs_latent
    _installed = layer_fn
