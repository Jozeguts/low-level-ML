from __future__ import annotations

from runtime import ModelConfig, DecoderModel


def parameter_memory_bytes(model: DecoderModel) -> int:
    return model.parameter_bytes()


def kv_cache_bytes(config: ModelConfig, batch_size: int, sequence_length: int, dtype_bytes: int = 4) -> int:
    if batch_size < 1 or sequence_length < 0:
        raise ValueError("invalid batch or sequence length")
    return (
        config.num_layers
        * batch_size
        * 2
        * config.num_kv_heads
        * sequence_length
        * config.head_dim
        * dtype_bytes
    )


def activation_estimate_bytes(config: ModelConfig, batch_size: int, sequence_length: int, dtype_bytes: int = 4) -> int:
    if batch_size < 1 or sequence_length < 1:
        raise ValueError("invalid activation dimensions")
    # Conservative baseline: input/output hidden states plus intermediate MLP.
    hidden = batch_size * sequence_length * config.hidden_size
    intermediate = batch_size * sequence_length * config.intermediate_size
    return (2 * hidden + intermediate) * dtype_bytes


def memory_report(model: DecoderModel, batch_size: int, sequence_length: int) -> dict[str, int]:
    params = parameter_memory_bytes(model)
    kv = kv_cache_bytes(model.config, batch_size, sequence_length)
    activations = activation_estimate_bytes(model.config, batch_size, sequence_length)
    return {
        "parameters_bytes": params,
        "kv_cache_bytes": kv,
        "activation_estimate_bytes": activations,
        "total_estimate_bytes": params + kv + activations,
    }
