# @title Transformer Model
# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer-based language model.

Reusing decoder only model from examples/wmt.
"""

# pylint: disable=attribute-defined-outside-init
# See issue #620.
# pytype: disable=wrong-arg-count
# pytype: disable=wrong-keyword-args
# pytype: disable=attribute-error

from typing import Dict
import flax

from flax import linen as nn
from flax import struct
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [10, 5]

import functools
from typing import Any, Callable, Optional, Tuple

from flax.linen.initializers import zeros
from flax.linen.linear import default_kernel_init
from flax.linen.linear import DenseGeneral
from flax.linen.linear import PrecisionLike
from flax.linen.module import compact
from flax.linen.module import merge_param
from flax.linen.module import Module
import optax

from jax import lax

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb
from brax.io import model
from brax.training.acme import running_statistics

from ppo import networks as ppo_networks

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any

from flax.linen import attention


def shard(xs):
    """Helper for pmap to shard a pytree of arrays by local_device_count.
    Args:
      xs: a pytree of arrays.
    Returns:
      A matching pytree with arrays' leading dimensions sharded by the
      local device count.
    """
    local_device_count = jax.local_device_count()
    return jax.tree_util.tree_map(lambda x: x.reshape((local_device_count, -1) + x.shape[1:]), xs)


# Note we need to redefine this here since we use a jnp.int32 in the
# MultiHeadDotProductAttention
class MultiHeadDotProductAttention(Module):
    """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation
        (default: infer from inputs and params)
      param_dtype: the dtype passed to parameter initializers (default: float32)
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rate: dropout rate
      deterministic: if false, the attention weight is masked randomly
        using dropout, whereas if true, the attention weights
        are deterministic.
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      use_bias: bool: whether pointwise QKVO dense transforms use bias.
      attention_fn: dot_product_attention or compatible function. Accepts
        query, key, value, and returns output of shape
        `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``
      decode: whether to prepare and use an autoregressive cache.
    """

    num_heads: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    broadcast_dropout: bool = True
    dropout_rate: float = 0.0
    deterministic: Optional[bool] = None
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
    use_bias: bool = True
    attention_fn: Callable[[Array, Array, Array], Array] = attention.dot_product_attention
    decode: bool = False

    @compact
    def __call__(
        self,
        inputs_q: Array,
        inputs_kv: Array,
        mask: Optional[Array] = None,
        deterministic: Optional[bool] = None,
    ):
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        Args:
          inputs_q: input queries of shape
            `[batch_sizes..., length, features]`.
          inputs_kv: key/values of shape
            `[batch_sizes..., length, features]`.
          mask: attention mask of shape
            `[batch_sizes..., num_heads, query_length, key/value_length]`.
            Attention weights are masked out if their corresponding mask value
            is `False`.
          deterministic: if false, the attention weight is masked randomly
            using dropout, whereas if true, the attention weights
            are deterministic.

        Returns:
          output of shape `[batch_sizes..., length, features]`.
        """
        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        assert (
            qkv_features % self.num_heads == 0
        ), "Memory dimension must be divisible by number of heads."
        head_dim = qkv_features // self.num_heads

        dense = functools.partial(
            DenseGeneral,
            axis=-1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            features=(self.num_heads, head_dim),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            precision=self.precision,
        )
        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]
        query, key, value = (
            dense(name="query")(inputs_q),
            dense(name="key")(inputs_kv),
            dense(name="value")(inputs_kv),
        )

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.decode:
            # detect if we're initializing by absence of existing cache data.
            is_initialized = self.has_variable("cache", "cached_key")
            cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
            cached_value = self.variable(
                "cache", "cached_value", jnp.zeros, value.shape, value.dtype
            )
            cache_index = self.variable(
                "cache", "cache_index", lambda: jnp.array(0, dtype=jint_dtype)
            )
            if is_initialized:
                *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
                # shape check of cached keys against query input
                expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
                if expected_shape != query.shape:
                    raise ValueError(
                        "Autoregressive cache shape error, "
                        "expected query shape %s instead got %s." % (expected_shape, query.shape)
                    )
                # update key, value caches with our new 1d spatial slices
                cur_index = cache_index.value
                indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
                key = lax.dynamic_update_slice(cached_key.value, key, indices)
                value = lax.dynamic_update_slice(cached_value.value, value, indices)
                cached_key.value = key
                cached_value.value = value
                cache_index.value = cache_index.value + 1
                # causal mask for cached decoder self-attention:
                # our single query position should only attend to those key
                # positions that have already been generated and cached,
                # not the remaining zero elements.
                mask = attention.combine_masks(
                    mask,
                    jnp.broadcast_to(
                        jnp.arange(max_length) <= cur_index, tuple(batch_dims) + (1, 1, max_length)
                    ),
                )

        dropout_rng = None
        if self.dropout_rate > 0.0:  # Require `deterministic` only if using dropout.
            m_deterministic = merge_param("deterministic", self.deterministic, deterministic)
            if not m_deterministic:
                dropout_rng = self.make_rng("dropout")
        else:
            m_deterministic = True

        # apply attention
        x = self.attention_fn(
            query,
            key,
            value,
            mask=mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=m_deterministic,
            dtype=self.dtype,
            precision=self.precision,
        )  # pytype: disable=wrong-keyword-args
        # back to the original inputs dimensions
        out = DenseGeneral(
            features=features,
            axis=(-2, -1),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            name="out",
        )(x)
        return out


class SelfAttention(MultiHeadDotProductAttention):
    """Self-attention special case of multi-head dot-product attention."""

    @compact
    def __call__(
        self, inputs_q: Array, mask: Optional[Array] = None, deterministic: Optional[bool] = None
    ):
        """Applies multi-head dot product self-attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        Args:
          inputs_q: input queries of shape
            `[batch_sizes..., length, features]`.
          mask: attention mask of shape
            `[batch_sizes..., num_heads, query_length, key/value_length]`.
            Attention weights are masked out if their corresponding mask value
            is `False`.
          deterministic: if false, the attention weight is masked randomly
            using dropout, whereas if true, the attention weights
            are deterministic.

        Returns:
          output of shape `[batch_sizes..., length, features]`.
        """
        return super().__call__(inputs_q, inputs_q, mask, deterministic=deterministic)


@struct.dataclass
class TransformerConfig:
    """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

    vocab_size: int
    output_vocab_size: int
    share_embeddings: bool = False
    logits_via_embedding: bool = True
    dtype: Any = jnp.float32
    emb_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    qkv_dim: int = 512
    mlp_dim: int = 2048
    max_len: int = 2048
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    deterministic: bool = False
    decode: bool = False
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    posemb_init: Optional[Callable] = None


def sinusoidal_init(max_len=2048, min_scale=1.0, max_scale=10000.0):
    """1D Sinusoidal Position Embedding Initializer.

    Args:
        max_len: maximum possible length for the input.
        min_scale: float: minimum frequency-scale in sine grating.
        max_scale: float: maximum frequency-scale in sine grating.

    Returns:
        output: init function returning `(1, max_len, d_feature)`
    """

    def init(key, shape, dtype=jnp.float32):
        """Sinusoidal init."""
        del key, dtype
        d_feature = shape[-1]
        pe = np.zeros((max_len, d_feature), dtype=jnp.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
        div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
        pe[:, : d_feature // 2] = np.sin(position * div_term)
        pe[:, d_feature // 2 : 2 * (d_feature // 2)] = np.cos(position * div_term)
        pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
        return jnp.array(pe)

    return init


class AddPositionEmbs(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs.

    Args:
      config: TransformerConfig dataclass containing hyperparameters.
      decode: whether to run in single-position autoregressive mode.
    """

    config: TransformerConfig
    decode: bool = False

    @nn.compact
    def __call__(self, inputs, inputs_positions=None):
        """Applies AddPositionEmbs module.

        By default this layer uses a fixed sinusoidal embedding table. If a
        learned position embedding is desired, pass an initializer to
        posemb_init in the configuration.

        Args:
          inputs: input data.
          inputs_positions: input position indices for packed sequences.

        Returns:
          output: `(bs, timesteps, in_dim)`
        """
        config = self.config
        # inputs.shape is (batch_size, seq_len, emb_dim)
        assert inputs.ndim == 3, "Number of dimensions should be 3," " but it is: %d" % inputs.ndim
        pos_emb_shape = (1, config.max_len, inputs.shape[-1])
        if config.posemb_init is None:
            # Use a fixed (non-learned) sinusoidal position embedding.
            pos_embedding = sinusoidal_init(max_len=config.max_len)(None, pos_emb_shape, None)
        else:
            pos_embedding = self.param("pos_embedding", config.posemb_init, pos_emb_shape)
        if inputs_positions is not None:
            pe = pos_embedding
        else:
            pe = pos_embedding[:, : inputs.shape[1], :]

        # We use a cache position index for tracking decoding position.
        if self.decode:
            is_initialized = self.has_variable("cache", "cache_index")
            cache_index = self.variable(
                "cache", "cache_index", lambda: jnp.array(0, dtype=juint_dtype)
            )
            if is_initialized:
                i = cache_index.value
                _, _, df = pos_embedding.shape
                cache_index.value = i + 1
                pe = lax.dynamic_slice(pos_embedding, jnp.array((0, i, 0)), (1, 1, df))
        if inputs_positions is None:
            # normal unpacked case:
            return inputs + pe
        else:
            # for packed data we need to use known position indices:
            return inputs + jnp.take(pe[0], inputs_positions, axis=0)


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block.

    Args:
      config: TransformerConfig dataclass containing hyperparameters.
      out_dim: optionally specify out dimension.
    """

    config: TransformerConfig
    out_dim: Optional[int] = None

    @nn.compact
    def __call__(self, inputs):
        """Applies Transformer MlpBlock module."""
        config = self.config
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
            config.mlp_dim,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
        )(inputs)
        x = nn.gelu(x)
        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=config.deterministic)
        output = nn.Dense(
            actual_out_dim,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
        )(x)
        output = nn.Dropout(rate=config.dropout_rate)(output, deterministic=config.deterministic)
        return output


class EncoderDecoder1DBlock(nn.Module):
    """Transformer encoder-decoder layer.

    Args:
      config: TransformerConfig dataclass containing hyperparameters.
    """

    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, decoder_mask=None, encoder_decoder_mask=None):
        """Applies EncoderDecoder1DBlock module.

        Args:
          inputs: input data for decoder
          decoder_mask: decoder self-attention mask.
          encoder_decoder_mask: encoder-decoder attention mask.

        Returns:
          output after transformer encoder-decoder block.
        """
        config = self.config

        # Decoder block.
        assert inputs.ndim == 3
        x = nn.LayerNorm(dtype=config.dtype)(inputs)
        x = SelfAttention(
            num_heads=config.num_heads,
            dtype=config.dtype,
            qkv_features=config.qkv_dim,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=config.attention_dropout_rate,
            deterministic=config.deterministic,
            decode=config.decode,
        )(x, decoder_mask)
        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=config.deterministic)
        x = x + inputs

        # MLP block.
        z = nn.LayerNorm(dtype=config.dtype)(x)
        z = MlpBlock(config=config)(z)

        return x + z


class Decoder(nn.Module):
    """Transformer Model Decoder for sequence to sequence translation.

    Args:
      config: TransformerConfig dataclass containing hyperparameters.
      shared_embedding: a shared embedding layer to use.
    """

    config: TransformerConfig
    shared_embedding: Any = None

    @nn.compact
    def __call__(
        self,
        inputs,
        inputs_positions=None,
        inputs_segmentation=None,
        decoder_mask=None,
        encoder_decoder_mask=None,
        both=False,
    ):
        """Applies Transformer model on the inputs.

        Args:
          encoded: encoded input data from encoder.
          inputs: input data.
          inputs_positions: input subsequence positions for packed examples.
          inputs_segmentation: input segmentation info for packed examples.
          decoder_mask: decoder self-attention mask.
          encoder_decoder_mask: encoder-decoder attention mask.
          both: whether to return both sets of logits, or the itemwise min

        Returns:
          output of a transformer decoder.
        """
        config = self.config
        # assert inputs.ndim == 2  # (batch, len)

        # Target Embedding
        if self.shared_embedding is None:
            output_embed = nn.Embed(
                num_embeddings=config.vocab_size,
                features=config.emb_dim,
                embedding_init=nn.initializers.normal(stddev=1.0),
            )
        else:
            output_embed = self.shared_embedding

        y = inputs
        # The way we have set this up, we will use unshifted inputs
        # r = y
        # if not config.decode:
        # y = shift_inputs(y, segment_ids=inputs_segmentation)
        # y = output_embed(y)
        # breakpoint()
        y = y @ output_embed.embedding
        # print(y.shape)
        y = AddPositionEmbs(config=config, decode=config.decode, name="posembed_output")(
            y, inputs_positions=inputs_positions
        )
        y = nn.Dropout(rate=config.dropout_rate)(y, deterministic=config.deterministic)

        y = y.astype(config.dtype)

        # Target-Input Decoder
        for lyr in range(config.num_layers):
            # print(y.shape)
            y = EncoderDecoder1DBlock(config=config, name=f"encoderdecoderblock_{lyr}")(
                y, decoder_mask=decoder_mask, encoder_decoder_mask=encoder_decoder_mask
            )
        y = nn.LayerNorm(dtype=config.dtype, name="encoderdecoder_norm")(y)

        # Decoded Logits
        if config.logits_via_embedding:
            # Use the transpose of embedding matrix for logit transform.
            logits = output_embed.attend(y.astype(jnp.float32))
            # Correctly normalize pre-softmax logits for this shared case.
            logits = logits / jnp.sqrt(y.shape[-1])
            return logits
        else:
            logits = nn.Dense(
                config.output_vocab_size,
                dtype=config.dtype,
                kernel_init=config.kernel_init,
                bias_init=config.bias_init,
                name="logitdense",
            )(y)
            return logits


class TransformerLM(nn.Module):
    """Transformer pure decoder stack for language modelling.

    Args:
      config: TransformerConfig dataclass containing hyperparameters.
    """

    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, inputs_positions=None, inputs_segmentation=None, both=False):
        """Applies TransformerLM on the inputs.

        Args:
          inputs: target data.
          inputs_positions: input subsequence positions for packed examples.
          inputs_segmentation: input segmentation info for packed examples.

        Returns:
          logits array from transformer decoder.
        """
        config = self.config

        # Make padding attention masks.
        if config.decode:
            # for fast autoregressive decoding we use no decoder mask
            decoder_mask = None
        else:
            # decoder_mask = nn.combine_masks(
            #     nn.make_attention_mask(inputs > 0, inputs > 0, dtype=config.dtype),
            #     nn.make_causal_mask(inputs, dtype=config.dtype))
            # breakpoint()
            # dummy
            decoder_mask = nn.make_causal_mask(inputs[:, :, 0], dtype=config.dtype)

        # Add segmentation block-diagonal attention masks if using segmented data.
        if inputs_segmentation is not None:
            decoder_mask = nn.combine_masks(
                decoder_mask,
                nn.make_attention_mask(
                    inputs_segmentation, inputs_segmentation, jnp.equal, dtype=config.dtype
                ),
            )
        # breakpoint()
        # print(decoder_mask)
        logits = Decoder(config=config, shared_embedding=None, name="decoder")(
            inputs,
            inputs_positions=inputs_positions,
            inputs_segmentation=inputs_segmentation,
            decoder_mask=decoder_mask,
            encoder_decoder_mask=None,
            both=both,
        )
        return jax.tree_map(lambda x: x.astype(self.config.dtype), logits)


from typing import NamedTuple

VOCAB_SIZE = None  # ACTION SIZE
SEQUENCE_LENGTH = None  # EPISODE LENGTH
OUTPUT_VOCAB_SIZE = 2  # OUTPUT SIZE (binary u - hidden/not hidden)
EMB_SIZE = 256
NUM_HEADS = 4
NUM_LAYERS = 6  # 8 then 6
DEVICE_COUNT = jax.device_count()
TOTAL_BATCHSIZE = 128  # 512
PER_DEVICE_BATCH_SIZE = TOTAL_BATCHSIZE // DEVICE_COUNT
LEARNING_RATE = 3e-4  # 2e-4
WEIGHT_DECAY = 1e-3
N_STEPS = 10_000  # 10_000
N_DATA = 10_000
DROPOUT_RATE = 0.1
GRAD_CLIP_VALUE = 5

assert TOTAL_BATCHSIZE % DEVICE_COUNT == 0

train_model = None
test_model = None
env = None


def decay_mask_fn(params):
    flat_params = flax.traverse_util.flatten_dict(params)  # type: ignore
    flat_mask = {
        path: (
            path[-1] != "bias"
            and path[-2:]
            not in [("LayerNorm_0", "scale"), ("LayerNorm_1", "scale"), ("LayerNorm_2", "scale")]
        )
        for path in flat_params
    }
    return flax.traverse_util.unflatten_dict(flat_mask)  # type: ignore


def w_init(*args):
    return nn.initializers.lecun_normal()(*args) * (2 / NUM_LAYERS)  # type: ignore


def init_transformers(model_config=None):
    if model_config is None:
        model_config = {
            "vocab_size": 7,  # ACTION SIZE
            "output_vocab_size": 2,  # OUTPUT SIZE
            "emb_dim": EMB_SIZE,
            "num_heads": NUM_HEADS,
            "qkv_dim": EMB_SIZE,
            "mlp_dim": EMB_SIZE,
            "num_layers": 6,  # default is 6
            "max_len": 100,
            "kernel_init": functools.partial(w_init, num_layers=6),
            "logits_via_embedding": False,
        }
    TrainConfig = TransformerConfig(
        **model_config, dropout_rate=DROPOUT_RATE, attention_dropout_rate=DROPOUT_RATE, decode=False
    )
    TestConfig = TransformerConfig(
        **model_config,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        decode=False,
        deterministic=True,
    )
    global train_model, test_model
    train_model = TransformerLM(TrainConfig)
    test_model = TransformerLM(TestConfig)


def get_logits(params, rng, batch, is_training, pos_ids=None, center_logits=False, both=False):
    if is_training:
        logits = train_model.apply(
            {"params": params},
            batch.inputs,
            inputs_positions=pos_ids,
            both=both,
            rngs={"dropout": rng},
        )
    else:
        logits = test_model.apply(
            {"params": params}, batch.inputs, inputs_positions=pos_ids, both=both
        )
    if center_logits:
        logits = logits - jnp.mean(logits, axis=(-1, -2))[..., None, None]
    return logits


# Training


class Batch(NamedTuple):
    inputs: np.ndarray  # Integer tokens, shape [B, T]
    targets: np.ndarray  # Integer tokens, shape [B, T]


class TrainingState(NamedTuple):
    """Container for the training state."""

    params: Any
    opt_state: optax.OptState
    rng: Any
    step: int


def create_learning_rate_fn(
    train_ds_size: int,
    train_batch_size: int,
    num_train_epochs: int,
    num_warmup_steps: int,
    learning_rate: float,
) -> Callable[[int], jnp.ndarray]:
    """Returns a linear warmup, linear_decay learning rate function."""
    # We need to multiply by grad_accum_substeps as we take k 'steps' in the
    # optimizer per actual gradient step
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=num_warmup_steps,
    )
    decay_fn = optax.linear_schedule(
        init_value=learning_rate,
        end_value=0,
        transition_steps=num_train_steps - num_warmup_steps,
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps]
    )
    return schedule_fn  # type: ignore


num_epochs = 10  # N_STEPS // (N_DATA // BATCH_SIZE)
linear_decay_lr_schedule_fn = create_learning_rate_fn(
    N_DATA, PER_DEVICE_BATCH_SIZE, num_epochs, 1_000, LEARNING_RATE
)

optimiser = optax.chain(
    optax.clip_by_global_norm(GRAD_CLIP_VALUE),
    optax.adamw(
        linear_decay_lr_schedule_fn, b1=0.9, b2=0.999, weight_decay=WEIGHT_DECAY, mask=decay_mask_fn  # type: ignore
    ),
)


# @functools.partial(jax.pmap, static_broadcasted_argnums=[2, 3])
@jax.jit
def init(rng: jnp.ndarray, data, sequence_length: int, action_size: int) -> TrainingState:
    rng, init_rng = jax.random.split(rng)  # type: ignore
    initial_params = flax.core.frozen_dict.unfreeze(
        train_model.init(
            {"params": rng, "dropout": init_rng},
            jnp.zeros((PER_DEVICE_BATCH_SIZE, SEQUENCE_LENGTH, VOCAB_SIZE), dtype=jnp.float32),
        )["params"]
    )
    initial_opt_state = optimiser.init(initial_params)
    return TrainingState(
        params=initial_params,
        opt_state=initial_opt_state,
        rng=rng,
        step=np.array(0),  # type: ignore
    )


def mle_loss_fn(params, rng, data: Batch, is_training=True) -> jnp.ndarray:
    """Computes the (scalar) LM loss on `data` w.r.t. params."""
    # The architecture gives out *NON-SHIFTED* logits.
    # So given an input [a, b, c], the outputs are [logits(target | a), logits(target | a,b), logits(target | a,b,c)]
    logits = get_logits(params, rng, data, is_training)
    # targets = jax.nn.one_hot(data.targets, VOCAB_SIZE)
    targets = data.targets
    assert logits.shape == targets.shape  # type: ignore

    log_likelihood = jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
    return -jnp.mean(log_likelihood)  # NLL per token.


def mle_loss_trajectory_fn(params, rng, data: Batch, is_training=True) -> jnp.ndarray:
    """Computes the (SEQ_LEN,) LM loss on `data` w.r.t. params."""
    logits = get_logits(params, rng, data, is_training)
    targets = data.targets
    assert logits.shape == targets.shape  # type: ignore
    log_likelihood = jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
    # return LL over trajectory
    return -jnp.mean(log_likelihood, axis=0)


@functools.partial(jax.pmap, axis_name="batch", donate_argnums=[0])
def update(state: TrainingState, data) -> TrainingState:
    """Does an SGD step and returns metrics."""
    rng, new_rng = jax.random.split(state.rng, 2)  # type: ignore
    loss_and_grad_fn = jax.value_and_grad(mle_loss_fn)
    loss, gradients = loss_and_grad_fn(state.params, rng, data)
    gradients = jax.lax.pmean(gradients, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")

    updates, new_opt_state = optimiser.update(gradients, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    new_state = TrainingState(
        params=new_params,
        opt_state=new_opt_state,
        rng=new_rng,
        step=state.step + 1,
    )
    return new_state


def load_rollouts(n_data, rng_key, loaded_params, model_config, layers: Dict, do_random):
    network_factory = ppo_networks.make_ppo_networks
    normalize = running_statistics.normalize
    ppo_network = network_factory(
        env.observation_size, env.action_size, preprocess_observations_fn=normalize, **layers
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    make_inference_fn = make_policy
    inference_fn = make_inference_fn(loaded_params)

    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(inference_fn)

    sequence_length = model_config["max_len"]

    def get_rollout(rng):
        curr_state = jit_env_reset(rng=rng)
        hidden_ball = curr_state.qp.pos[env._hidden_idx][0]
        targets = jnp.tile(jax.nn.one_hot(hidden_ball, 2)[None, ...], (sequence_length, 1))

        def inner(state_i, rng_i):
            act_rng, rng = jax.random.split(rng_i)
            if do_random:
                action = jax.random.uniform(act_rng, low=-1, high=1, shape=(env.action_size,))
            else:
                action, _ = jit_inference_fn(state_i.obs, act_rng)
            state_i = jit_env_step(state_i, action)
            return state_i, action

        mykeys = jax.random.split(rng, sequence_length)
        _, out = jax.lax.scan(inner, curr_state, mykeys)
        return out, targets

    keys = jax.random.split(rng_key, n_data)
    return jax.jit(jax.vmap(get_rollout))(keys)


def get_dataloader(n_data, seed, params_path, model_config, layers, do_random):
    loaded_model = model.load_params(params_path)
    inputs, targets = load_rollouts(
        n_data, jax.random.PRNGKey(seed), loaded_model, model_config, layers, do_random
    )
    i = 0

    TOTAL_BATCHES = n_data // TOTAL_BATCHSIZE
    while True:
        i = (i + 1) % TOTAL_BATCHES
        inputs_batch = jnp.array(inputs[TOTAL_BATCHSIZE * i : (TOTAL_BATCHSIZE) * (i + 1), :-1])
        targets_batch = jnp.array(targets[TOTAL_BATCHSIZE * i : (TOTAL_BATCHSIZE) * (i + 1), :-1])
        inputs_batch = shard(inputs_batch)
        targets_batch = shard(targets_batch)
        yield Batch(inputs=inputs_batch, targets=targets_batch)


def full_trajectory_MI(
    model_path=None,
    seed: int = 0,
    model_config=None,
    warmup_state=None,
    layers=None,
    do_random=False,
):
    """
    Trains predictor (transformer) to compute the full-trajectory MI of a model
    """
    init_transformers(model_config)
    trainloader = get_dataloader(N_DATA, 0, model_path, model_config, layers, do_random)
    testloader = get_dataloader(N_DATA, 1, model_path, model_config, layers, do_random)
    if warmup_state is None:
        state = init(
            jax.random.PRNGKey(seed),
            flax.jax_utils.unreplicate(next(trainloader)),
            model_config["max_len"],
            model_config["vocab_size"],
        )
        state = flax.jax_utils.replicate(state)
    else:
        state = warmup_state
    best_eval_loss = np.inf
    no_improvement_limit = 3
    no_improvement_count = 0
    epochs_per_eval = 10
    steps_per_eval = epochs_per_eval * N_DATA // TOTAL_BATCHSIZE
    for i in range(steps_per_eval * 10):
        train_batch = next(trainloader)
        state = update(state, train_batch)
        if i % steps_per_eval == 0:
            trainloader = get_dataloader(
                N_DATA, i + 1234, model_path, model_config, layers, do_random
            )
            eval_batch = next(testloader)
            eval_loss = jax.jit(mle_loss_fn)(
                flax.jax_utils.unreplicate(state.params),
                flax.jax_utils.unreplicate(state.rng),
                flax.jax_utils.unreplicate(eval_batch),
            )
            eval_loss_trajectory = jax.jit(mle_loss_trajectory_fn)(
                flax.jax_utils.unreplicate(state.params),
                flax.jax_utils.unreplicate(state.rng),
                flax.jax_utils.unreplicate(eval_batch),
            )
            print(f"Step {i}, eval loss is {eval_loss}")

            eval_loss_trajectory = np.array(jnp.exp(-eval_loss_trajectory))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            if no_improvement_count > no_improvement_limit:
                break
    plt.clf()
    plt.bar(range(len(eval_loss_trajectory)), eval_loss_trajectory)
    plt.title(f"step{i // 1000}")
    plt.savefig("bar.png")
    dic = {
        "eval_loss": eval_loss,
        "trajectory_loss_img": wandb.Image("bar.png"),
        "full_trajectory_MI": float(-eval_loss + np.log(2)),
    }
    return dic, state


if __name__ == "__main__":
    # TODO: be wary of the model config
    train()
