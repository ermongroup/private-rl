# Copyright 2022 The Brax Authors.
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

"""PPO networks."""

from typing import Sequence, Tuple, Callable, Any
import dataclasses

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class FeedForwardNetwork:
    init: Callable[..., Any]
    apply: Callable[..., Any]


# Too much effort at the moment to fill out all the methods
# for distribution.ParametricDistribution
class MultiNomial:
    def __init__(self, event_size):
        self.param_size = event_size

    def log_prob(self, logits, x):
        # x needs to be fed in as one-hot
        assert logits.shape == x.shape
        all_log_probs = jax.nn.log_softmax(logits, axis=-1)
        return jnp.sum(all_log_probs * x, axis=-1)


@flax.struct.dataclass
class PPONetworks:
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_inference_fn(ppo_networks: PPONetworks):
    """Creates params and inference function for the PPO agent."""

    def make_policy(params: types.PolicyParams) -> types.Policy:
        policy_network = ppo_networks.policy_network
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def policy(
            observations: types.Observation, key_sample: PRNGKey
        ) -> Tuple[types.Action, types.Extra]:
            logits = policy_network.apply(*params, observations)
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(raw_actions)
            return postprocessed_actions, {"log_prob": log_prob, "raw_action": raw_actions}

        return policy

    return make_policy


def make_ppo_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: networks.ActivationFn = linen.swish,
    discrete_target_rv: bool = False,
    min_std: float = 0.01,
) -> PPONetworks:
    """Make PPO networks with preprocessor."""
    if discrete_target_rv:
        parametric_action_distribution = MultiNomial(event_size=action_size)
    else:
        parametric_action_distribution = distribution.NormalTanhDistribution(
            event_size=action_size, min_std=0.05
        )
    policy_network = networks.make_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        activation=activation,
    )
    value_network = networks.make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
    )

    return PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_policy_network(
    param_size: int,
    obs_size: int,
    preprocess_observations_fn,
    hidden_layer_sizes=(256, 256),
    activation=linen.relu,
):
    """Creates a policy network."""
    policy_module = D2MLP(
        layer_sizes=list(hidden_layer_sizes) + [param_size],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs)

    dummy_obs = jnp.zeros((1, obs_size))
    return FeedForwardNetwork(init=lambda key: policy_module.init(key, dummy_obs), apply=apply)


def make_value_network(
    obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation=linen.relu,
):
    """Creates a policy network."""
    value_module = D2MLP(
        layer_sizes=list(hidden_layer_sizes) + [1],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return jnp.squeeze(value_module.apply(policy_params, obs), axis=-1)

    dummy_obs = jnp.zeros((1, obs_size))
    return FeedForwardNetwork(init=lambda key: value_module.init(key, dummy_obs), apply=apply)


class D2MLP(linen.Module):
    """D2MLP module."""

    layer_sizes: Sequence[int]
    activation: linen.relu
    kernel_init: jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            if i != 0:
                hidden = jnp.concatenate((hidden, data), axis=1)
            hidden = linen.Dense(
                hidden_size, name=f"hidden_{i}", kernel_init=self.kernel_init, use_bias=self.bias
            )(hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                hidden = self.activation(hidden)
        return hidden
