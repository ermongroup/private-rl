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

"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

import functools
import time
from typing import Callable, Optional, Tuple, Sequence, Any

from absl import logging
from brax import envs
from brax.envs import wrappers
from brax.training import acting
from brax.training import gradients
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.training.types import Metrics
from brax.training.types import Policy
from brax.training.types import Transition
from brax.envs.env import State
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from ppo import losses as ppo_losses
from ppo import networks as ppo_networks
import predictor
from brax.io import model
from optax._src.transform import ScaleByAdamState
from optax._src.wrappers import MultiStepsState
from optax._src.base import EmptyState
from ppo.losses import TrainingParams
from private_envs import turningant
import brax_utils

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = "i"


def loss_and_pgrad(
    loss_fn: Callable[..., float], pmap_axis_name: Optional[str], has_aux: bool = False
):
    g = jax.value_and_grad(loss_fn, has_aux=has_aux)

    def h(*args, **kwargs):
        value, grad = g(*args, **kwargs)
        return value, jax.lax.pmean(grad, axis_name=pmap_axis_name)

    return g if pmap_axis_name is None else h


def gradient_update_fn_custom(
    loss_fn: Callable[..., float],
    optimizer: optax.GradientTransformation,
    pmap_axis_name: Optional[str],
    has_aux: bool = False,
):
    """Wrapper of the loss function that apply gradient updates.

    Args:
      loss_fn: The loss function.
      optimizer: The optimizer to apply gradients.
      pmap_axis_name: If relevant, the name of the pmap axis to synchronize
        gradients.
      has_aux: Whether the loss_fn has auxiliary data.

    Returns:
      A function that takes the same argument as the loss function plus the
      optimizer state. The output of this function is the loss, the new parameter,
      and the new optimizer state.
    """
    loss_and_pgrad_fn = loss_and_pgrad(loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux)

    def f(*args, optimizer_state):
        params = args[0]
        value, grads = loss_and_pgrad_fn(*args)
        params_update, optimizer_state = optimizer.update(grads, optimizer_state, params)
        params = optax.apply_updates(args[0], params_update)
        return value, params, optimizer_state

    return f


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    # Contains opt state, (I, MI_prev) for the PID controller
    joint_optimizer_state: Tuple[optax.OptState, ppo_losses.PID_state]
    params_both: ppo_losses.TrainingParams
    normalizer_params: ppo_losses.NormalizerParams
    env_steps: jnp.ndarray


def _unpmap(v):
    return jax.tree_map(lambda x: x[0], v)


def generate_unroll_full(
    env: envs.Env,
    env_state: envs.State,
    policy: Policy,
    key: PRNGKey,
    unroll_length: int,
    extra_fields: Sequence[str] = (),
) -> Tuple[envs.State, envs.State, Transition]:
    """Collect trajectories AND STATES of given unroll_length."""

    @jax.jit
    def f(carry, unused_t):
        state, current_key = carry
        current_key, next_key = jax.random.split(current_key)
        nstate, transition = acting.actor_step(
            env, state, policy, current_key, extra_fields=extra_fields
        )
        return (nstate, next_key), (state, transition)

    (final_state, _), (states, data) = jax.lax.scan(f, (env_state, key), (), length=unroll_length)
    return final_state, data, states


def train(
    environment: envs.Env,
    num_timesteps: int,
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    max_devices_per_host: Optional[int] = None,
    num_eval_envs: int = 128,
    learning_rate=1e-4,
    entropy_cost=1e-4,
    discounting=0.9,
    seed=0,
    unroll_length=10,
    batch_size=32,
    num_minibatches=16,
    num_updates_per_batch=2,
    num_evals=1,
    normalize_observations=False,
    reward_scaling=1.0,
    get_us_fn=None,  # index of u in flat obs
    network_factory: types.NetworkFactory[
        ppo_networks.PPONetworks
    ] = ppo_networks.make_ppo_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    pid_parameters=None,
    horizon=10,
    n_truncated_rollouts=32,
    subsample_batch_size=5,
    MI_eps: float = 10,
    run_name: str = None,
    transformer_config: dict = None,
    target_kl: float = 0.1,
    debug: bool = False,
):
    """PPO training."""
    if pid_parameters is None:
        k_p, k_i, k_d = 1.0, 0.1, 0.001
    else:
        k_p, k_i, k_d = pid_parameters
    assert batch_size * num_minibatches % num_envs == 0
    xt = time.time()

    config = {
        "num_timesteps": num_timesteps,
        "episode_length": episode_length,
        "action_repeat": action_repeat,
        "num_envs": num_envs,
        "max_devices_per_host": max_devices_per_host,
        "num_eval_envs": num_eval_envs,
        "learning_rate": learning_rate,
        "entropy_cost": entropy_cost,
        "discounting": discounting,
        "seed": seed,
        "unroll_length": unroll_length,
        "batch_size": batch_size,
        "num_minibatches": num_minibatches,
        "num_updates_per_batch": num_updates_per_batch,
        "num_evals": num_evals,
        "normalize_observations": normalize_observations,
        "reward_scaling": reward_scaling,
        "k_p": k_p,
        "k_i": k_i,
        "k_d": k_d,
        "horizon": horizon,
        "n_truncated_rollouts": n_truncated_rollouts,
        "subsample_batch_size": subsample_batch_size,
        "MI_eps": MI_eps,
        "target_kl": target_kl,
        "debug": debug,
    }

    wandb.run.config.update(config)
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    logging.info(
        "Device count: %d, process count: %d (id %d), local device count: %d, "
        "devices to be used count: %d",
        jax.device_count(),
        process_count,
        process_id,
        local_device_count,
        local_devices_to_use,
    )
    device_count = local_devices_to_use * process_count

    # The number of environment steps executed for every training step.
    env_step_per_training_step = batch_size * unroll_length * num_minibatches * action_repeat
    env_step_per_training_step += n_truncated_rollouts * num_minibatches * action_repeat * horizon
    num_evals_after_init = max(num_evals - 1, 1)
    # The number of training_step calls per training_epoch call.
    # equals to ceil(num_timesteps / (num_evals * env_step_per_training_step))
    num_training_steps_per_epoch = -(
        -num_timesteps // (num_evals_after_init * env_step_per_training_step)
    )
    print(f"Num training steps per eval step: {num_training_steps_per_epoch}")
    print(f"Num gradient steps per training step: {num_updates_per_batch}")

    assert num_envs % device_count == 0
    serial_env = environment
    env = environment
    env = wrappers.EpisodeWrapper(env, episode_length, action_repeat)
    env = wrappers.VmapWrapper(env)
    env = wrappers.AutoResetWrapper(env)
    # reset_fn = jax.jit(jax.vmap(env.reset))
    reset_fn = jax.pmap(env.reset)

    normalize = lambda x, y: x
    if normalize_observations:
        normalize = running_statistics.normalize
    
    layers = {"policy_hidden_layer_sizes": (32,) * 4, "value_hidden_layer_sizes": (256,) * 5}
    ppo_network = network_factory(
        env.observation_size, env.action_size, preprocess_observations_fn=normalize, **layers
    )

    make_policy = ppo_networks.make_inference_fn(ppo_network)
    total_steps = num_training_steps_per_epoch * num_minibatches * num_evals

    warmup_cosine_decay_scheduler = brax_utils.warmup_flat_cosine_decay_schedule(
        init_value=learning_rate / 100,
        peak_value=learning_rate,
        warmup_steps=int(total_steps * 0.1),
        flat_steps=int(total_steps * 0.5),
        total_steps=total_steps,
        end_value=learning_rate / 10,
    )

    optlist = [
        optax.clip_by_global_norm(10),
        optax.adamw(learning_rate=warmup_cosine_decay_scheduler, weight_decay=1e-8),
    ]
    optimizer = optax.chain(*optlist)
    skip_update_norm_square = 1e16
    grad_accumulation_steps = 1
    optimizer = optax.MultiSteps(
        optimizer,
        grad_accumulation_steps,
        should_skip_update_fn=functools.partial(
            optax.skip_large_updates,
            max_squared_norm=skip_update_norm_square,
        ),
    )
    # only_predictor_optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=1e-5)

    if MI_eps != 10.0:
        loss_fn = functools.partial(
            ppo_losses.compute_ppo_loss,
            ppo_network=ppo_network,
            entropy_cost=entropy_cost,
            discounting=discounting,
            reward_scaling=reward_scaling,
            get_us_fn=get_us_fn,
            n_truncated_rollouts=n_truncated_rollouts,
            subsample_batch_size=subsample_batch_size,
            horizon=horizon,
            env=env,
            action_repeat=action_repeat,
            T=episode_length,
            do_regularization=MI_eps != 10.0,
        )
    else:
        loss_fn = functools.partial(
            ppo_losses.compute_ppo_loss_unconstrained,
            ppo_network=ppo_network,
            entropy_cost=entropy_cost,
            discounting=discounting,
            reward_scaling=reward_scaling,
            get_us_fn=get_us_fn,
            n_truncated_rollouts=n_truncated_rollouts,
            subsample_batch_size=subsample_batch_size,
            horizon=horizon,
            env=env,
            action_repeat=action_repeat,
            T=episode_length,
            do_regularization=MI_eps != 10.0,
        )

    gradient_update_fn = gradient_update_fn_custom(
        loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
    )

    mle_loss_fn = functools.partial(
        ppo_losses.mle_loss_fn_from_both,
        horizon=horizon,
        subsample_batch_size=subsample_batch_size,
        get_us_fn=get_us_fn,
        is_training=False,
    )

    mle_gradient_update_fn = gradient_update_fn_custom(
        mle_loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=False
    )

    def mle_minibatch_step(carry, _xs, data, states):
        optimizer_state, params, key = carry
        _, params, optimizer_state_0 = mle_gradient_update_fn(
            params,
            key,
            data,
            states.info["steps"],
            optimizer_state=optimizer_state[0],
        )
        return ((optimizer_state_0, optimizer_state[1]), params, jax.random.split(key, 1)[0]), None

    def minibatch_step(
        carry,
        inputs: Tuple[types.Transition, envs.State, Any],
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        data, states, log_baseline_probs = inputs
        optimizer_state, params, key, should_skip = carry
        key, key_loss = jax.random.split(key)

        def do_update_fn(params, normalizer_params, data, states, optimizer_state):
            (_, metrics), params, optimizer_state_0 = gradient_update_fn(
                params,
                normalizer_params,
                data,
                states,
                key_loss,
                optimizer_state[1],
                optimizer_state=optimizer_state[0],
            )
            optimizer_state = (optimizer_state_0, metrics["PID_state"])
            logits = ppo_network.policy_network.apply(
                normalizer_params.ppo_norm_params, params.ppo_params.policy, data.observation
            )
            action_log_probs = ppo_network.parametric_action_distribution.log_prob(
                logits, data.extras["policy_extras"]["raw_action"]
            )
            # kl = jnp.mean(log_baseline_probs - action_log_probs)
            log_r = action_log_probs - log_baseline_probs
            kl = jnp.mean(jnp.exp(log_r) - 1 - log_r)
            metrics["kl"] = kl
            metrics["skipped"] = False
            metrics["mean_grad_norm_squared"] = optimizer_state_0.skip_state["norm_squared"]  # type: ignore
            should_skip = kl > target_kl * 1.5

            return metrics, params, optimizer_state, should_skip

        def do_nothing_fn(params, normalizer_params, data, states, optimizer_state):
            metrics = {
                "PID_state": ppo_losses.PID_state(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                "d_lm_contribution": 0.0,
                "effective_truncated_loss": 0.0,
                "entropy_loss": 0.0,
                "i_lm_contribution": 0.0,
                "kl": 0.0,
                "lm": 0.0,
                "p_lm_contribution": 0.0,
                "policy_loss": 0.0,
                "predictor_loss": 0.0,
                "total_loss": 0.0,
                "truncated_MI": 0.0,
                "v_loss": 0.0,
                "skipped": True,
                "mean_grad_norm_squared": jnp.inf,
            }
            return metrics, params, optimizer_state, should_skip

        metrics, params, optimizer_state, should_skip = jax.lax.cond(
            should_skip,
            do_nothing_fn,
            do_update_fn,
            params,
            normalizer_params,
            data,
            states,
            optimizer_state,
        )

        # Compute KL
        # log_baseline_probs - log ppo_network(states, actions)

        return (optimizer_state, params, key, should_skip), metrics

    def sgd_step(
        carry,
        unused_t,
        data: types.Transition,
        states: envs.State,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        # probs = ppo_network.apply(data.observation, data.action)
        shuffled_data = jax.tree_map(convert_data, data)
        shuffled_states = jax.tree_map(convert_data, states)
        logits = ppo_network.policy_network.apply(
            normalizer_params.ppo_norm_params, params.ppo_params.policy, data.observation
        )
        action_log_probs = ppo_network.parametric_action_distribution.log_prob(
            logits, data.extras["policy_extras"]["raw_action"]
        )
        shuffled_log_probs = jax.tree_map(convert_data, action_log_probs)

        # Get states, actions, store log_probs and pass into minibatch step
        # In minibatch step, if KL > delta, return identity

        (optimizer_state, params, _, _), metrics = jax.lax.scan(
            functools.partial(minibatch_step, normalizer_params=normalizer_params),
            (optimizer_state, params, key_grad, False),
            (
                shuffled_data,
                shuffled_states,
                shuffled_log_probs,
            ),
            length=num_minibatches,
        )

        return (optimizer_state, params, key), metrics

    def training_step(
        carry: Tuple[TrainingState, envs.State, PRNGKey, int], unused_t
    ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
        training_state, state, key, num_minibatches_last = carry
        key_sgd, key_mle, key_generate_unroll, new_key = jax.random.split(key, 4)

        policy = make_policy(
            (
                training_state.normalizer_params.ppo_norm_params,
                training_state.params_both.ppo_params.policy,
            )
        )

        def f(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data, all_states = generate_unroll_full(
                env,
                current_state,
                policy,
                current_key,
                unroll_length,
                extra_fields=("truncation",),
            )
            return (next_state, next_key), (data, all_states)

        (state, _), (data, states) = jax.lax.scan(
            f, (state, key_generate_unroll), (), length=batch_size * num_minibatches // num_envs
        )
        # Have leading dimentions (batch_size * num_minibatches, unroll_length)
        data = jax.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)

        states = jax.tree_map(lambda x: jnp.swapaxes(x, 1, 2), states)
        states = jax.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), states)

        assert data.discount.shape[1:] == (unroll_length,)

        # Update normalization params and normalize observations.
        ppo_norm_params = running_statistics.update(
            training_state.normalizer_params.ppo_norm_params,
            data.observation,
            pmap_axis_name=_PMAP_AXIS_NAME,
            weights=jnp.ones(data.observation.shape[:-1]) * num_minibatches_last / num_minibatches,
        )
        predictor_norm_params = running_statistics.update(
            training_state.normalizer_params.predictor_norm_params,
            data.extras["policy_extras"]["raw_action"],
            pmap_axis_name=_PMAP_AXIS_NAME,
        )
        normalizer_params = ppo_losses.NormalizerParams(ppo_norm_params, predictor_norm_params)

        # Predictor Update
        # ----------
        if MI_eps != 10.0:
            (_new_optimizer_state, new_params_both, _), _ = jax.lax.scan(
                functools.partial(
                    mle_minibatch_step,
                    data=data,
                    states=states,
                ),
                (training_state.joint_optimizer_state, training_state.params_both, key_mle),
                (),
                length=num_minibatches // 2,  # Update predictor more frequently than policy
            )
            params_both = ppo_losses.TrainingParams(
                ppo_params=training_state.params_both.ppo_params,
                predictor_params=new_params_both.predictor_params,
            )
            old_opt_state = training_state.joint_optimizer_state[0]
            old_inner_adam_state = old_opt_state.inner_opt_state[1][0]
            new_inner_adam_state = _new_optimizer_state[0].inner_opt_state[1][0]
            # MultiStepsState(mini_step=old_opt_state.mini_step, gradient_step=old_opt_state.gradient_step, new_optimizer_state[0].inner_opt_state,
            new_inner_opt_state = (
                EmptyState(),
                (
                    ScaleByAdamState(
                        count=old_inner_adam_state.count,
                        mu=TrainingParams(
                            ppo_params=old_inner_adam_state.mu.ppo_params,
                            predictor_params=new_inner_adam_state.mu.predictor_params,
                        ),
                        nu=TrainingParams(
                            ppo_params=old_inner_adam_state.nu.ppo_params,
                            predictor_params=new_inner_adam_state.nu.predictor_params,
                        ),
                    ),
                    EmptyState(),
                ),
                EmptyState(),
            )
            new_opt_state = MultiStepsState(
                mini_step=old_opt_state.mini_step,
                gradient_step=old_opt_state.gradient_step,
                inner_opt_state=new_inner_opt_state,
                acc_grads=old_opt_state.acc_grads,
                skip_state=old_opt_state.skip_state,
            )
            optimizer_state = (new_opt_state, training_state.joint_optimizer_state[1])
        # ----------
        # End predictor update

        (optimizer_state, params_both, _), metrics = jax.lax.scan(
            functools.partial(
                sgd_step, data=data, states=states, normalizer_params=normalizer_params
            ),
            (training_state.joint_optimizer_state, training_state.params_both, key_sgd),
            (),
            length=num_updates_per_batch,
        )

        fraction_completed = jnp.sum(jnp.bitwise_not(metrics["skipped"])) / num_minibatches
        new_training_state = TrainingState(
            joint_optimizer_state=optimizer_state,
            params_both=params_both,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps
            + (fraction_completed * env_step_per_training_step).astype(jnp.int32),
        )
        mean_mask = jnp.bitwise_not(metrics["skipped"])
        nonskipped_minibatches = jnp.cumsum(mean_mask, axis=-1)[-1, -1]

        return (new_training_state, state, new_key, nonskipped_minibatches), metrics

    def training_epoch(
        training_state: TrainingState, state: envs.State, key: PRNGKey
    ) -> Tuple[TrainingState, envs.State, Metrics]:
        (training_state, state, _, _), loss_metrics = jax.lax.scan(
            training_step,
            (training_state, state, key, num_minibatches),
            (),
            length=num_training_steps_per_epoch,
        )
        mean_mask = jnp.bitwise_not(loss_metrics["skipped"])
        masked_mean = lambda x: jnp.sum(mean_mask * x) / jnp.sum(mean_mask)
        nonskipped_minibatches = jnp.cumsum(mean_mask, axis=-1)[:, :, -1]
        # jax.debug.breakpoint()
        loss_metrics["min_grad_norm_squared"] = jnp.min(loss_metrics["mean_grad_norm_squared"])  # type: ignore
        loss_metrics = jax.tree_map(masked_mean, loss_metrics)
        loss_metrics["n_minibatches_nonskipped"] = nonskipped_minibatches

        return training_state, state, loss_metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
        training_state: TrainingState, env_state: envs.State, key: PRNGKey
    ) -> Tuple[TrainingState, envs.State, Metrics]:
        nonlocal training_walltime
        t = time.time()
        (training_state, env_state, metrics) = training_epoch(training_state, env_state, key)
        # This line also guarantees the values are ready.
        # Chuck away skipped steps
        # breakpoint()
        # skip_id = jnp.where(metrics["skipped"])[0]
        # if len(skip_id) == 0:
        #     pass
        # else:
        #     metrics = {k: v[:skip_id] for k, v in metrics.items()}
        metrics = jax.tree_map(jnp.mean, metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        fraction_completed = metrics["n_minibatches_nonskipped"] / num_minibatches
        sps = (
            fraction_completed
            * (num_training_steps_per_epoch * env_step_per_training_step)
            / epoch_training_time
        )
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return training_state, env_state, metrics

    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    # key_networks should be global, so that networks are initialized the same
    # way for different processes.
    key_policy, key_value, key_predictor = jax.random.split(global_key, 3)
    del global_key

    # Initialize Policy and Predictor Networks
    init_params = ppo_losses.PPONetworkParams(
        policy=ppo_network.policy_network.init(key_policy),
        value=ppo_network.value_network.init(key_value),
    )
    predictor.init_transformers(transformer_config)
    init_predictor_params = ppo_losses.UPredictorNetworkParams(
        transformer_params=predictor.init(
            key_predictor, None, transformer_config["max_len"], transformer_config["vocab_size"]
        ).params
    )

    init_params_both = ppo_losses.TrainingParams(
        ppo_params=init_params, predictor_params=init_predictor_params
    )
    init_norm_params = ppo_losses.NormalizerParams(
        ppo_norm_params=running_statistics.init_state(
            specs.Array((env.observation_size,), jnp.float32)
        ),
        predictor_norm_params=running_statistics.init_state(
            specs.Array((env.action_size,), jnp.float32)
        ),
    )
    init_PID = ppo_losses.PID_state(
        MI_prev=jnp.array(1.0, dtype=jnp.float32),
        I=jnp.array(0.0, dtype=jnp.float32),
        k_p=jnp.array(k_p, dtype=jnp.float32),
        k_i=jnp.array(k_i, dtype=jnp.float32),
        k_d=jnp.array(k_d, dtype=jnp.float32),
        MI_eps=jnp.array(MI_eps, dtype=jnp.float32),
        lm=jnp.array(1.0, dtype=jnp.float32),
    )
    training_state = TrainingState(
        joint_optimizer_state=(optimizer.init(init_params_both), init_PID),
        params_both=init_params_both,
        normalizer_params=init_norm_params,
        env_steps=jnp.array(0.0, dtype=jnp.float32),
    )
    training_state = jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )

    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])

    # This is a bunch of stupid stuff necessary to get the compilation to recognise that
    # the output of the reset_fn has the same type signature as the output of 'update'
    #
    env_state = reset_fn(key_envs)
    env_state.info["truncation"] = jax.device_put_replicated(
        jax.lax.convert_element_type_p.bind(
            env_state.info["truncation"][0], new_dtype=jnp.float32, weak_type=True
        ),
        jax.local_devices(),
    )
    if type(env) is turningant.TurningAnt:
        env_state.metrics["reward_survive"] = jax.device_put_replicated(
            jax.lax.convert_element_type_p.bind(
                env_state.metrics["reward_survive"][0], new_dtype=jnp.float32, weak_type=True
            ),
            jax.local_devices(),
        )

    env_dones = jax.device_put_replicated(
        jax.lax.convert_element_type_p.bind(
            env_state.done[0], new_dtype=jnp.float32, weak_type=True
        ),
        jax.local_devices(),
    )
    env_state = State(
        qp=env_state.qp,
        obs=env_state.obs,
        reward=env_state.reward,
        done=env_dones,
        metrics=env_state.metrics,
        info=env_state.info,
    )

    evaluator = acting.Evaluator(
        env,
        make_policy,
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

    # get_type_diff = lambda l1, l2: "same" if get_dtype(l1) == get_dtype(l2) else f"DIFF -- {get_dtype(l1)} vs {get_dtype(l2)}"
    # Run initial eval
    # if process_id == 0 and num_evals > 1:
    #     metrics = evaluator.run_evaluation(
    #         _unpmap(
    #             (
    #                 training_state.normalizer_params.ppo_norm_params,
    #                 training_state.params_both.ppo_params.policy,
    #             )
    #         ),
    #         training_metrics={},
    #     )
    #     logging.info(metrics)
    #     progress_fn(0, metrics)

    training_walltime = 0
    current_step = 0
    eval_predictor_state = None
    carry_out_full_eval = True
    time_taken_for_epoch = 0.0
    time_taken_for_eval = 200
    last_video_time = time.time()
    for it in range(num_evals_after_init):
        print(f"starting iteration {it} at time {round(time.time() - xt, 5)}")

        # optimization
        epoch_key, local_key = jax.random.split(local_key)
        epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
        t0 = time.time()
        # ts_0 = training_state
        # es_0 = env_state
        (training_state, env_state, training_metrics) = training_epoch_with_timing(
            training_state, env_state, epoch_keys
        )
        time_taken_for_epoch = time.time() - t0
        if it == 0:
            print(f"Compiled and ran first epoch in {round(time_taken_for_epoch, 5)}")
        else:
            print(f"Finished epoch {it} in {round(time_taken_for_epoch, 5)}")

        current_step = int(_unpmap(training_state.env_steps))
        # print(training_metrics, "\n\n\n")
        if process_id == 0:
            # Run evals.
            params = _unpmap(
                (
                    training_state.normalizer_params.ppo_norm_params,
                    training_state.params_both.ppo_params.policy,
                )
            )
            metrics = evaluator.run_evaluation(
                params,
                training_metrics,
            )
            logging.info(metrics)
            progress_fn(current_step, metrics)
            carry_out_full_eval = time_taken_for_epoch > time_taken_for_eval
            if carry_out_full_eval:
                t00 = time.time()
                current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
                model_path = "./params/tmp_params_" + run_name + "_" + current_time
                print(f"Saving to {model_path}")
                model.save_params(model_path, params)
                full_eval_dict, eval_predictor_state = predictor.full_trajectory_MI(
                    model_path,
                    seed=seed,
                    model_config=transformer_config,
                    warmup_state=eval_predictor_state,
                    layers=layers,
                )
                time_taken_for_eval = time.time() - t00
            else:
                print("skipping eval since eval took much longer than training")
            wandb_keys = [
                "training/policy_loss",
                "training/predictor_loss",
                "training/entropy_loss",
                "training/total_loss",
                "eval/episode_reward",
                "eval/episode_reward_ctrl",
                "training/lm",
                "training/truncated_MI",
                "training/effective_truncated_loss",
                "training/p_lm_contribution",
                "training/i_lm_contribution",
                "training/d_lm_contribution",
                "training/sps",
                "training/walltime",
                "training/kl",
                "training/n_minibatches_nonskipped",
                "training/mean_grad_norm_squared",
                "training/min_grad_norm_squared",
                "eval/episode_t",
                "eval/avg_episode_length",
                "eval/epoch_eval_time",
                "eval/sps",
            ]
            if type(serial_env) is turningant.TurningAnt:
                wandb_keys += [
                    "eval/episode_reward_forward",
                    "eval/episode_reward_survive",
                    "eval/episode_reward_contact",
                ]
                hfov = 100.0
            else:
                wandb_keys += ["eval/episode_reward_dist", "eval/episode_reward_near"]
                hfov = 29.0

            # dict = {key.split("/")[1]: metrics[key] for key in wandb_keys}
            _dict = {key: round(metrics[key], 5) for key in wandb_keys}
            if carry_out_full_eval:
                _dict = {**_dict, **full_eval_dict}  # type: ignore
            _dict["training/env_steps"] = training_state.env_steps[0]
            _dict["training/normalizer_mean_mean"] = np.mean(
                training_state.normalizer_params.ppo_norm_params.mean
            )
            _dict["training/normalizer_mean_std"] = np.mean(
                training_state.normalizer_params.ppo_norm_params.std
            )
            lr = warmup_cosine_decay_scheduler(
                training_state.joint_optimizer_state[0].inner_opt_state[1][2].count[0]
            )
            _dict["training/lr"] = lr
            # print(dict)
            # dict["opt/update_norm_sq"] = training_state.joint_optimizer_state[0].skip_state['norm_squared'][0]
            # If it's been more than two hours since the last video,
            # we record one
            if time.time() - last_video_time > 3600:
                last_video_time = time.time()
                video = brax_utils.make_video(
                    params,
                    make_policy,
                    serial_env,
                    episode_length,
                    not type(serial_env) is turningant.TurningAnt,
                    it,
                    n_seeds=5,
                    hfov=hfov,
                )
                _dict["vid"] = video
            print(_dict)
            wandb.log(_dict)
            if debug:
                return (make_policy, params, metrics)

    total_steps = current_step
    # assert total_steps >= num_timesteps

    # If there was no mistakes the training_state should still be identical on all
    # devices.
    pmap.assert_is_replicated(training_state)
    params = _unpmap(
        (
            training_state.normalizer_params.ppo_norm_params,
            training_state.params_both.ppo_params.policy,
        )
    )
    logging.info("total steps: %s", total_steps)
    pmap.synchronize_hosts()
    return (make_policy, params, metrics)
