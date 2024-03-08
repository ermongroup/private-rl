import jax.numpy as np
import jax.ops as ops
import jax.random as rnd
from jax import grad, jit, vmap
from jax.experimental import optimizers
from environments import VPNEnvironment, CustomerEnvironment, SNAPEnvironment
from utils import (
    calculate_return,
    fit_general_psi_predictor,
    predict_general_psi_predictor,
    initialize_trajectory,
    plot_customer_trajectory,
    plot_vpn_fig,
    plot_distribution_t_MI_fig,
)
from analyse_acs import (
    compute_KDEs,
    load_race_income_dc,
)

from jax.experimental.stax import softmax, logsoftmax
from jax.tree_util import tree_multimap, tree_flatten
from networks import get_model
import matplotlib.pyplot as plt
import numpy as onp
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--experiment_name", type=str, default="snap")
args = parser.parse_args()
if args.experiment_name == "vpn":
    environment = VPNEnvironment(4, 1, 0.9, 0.5, 0.9, 0, 0.0)
    draw_p0 = environment.draw_p0
    transition = environment.transition
    public_state_dim = environment.public_state_dim
    private_state_dim = environment.private_state_dim
    action_dim = environment.action_dim
    state_dim = public_state_dim + private_state_dim
    use_timestep_indicator = False
    T = 10
    batch_size = 32
    odm = environment.oracle_dynamics_model
    L2_PARAM = 1e-8
    ENTROPY_PARAM = 10.0
    num_epochs = 5000

    critic, critic_params = get_model(state_dim, [64, 64, 1], batch_size)
    policy_logits, policy_params = get_model(state_dim, [256, 256, action_dim], batch_size)

    lr_0 = 3e-3
    # schedule = lambda i: (lr_0 * 0.5) * (1 + np.cos(np.pi * i / NUM_EPOCHS))
    policy_opt_init, policy_opt_update, policy_get_params = optimizers.adam(lr_0, b1=0.9)
    critic_opt_init, critic_opt_update, critic_get_params = optimizers.adam(lr_0, b1=0.9)

if args.experiment_name == "customer":
    environment = CustomerEnvironment(2, 0.25, 0.0, 0.5, 0.0)
    draw_p0 = environment.draw_p0
    transition = environment.transition
    public_state_dim = environment.public_state_dim
    private_state_dim = environment.private_state_dim
    action_dim = environment.action_dim
    state_dim = public_state_dim + private_state_dim
    T = 6
    batch_size = 12
    odm = environment.oracle_dynamics_model
    L2_PARAM = 1e-8
    ENTROPY_PARAM = 0.0
    num_epochs = 5000
    use_timestep_indicator = True

    critic, critic_params = get_model(state_dim, [128, 128, 1], batch_size)
    policy_logits, policy_params = get_model(state_dim + T, [128, 128, action_dim], batch_size)

    lr_0 = 1e-4
    # schedule = lambda i: (lr_0 * 0.5) * (1 + np.cos(np.pi * i / NUM_EPOCHS))
    policy_opt_init, policy_opt_update, policy_get_params = optimizers.adam(lr_0, b1=0.9)
    critic_opt_init, critic_opt_update, critic_get_params = optimizers.adam(lr_0, b1=0.9)


if args.experiment_name == "snap":
    white_KDE, black_KDE, u0_frac = compute_KDEs(load_race_income_dc(), (10000, 10000))
    environment = SNAPEnvironment(white_KDE, black_KDE, u0_frac)
    u0_fraction = u0_frac
    draw_p0 = environment.draw_p0
    transition = environment.transition
    public_state_dim = environment.public_state_dim
    private_state_dim = environment.private_state_dim
    action_dim = environment.action_dim
    state_dim = public_state_dim + private_state_dim
    T = 4
    batch_size = 512
    odm = environment.oracle_dynamics_model
    L2_PARAM = 1e-8
    ENTROPY_PARAM = 0.1
    num_epochs = 1000
    use_timestep_indicator = False

    critic, critic_params = get_model(state_dim, [128, 128, 1], batch_size)
    policy_logits, policy_params = get_model(state_dim, [256, 256, action_dim], batch_size)

    lr_0 = 1e-3
    # schedule = lambda i: (lr_0 * 0.5) * (1 + np.cos(np.pi * i / NUM_EPOCHS))
    policy_opt_init, policy_opt_update, policy_get_params = optimizers.adam(lr_0, b1=0.9)
    critic_opt_init, critic_opt_update, critic_get_params = optimizers.adam(lr_0, b1=0.9)


def l2_parameter_loss(theta):
    return L2_PARAM * 0.5 * sum([np.sum(x**2) for x in tree_flatten(theta)[0]])


def entropy_loss(policy_params, trajectories):
    if use_timestep_indicator:
        T_indicators = np.tile(np.eye(T).reshape((1, T, T)), (batch_size, 1, 1))
        inputs = np.concatenate((trajectories[:, :, :state_dim], T_indicators), axis=-1)
    else:
        inputs = trajectories[:, :, :state_dim]
    actions = policy_logits(policy_params, inputs)
    entropy = -np.sum((logsoftmax(actions, axis=-1) * softmax(actions, axis=-1)), axis=-1)
    return -np.mean(ENTROPY_PARAM * entropy)


def sample_trajectory(theta, rng):
    """Samples a single trajectory from theta-distribution.

    We sample a single trajectory from the distribution
    induced by the parameter theta and the function `policy_logits`.

    Args:
        theta: dictionary of parameters for `policy_logits`
        rng: jax.random.PRNGKey object

    Returns:
        A (Tx(state_dim + action_dim)) matrix of the state
        and action at each timestep
    """
    rewards = np.zeros(
        T,
    )
    rng_1, rng_2, rng_3 = rnd.split(rng, 3)

    state_0 = draw_p0(rng_1)
    trajectory = initialize_trajectory(state_0, T, state_dim, action_dim)

    for t in range(T):
        rng_1, rng_2 = rnd.split(rng_1, 2)
        if use_timestep_indicator:
            inputs = np.concatenate(
                (
                    trajectory[t, :state_dim],
                    np.zeros(
                        T,
                    ),
                )
            )
            inputs = ops.index_update(inputs, ops.index[state_dim + t], 1)
        else:
            inputs = trajectory[t, :state_dim]
        action_logits = policy_logits(theta, inputs)
        action = rnd.categorical(rng_1, action_logits)
        trajectory = ops.index_update(trajectory, ops.index[t, state_dim + action], 1)
        new_state, r = transition(
            trajectory[t, :state_dim],
            action,
            rng_2,
        )  # (x,u), a
        if t != T - 1:
            trajectory = ops.index_update(trajectory, ops.index[t + 1, :state_dim], new_state)
        rewards = ops.index_update(rewards, ops.index[t], r)

    return trajectory, rewards


def sample_trajectories(theta, rng, n):
    rngs = rnd.split(rng, n)
    trajectories, rewards = vmap(sample_trajectory, in_axes=(None, 0))(theta, rngs)
    returns = calculate_return(rewards)
    return trajectories, rewards, returns


def policy_gradient_fn(params, ts, advantages):
    """Function whose gradient gives the policy gradient

    This function computes log p(theta, action) for actions
    observed in the trajectory. Its derivative is grad(log p...)
    which is the policy gradient

    Args:
        params: dictionary of parameters for `policy_logits`
        ts: (n x state_dim + action_dim) array of trajectories
            sampled from the distribution induced by `params`

    Returns:
        A loss whose derivative is the policy gradient.
    """
    # For the customer service example we also provide a timestep indicator
    if use_timestep_indicator:
        T_indicators = np.tile(np.eye(T).reshape((1, T, T)), (batch_size, 1, 1))
        inputs = np.concatenate((ts[:, :, :state_dim], T_indicators), axis=-1)
    else:
        inputs = ts[:, :, :state_dim]
    log_probs = logsoftmax(policy_logits(params, inputs))
    log_prob_action = np.sum(log_probs * ts[:, :, state_dim:], axis=2)
    reward_term = np.mean(np.sum(advantages * log_prob_action, axis=1))
    l2_term = l2_parameter_loss(params)
    return -reward_term + l2_term


def critic_loss(critic_params, trajectories, returns):
    """Computes the loss from the critic on given trajectories

    The critic predicts using critic_params and inputs
    of the state part of the data, we then return the
    squared loss on the returns.

    Args:
        critic_params: Jax dictionary of parameters of the critic
        trajectories: (n x state_dim + action_dim) array of
                      trajectories

    Returns:
        Squared error when using critic params to predict the
        return.
    """
    return np.mean((critic(critic_params, trajectories[:, :, :state_dim]).squeeze() - returns) ** 2)


def compute_model_factors(policy_params, trajectories):
    # Compute p(x_{t+1}|a_t,x_t,u_t) / q(x_{t+1}|x_t,u_t)
    # We want the oracle dynamics model to give us p(x_{t+1}|x_t, u_t, a_t)
    # for all values of a_t
    ps_given_as = odm.conditional_density(
        trajectories[:, 1:, :state_dim],
        trajectories[:, :-1, :state_dim],
    )
    actions = trajectories[:, :-1, state_dim:]
    if use_timestep_indicator:
        T_indicators = np.tile(np.eye(T).reshape((1, T, T)), (batch_size, 1, 1))
        inputs = np.concatenate((trajectories[:, :, :state_dim], T_indicators), axis=-1)
    else:
        inputs = trajectories[:, :, :state_dim]
    qs = policy_logits(policy_params, inputs)
    ps = np.sum(ps_given_as * actions, axis=-1)
    q_xs = np.sum(softmax(qs[:, :-1, :], axis=-1) * ps_given_as, axis=-1)
    return ps / q_xs


def regularizer_loss(policy_params, trajectories, approx_MIs, model_factors, lm_vector):
    """Computes the loss from the mutual information regularizer

    The loss from the regularizer is approx_MIs (the actual
    value of the MI) multiplied by grad(log q(a|x,u)) at the
    current timestep plus the sum of grad(log q(a_t|x_t,u_t))
    from the previous timesteps, weighted by the `model_factors`
    p(x_t, u_t|a_{t-1}, x_{t-1}, u_{t-1})/q(x_t,u_t|x_{t-1}, u_{t-1})

    Args:
        policy_params: dictionary of parameters for `policy_logits`
        trajectories: (n x state_dim + action_dim) array of
            trajectories
        approx_MIs: The computed values of I(a_t;u_t), of shape
            (n x T)
        model_factors: (n x T) array of model factors as described
            above
        lm_vector: The (n,) dimensional Lagrange multiplier

    Returns:
        A loss whose gradient is the gradient of the MI
        regularizer
    """
    actions = trajectories[:, :, state_dim:]
    # For Markov, inputs are bs x num_repeats x T x state_dim
    if use_timestep_indicator:
        T_indicators = np.tile(np.eye(T).reshape((1, T, T)), (batch_size, 1, 1))
        inputs = np.concatenate((trajectories[:, :, :state_dim], T_indicators), axis=-1)
    else:
        inputs = trajectories[:, :, :state_dim]
    log_q_as = logsoftmax(policy_logits(policy_params, inputs))
    # mask out and sum to get p(action|x,u)
    expected_grad_log_1 = (log_q_as * actions).sum(axis=-1)
    first_term = np.mean(expected_grad_log_1 * approx_MIs, axis=0)
    regularizer_returns = approx_MIs[:, 1:] * np.mean(
        np.cumsum(model_factors * expected_grad_log_1[:, 1:], axis=-1), axis=0
    )
    second_factor = np.concatenate((np.zeros((1,)), regularizer_returns.mean(axis=0)), axis=0)
    return np.sum(lm_vector * (first_term + second_factor))


def step_policy_params(i, policy_opt_state, critic_params, lagrange_vec, trajectories, returns):
    """Does one gradient step for parameters in `policy_opt_state`

    Given trajectories, we compute the policy gradient from the
    reward, and the policy gradient from the I(a_t;u_t), add them
    up weighted by `lagrange_vec` and update `policy_opt_state`

    Args:
        i: current iteration
        policy_opt_state: jax.experimental.optimizers object containing
            the parameters for the policy
        critic_params: jax.experimental.optimizers object containing
            the parameters for the critic
        lagrange_vec: A vector of Lagrange multipliers applied to
            each of the mutual information terms
        trajectories: (batch_size x T x (state_dim + action_dim))
            trajectories sampled from the policy
        returns: (batch_size x T) returns corresponding to the policy

    Returns:
        A jax.experimental.optimizers object consisting of the
        policy parameters after the gradient step
    """
    policy_params = policy_get_params(policy_opt_state)
    return_baseline = critic(critic_params, trajectories[:, :, :state_dim].reshape((-1, state_dim)))
    advantage = returns - return_baseline.reshape((batch_size, T))
    policy_grad = grad(policy_gradient_fn)(policy_params, trajectories, advantage)

    # Now compute mutual information regularizer
    etas = fit_general_psi_predictor(trajectories, public_state_dim, private_state_dim, action_dim)
    psi_preds = predict_general_psi_predictor(trajectories, public_state_dim, etas)
    if args.experiment_name == "snap":
        # Here we have non-uniform baseline u fractions
        baseline_log_likelihoods = np.array(
            [np.log(environment.u_frac()), np.log(1 - environment.u_frac())]
        )
        baseline_log_likelihoods = np.sum(
            trajectories[:, :, public_state_dim : public_state_dim + private_state_dim]
            * baseline_log_likelihoods,
            axis=-1,
        )
    else:
        baseline_log_likelihoods = np.log(environment.u_frac())

    Rs = np.log(psi_preds) - baseline_log_likelihoods.squeeze()
    model_factors = compute_model_factors(policy_params, trajectories)
    regularizer_grad = grad(regularizer_loss)(
        policy_params, trajectories, Rs, model_factors, lagrange_vec
    )

    # Compute entropy reward term
    entropy_grad = grad(entropy_loss)(policy_params, trajectories)

    # Add the three gradients, anneal the entropy exploration
    total_gradient = tree_multimap(
        lambda x, y, z: x + y + (1 / np.log(i + 2)) * z,
        regularizer_grad,
        policy_grad,
        entropy_grad,
    )
    # breakpoint()

    return policy_opt_update(i, total_gradient, policy_opt_state)


def step_critic_params(i, critic_opt_state, rng, trajectories, returns):
    """Updates critic parameters with gradient step wrt `critic_loss`"""
    critic_params = critic_get_params(critic_opt_state)
    critic_grad = grad(critic_loss)(critic_params, trajectories, returns)
    return critic_opt_update(i, critic_grad, critic_opt_state)


@jit
def full_step(i, policy_opt_state, critic_opt_state, rng, lagrange_vec):
    """Takes a gradient step for the policy, then for the critic."""
    rng_1, rng_2, rng_3 = rnd.split(rng, 3)
    policy_params = policy_get_params(policy_opt_state)
    critic_params = critic_get_params(critic_opt_state)
    trajectories, rewards, returns = sample_trajectories(policy_params, rng_1, batch_size)
    policy_opt_state = step_policy_params(
        i, policy_opt_state, critic_params, lagrange_vec, trajectories, returns
    )
    critic_opt_state = step_critic_params(i, critic_opt_state, rng_2, trajectories, returns)
    return policy_opt_state, critic_opt_state, rng_1


def train(num_epochs, lagrange_vec):
    policy_opt_state = policy_opt_init(policy_params)
    critic_opt_state = critic_opt_init(critic_params)
    rng = rnd.PRNGKey(0)
    for k in range(num_epochs):
        policy_opt_state, critic_opt_state, rng = full_step(
            k, policy_opt_state, critic_opt_state, rng, lagrange_vec
        )
        if k % 50 == 0:
            ts, rws, rts = sample_trajectories(policy_get_params(policy_opt_state), rng, 128)
            print(f"At iteration {k}, return: {np.mean(rts[:, 0])}")
    return policy_get_params(policy_opt_state)


def get_MIs(trajectories):
    """Gets the mutual information in the tabular case

    Uses the empirical frequencies to estimate p(u_t, a_t)

    Args:
        trajectories: (batch_size x T x (state_dim + action_dim))
            trajectories sampled from the policy

    Returns:
        A vector of size T giving the mutual information I(u_t;a_t) at
        each timestep
    """
    etas = fit_general_psi_predictor(trajectories, public_state_dim, private_state_dim, action_dim)
    psi_preds = predict_general_psi_predictor(trajectories, public_state_dim, etas)
    if args.experiment_name == "snap":
        # Here we have non-uniform baseline u fractions, need to mask out with corresponding values
        baseline_log_likelihoods = np.array(
            [np.log(environment.u_frac()), np.log(1 - environment.u_frac())]
        )
        baseline_log_likelihoods = np.sum(
            trajectories[:, :, public_state_dim : public_state_dim + private_state_dim]
            * baseline_log_likelihoods,
            axis=-1,
        )
    else:
        baseline_log_likelihoods = np.log(environment.u_frac())
    Rs = np.log(psi_preds) - baseline_log_likelihoods.squeeze()
    return Rs.mean(axis=0)


if args.experiment_name == "snap":
    new_policy_params = train(
        num_epochs,
        0.0
        * np.ones(
            T,
        ),
    )
    # Need quite a few samples to get a good kernel density estimator for the plot
    plot_bs = 10000
    ts, rws, rts = sample_trajectories(new_policy_params, rnd.PRNGKey(0), plot_bs)
    MIs = get_MIs(ts)
    print(f"Mutual information is {MIs.mean()}")
    plot_distribution_t_MI_fig(MIs, ts, environment.x_scaling, "nonprivate_snap")

    new_policy_params = train(
        num_epochs,
        1.0
        * np.ones(
            T,
        ),
    )
    ts, rws, rts = sample_trajectories(new_policy_params, rnd.PRNGKey(0), plot_bs)
    MIs = get_MIs(ts)
    print(f"Mutual information is {MIs.mean()}")
    plot_distribution_t_MI_fig(MIs, ts, environment.x_scaling, "private_snap")

if args.experiment_name == "vpn":
    new_policy_params = train(
        num_epochs,
        np.zeros(
            T,
        ),
    )
    plot_bs = 128
    ts, rws, rts = sample_trajectories(new_policy_params, rnd.PRNGKey(0), plot_bs)
    MIs = get_MIs(ts)
    print(f"Mutual information is {MIs.mean()}")
    plot_vpn_fig(MIs, ts, state_dim, "vpn_unconstrained")

    new_policy_params = train(
        num_epochs,
        np.ones(
            T,
        ),
    )
    plot_bs = 128
    ts, rws, rts = sample_trajectories(new_policy_params, rnd.PRNGKey(0), plot_bs)
    MIs = get_MIs(ts)
    print(f"Mutual information is {MIs.mean()}")
    plot_vpn_fig(MIs, ts, state_dim, "vpn_constrained")


if args.experiment_name == "customer":
    new_policy_params = train(num_epochs, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    plot_bs = 128
    ts, rws, rts = sample_trajectories(new_policy_params, rnd.PRNGKey(0), plot_bs)
    MIs = get_MIs(ts)
    print(f"Mutual information is {MIs.mean()}")
    plot_customer_trajectory(MIs, ts, "left_panel")

    new_policy_params = train(num_epochs, np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    plot_bs = 128
    ts, rws, rts = sample_trajectories(new_policy_params, rnd.PRNGKey(0), plot_bs)
    MIs = get_MIs(ts)
    print(f"Mutual information is {MIs.mean()}")
    plot_customer_trajectory(MIs, ts, "middle_panel")

    new_policy_params = train(num_epochs, np.array([0.0, 10.0, 10, 10, 10, 10]))
    plot_bs = 128
    ts, rws, rts = sample_trajectories(new_policy_params, rnd.PRNGKey(0), plot_bs)
    MIs = get_MIs(ts)
    print(f"Mutual information is {MIs.mean()}")
    plot_customer_trajectory(MIs, ts, "right_panel")

# plt.show()
# from utils import plot_trajectory_figure
# import matplotlib.pyplot as plt

# fig = plot_trajectory_figure(ts, 2, 2, False)
