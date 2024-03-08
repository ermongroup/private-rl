import jax.numpy as np
import jax.ops as ops
import jax.random as rnd
from jax import grad, jit, vmap
from jax.experimental import optimizers
from environments import VPNEnvironment, QuadraticImpulseEnvironment
from utils import (
    calculate_return,
    fit_general_psi_predictor,
    predict_general_psi_predictor,
    initialize_trajectory,
    plot_2d_fig,
)

from jax.experimental.stax import softmax, logsoftmax
from jax.tree_util import tree_multimap
from networks import get_model
import matplotlib.pyplot as plt

num_epochs = 4000
environment = QuadraticImpulseEnvironment(0.5, 1.0)
draw_p0 = environment.draw_p0
transition = environment.transition
public_state_dim = environment.public_state_dim
private_state_dim = environment.private_state_dim
action_dim = environment.action_dim
state_dim = public_state_dim + private_state_dim
T = 10
batch_size = 128
odm = environment.oracle_dynamics_model
ENTROPY_PARAM = 0.0

critic, critic_params = get_model(state_dim, [256, 256, 1], batch_size)
policy_logits, policy_params = get_model(state_dim, [256, 256, action_dim], batch_size)
predict_u_a, predict_u_a_params = get_model(action_dim + T, [256, 256, 2], batch_size)

policy_opt_init, policy_opt_update, policy_get_params = optimizers.adam(
    step_size=1e-2, b1=0.9
)
critic_opt_init, critic_opt_update, critic_get_params = optimizers.adam(
    step_size=1e-2, b1=0.9
)
predictor_opt_init, predictor_opt_update, predictor_get_params = optimizers.adam(
    step_size=1e-2, b1=0.9
)


def fit_max_likelihood_u(trajectories):
    """Returns the max likelihood estimate for means and standard deviations of u_t.

    Args:
        trajectories: (n x state_dim + action_dim) array of
            trajectories
    Returns:
        means: A (T)-dimensional array with means of u at each t
        stds: A (T)-dimensional array with standard deviation of u at each t

    Given some trajectories, we return the mean
    and covariance of u for each timestep"""
    us = trajectories[:, :, 3]
    means = np.mean(us, axis=0)
    stds = np.std(us, axis=0)
    return means, stds


def compute_max_likelihood(baseline_params, trajectories):
    means, stds = baseline_params
    us = trajectories[:, :, 3]
    prefactor = -0.5 * np.log(2 * np.pi) - np.log(stds)
    exponent = -0.5 * ((means - us) ** 2) / (stds ** 2)
    return prefactor + exponent


def entropy_loss(policy_params, trajectories):
    actions = policy_logits(policy_params, trajectories[:, :, :state_dim])
    entropy = -np.sum(
        (logsoftmax(actions, axis=-1) * softmax(actions, axis=-1)), axis=-1
    )
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
    rewards = np.zeros(T,)
    rng_1, rng_2, rng_3 = rnd.split(rng, 3)

    state_0 = draw_p0(rng_1)
    trajectory = initialize_trajectory(state_0, T, state_dim, action_dim)

    for t in range(T):
        rng_1, rng_2 = rnd.split(rng_1, 2)
        action_logits = policy_logits(theta, trajectory[t, :state_dim])
        action = rnd.categorical(rng_1, action_logits)
        trajectory = ops.index_update(trajectory, ops.index[t, state_dim + action], 1)
        new_state, r = transition(trajectory[t, :state_dim], action, rng_2,)  # (x,u), a
        if t != T - 1:
            trajectory = ops.index_update(
                trajectory, ops.index[t + 1, :state_dim], new_state
            )
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
    log_probs = logsoftmax(policy_logits(params, ts[:, :, :state_dim]))
    log_prob_action = np.sum(log_probs * ts[:, :, state_dim:], axis=2)
    reward_term = np.mean(np.sum(advantages * log_prob_action, axis=1))
    return -reward_term


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
    return np.mean(
        (critic(critic_params, trajectories[:, :, :state_dim]).squeeze() - returns) ** 2
    )


def u_a_loss(predict_u_a_params, trajectories):
    log_likelihoods = u_a_log_likelihoods(predict_u_a_params, trajectories)
    return -np.mean(log_likelihoods)


def u_a_log_likelihoods(predict_u_a_params, trajectories):
    """Computes the log_likelihood of p(u_t|a_t) with given predictor params

    predict_u_a computes a mean and log-covariance given the action 
    and a one-hot vector indicating which timestep we are currently on.

    Args:
        predict_u_a_params: a Jax dictionary of parameters of the predictor
        trajectories: (n x state_dim + action_dim) array of
                      trajectories

    Returns:
        An (n x T) array containing the log-likelihoods of the observed us given
        the observed actions.
    """
    T_indicators = np.tile(np.eye(T).reshape((1, T, T)), (batch_size, 1, 1))
    inputs = np.concatenate((trajectories[:, :, state_dim:], T_indicators), axis=-1)
    us = trajectories[:, :, 3]
    u_a_distribution_params = predict_u_a(predict_u_a_params, inputs)
    means = u_a_distribution_params[:, :, 0]
    log_stds = u_a_distribution_params[:, :, 1]
    prefactor = -0.5 * np.log(2 * np.pi) - log_stds
    exponents = -0.5 * (((means - us) ** 2) / (np.exp(log_stds) ** 2))
    return prefactor + exponents


def compute_model_factors(policy_params, trajectories):
    """Computes the model factors p(x_{t+1}|a_t,x_t,u_t) / q(x_{t+1}|x_t,u_t)

    We call ps_given_as to compute the set of p(x_{t+1}|a_t,x_t,u_t)
    for each a_t, then we multiply by q(a_t|x_t,u_t) and sum to 
    marginalise out the actions. We don't do this for the last 
    timestep as we don't know the next state.

    Args:
        policy_params: dictionary of parameters for `policy_logits`
        trajectories: (n x state_dim + action_dim) array of
            trajectories
    Returns:
        ps / q_xs: an (n x (T-1)) array of the model factors described above.
    """
    ps_given_as = vmap(odm.conditional_density)(
        trajectories[:, 1:, :state_dim].reshape((-1, state_dim)),
        trajectories[:, :-1, :state_dim].reshape((-1, state_dim)),
    ).reshape((batch_size, T - 1, action_dim))
    actions = trajectories[:, :-1, state_dim:]
    qs = policy_logits(policy_params, trajectories[:, :, :state_dim])
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
    inputs = trajectories[:, :, :state_dim]
    actions = trajectories[:, :, state_dim:]
    log_probs = logsoftmax(policy_logits(policy_params, trajectories[:, :, :state_dim]))
    expected_grad_log_1 = np.sum(log_probs * trajectories[:, :, state_dim:], axis=2)
    # mask out and sum to get p(action|x,u)
    first_term = np.mean(expected_grad_log_1 * approx_MIs, axis=0)
    regularizer_returns = approx_MIs[:, 1:] * np.mean(
        np.cumsum(model_factors * expected_grad_log_1[:, 1:], axis=-1), axis=0
    )
    second_factor = np.concatenate(
        (np.zeros((1,)), regularizer_returns.mean(axis=0)), axis=0
    )
    return np.sum(lm_vector * (first_term + second_factor))


def step_policy_params(
    i, policy_opt_state, critic_params, lagrange_vec, trajectories, returns
):
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
    # First compute policy gradient
    policy_params = policy_get_params(policy_opt_state)
    return_baseline = critic(
        critic_params, trajectories[:, :, :state_dim].reshape((-1, state_dim))
    )
    advantage = returns - return_baseline.reshape((batch_size, T))
    policy_grad = grad(policy_gradient_fn)(policy_params, trajectories, advantage)

    # Now compute mutual information regularizer
    baseline_params = fit_max_likelihood_u(trajectories)
    baseline_log_likelihoods = compute_max_likelihood(baseline_params, trajectories)
    psi_log_likelihoods = u_a_log_likelihoods(predict_u_a_params, trajectories)
    Rs = psi_log_likelihoods - baseline_log_likelihoods
    model_factors = compute_model_factors(policy_params, trajectories)
    regularizer_grad = grad(regularizer_loss)(
        policy_params, trajectories, Rs, model_factors, lagrange_vec
    )
    # Compute entropy reward term
    entropy_grad = grad(entropy_loss)(policy_params, trajectories)

    # Add the three gradients
    total_gradient = tree_multimap(
        lambda x, y, z: x + y + (1 / np.log(i + 2)) * z,
        regularizer_grad,
        policy_grad,
        entropy_grad,
    )
    return policy_opt_update(i, total_gradient, policy_opt_state)


def step_critic_params(i, critic_opt_state, rng, trajectories, returns):
    """Updates critic parameters with gradient step wrt `critic_loss`"""
    critic_params = critic_get_params(critic_opt_state)
    critic_grad = grad(critic_loss)(critic_params, trajectories, returns)
    return critic_opt_update(i, critic_grad, critic_opt_state)


def step_predictor_params(i, predictor_opt_state, rng, trajectories):
    """Updates predictor parameters with gradient step wrt `u_a_loss`"""
    predictor_params = predictor_get_params(predictor_opt_state)
    predictor_grad = grad(u_a_loss)(predictor_params, trajectories)
    return predictor_opt_update(i, predictor_grad, predictor_opt_state)


@jit
def full_step(
    i, policy_opt_state, critic_opt_state, predictor_opt_state, rng, lagrange_vec
):
    """Takes a gradient step for the policy, critic and predictor"""
    rng_1, rng_2, rng_3, rng_4 = rnd.split(rng, 4)
    policy_params = policy_get_params(policy_opt_state)
    critic_params = critic_get_params(critic_opt_state)
    trajectories, rewards, returns = sample_trajectories(
        policy_params, rng_1, batch_size
    )
    policy_opt_state = step_policy_params(
        i, policy_opt_state, critic_params, lagrange_vec, trajectories, returns
    )
    critic_opt_state = step_critic_params(
        i, critic_opt_state, rng_2, trajectories, returns
    )
    predictor_opt_state = step_predictor_params(
        i, predictor_opt_state, rng_3, trajectories
    )
    return policy_opt_state, critic_opt_state, predictor_opt_state, rng_1


def train(num_epochs, lagrange_multiplier):
    policy_opt_state = policy_opt_init(policy_params)
    critic_opt_state = critic_opt_init(critic_params)
    predictor_opt_state = predictor_opt_init(predict_u_a_params)
    lagrange_vec = np.ones(T,) * lagrange_multiplier
    rng = rnd.PRNGKey(0)
    ts, rws, rts = sample_trajectories(policy_get_params(policy_opt_state), rng, 16)
    print(f"Loss with initial params: {np.mean(rts[:, 0])}")
    for k in range(num_epochs):
        policy_opt_state, critic_opt_state, predictor_opt_state, rng = full_step(
            k,
            policy_opt_state,
            critic_opt_state,
            predictor_opt_state,
            rng,
            lagrange_vec,
        )
        if k % 50 == 0:
            ts, rws, rts = jit(sample_trajectories, static_argnums=(2,))(
                policy_get_params(policy_opt_state), rng, 256
            )
            print(f"At iteration {k}, loss: {np.mean(rts[:, 0])}")

    return (
        policy_get_params(policy_opt_state),
        predictor_get_params(predictor_opt_state),
    )


def get_MIs(trajectories, predictor_params):
    """Gets the mutual information in the continuous setting

    Args:
        trajectories: (batch_size x T x (state_dim + action_dim))
            trajectories sampled from the policy
        predictor_params: a Jax dictionary of parameters of the predictor

    Returns:
        A vector of size T giving the mutual information I(u_t;a_t) at
        each timestep
    """
    baseline_params = fit_max_likelihood_u(trajectories)
    baseline_log_likelihoods = compute_max_likelihood(baseline_params, trajectories)

    psi_log_likelihoods = u_a_log_likelihoods(predictor_params, trajectories)
    Rs = psi_log_likelihoods - baseline_log_likelihoods
    return Rs.mean(axis=0)


# For the constrained case:
new_policy_params, new_predictor_params = train(num_epochs, 20.0)
plot_batch_size = 128
ts, rws, rts = sample_trajectories(new_policy_params, rnd.PRNGKey(0), plot_batch_size)
MIs = get_MIs(ts, new_predictor_params)
print(f"Average mutual information is {MIs.mean()}")
plot_2d_fig(MIs, ts, "2d_constrained")

# # For the unconstrained case
new_policy_params, new_predictor_params = train(num_epochs, 0.0)
plot_batch_size = 128
ts, rws, rts = sample_trajectories(new_policy_params, rnd.PRNGKey(0), plot_batch_size)
MIs = get_MIs(ts, new_predictor_params)
print(f"Average mutual information is {MIs.mean()}")
plot_2d_fig(MIs, ts, "2d_unconstrained")
