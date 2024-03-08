from __future__ import division
import jax.numpy as np
import jax.random as jaxrand
import numpy as onp
from jax.experimental import optimizers
import argparse
from jax import vmap
from jax import grad, jit, vmap
import jax.ops as jaxops
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from analyse_acs import compute_KDEs
from jax.scipy.special import ndtr


def calculate_return(rewards):
    """Given a batch x episode length matrix of
    rewards, return the return for each episode """
    returns = np.cumsum(rewards[:, ::-1], axis=1)[:, ::-1]
    # Set up to accumulate inputs from the trajectory
    return returns


def fit_general_psi_predictor(
    trajectories, public_state_dim, private_state_dim, action_dim
):
    """Computes empirical estimates of p(U_t=u|A_t=a).

    To do this we use einsum: we want to calculate frequencies 
    from an n x T x action_dim matrix and an n x T x private_state_dim
    matrix.

    Args:
        trajectories: an n x T x (public_state_dim + private_state_dim + action_dim)
            array of trajectories
        public_state_dim, etc: dimensions of the public, private state and actions
    Returns:
        A T x action_dim x private_state_dim array W
        where W[t, i, j] = P(U_t=u_i|a_j)
    """
    n, T, _ = np.shape(trajectories)
    etas = np.zeros((T, private_state_dim, action_dim))
    actions = trajectories[:, :, private_state_dim + public_state_dim :]
    us = trajectories[:, :, public_state_dim : public_state_dim + private_state_dim]
    counts = np.einsum("ijp,ijq->ijpq", actions, us, optimize=True)
    etas = counts.mean(axis=0)
    sums = etas.sum(axis=2)[:, :, None]
    etas = np.where(sums != 0, etas / sums, 0)
    return etas


def predict_general_psi_predictor(trajectories, public_state_dim, etas):
    """Computes empirical guess of p(U_t=u|A_t=a)
    Args:
        a_array: n x T x action_dim array of actions
        u_array: n x T x private_state_dim array of private state
        etas: empirical frequencies from `fit_general_psi_predictor`
    Returns:
        n x T array of probabilities that the predictor chose
        the correct u
    """
    T, action_dim, private_state_dim = etas.shape
    actions = trajectories[:, :, private_state_dim + public_state_dim :]
    us = trajectories[:, :, public_state_dim : public_state_dim + private_state_dim]
    # We want to get a mask of dimension n x T x action_dim x private_state_dim
    # with a 1 iff we have that action and u at that T in the nth trajectory
    mask = np.einsum("ijk,ijl->ijkl", actions, us, optimize=True)
    correct_probs = (mask * etas[None, :]).sum(axis=(2, 3))
    return correct_probs


class OracleVPNDynamicsModel:
    def __init__(self, environment):
        self.p_change_u = environment.p_change_u
        self.num_servers = environment.num_servers

    def density(self, state_ts, action_prev, state_prev):
        """Return p(x_t, u_t|a_{t-1}, x_{t-1}, u_{t-1})"""
        # If a == `activate VPN` then x_t is certain to be
        # [0, 1], else certain to be x
        activates_vpn = action_prev[0] == 1
        cur_x_is_prev_x = state_ts[:, :2] == state_prev[:, :2]
        x_density_term = cur_x_is_prev_x + (1 - cur_x_is_prev_x) * activates_vpn

        cur_u_is_prev_u = state_ts[:, 2:] == state_prev[:, 2:]
        u_density_term = cur_u_is_prev_u * (
            1 - self.p_change_u + self.p_change_u * (1 / self.num_servers)
        ) + (1 - cur_u_is_prev_u) * self.p_change_u * (1 / self.num_servers)
        return u_density_term * x_density_term

    def conditional_density(self, state_ts, state_prev):
        """Computes p(x_t, u_t | a_{t-1}, x_{t-1}, u_{t-1}) for all a_{t-1}.

        Computes the above quantity. Note that here the dynamics are
        independent of u, so we don't actually use u at all.

        Args:
        state_ts: An n x (state_dim) matrix of xs, us at the current timestep
        state_prevs: An n x (state_dim) matrix of xs, us at the previous timestep

        Returns:
            An n x action_dim vector of probabilities for each action

        """
        # There are num_servers + 1 actions, they all have the same density
        # except for action 0, activate vpn
        batch_size, T_1, _ = state_ts.shape
        cur_x_is_vpn = state_ts[:, :, 1] == 1
        prev_x_is_vpn = state_prev[:, :, 1] == 1

        # If previous is vpn, any action is ok
        # If previous is not vpn and current is vpn, need vpn
        # If previous is not vpn and current is not vpn, can't have vpn

        any_action_ok = prev_x_is_vpn
        only_vpn_ok = cur_x_is_vpn * (
            1 - prev_x_is_vpn
        )  # Prev x is not vpn, need vpn action

        vpn_not_ok = (1 - cur_x_is_vpn) * (1 - prev_x_is_vpn)

        vpns = np.concatenate(
            (
                np.ones((batch_size, T_1, 1)),
                np.zeros((batch_size, T_1, self.num_servers)),
            ),
            axis=2,
        )

        no_vpns = np.concatenate(
            (
                np.zeros((batch_size, T_1, 1)),
                np.ones((batch_size, T_1, self.num_servers)),
            ),
            axis=2,
        )

        all_actions = np.ones((batch_size, T_1, self.num_servers + 1))
        return (
            all_actions * any_action_ok[:, :, None]
            + vpns * only_vpn_ok[:, :, None]
            + vpn_not_ok[:, :, None] * no_vpns
        )


class OracleQuadraticImpulseDynamicsModel:
    def __init__(self, environment):
        self.force_noise_std = environment.force_noise_std

    def state_transition_w_action_density(self, state_ts, state_prevs, action):
        sigma_e = np.eye(2) * self.force_noise_std
        sigma_e_inv = np.eye(2) * (1.0 / self.force_noise_std)
        prefactor = (1.0 / (2 * np.pi)) * (1.0 / (self.force_noise_std ** 2))
        e_means = np.array(
            [
                state_ts[0]
                - state_prevs[0]
                - state_prevs[1]
                - (action == 0)
                + (action == 1),
                state_ts[3]
                - state_prevs[3]
                - state_prevs[2]
                - (action == 2)
                + (action == 3),
            ]
        )
        exponent = np.exp(-0.5 * e_means.T @ (sigma_e_inv) @ e_means)
        return prefactor * exponent

    def conditional_density(self, state_ts, state_prevs):
        """Computes p(x_t, u_t | a_{t-1}, x_{t-1}, u_{t-1}) for all a_{t-1}.

        Computes the above quantity. Note that here the dynamics are
        independent of u, so we don't actually use u at all.

        Args:
        state_ts: An n x (state_dim) matrix of xs, us at the current timestep
        state_prevs: An n x (state_dim) matrix of xs, us at the previous timestep

        Returns:
            An n x action_dim vector of probabilities for each action

        """
        # There are num_servers + 1 actions, they all have the same density
        # except for action 0, activate vpn
        sigma_e = np.eye(2) * self.force_noise_std
        sigma_e_inv = np.eye(2) * (1.0 / self.force_noise_std)
        e_means_0 = np.array(
            [
                state_ts[0] - state_prevs[0] - state_prevs[1] - 1,
                state_ts[3] - state_prevs[3] - state_prevs[2],
            ]
        )
        prefactor = (1.0 / (2 * np.pi)) * (1.0 / (self.force_noise_std ** 2))
        exponent_0 = np.exp(-0.5 * e_means_0.T @ (sigma_e_inv) @ e_means_0)

        e_means_1 = np.array(
            [
                state_ts[0] - state_prevs[0] - state_prevs[1] + 1,
                state_ts[3] - state_prevs[3] - state_prevs[2],
            ]
        )
        prefactor = (1.0 / (2 * np.pi)) * (1.0 / (self.force_noise_std ** 2))
        exponent_1 = np.exp(-0.5 * e_means_1.T @ (sigma_e_inv) @ e_means_1)

        e_means_2 = np.array(
            [
                state_ts[0] - state_prevs[0] - state_prevs[1],
                state_ts[3] - state_prevs[3] - state_prevs[2] - 1,
            ]
        )
        prefactor = (1.0 / (2 * np.pi)) * (1.0 / (self.force_noise_std ** 2))
        exponent_2 = np.exp(-0.5 * e_means_2.T @ (sigma_e_inv) @ e_means_2)

        e_means_3 = np.array(
            [
                state_ts[0] - state_prevs[0] - state_prevs[1],
                state_ts[3] - state_prevs[3] - state_prevs[2] + 1,
            ]
        )
        prefactor = (1.0 / (2 * np.pi)) * (1.0 / (self.force_noise_std ** 2))
        exponent_3 = np.exp(-0.5 * e_means_3.T @ (sigma_e_inv) @ e_means_3)
        return (
            np.concatenate(
                (
                    exponent_0.reshape((1,)),
                    exponent_1.reshape((1,)),
                    exponent_2.reshape((1,)),
                    exponent_3.reshape((1,)),
                )
            )
            * prefactor
        )


class OracleSNAPDynamicsModel:
    def __init__(self, environment):
        self.sigma = environment.process_noise_level
        prefactor = 1.0 / (np.sqrt(2 * np.pi * self.sigma ** 2))
        gaussian_density_func = lambda x, y: prefactor * np.exp(
            -((x - y) ** 2) / (2 * self.sigma ** 2)
        )
        self.vmap_gaussian_density_func = vmap(gaussian_density_func)
        self.x_scaling = environment.x_scaling
        self.uniform_upper_bound = environment.snap_benefit_range[1]
        self.uniform_lower_bound = environment.snap_benefit_range[0]
        self.snap_benefit_range = self.uniform_upper_bound - self.uniform_lower_bound

    def conditional_density(self, state_ts, state_prev):
        """Computes p(x_t, u_t | a_{t-1}, x_{t-1}, u_{t-1}) for all a_{t-1}.

        Computes the above quantity. Note that here the dynamics are
        independent of u, so we don't actually use u at all.

        Args:
        state_ts: An n x (state_dim) matrix of xs, us at the current timestep
        state_prevs: An n x (state_dim) matrix of xs, us at the previous timestep

        Returns:
            An n x action_dim vector of probabilities for each action
        """
        x_curs = state_ts[:, :, 0] * self.x_scaling
        x_prevs = state_prev[:, :, 0] * self.x_scaling
        prefactor = 1.0 / self.snap_benefit_range
        action_0_densities = self.vmap_gaussian_density_func(x_curs, x_prevs)
        x_change = x_curs - x_prevs
        # Not sure about the constants here, but they just cancel..?
        action_1_densities = ndtr(
            (x_change - self.uniform_lower_bound) / self.sigma
        ) - ndtr((x_change - self.uniform_upper_bound) / self.sigma)

        action_1_densities = action_1_densities * prefactor
        # greater_action_1_densities = (
        #     prefactor
        #     * self.vmap_gaussian_density_func(
        #         x_curs, self.uniform_upper_bound + x_prevs
        #     )
        #     * ((x_curs - x_prevs) > self.uniform_upper_bound)
        # )
        # # Get densities where x_ts > x_t1 + lower_limit:
        # lower_action_1_densities = (
        #     prefactor
        #     * self.vmap_gaussian_density_func(
        #         x_curs, self.uniform_lower_bound + x_prevs
        #     )
        #     * ((x_curs - x_prevs) < self.uniform_lower_bound)
        # )
        # # Get densities where upper_limit < x_ts < lower_limit
        # uniform_densities = (
        #     prefactor
        #     * np.ones_like(x_curs)
        #     * ((x_curs - x_prevs) < self.uniform_upper_bound)
        #     * ((x_curs - x_prevs) > self.uniform_lower_bound)
        # )

        # action_1_densities = (
        #     lower_action_1_densities + greater_action_1_densities + uniform_densities
        # )
        return np.concatenate(
            (action_0_densities[:, :, None], action_1_densities[:, :, None]), axis=-1
        )

    def density(self, x_ts_, a_t1s, x_t1s_, us):
        """Return p(x_t|a_{t-1}, x_{t-1}, u) which is a Gaussian
        with center x_{t-1} + a_{t-1} and standard deviation self.process_noise_level
        """
        # First get densities where a = 0:
        x_ts = x_ts_ * self.x_scaling
        x_t1s = x_t1s_ * self.x_scaling
        x_a0s_ts_center = x_t1s
        a0_densities = self.vmap_gaussian_density_func(x_ts, x_a0s_ts_center) * (
            1 - a_t1s
        )
        # Get densities where x_ts > x_t1 + upper_limit:
        prefactor = 1.0 / self.snap_benefit_range

        prefactor = 1.0 / self.snap_benefit_range

        greater_densities = (
            prefactor
            * self.vmap_gaussian_density_func(x_ts, self.uniform_upper_bound + x_t1s)
            * ((x_ts - x_t1s) > self.uniform_upper_bound)
            * a_t1s
        )
        # Get densities where x_ts > x_t1 + lower_limit:
        lower_densities = (
            prefactor
            * self.vmap_gaussian_density_func(x_ts, self.uniform_lower_bound + x_t1s)
            * ((x_ts - x_t1s) < self.uniform_lower_bound)
            * a_t1s
        )
        # Get densities where upper_limit < x_ts < lower_limit
        uniform_densities = (
            prefactor
            * np.ones(len(x_ts))
            * a_t1s
            * ((x_ts - x_t1s) < self.uniform_upper_bound)
            * ((x_ts - x_t1s) > self.uniform_lower_bound)
        )

        return greater_densities + lower_densities + uniform_densities + a0_densities


class OracleCustomerDynamicsModel:
    def __init__(self, environment):
        self.sigma = environment.process_noise_level
        prefactor = 1.0 / (np.sqrt(2 * np.pi * self.sigma ** 2))
        gaussian_density_func = lambda x, y: prefactor * np.exp(
            -((x - y) ** 2) / (2 * self.sigma ** 2)
        )
        self.vmap_gaussian_density_func = vmap(gaussian_density_func)
        self.u_transition_dependence = environment.u_transition_dependence

    def conditional_density(self, state_ts, state_prev):
        """Computes p(x_t, u_t | a_{t-1}, x_{t-1}, u_{t-1}) for all a_{t-1}.
        Computes the above quantity.

        Args:
        state_ts: An n x (state_dim) matrix of xs, us at the current timestep
        state_prevs: An n x (state_dim) matrix of xs, us at the previous timestep

        Returns:
            An n x action_dim vector of probabilities for each action

        """
        x_curs = state_ts[:, :, 0]
        x_prevs = state_prev[:, :, 0]
        u_prev = np.argmax(state_prev[:, :, 1:3], axis=-1)
        action_0_centers = x_prevs - 1 + (u_prev - 0.5) * self.u_transition_dependence
        action_0_densities = self.vmap_gaussian_density_func(x_curs, action_0_centers)
        action_1_centers = x_prevs + 1 + (u_prev - 0.5) * self.u_transition_dependence
        action_1_densities = self.vmap_gaussian_density_func(x_curs, action_1_centers)
        return np.concatenate(
            (action_0_densities[:, :, None], action_1_densities[:, :, None]), axis=-1
        )


def initialize_trajectory(state_0, T, state_dim, action_dim):
    states = np.ones((T, state_dim)) * state_0  # Must make sure we update this
    actions = np.zeros((T, action_dim))
    return np.concatenate((states, actions), axis=1)


def plot_trajectory_figure(figure_trajectories, state_dim, action_dim, plot_hists):
    """Runs a few rollouts and plots the figure of trajectory"""
    _, T, _ = figure_trajectories.shape
    if not plot_hists:
        u0s = figure_trajectories[figure_trajectories[:, :, 1] == 0].reshape(
            (-1, T, state_dim + action_dim)
        )
        u1s = figure_trajectories[figure_trajectories[:, :, 1] == 1].reshape(
            (-1, T, state_dim + action_dim)
        )

        u0_means = onp.array(onp.mean(u0s[:, :, 0], axis=0))
        u1_means = onp.array(onp.mean(u1s[:, :, 0], axis=0))
        u0_stds = onp.array(onp.std(u0s[:, :, 0], axis=0))
        u1_stds = onp.array(onp.std(u1s[:, :, 0], axis=0))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(T)),
                y=u0_means,
                error_y=dict(type="data", array=u0_stds, visible=True),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(T)),
                y=u1_means,
                error_y=dict(type="data", array=u1_stds, visible=True),
            )
        )
    else:
        # Plot the distributions at different times
        u0s = figure_trajectories[figure_trajectories[:, :, 1] == 0].reshape(
            (-1, T, state_dim + action_dim)
        )
        u1s = figure_trajectories[figure_trajectories[:, :, 1] == 1].reshape(
            (-1, T, state_dim + action_dim)
        )
        labels = []
        fig = go.Figure()
        for t in list(range(T)):
            labels.append("White T={}".format(t))
            labels.append("Black T={}".format(t))
            nums, bin_edges = onp.histogram(
                onp.array(u0s[:, t, 0]), onp.arange(0, 10, 1), density=True
            )
            fig.add_trace(
                go.Bar(
                    y=nums,
                    x=bin_edges,
                    name="Black, T={}".format(t),
                    marker_color="black",
                    marker=dict(line=dict(width=0.5, color="crimson")),
                )
            )
            nums, bin_edges = onp.histogram(
                onp.array(u1s[:, t, 0]), onp.arange(0, 10, 1), density=True,
            )
            fig.add_trace(
                go.Bar(
                    y=nums,
                    x=bin_edges,
                    name="White, T={}".format(t),
                    marker_color="white",
                    marker=dict(line=dict(width=0.5, color="crimson")),
                )
            )
            # fig.add_trace(go.Histogram(name='t = {}, White Incomes'.format(t),
            #                            x=np.array(u0s[:, t, 0])))
            # fig.add_trace(go.Histogram(name='t = {}, Black Incomes'.format(t),
            #                            x=np.array(u1s[:, t, 0])))
        # fig = ff.create_distplot(incomes, labels)
        # import pdb; pdb.set_trace()

    return fig


def plot_zoom_in_figure(figure_trajectories, state_dim, action_dim, plot_hists):
    """Runs a few rollouts and plots the figure of trajectory"""
    _, T, _ = figure_trajectories.shape
    # Plot the distributions at different times
    u0s = figure_trajectories[figure_trajectories[:, :, 1] == 0].reshape(
        (-1, T, state_dim + (action_dim + 1) * T)
    )
    u1s = figure_trajectories[figure_trajectories[:, :, 1] == 1].reshape(
        (-1, T, state_dim + (action_dim + 1) * T)
    )
    u0s = u0s * 1e5
    u1s = u1s * 1e5
    fig = go.Figure()
    if plot_hists:
        for t in list(range(T)):
            nums, bin_edges = onp.histogram(
                onp.array(u0s[:, t, 0]), onp.arange(0, 1e5, 1e4), density=True
            )
            fig.add_trace(
                go.Bar(
                    y=nums,
                    x=bin_edges,
                    name="Black, T={}".format(t),
                    marker_color="black",
                    marker=dict(line=dict(width=0.5, color="crimson")),
                )
            )
            nums, bin_edges = onp.histogram(
                onp.array(u1s[:, t, 0]), onp.arange(0, 1e5, 1e4), density=True
            )
            fig.add_trace(
                go.Bar(
                    y=nums,
                    x=bin_edges,
                    name="White, T={}".format(t),
                    marker_color="white",
                    marker=dict(line=dict(width=0.5, color="crimson")),
                )
            )
    return fig


def plot_actions_xs_figure(
    figure_xs, figure_one_hot_actions, x_scaling, state_dim, action_dim
):
    xs = onp.array(figure_xs * x_scaling)
    actions = onp.array(figure_one_hot_actions)
    x_indices = onp.linspace(0, 8e4, 20)
    cell_proportions = []
    for index, x_floor in enumerate(x_indices[:-1]):
        cell_actions = actions[(x_indices[index + 1] > xs) & (xs > x_floor)]
        cell_proportions.append(
            float(np.sum(cell_actions[:, 1])) / len(cell_actions[:, 1])
        )
    fig = go.Figure()
    nums, bin_edges = onp.histogram(cell_proportions, x_indices, density=True)
    fig.add_trace(
        go.Bar(
            y=cell_proportions,
            x=x_indices,
            marker_color="white",
            marker=dict(line=dict(width=0.5, color="crimson")),
        )
    )
    return fig


def plot_discriminator_figure(discriminator_predictions):
    """Runs a few rollouts and plots the figure of the per-timestep
    predictions. We have to use plotly because it is much better with
    weights and biases"""
    _, T = discriminator_predictions[0].shape
    u0_preds, u1_preds = discriminator_predictions
    u0_preds_means = np.mean(u0_preds, axis=0)
    u1_preds_means = np.mean(u1_preds, axis=0)
    u0_preds_stds = np.std(u0_preds, axis=0)
    u1_preds_stds = np.std(u1_preds, axis=0)

    # fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    # fig, axs = plt.subplots()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="P(U=0) for U=0",
            x=list(range(T)),
            y=1 - u0_preds_means,
            error_y=dict(type="data", array=u0_preds_stds),
        )
    )
    fig.add_trace(
        go.Bar(
            name="P(U=1) for U=1",
            x=list(range(T)),
            y=u1_preds_means,
            error_y=dict(type="data", array=u1_preds_stds),
        )
    )
    return fig


def plot_discriminator_figure_tabular(etas, joint_freqs):
    """We plot the expected value of q(u|a) over the actions for
    both u values"""
    # Have eta as a T x (action_dim) matrix, with eta[t, a] giving
    # p(u = 1 | a_t = 1)
    # We want to show P(U=0) when U=0 and P(U=1) when U=1.
    # We have P(u=1) = sum [P(a_t = a_t) * p(u=1|a_t = a_t)]
    T, action_dim = etas.shape
    a_preds = onp.zeros((T, action_dim))
    for t in list(range(T)):
        for a in list(range(action_dim)):
            a_preds[t, a] = etas[t, a]

    # fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    # fig, axs = plt.subplots()
    fig = go.Figure()
    for a in list(range(action_dim)):
        fig.add_trace(
            go.Bar(name="P(U=1) for a={}".format(a), x=list(range(T)), y=a_preds[:, a])
        )
        fig.update_yaxes(range=[0, 1.0])
    return fig


def plot_mutual_information(discriminator_predictions):
    """Runs a few rollouts and plots the figure of the per-timestep
    predictions. We have to use plotly because it is much better with
    weights and biases"""
    u0_preds, u1_preds = discriminator_predictions
    total_samples = len(u0_preds) + len(u1_preds)
    _, T = discriminator_predictions[0].shape
    u1_term = np.sum(np.log(u1_preds) + np.log(2), axis=0)
    u0_term = np.sum(np.log(1 - u0_preds) + np.log(2), axis=0)
    MIs = (u1_term + u0_term) / total_samples

    # fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    # fig, axs = plt.subplots()
    fig = go.Figure()
    fig.add_trace(go.Bar(name=r"$I(a_t;u)$", x=list(range(T)), y=MIs))
    return fig


def plot_mutual_information_tabular(etas, joint_freqs, u0_frac):
    """Plots the mutual informations from tabular inputs"""
    T, private_state_dim, action_dim = joint_freqs.shape
    MIs = []
    for t in list(range(T)):
        MI = 0
        for a in list(range(action_dim)):
            MI += joint_freqs[t, 1, a] * (np.log(etas[t, a]) - np.log(1 - u0_frac))
            MI += joint_freqs[t, 0, a] * (np.log(1 - etas[t, a]) - np.log(u0_frac))
        MIs.append(MI)
    fig = go.Figure()
    fig.add_trace(go.Bar(name=r"$I(a_t;u)$", x=list(range(T)), y=MIs))
    fig.update_yaxes(range=[0, 0.3])
    return fig


def plot_actions(trajectories):
    """Given a set of trajectories, plot the frequency of actions
    chosen at each timestep"""
    u0_actions, u1_actions = trajectories
    u0_action_choices = np.argmax(u0_actions, axis=2)
    frac_a1_u0 = u0_action_choices.mean(axis=0)
    T = len(frac_a1_u0)

    u1_action_choices = np.argmax(u1_actions, axis=2)
    frac_a1_u1 = u1_action_choices.mean(axis=0)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="Fraction with action 1 for U=0", x=list(range(T)), y=frac_a1_u0)
    )
    fig.add_trace(
        go.Bar(name="Fraction with action 1 for U=1", x=list(range(T)), y=frac_a1_u1)
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def plot_2d_fig(MIs, trajectories, filename=None):
    fig = plt.figure(figsize=(6, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axs = (plt.subplot(gs[0]), plt.subplot(gs[1]))
    plot_batch_size, T, _ = trajectories.shape
    for k in range(plot_batch_size):
        axs[0].plot(trajectories[k, :, 0], trajectories[k, :, 3], ".c-", alpha=0.3)
    axs[0].plot([], [], ".c-", alpha=0.3, label="Trajectories")
    # from utils import plot_trajectory_figure
    all_points = np.concatenate(
        (trajectories[:, :, 0, None], trajectories[:, :, 3, None]), axis=-1
    ).reshape((-1, 2))
    all_points_cov = np.cov(all_points.T)
    all_points_mean = np.mean(all_points, axis=0)
    print(f"Covariance is {all_points_cov}")
    circle_points = (
        np.concatenate(
            (
                np.cos(np.linspace(0, 2 * np.pi, 200))[:, None],
                np.sin(np.linspace(0, 2 * np.pi, 200))[:, None],
            ),
            axis=1,
        )
        @ np.linalg.cholesky(all_points_cov)
        + all_points_mean
    )
    axs[0].plot(
        circle_points[:, 0],
        circle_points[:, 1],
        "k-",
        linewidth=2,
        label=r"One $\sigma$ Gaussian Fit",
    )
    axs[0].legend(edgecolor="k", loc="upper right")
    axs[0].set_ylim([-10, 10])
    axs[0].set_xlim([-10, 10])
    axs[0].set_xlabel(r"$x$")
    axs[0].set_ylabel(r"$u$")
    axs[0].spines["right"].set_visible(False)
    axs[0].spines["top"].set_visible(False)
    bar1 = axs[1].bar(
        range(T),
        MIs,
        label=r"$I(a_t;u_t)$",
        color="tab:blue",
        alpha=0.5,
        edgecolor="k",
        linewidth=1.0,
    )
    axs[1].set_ylim([0, 0.5])
    axs[1].set_xlabel("Timesteps")
    axs[1].set_ylabel(r"$I(a_t;u_t)$")
    axs[1].spines["right"].set_visible(False)
    axs[1].spines["top"].set_visible(False)
    axs[1].legend(edgecolor="k", loc="upper right")
    axs[1].set_xticks(np.arange(0, T, step=1.0))
    plt.tight_layout()
    if filename is not None:
        plt.savefig(f"./{filename}.pdf", dpi=600)
    else:
        plt.show()


def plot_vpn_fig(MIs, trajectories, state_dim, filename=None):
    plot_bs, T, _ = trajectories.shape
    public_state_dim = 2
    action_dim = state_dim - 1
    fig, axs = plt.subplots(2, 1, figsize=(6, 6))
    actions = trajectories[:, :, state_dim:]
    actions_idx = (
        np.argmax(actions, axis=-1) + onp.random.normal(size=(plot_bs, T)) * 0.01
    )
    # Work out how to get nice labels
    for k in range(plot_bs):
        if trajectories[k, 0, public_state_dim] == 1:
            axs[0].plot(range(T), actions_idx[k], "xk-", alpha=0.3)
        if trajectories[k, 0, public_state_dim + 1] == 1:
            axs[0].plot(range(T), actions_idx[k], "+b-", alpha=0.3)
        if trajectories[k, 0, public_state_dim + 2] == 1:
            axs[0].plot(range(T), actions_idx[k], "oc-", alpha=0.3)
        if trajectories[k, 0, public_state_dim + 3] == 1:
            axs[0].plot(range(T), actions_idx[k], "*r-", alpha=0.3)
        # plt.legend()
    axs[0].plot([], [], "xk-", alpha=1.0, label="u=0")
    axs[0].plot([], [], "+b-", alpha=1.0, label="u=1")
    axs[0].plot([], [], "oc-", alpha=1.0, label="u=2")
    axs[0].plot([], [], "*r-", alpha=1.0, label="u=3")
    axs[0].spines["right"].set_visible(False)
    axs[0].spines["top"].set_visible(False)
    axs[0].legend(edgecolor="k", loc="upper right")
    axs[0].grid()
    axs[0].set_ylim([0, action_dim + 1])
    axs[0].set_xticks(np.arange(0, T, step=1.0))
    labels = ["Activate VPN", "Mirror 0", "Mirror 1", "Mirror 2", "Mirror 3"]
    axs[0].set_yticks(range(len(labels)))
    axs[0].set_yticklabels(labels)

    bar1 = axs[1].bar(
        range(T),
        MIs,
        label=r"$I(a_t;u_t)$",
        color="tab:blue",
        alpha=0.5,
        edgecolor="k",
        linewidth=1.0,
    )
    axs[1].set_ylim([0, 2.0])
    axs[1].set_xlabel("Timesteps")
    axs[1].set_ylabel(r"$I(a_t;u_t)$")
    axs[1].spines["right"].set_visible(False)
    axs[1].spines["top"].set_visible(False)
    axs[1].legend(edgecolor="k", loc="upper right")
    axs[1].set_xticks(np.arange(0, T, step=1.0))
    plt.tight_layout()
    if filename is not None:
        plt.savefig(f"./{filename}.pdf", dpi=600)
    else:
        plt.show()


def plot_customer_trajectory(MIs, trajectories, filename=None):
    fig, axs = plt.subplots(2, 1)
    _, T, _ = trajectories.shape
    trajectories = onp.array(trajectories)
    u0s = trajectories[trajectories[:, :, 1] == 1].reshape((-1, T, 5))
    u1s = trajectories[trajectories[:, :, 1] == 0].reshape((-1, T, 5))
    u0_means = np.array(np.mean(u0s[:, :, 0], axis=0))
    u1_means = np.array(np.mean(u1s[:, :, 0], axis=0))
    u0_stds = np.array(np.std(u0s[:, :, 0], axis=0))
    u1_stds = np.array(np.std(u1s[:, :, 0], axis=0))

    axs[0].errorbar(
        range(T),
        u0_means,
        yerr=u0_stds,
        linestyle="-",
        color="tab:blue",
        label="U=0",
        capsize=5,
        marker=".",
        markersize=10,
    )
    axs[0].errorbar(
        range(T),
        u1_means,
        yerr=u1_stds,
        linestyle="--",
        color="tab:orange",
        label="U=1",
        capsize=5,
        marker=".",
        markersize=10,
    )
    axs[0].legend(edgecolor="k", loc="upper right")
    axs[0].set_ylim([-2.5, 2.5])
    axs[0].set_title("Average Client Distance from Service Center")
    axs[0].set_ylabel("x - w")
    plt.xticks(np.arange(0, T, step=1.0))
    axs[0].axhline(0, linestyle="-", color="k", alpha=0.4)
    axs[0].spines["right"].set_visible(False)
    axs[0].spines["top"].set_visible(False)

    bar1 = axs[1].bar(
        range(T), MIs, color="tab:blue", alpha=0.5, edgecolor="k", linewidth=1.0
    )
    axs[1].set_ylim([0, 0.3])
    axs[1].set_xlabel("Timesteps")
    axs[1].set_ylabel(r"$I(a_t;u)$")
    axs[1].spines["right"].set_visible(False)
    axs[1].spines["top"].set_visible(False)
    axs[1].set_xticks(np.arange(0, T, step=1.0))
    plt.tight_layout()
    if filename is not None:
        plt.savefig(f"./{filename}.pdf", dpi=600)
    else:
        plt.show()


def plot_customer_no_MIs(trajectories, filename=None):
    fig, ax = plt.subplots()
    _, T, _ = trajectories.shape
    trajectories = onp.array(trajectories)
    u0s = trajectories[trajectories[:, :, 1] == 1].reshape((-1, T, 5))
    u1s = trajectories[trajectories[:, :, 1] == 0].reshape((-1, T, 5))
    u0_means = np.array(np.mean(u0s[:, :, 0], axis=0))
    u1_means = np.array(np.mean(u1s[:, :, 0], axis=0))
    u0_stds = np.array(np.std(u0s[:, :, 0], axis=0))
    u1_stds = np.array(np.std(u1s[:, :, 0], axis=0))

    ax.errorbar(
        range(T),
        u0_means,
        yerr=u0_stds,
        linestyle="-",
        color="tab:blue",
        label="U=0",
        capsize=5,
        marker=".",
        markersize=10,
    )
    ax.errorbar(
        range(T),
        u1_means,
        yerr=u1_stds,
        linestyle="--",
        color="tab:orange",
        label="U=1",
        capsize=5,
        marker=".",
        markersize=10,
    )
    ax.legend(edgecolor="k", loc="upper right")
    ax.set_ylim([-2.5, 2.5])
    ax.set_ylabel("x - w")
    plt.xticks(np.arange(0, T, step=1.0))
    ax.axhline(0, linestyle="-", color="k", alpha=0.4)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(f"./{filename}.pdf", dpi=600)
    else:
        plt.show()


def plot_distribution_t_MI_fig(MIs, trajectories, x_scaling, filename=None):
    T, _ = trajectories.shape[1:]

    trajs = onp.array(trajectories[:, -1, :2])

    trajs[:, [0, 1]] = trajs[:, [1, 0]]
    trajs[:, 0] = 1 - trajs[:, 0]
    trajs[:, 1] = trajs[:, 1] * x_scaling
    traj_pd = pd.DataFrame(trajs)
    traj_pd = traj_pd.replace([1, 0], [1, 2])  # Recode to the ACS codes
    traj_pd.columns = ["RAC1P", "HINCP"]

    xlim = 1e5
    x_grid = np.arange(0, xlim, 1e3).reshape(-1, 1)
    bandwidths = (10000, 10000)
    white_KD, black_KD, _ = compute_KDEs(traj_pd, bandwidths)
    white_scores = white_KD.score_samples(x_grid)
    black_scores = black_KD.score_samples(x_grid)

    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    axs = (plt.subplot(gs[0]), plt.subplot(gs[1]))
    axs[0].plot(x_grid, np.exp(white_scores), "g--", label="RAC1P = 1")
    axs[0].set_ylim([0, np.exp(max(white_scores))])
    axs[0].set_xlim([0, xlim])
    highest_density = np.max(np.concatenate((black_scores, white_scores)))
    axs[0].plot(x_grid, np.exp(black_scores), "b-", label="RAC1P = 2")
    axs[0].set_ylim([0, np.exp(highest_density)])
    axs[0].set_xlabel("Income/$")
    axs[0].set_ylabel("Income Density")
    axs[0].spines["right"].set_visible(False)
    axs[0].spines["top"].set_visible(False)
    axs[0].legend(edgecolor="k", loc="upper right")
    axs[0].tick_params(
        axis="y",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelleft=False,
    )  # labels along the bottom edge are off
    # plt.xlim([0, xlim])
    axs[0].legend()

    bar1 = axs[1].bar(
        range(T), MIs, color="tab:blue", alpha=0.5, edgecolor="k", linewidth=1.0
    )
    axs[1].set_ylim([0, 0.1])
    axs[1].set_xlabel("Timesteps")
    axs[1].set_ylabel(r"$I(a_t;u)$")
    axs[1].spines["right"].set_visible(False)
    axs[1].spines["top"].set_visible(False)
    axs[1].set_xticks(np.arange(0, T, step=1.0))
    plt.tight_layout()
    if filename is not None:
        plt.savefig(f"./{filename}.pdf", dpi=600)
    else:
        plt.show()
