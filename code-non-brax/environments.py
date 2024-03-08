from __future__ import division
from jax import ops
import jax.numpy as np
import jax.random as rnd
from utils import (
    OracleVPNDynamicsModel,
    OracleQuadraticImpulseDynamicsModel,
    OracleCustomerDynamicsModel,
    OracleSNAPDynamicsModel,
)
import numpy as onp


class QuadraticImpulseEnvironment:
    def __init__(self, force_noise_std, p0_std):
        self.force_noise_std = force_noise_std
        self.p0_std = p0_std
        self.public_state_dim = 3
        self.private_state_dim = 1
        self.action_dim = 4
        self.oracle_dynamics_model = OracleQuadraticImpulseDynamicsModel(self)

    def draw_p0(self, rand_key):
        x_0, u_0 = rnd.normal(rand_key, (2,)) * self.p0_std
        return np.array([x_0, 0.0, 0.0, u_0])

    def transition(self, state, action, rng):
        # State is [x, x_dot, u_dot, u]
        # We have some amount of random force applied to the ball
        # Action is one-hot, [0, 0, 0, 0]
        # Actions correspond to right, left, up, down
        # Then we have the standard Newton updates
        # x_dot = x_dot + noise[0]
        # u_dot = u_dot + noise[1]
        # x = x + x_dot
        # u = u + u_dot
        x, x_dot, u_dot, u = state
        force_noise = rnd.normal(rng, (2,)) * self.force_noise_std
        x_dot_next = x_dot + force_noise[0] + (action == 0) - (action == 1)
        u_dot_next = u_dot + force_noise[1] + (action == 2) - (action == 3)
        x_next = x + x_dot_next
        u_next = u + u_dot_next
        reward = -(x_next ** 2) - u_next ** 2
        return np.array([x_next, x_dot_next, u_dot_next, u_next]), reward

    def __str__(self):
        return "Quadratic Impulse Environment"


class VPNEnvironment:
    """Encapsulates the VPN example.

    Attributes:
        num_servers: number of servers
        r_correct_no_vpn: reward for correct action, no vpn
        r_correct_vpn: reward for correct action, vpn
        r_wrong_no_vpn: reward for wrong action, no vpn
        r_wrong_vpn: reward for wrong action, vpn
        r_vpn: reward for activating the vpn
        p_change_u: probability u changes in the next timestep

    """

    def __init__(
        self,
        num_servers,
        r_correct_no_vpn,
        r_correct_vpn,
        r_wrong_no_vpn,
        r_wrong_vpn,
        r_vpn,
        p_change_u,
    ):
        self.action_dim = num_servers + 1
        self.public_state_dim = 2
        self.private_state_dim = num_servers
        self.num_servers = num_servers
        self.r_correct_no_vpn = r_correct_no_vpn
        self.r_correct_vpn = r_correct_vpn
        self.r_wrong_no_vpn = r_wrong_no_vpn
        self.r_wrong_vpn = r_wrong_vpn
        self.r_vpn = r_vpn
        if p_change_u != 0:
            raise NotImplementedError
        self.p_change_u = p_change_u
        self.oracle_dynamics_model = OracleVPNDynamicsModel(self)

    def u_frac(self):
        return 1.0 / self.num_servers

    def draw_p0(self, rand_key):
        """Draw the initial state from the distribution"""
        rng_0, rng_1, rng_2 = rnd.split(rand_key, 3)
        u_0_idx = rnd.categorical(
            rng_0, logits=np.ones(self.num_servers) / self.num_servers
        )
        u_0 = ops.index_update(np.zeros(self.num_servers), u_0_idx, 1)
        x_0 = np.array([1, 0])
        return np.concatenate((x_0, u_0))

    def transition(self, state, action, rng):
        """Return the reward and next state after taking action
        in state

        Args:
            action: An integer specifying the action to take
                note that we use an integer since it's the output of 
                the jax.random categorical.

        Recall that the action is of dimensionality `num_servers + 1`,
        since the first possible action denotes activating the VPN"""
        rng_1, rng_2 = rnd.split(rng)
        x = state[: self.public_state_dim]  # VPN or not
        u = state[
            self.public_state_dim : self.public_state_dim + self.private_state_dim
        ]
        activated_vpn = action == 0
        has_vpn = x[1] == 1
        correct_server_chosen = np.argmax(u) + 1 == action

        reward = activated_vpn * self.r_vpn + (
            (1 - activated_vpn)
            * (
                has_vpn
                * (
                    correct_server_chosen * self.r_correct_vpn
                    + (1 - correct_server_chosen) * self.r_wrong_vpn
                )
                + (1 - has_vpn)
                * (
                    correct_server_chosen * self.r_correct_no_vpn
                    + (1 - correct_server_chosen) * self.r_wrong_no_vpn
                )
            )
        )

        change_u = rnd.bernoulli(rng_1, self.p_change_u)
        u_next_idx = (
            (
                change_u * rnd.categorical(rng_2, np.ones(self.num_servers,) * 1.0)
                + (1 - change_u) * np.argmax(u)
            )
        ).astype(np.int32)
        u_next = ops.index_update(np.zeros(self.num_servers), ops.index[u_next_idx], 1)
        x_next = activated_vpn * np.array([0, 1]) + (1 - activated_vpn) * x

        return np.concatenate((x_next, u_next)), reward

    def __str__(self):
        return "VPN Environment"


class SNAPEnvironment:
    def __init__(
        self, white_income_kde, black_income_kde, u0_frac, poverty_level=24900
    ):
        self.u0_fraction = u0_frac
        self.black_fraction = self.u0_fraction
        self.white_kde = white_income_kde
        self.black_kde = black_income_kde
        self.poverty_level = poverty_level
        self.income_noise_std = 1000
        self.snap_benefit_range = [12 * 126, 12 * 600]
        self.snap_admin_cost = 5e-2  # Add some cost to do the program
        self.reward_scaling = 1e8
        self.x_scaling = 1e5
        self.transform_kdes()

        self.action_dim = 2
        self.public_state_dim = 1  # Income
        self.private_state_dim = 2  # race

        self.process_noise_level = self.income_noise_std
        self.oracle_dynamics_model = OracleSNAPDynamicsModel(self)

    def u_frac(self):
        return self.u0_fraction

    def __str__(self):
        return "SNAP Environment"

    def transform_kdes(self):
        """Turn the KDEs from scikit-learn objects into jax
        objects that we can use in native jax. We only need the
        data objects (and bandwidths) after this"""
        # assert self.white_kde.tree_.sample_weight is None
        # assert self.black_kde.tree_.sample_weight is None
        self.white_data = np.array(onp.asarray(self.white_kde.tree_.data))
        self.black_data = np.array(onp.asarray(self.black_kde.tree_.data))
        self.white_bandwidth = np.array(self.white_kde.bandwidth)
        self.black_bandwidth = np.array(self.black_kde.bandwidth)
        self.jax_white_data = np.ones(len(self.white_data))
        self.jax_black_data = np.ones(len(self.black_data))

        self.jax_white_data = self.jax_white_data * self.white_data.reshape(-1)
        self.jax_black_data = self.jax_black_data * self.black_data.reshape(-1)

        self.white_data = self.jax_white_data
        self.black_data = self.jax_black_data

    def sample_white_kde(self, rng):
        rng1, rng2 = rnd.split(rng, 2)
        u = rnd.uniform(rng1, (1,), minval=0, maxval=1)
        i = np.array(u * self.white_data.shape[0], dtype=np.int32)
        return (
            rnd.normal(rng2, (1,)) * np.sqrt(self.white_bandwidth) + self.white_data[i]
        )

    def sample_black_kde(self, rng):
        rng1, rng2 = rnd.split(rng, 2)
        u = rnd.uniform(rng1, (1,), minval=0, maxval=1)
        i = np.array(u * self.black_data.shape[0], dtype=np.int32)
        return (
            rnd.normal(rng2, (1,)) * np.sqrt(self.black_bandwidth) + self.black_data[i]
        )

    def draw_p0(self, rand_key):
        """Draw a sample from the KDE"""
        # if u == [0, 1], sampled person is white
        # Compute p(race, income) = p(income|race)*p(race)
        rand_key1, rand_key2, rand_key3, rand_key4 = rnd.split(rand_key, 4)
        u_idx = rnd.bernoulli(rand_key1, 1 - self.black_fraction)
        white_sample = self.sample_white_kde(rand_key3)
        black_sample = self.sample_black_kde(rand_key4)
        x = np.where(u_idx, white_sample, black_sample)
        u = np.array([0, 1]) * u_idx + np.array([1, 0]) * (1 - u_idx)
        return np.concatenate((x / self.x_scaling, u))

    def transition(self, state, give_snap, rng_key):
        """Return the reward and next state after giving
        or not giving SNAP in state x"""
        rng1, rng2, rng3 = rnd.split(rng_key, 3)
        x, u = state[0], state[1:3]
        x = x * self.x_scaling  # Get back to real (unnormalized) value_counts
        sigma = rnd.normal(rng_key, (1,))[0] * self.income_noise_std
        x_next_snap = (
            sigma
            + x
            + rnd.uniform(
                rng1,
                (1,),
                minval=self.snap_benefit_range[0],
                maxval=self.snap_benefit_range[1],
            )
        )
        x_next_no_snap = sigma + x
        x_next = np.where(give_snap, x_next_snap, x_next_no_snap)
        cost = (np.maximum(self.poverty_level - x_next, 0) ** 2)[0]
        snap_admin_cost = np.where(give_snap, self.snap_admin_cost, 0)

        return (
            np.concatenate((x_next / self.x_scaling, u)),
            -cost / self.reward_scaling - snap_admin_cost,
        )


class CustomerEnvironment:
    def __init__(
        self,
        separation=10,
        process_noise_level=1.0 / 4,
        reward_adjustment=0.2,
        init_noise_std=3,
        u_transition_dependence=0.0,
    ):
        self.action_dim = 2
        self.public_state_dim = 1
        self.private_state_dim = 2
        self.separation = separation
        self.process_noise_level = process_noise_level
        self.init_noise_std = init_noise_std
        self.reward_adjustment = reward_adjustment
        self.u_transition_dependence = u_transition_dependence
        self.oracle_dynamics_model = OracleCustomerDynamicsModel(self)

    def u_frac(self):
        return 0.5

    def draw_p0(self, rand_key):
        """Draw initial state from the distribution"""
        rand_key1, rand_key2 = rnd.split(rand_key, 2)
        u_idx = rnd.bernoulli(rand_key1, 0.5)  # either 1 or 0
        u = u_idx * np.array([0, 1]) + (1 - u_idx) * np.array([1, 0])
        x = (
            rnd.normal(rand_key2, (1,))[0] * self.init_noise_std
            + self.separation * u_idx
        )
        return np.concatenate((x.reshape((1,)), u.reshape(2,)))

    def transition(self, state, give_loan, rng_key):
        """Return the reward and next state after taking action a
        in state x"""
        sigma = rnd.normal(rng_key, (1,))[0] * np.sqrt(self.process_noise_level)
        # give_loan is a bool, with `take_action_1` equal to true (i.e. 1)
        # `take_action_0` equal to false, (i.e. 0)
        x, u = state[0], state[1:]
        u_idx = np.argmax(u)
        x_next = (
            x
            + (give_loan - 0.5) * 2
            + sigma
            + (u_idx - 0.5) * self.u_transition_dependence
        )
        # reward = -np.abs(x_next) + self.reward_adjustment
        reward = -np.abs(x_next) + self.reward_adjustment
        return np.concatenate((x_next.reshape((1,)), u.reshape((2,)))), reward

    def __str__(self):
        return "Customer Service Environment"
