import pickle
import os
import uuid

import numpy as np
# import tensorflow as tf

from gps.algorithm.policy.policy import Policy
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.ignasi.envs.pr2_disk_env import Pr2DiskEnv
from rllab.envs.normalized_env import normalize


class ThPolicy(Policy):
    """
    A neural network policy implemented in tensor flow. The network output is
    taken to be the mean, and Gaussian noise is added on top of it.
    U = net.forward(obs) + noise, where noise ~ N(0, diag(var))
    Args:
        obs_tensor: tensor representing tf observation. Used in feed dict for forward pass.
        act_op: tf op to execute the forward pass. Use sess.run on this op.
        var: Du-dimensional noise variance vector.
        sess: tf session.
        device_string: tf device string for running on either gpu or cpu.
    """
    def __init__(self, npz_path, recurrent=True, *shit, **random):
        Policy.__init__(self)
        self.policy = self.load_policy(npz_path, recurrent)
        self.dU = 1

    def act(self, more_shit, obs, *garbage, **more_garbage):
        """
        Return an action for a state.
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: Action noise. This will be scaled by the variance.
        """

        # Normalize obs.
        obs = np.array(obs).reshape(-1)
        action, action_var = self.policy.get_action(obs)
        return action_var['mean']

    @classmethod
    def load_policy(cls, npz_path, recurrent):
        """
        For when we only need to load a policy for the forward pass. For instance, to run on the robot from
        a checkpointed policy.
        """
        env = normalize(Pr2DiskEnv())
        if recurrent:
            policy = GaussianGRUPolicy(env.spec, npz_path=npz_path)
        else:
            policy = GaussianMLPPolicy(env.spec, npz_path=npz_path)
        return policy
