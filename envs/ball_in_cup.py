from dm_control.suite.ball_in_cup import BallInCup, Physics
from mp_env_api.envs.positional_env import PositionalEnv
from mp_env_api.envs.mp_env import MpEnv
from dm_control import suite
from dm_control.rl.control import Environment
import numpy as np
from typing import Union
import gym
import collections

import mujoco_py


class BICEnv(BallInCup, PositionalEnv, MpEnv, gym.Env):
    def __init__(self):
        BallInCup.__init__(self)
        PositionalEnv.__init__(self)
        MpEnv.__init__(self)
        gym.Env.__init__(self)
        self.env = suite.load(domain_name="ball_in_cup", task_name="catch")

        self._start_pos = np.zeros(8)
        self._start_pos[0:4] = [0,         -0.00151631,  0,         -0.0021582]
        self._start_pos[4:8] = np.zeros(4)
        #print("self._start_pos",self._start_pos)



        #self._start_pos = np.ones(8)
        self._start_vel = np.zeros(2)

        self.physics = self.env.physics
        self.env.initialize_episode = self.initialize_episode(self.physics)
        self.n_substeps = 4
        #self.sim = mujoco_py.MjSim(self.physics.model, nsubsteps=self.n_substeps)
        #self.physics.set_state = self.set_state(self._start_pos)

    def set_state(self, physics_state):
        """Sets the physics state.

        Args:
          physics_state: NumPy array containing the full physics simulation state.

        Raises:
          ValueError: If `physics_state` has invalid size.
        """
        # print("physics_state",physics_state)
        state_items = self.physics._physics_state_items()
        # assert 1== 0

        expected_shape = (sum(item.size for item in state_items),)
        if expected_shape != physics_state.shape:
            raise ValueError('Input physics state has shape {}. Expected {}.'.format(
                physics_state.shape, expected_shape))

        start = 0
        state_items_total = []
        for state_item in state_items:
            size = state_item.size
            np.copyto(state_item, physics_state[start:start + size])
            state_items_total.append(state_item)
            start += size
        self.physics.data.qpos[0:] = state_items_total[0]
        self.physics.data.qvel[0:] = state_items_total[1]

        #self.physics.data.qpos = [0,0,0,0]
        self.physics.forward()
        # self.data.qvel
        #assert 1 == 0

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.
        Args:
          physics: An instance of `Physics`.
        """
        # Find a collision-free random initial position of the ball.
        penetrating = True
        while penetrating:
            # Assign a random ball position.
            # physics.named.data.qpos['ball_x'] = self.random.uniform(-.2, .2)
            # physics.named.data.qpos['ball_z'] = self.random.uniform(.2, .5)
            physics.named.data.qpos['ball_x'] = 0 * np.ones(1)
            physics.named.data.qpos['ball_z'] = 0 * np.ones(1)
            # Check for collisions.
            physics.after_reset()
            penetrating = physics.data.ncon > 0
        super().initialize_episode(physics)


    def step(self, action):
        print(self.physics.position())
        #assert 1==0
        #self.reset()
        self.physics.step()
        return self.env.step(action)


    def reset(self):
        self.physics.reset()
        self.env.reset()
        start_pos = self.start_pos
        self.set_state(start_pos)

        #print("self.physics.data.qpos",self.physics.data.qpos)
        #
        super(BICEnv, self).__init__()
        #Environment.reset(self)


    @property
    def start_pos(self) -> Union[float, int, np.ndarray]:
        """
        Returns the starting position of the joints
        """
        return self._start_pos

    @property
    def dt(self) -> Union[float, int]:
        """
        Returns the time between two simulated steps of the environment
        """
        return 0.1

    def get_observation(self):
        """Returns an observation of the state."""
        obs = collections.OrderedDict()
        obs['position'] = self.physics.position()

        #ball_id = self.physics.model._body_name2id["ball"]
        #ball_pos = self.physics.model.body_xpos[ball_id]
        #print("ball_pos ",ball_pos )
        #print("obs['position'] ",obs['position'] )
        #assert 1== 0
        obs['velocity'] = self.physics.velocity()
        #self.set_state()
        return obs



