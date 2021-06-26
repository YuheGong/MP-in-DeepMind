import gym
from dm_control.rl import control
import dm_env
import numpy as np
from typing import Callable, Tuple, Optional
from collections import OrderedDict


def _default_construct_obs_fn(observation: OrderedDict):
    obs = []
    for _, v in observation.items():
        obs.append(v.ravel())
    return np.concatenate(obs, -1)


class DMEnvWrapper(gym.Env):

    def __init__(self,
                 base_env: control.Environment,
                 construct_obs_fn: Callable = None,
                 terminate_fn: Callable = None,
                 dtype=np.float32,
                 render_size: Tuple[int, int] = (64, 64),
                 camera_id: int = 0,
                 extract_input_fn: Optional[Callable] = None,
                 extract_target_fn: Optional[Callable] = None):

        self._base_env = base_env
        self._render_size = render_size
        self._camera_id = camera_id
        self._extract_input_fn = extract_input_fn
        self._extract_target_fn = extract_target_fn
        #self.physics = base_env.physics

        self._construct_obs = _default_construct_obs_fn if construct_obs_fn is None else construct_obs_fn
        self._terminate_fn = terminate_fn
        test_obs = self.reset()
        #print("test_obs",test_obs)

        self.observation_space = gym.spaces.Box(low=-float('inf'),
                                                high=float('inf'),
                                                shape=test_obs['position'].shape,
                                                dtype=dtype)
        action_space = self._base_env.action_spec(self._base_env.physics)
        self.action_space = gym.spaces.Box(low=action_space.minimum,
                                           high=action_space.maximum,
                                           shape=action_space.shape,
                                           dtype=dtype)

        self.metadata = {}

    def step(self, action):
        time_step = self._base_env.step(action)
        if self._terminate_fn is None:
            done = time_step.step_type == dm_env.StepType.LAST
        else:
            done = time_step.step_type == dm_env.StepType.LAST or self._terminate_fn(self._base_env.physics)
        return self._construct_obs(time_step.observation), time_step.reward, done, {"time_step": time_step}

    def reset(self):
        #return self._construct_obs(self._base_env.reset().observation)
        self._base_env.reset()
        return self._base_env.get_observation()

    def render(self, mode='rgb_array'):
        if mode != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._base_env.physics.render(*self._render_size, camera_id=self._camera_id)

    def get_inputs_and_target(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self._extract_input_fn is not None and self._extract_target_fn is not None
        return self._extract_input_fn(self._base_env), self._extract_target_fn(self._base_env)
