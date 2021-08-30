import gym
import logging
import pandas as pd
import numpy as np
import clr
import sys
import time
import datetime

assembly_path = r"path of the simulator dll..."
sys.path.append(assembly_path)
clr.AddReference("dllSchedulerPrueba")
from schedulerPrueba import Scheduler

from gym import spaces
from typing import Union, Tuple, List
from gym import spaces
from PIL import Image

from stable_baselines.common.policies import CnnLstmPolicy, MlpLnLstmPolicy, LstmPolicy, MlpLstmPolicy, \
    FeedForwardPolicy, CnnPolicy, nature_cnn
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines import PPO2, ACER
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds

import os.path

class SchedulingEnvironment(gym.Env):
    """A scheduling environment made for use with Gym-compatible reinforcement learning algorithms."""

    def __init__(self, temp_folder: str, picture_path: str, fixed_sch_path: str, even_cons: str, goal_sch: str,
                 rand_numb_task: bool, min_tar: int, max_tar: int, min_kil: int, max_kil: int, min_dd: int, max_dd: int,
                 train: bool, log_dir, **kwargs):
        """
        Arguments:

        kwargs (optional): Additional arguments for tuning the environments, logging, etc.
        """

        super(SchedulingEnvironment, self).__init__()

        self.observation_space = spaces.Box(low=0, high=255, shape=(100, 50, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(14)

        self._log_dir = log_dir
        self._temp_folder = temp_folder
        self._reduced_tardiness = []
        self._invalid_ops = []
        self._invalid_ops_counter = 0
        #instance of Scheduler
        self._scheduler = Scheduler(picture_path, fixed_sch_path, even_cons, goal_sch, rand_numb_task, min_tar,
                                    max_tar, min_kil, max_kil, min_dd, max_dd, train)

        #self._bankrupt_penalty = kwargs.get('bankrupt_penalty', 0)
        #self._step_bonus = kwargs.get('step_bonus', False)

        self.logger = logging.getLogger(kwargs.get('logger_name', __name__))
        self.logger.setLevel(kwargs.get('log_level', logging.DEBUG))

        logging.getLogger('tensorflow').disabled = kwargs.get('disable_tensorflow_logger', True)

        #self.reset()
    def get_invalid_ops(self):
        return self._invalid_ops

    def get_reduced_tardiness(self):
        return self._reduced_tardiness

    def save_tardiness_reduction(self, item):
        with open(os.path.join(self._log_dir, 'tardiness_reduction.txt'), 'a+') as f:
            f.write("%s\n" % item)

    def save_illegal_ops_counter(self, item):
        with open(os.path.join(self._log_dir, 'illegal_ops_counter.txt'), 'a+') as f:
           f.write("%s\n" % item)

    def get_image(self, image_bitmap):
        """Get a numpy array of an image in order to access values[x][y]."""

        data_folder = os.path.join(self._temp_folder)
        image_path = os.path.join(data_folder, "imagenreduced.bmp")
        image = Image.open(image_path, "r")
        width, height = image.size
        pixel_values = list(image.getdata())

        if image.mode == "RGB":
            channels = 3
        elif image.mode == "L":
            channels = 1
        else:
            print("Unknown mode: %s" % image.mode)
            return None
        pixel_values = np.array(pixel_values).reshape((width, height, channels))
        #delete temporal image
        os.remove(image_path)

        return pixel_values

    def step(self, action):
        """Run one timestep within the environments based on the specified action.

        Arguments:
        action: The rescheduling action provided by the agent for this timestep.

        Returns:
        observation (pandas.DataFrame): Provided by the environments's scheduler.
        reward (float): An amount corresponding to the benefit earned by the action taken this timestep.
        done (bool): If `True`, the environments is complete and should be restarted.
        info (dict): Any auxiliary, diagnostic, or debugging information to output.
        """

        result = self._scheduler._step(action)

        observation = self.get_image(result[0])
        reward = result[1]
        done = result[2]
        info = {}
        if reward == -1:
            self._invalid_ops_counter = self._invalid_ops_counter + 1

        if done:
            self._reduced_tardiness.append(reward)
            self._invalid_ops.append(self._invalid_ops_counter)
            self.save_tardiness_reduction(str(reward))
            self.save_illegal_ops_counter(str(self._invalid_ops_counter))

        return observation, reward, done, info

    def reset(self):
        """Resets the state of the environments and returns an initial observation.
        Returns:
        observation: the initial observation.
        """
        self._current_step = 0
        self._invalid_ops_counter = 0

        image = self._scheduler._reset()
        obs = self.get_image(image)
        return obs

    def render(self, mode='none'):
        """Renders the environments."""

    pass

class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse, act_fun= tf.tanh,
                         net_arch=[256, 'lstm', dict(vf=[128, 256, 128], pi=[128, 256, 128])],
                         layer_norm=False, feature_extraction="mlp", **_kwargs)


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[512, dict(pi=[256, 512, 256],
                                                          vf=[256, 512, 256])],
                                           feature_extraction="cnn", cnn_extractor=nature_cnn)

class CustomPolicy2(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy2, self).__init__(*args, **kwargs,
                                           net_arch=[256, dict(pi=[128, 256, 128],
                                                          vf=[128, 256, 128])],
                                           feature_extraction="cnn", cnn_extractor=nature_cnn)

def save_test_episode_reward(list, log_dir):
        with open(os.path.join(log_dir, 'episode_reward_test.txt'), 'a+') as f:
            for item in list:
                f.write("%s\n" % item)


def save_test_tardiness_reduction(list, log_dir):
    with open(os.path.join(log_dir, 'tardiness_reduction_test.txt'), 'a+') as f:
        for item in list:
            f.write("%s\n" % item)

if __name__ == '__main__':
    event = 'Order Insertion'
    event_dir = 'insertion_normal'
    goal = 'Optimizar'
    log_dir = "./log/experiment/training/{}".format(event_dir)
    log_dir_test = "./log/experiment/test/{}".format(event_dir)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir_test, exist_ok=True)

    #env = SchedulingEnvironment("temp", "F:\Schedules", "F:\Schedules", event, goal, True, 25
    #                            , 30, 100, 1001, 30, 100, True, log_dir)

    env = SchedulingEnvironment("temp", "F:\Schedules", "F:\Schedules", event, goal, True, 15
                                , 20, 100, 1001, 0, 0, True, log_dir)

    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    
    #env = VecFrameStack(env, 3)
    # Automatically normalize the input features and reward

    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=1e6, clip_reward=1e5)

    model_train = PPO2(CustomPolicy, env, n_steps=128, learning_rate=0.0001,
                       nminibatches=4, noptepochs=10, verbose=2, ent_coef=0.02, cliprange=0.2,
                       tensorboard_log="./log")

    # model_train = ACER(MlpLstmPolicy, env, n_steps=256, verbose=2,
    #                   tensorboard_log="./log")

    model_train.learn(total_timesteps=300000)

    # Save the VecNormalize statistics when saving the agent
    model_train.save(log_dir + "ppo_Ord_Ins")
    stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    env.save(stats_path)

    del model_train, env

    # Load the agent
    model_test = PPO2.load(log_dir + "ppo_Ord_Ins")
    #env = SchedulingEnvironment("temp", "F:\Schedules", "F:\Schedules", event, goal, True, 25
    #                            , 30, 100, 1001, 30, 100, False, log_dir_test)

    env = SchedulingEnvironment("temp", "F:\Schedules", "F:\Schedules", event, goal, True, 15
                                , 20, 100, 1001, 0, 0, False, log_dir_test)

    env = Monitor(env, log_dir_test, allow_early_resets=True,)
    # Load the saved statistics
    env = DummyVecEnv([lambda: env])
    #env = VecFrameStack(env, 3)
    env = VecNormalize.load(stats_path, env)
    #  do not update them at test time
    env.training = False
    # reward normalization is not needed at test time
    env.norm_reward = False

    #Policy Test
    episode_rewards = []
    tardiness_reduced = []
    for _ in range(0, 1000):
        reward_sum = 0
        obs = env.reset()
        # Passing state=None to the predict function means
        # it is the initial state
        state = None
        # When using VecEnv, done is a vector
        done = False
        while not done:
            # We need to pass the previous state and a mask for recurrent policies
            # to reset lstm state when a new episode begin
            action, state = model_test.predict(obs)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward[0]
            if done:
                tardiness_reduced.append(reward[0])
            # Note: with VecEnv, env.reset() is automatically called
        episode_rewards.append(reward_sum)
    save_test_episode_reward(episode_rewards, log_dir_test)
    save_test_tardiness_reduction(tardiness_reduced, log_dir_test)
    print("Fin")