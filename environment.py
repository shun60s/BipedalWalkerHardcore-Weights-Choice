# -*- coding: utf-8 -*-

"""
-----------------------------------------------------------------------------
   Copyright 2017 David Griffis

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
-----------------------------------------------------------------------------
Changed:
        Add PassThr():
        Add custom env
"""

from __future__ import division
import custom_env  # add
import gym
import numpy as np
from collections import deque
from gym import spaces


def create_env(env_id, args):
    env = gym.make(env_id)
    env = frame_stack(env, args)
    return env

"""
Observation wrapper that stacks the observations in a rolling manner.
    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v0', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].

"""

class frame_stack(gym.Wrapper):   # gym Wrapper has frame_stack
    def __init__(self, env, args):
        super(frame_stack, self).__init__(env)
        self.stack_frames = args.stack_frames
        self.frames = deque([], maxlen=self.stack_frames)

        # 3 choices
        #   MaxMinFilter()
        #   NormalizedEnv() alternative or can 
        #   just not normalize observations as environment is already kinda normalized
        self.obs_norm = MaxMinFilter() 


    def reset(self):
        ob = self.env.reset()
        ob = np.float32(ob)
        ob = self.obs_norm(ob)
        for _ in range(self.stack_frames):
            self.frames.append(ob)
        return self.observation()

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        #print ('ob', ob)  # dump ob
        ob = np.float32(ob)
        ob = self.obs_norm(ob)
        #print ('ob', ob)  # dump ob
        self.frames.append(ob)
        return self.observation(), rew, done, info

    def observation(self):
        assert len(self.frames) == self.stack_frames

        #print ('np.stack self.frames', np.stack(self.frames, axis=0) )
        #                                             stack_frame order is older, old, present
        return np.stack(self.frames, axis=0)      #  [self.stack_frames, 24]


# trans max-min from clip (-/+3.15) to -/+10
class MaxMinFilter():
    def __init__(self):
        self.mx_d = 3.15
        self.mn_d = -3.15
        self.new_maxd = 10.0
        self.new_mind = -10.0

    def __call__(self, x):
        obs = x.clip(self.mn_d, self.mx_d)
        new_obs = (((obs - self.mn_d) * (self.new_maxd - self.new_mind)
                    ) / (self.mx_d - self.mn_d)) + self.new_mind
        return new_obs


# moving mean and std, trans to mean =0 and std is 1?
class NormalizedEnv():
    def __init__(self):
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def __call__(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)

# Add, not normalize observations, just return observation
class PassThr():
    def __call__(self, observation):
        return observation

