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
       Add to use custom env
       Add CONV6_Net
       Add load two basis models
       Add CONV_Choice1_Net
       Add observation_num for state_out, BipedalWalkerHardcoreStateout-v2

"""
from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1" # numpy...をimportする前に使えるスレッド数を制限
import argparse
import torch
from environment import create_env
from utils import setup_logger
from model import *  # change to import any models
from player_util import Agent
from torch.autograd import Variable
import gym
import logging

from gym.wrappers.monitor import Monitor  # add to use custom env

parser = argparse.ArgumentParser(description='A3C_EVAL')
parser.add_argument(
    '--env',
    default='BipedalWalkerHardcore-v2',
    metavar='ENV',
    help='environment to train on (default: BipedalWalker-v2)')
parser.add_argument(
    '--num-episodes',
    type=int,
    default=100,
    metavar='NE',
    help='how many episodes in evaluation (default: 100)')
parser.add_argument(
    '--load-model-dir',
    default='trained_models/',
    metavar='LMD',
    help='folder to load trained models from')
parser.add_argument(
    '--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument(
    '--render',
    default=False,
    metavar='R',
    help='Watch game as it being played')
parser.add_argument(
    '--render-freq',
    type=int,
    default=1,
    metavar='RF',
    help='Frequency to watch rendered game play')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=10000,
    metavar='M',
    help='maximum length of an episode (default: 10000)')
parser.add_argument(
    '--model',
    default='CONV_Choice1',
    metavar='M',
    help='Model type to use')
parser.add_argument(
    '--stack-frames',
    type=int,
    default=4,
    metavar='SF',
    help='Choose whether to stack observations')
parser.add_argument(
    '--new-gym-eval',
    default=True,
    metavar='NGE',
    help='Create a gym evaluation for upload')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--gpu-id',
    type=int,
    default=-1,
    help='GPU to use [-1 CPU only] (default: -1)')
parser.add_argument(
    '--basis-model1',
    default='CONV6',
    metavar='M1',
    help='Model type to use')
parser.add_argument(
    '--basis-model1-file',
    default='BipedalWalkerHardcore-v2_CONV6_Net_mix_trained.dat',
    metavar='MD1',
    help='trained models file')
parser.add_argument(
    '--basis-model2',
    default='CONV6',
    metavar='M2',
    help='Model type to use')
parser.add_argument(
    '--basis-model2-file',
    default='BipedalWalkerHardcoreStump1-v0_CONV6_Net_mix_trained.dat',
    metavar='MD2',
    help='trained models file')
parser.add_argument(
    '--discrete-number',
    type=int,
    default=4,
    metavar='NS2',
    help='number of discrete model which model CONV_Choice handles')
parser.add_argument(
    '--use-discrete-model',
    default=True,
    metavar='AM2',
    help='use discreate model for train')

args = parser.parse_args()

torch.set_default_tensor_type('torch.FloatTensor')

saved_state = torch.load(
    '{0}{1}.dat'.format(args.load_model_dir, args.env),
    map_location=lambda storage, loc: storage)

log = {}
setup_logger('{}_mon_log'.format(args.env), r'{0}{1}_mon_log'.format(
    args.log_dir, args.env))
log['{}_mon_log'.format(args.env)] = logging.getLogger(
    '{}_mon_log'.format(args.env))

gpu_id = args.gpu_id

torch.manual_seed(args.seed)
if gpu_id >= 0:
    torch.cuda.manual_seed(args.seed)


d_args = vars(args)
for k in d_args.keys():
    log['{}_mon_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

env = create_env("{}".format(args.env), args)
num_tests = 0
reward_total_sum = 0
player = Agent(None, env, args, None)

if args.model == 'CONV_Choice1':
     player.model = CONV_Choice1_Net(args.stack_frames, env.action_space, args.discrete_number, env.observation_space.shape[0]) # change
#--- set basis_models ---
if args.basis_model1 == 'CONV6':
    player.bm1_model = CONV6_Net(args.stack_frames, env.action_space,  env.observation_space.shape[0]) # change
    bm1_saved_state = torch.load('{0}{1}'.format(
        args.load_model_dir, args.basis_model1_file), map_location=lambda storage, loc: storage)
if args.basis_model1 == 'CONV6':
    player.bm2_model = CONV6_Net(args.stack_frames, env.action_space,  env.observation_space.shape[0]) # change
    bm2_saved_state = torch.load('{0}{1}'.format(
        args.load_model_dir, args.basis_model2_file), map_location=lambda storage, loc: storage)
#--- set basis_models ---


player.gpu_id = gpu_id
if gpu_id >= 0:
    with torch.cuda.device(gpu_id):
        player.model = player.model.cuda()
        player.bm1_model = player.bm1_model.cuda()
        player.bm2_model = player.bm2_model.cuda()
if args.new_gym_eval:
    player.env = gym.wrappers.Monitor(
        player.env, "{}_monitor".format(args.env), force=True)

#-if output mp4 every episode.
#
#if args.new_gym_eval:
#    player.env = gym.wrappers.Monitor(
#        player.env, "{}_monitor".format(args.env), force=True, video_callable=(lambda ep: ep % 1 == 0))
#
# -force (bool): Clear out existing training data from this directory


if gpu_id >= 0:
    with torch.cuda.device(gpu_id):
        player.model.load_state_dict(saved_state)
        player.bm1_model.load_state_dict(bm1_saved_state)
        print ('set basis-model1 and load ', args.basis_model1_file )
        player.bm2_model.load_state_dict(bm2_saved_state)
        print ('set basis-model2 and load ', args.basis_model2_file )
else:
    player.model.load_state_dict(saved_state)
    player.bm1_model.load_state_dict(bm1_saved_state)
    print ('set basis-model1 and load ', args.basis_model1_file )
    player.bm2_model.load_state_dict(bm2_saved_state)
    print ('set basis-model2 and load ', args.basis_model2_file )

player.model.eval()
player.bm1_model.eval()
player.bm2_model.eval()

for i_episode in range(args.num_episodes):
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
    player.eps_len = 0
    reward_sum = 0
    state_out_loss_sum = 0 # add
    state_out_hit = 0      # add
    
    while True:
        if args.render:
            if i_episode % args.render_freq == 0:
                player.env.render()

        player.action_test()
        reward_sum += player.reward
        
        # add
        if args.use_discrete_model:
            if player.env.observation_space.shape[0] == 28 and args.discrete_number == 4:
                state_out_loss_sum += player.loss_state_out[-1].detach().numpy()
                
                if player.hit:
                    state_out_hit += 1

        if player.done:
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            
            # add
            if args.use_discrete_model:
                if player.env.observation_space.shape[0] == 28 and args.discrete_number == 4:
                    state_out_loss_mean = state_out_loss_sum / player.eps_len 
                    hit_ratio = state_out_hit / player.eps_len * 100.0 # percentage
                    
                    log['{}_mon_log'.format(args.env)].info(
                        "Episode_length, {0}, reward_sum, {1}, reward_mean, {2:.4f}, state_out_loss_mean, {3:.8f}, hit_ratio, {4:.5f} %".format(player.eps_len, reward_sum, reward_mean, state_out_loss_mean, hit_ratio))
            else:
                log['{}_mon_log'.format(args.env)].info(
                    "Episode_length, {0}, reward_sum, {1}, reward_mean, {2:.4f}".format(player.eps_len, reward_sum, reward_mean))
            
            break
