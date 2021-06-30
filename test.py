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
        Add state_to_save message.
        Add args.save last
        Add CONV6_Net
        Add load two basis models
        Add CONV_Choice1_Net
"""

from __future__ import division
from setproctitle import setproctitle as ptitle
import numpy as np
import torch
from environment import create_env
from utils import setup_logger
from model import *  # change to import any models
from player_util import Agent
from torch.autograd import Variable
import time
import logging
import gym


def test(args, shared_model, optimizer,  shared_bm1_model, shared_bm2_model):  # change
    ptitle('Test Agent')
    gpu_id = args.gpu_ids[-1]
    log = {}
    setup_logger('{}_log'.format(args.env),
                 r'{0}{1}_log'.format(args.log_dir, args.env))
    log['{}_log'.format(args.env)] = logging.getLogger(
        '{}_log'.format(args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
    env = create_env(args.env, args)
    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    
    
    if args.model == 'CONV_Choice1':
        player.model = CONV_Choice1_Net(args.stack_frames, player.env.action_space, args.discrete_number, player.env.observation_space.shape[0]) # change
    if args.basis_model1 == 'CONV6':
        player.bm1_model = CONV6_Net(args.stack_frames, player.env.action_space, player.env.observation_space.shape[0]) # change
    if args.basis_model2 == 'CONV6':
        player.bm2_model = CONV6_Net(args.stack_frames, player.env.action_space, player.env.observation_space.shape[0]) # change
    
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()
            player.bm1_model = player.bm1_model.cuda()
            player.bm2_model = player.bm2_model.cuda()
            player.state = player.state.cuda()
    
    player.bm1_model.eval()
    player.bm2_model.eval()
    player.model.eval()
    max_score = 0
    state_out_loss_sum = 0 # add
    state_out_hit = 0      # add
    
    while True:
        if player.done:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.model.load_state_dict(shared_model.state_dict())
                    player.bm1_model.load_state_dict(shared_bm1_model.state_dict())
                    player.bm2_model.load_state_dict(shared_bm2_model.state_dict())
            else:
                player.model.load_state_dict(shared_model.state_dict())
                player.bm1_model.load_state_dict(shared_bm1_model.state_dict())
                player.bm2_model.load_state_dict(shared_bm2_model.state_dict())

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
                    
                    log['{}_log'.format(args.env)].info(
                        "Time {0}, episode reward, {1}, episode length, {2}, state_out_loss_mean, {3:.8f}, hit_ratio, {4:.5f} % ".
                        format(
                            time.strftime("%Hh %Mm %Ss",
                                          time.gmtime(time.time() - start_time)),
                            reward_sum, player.eps_len, state_out_loss_mean, hit_ratio))
                else:
                    pass
            else:
                log['{}_log'.format(args.env)].info(
                    "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}".
                    format(
                        time.strftime("%Hh %Mm %Ss",
                                      time.gmtime(time.time() - start_time)),
                        reward_sum, player.eps_len, reward_mean))

            if args.save_max and reward_sum >= max_score:
                max_score = reward_sum
                
                # add state_to_save message
                log['{}_log'.format(args.env)].info(
                "Time {0}  state_to_save".
                format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)) ))



                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        state_to_save = player.model.state_dict()
                        torch.save(state_to_save, '{0}{1}.dat'.format(args.save_model_dir, args.env))
                else:
                    state_to_save = player.model.state_dict()
                    torch.save(state_to_save, '{0}{1}.dat'.format(args.save_model_dir, args.env))
            
            # add save last model dict
            if args.save_last:
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        state_to_save = player.model.state_dict()
                        torch.save(state_to_save, '{0}{1}_last.dat'.format(args.save_model_dir, args.env))
                else:
                    state_to_save = player.model.state_dict()
                    torch.save(state_to_save, '{0}{1}_last.dat'.format(args.save_model_dir, args.env))

            # add save last optimizer dict
            if args.save_last:
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        state_to_save = optimizer.state_dict()
                        torch.save(state_to_save, '{0}{1}_last_opt.dat'.format(args.save_model_dir, args.env))
                else:
                    state_to_save = optimizer.state_dict()
                    torch.save(state_to_save, '{0}{1}_last_opt.dat'.format(args.save_model_dir, args.env))
            
            reward_sum = 0
            state_out_loss_sum = 0 # add
            state_out_hit = 0 # add
            player.eps_len = 0
            state = player.env.reset()
            time.sleep(60)
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
