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
        Add args.save last
        Add args.entropy, args.value
        Add args.last_load
        Add CONV6_Net
        Add second environment and its worker
        Add load two basis models
        Add CONV_Choice1_Net
"""


from __future__ import print_function, division
import os
os.environ["OMP_NUM_THREADS"] = "1" # numpy...をimportする前に使えるスレッド数を制限
import argparse
import torch
import torch.multiprocessing as mp
from environment import create_env
from model import *  # change to import any models
from train import train
from test import test
from shared_optim import SharedRMSprop, SharedAdam
import time


parser = argparse.ArgumentParser(description='A3C')
parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate (default: 0.0001)')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='G',
    help='discount factor for rewards (default: 0.99)')
parser.add_argument(
    '--tau',
    type=float,
    default=1.00,
    metavar='T',
    help='parameter for GAE (default: 1.00)')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--workers',
    type=int,
    default=6,
    metavar='W',
    help='how many training processes to use (default: 6)')
parser.add_argument(
    '--workers2',
    type=int,
    default=0,
    metavar='W2',
    help='how many training processes to use for second environment(default: 0 off)')
parser.add_argument(
    '--num-steps',
    type=int,
    default=20,
    metavar='NS',
    help='number of forward steps in A3C (default: 300)')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=4000,
    metavar='M',
    help='maximum length of an episode (default: 4000)')
parser.add_argument(
    '--env',
    default='BipedalWalkerHardcore-v2',
    metavar='ENV',
    help='environment to train on (default: BipedalWalker-v2)')
parser.add_argument(
    '--env2',
    default='BipedalWalkerHardcoreStump1-v0',
    metavar='ENV2',
    help='second environment to train on (default: BipedalWalkerHardcoreStump1-v0)')
parser.add_argument(
    '--shared-optimizer',
    default=True,
    metavar='SO',
    help='use an optimizer without shared statistics.')
parser.add_argument(
    '--load',
    default=False,
    metavar='L',
    help='load a trained model')
parser.add_argument(
    '--load-last',
    default=False,
    metavar='LL',
    help='load model dict and optimizer dicts when shared_optimizer.')
parser.add_argument(
    '--save-max',
    default=True,
    metavar='SM',
    help='Save model on every test run high score matched or bested')
parser.add_argument(
    '--save-last',
    default=True,
    metavar='SL',
    help='Save last model on every test')
parser.add_argument(
    '--optimizer',
    default='Adam',
    metavar='OPT',
    help='shares optimizer choice of Adam or RMSprop')
parser.add_argument(
    '--load-model-dir',
    default='trained_models/',
    metavar='LMD',
    help='folder to load trained models from')
parser.add_argument(
    '--save-model-dir',
    default='trained_models/',
    metavar='SMD',
    help='folder to save trained models')
parser.add_argument(
    '--log-dir',
    default='logs/',
    metavar='LG',
    help='folder to save logs')
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
    help='Choose number of observations to stack')
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
    help='number of discrete model which model CONV_Choice handles')  # GRASS=0, STUMP=1, STAIRS=2, PIT=3
parser.add_argument(
    '--use-discrete-model',
    default=True,
    metavar='AM2',
    help='use discreate model for train')
parser.add_argument(
    '--gpu-ids',
    type=int,
    default=-1,
    nargs='+',
    help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument(
    '--amsgrad',
    default=True,
    metavar='AM',
    help='Adam optimizer amsgrad parameter')
parser.add_argument(
    '--entropy',
    type=float,
    default=0.01,
    metavar='EN',
    help='entropy rate (default: 0.01)')
parser.add_argument(
    '--value',
    type=float,
    default=0.5,
    metavar='VA',
    help='value rate in Loss function (default: 0.5)')


# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior

if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method('spawn')
    env = create_env(args.env, args)
    
    if args.model == 'CONV_Choice1':
        shared_model = CONV_Choice1_Net(args.stack_frames, env.action_space, args.discrete_number, env.observation_space.shape[0]) # change
    
    #--- set and load basis_models ---
    if args.basis_model1 == 'CONV6':
        shared_bm1_model = CONV6_Net(args.stack_frames, env.action_space, env.observation_space.shape[0]) # change
        saved_state = torch.load('{0}{1}'.format(
            args.load_model_dir, args.basis_model1_file), map_location=lambda storage, loc: storage)
        shared_bm1_model.load_state_dict(saved_state)
        shared_bm1_model.share_memory()
        print ('set basis-model1 and load ', args.basis_model1_file )
    
    if args.basis_model2 == 'CONV6':
        shared_bm2_model = CONV6_Net(args.stack_frames, env.action_space, env.observation_space.shape[0]) # change
        saved_state = torch.load('{0}{1}'.format(
            args.load_model_dir, args.basis_model2_file), map_location=lambda storage, loc: storage)
        shared_bm2_model.load_state_dict(saved_state)
        shared_bm2_model.share_memory()
        print ('set basis-model2 and load ', args.basis_model2_file )
    #--- end of set and load basis_models ---
    
    # choice one of args.load or args.load_last
    if args.load:
        saved_state = torch.load('{0}{1}.dat'.format(
            args.load_model_dir, args.env), map_location=lambda storage, loc: storage)
        shared_model.load_state_dict(saved_state)
    elif args.load_last:
        saved_state = torch.load('{0}{1}_last.dat'.format(
            args.load_model_dir, args.env), map_location=lambda storage, loc: storage)
        shared_model.load_state_dict(saved_state)
    
    shared_model.share_memory()
    
    
    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        
        if args.load_last:
            saved_state = torch.load('{0}{1}_last_opt.dat'.format(
                args.load_model_dir, args.env), map_location=lambda storage, loc: storage)
            optimizer.load_state_dict(saved_state)
        
        optimizer.share_memory()
    else:
        optimizer = None

    processes = []

    # testを起動　これが定期的に結果を表示してMAX SAVEする
    p = mp.Process(target=test, args=(args, shared_model, optimizer, shared_bm1_model, shared_bm2_model)) # change
    p.start()
    processes.append(p)
    time.sleep(0.1)

    # trainをworkers個 起動
    for rank in range(0, args.workers + args.workers2):
        p = mp.Process(target=train, args=(
            rank, args, shared_model, optimizer, shared_bm1_model, shared_bm2_model))
        p.start()
        processes.append(p)
        time.sleep(0.1)
    for p in processes:
        time.sleep(0.1)
        p.join()
