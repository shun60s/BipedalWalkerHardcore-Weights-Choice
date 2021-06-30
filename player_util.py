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
        Add CONV6_Net
        Add load two basis models
        Add CONV_Choice1_Net

"""

from __future__ import division
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils import normal  # , pi


class Agent(object):
    def __init__(self, model, env, args, state, bm1_model=None, bm2_model=None):
        self.model = model
        self.bm1_model = bm1_model
        self.bm2_model = bm2_model
        self.env = env
        self.state = state
        self.state1 = state
        self.state2 = state
        
        self.hx = None
        self.cx = None
        
        self.bm1_hx1 = None   # add
        self.bm1_cx1 = None   # add
        self.bm1_hx2 = None   # add
        self.bm1_cx2 = None   # add
        self.bm2_hx1 = None   # add
        self.bm2_cx1 = None   # add
        self.bm2_hx2 = None   # add
        self.bm2_cx2 = None   # add
        
        self.loss_state_out = [] # add
        #self.loss_func = torch.nn.CrossEntropyLoss() # add
        self.loss_func = torch.nn.BCELoss() # add
        
        self.hit = None
        
        
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = -1

    def action_train(self):
        # This is no train about two basis models
        with torch.no_grad():  # without save parameters, only forward
            #
            self.state1= self.state
            self.state2= self.state
            if self.args.basis_model1 == 'CONV6':
                self.state1 = self.state1.unsqueeze(0)  #### reshape(mini-batch,...) for picture input
                value1, mu1, sigma1, (self.bm1_hx1, self.bm1_cx1, self.bm1_hx2, self.bm1_cx2) = self.bm1_model(
                    (Variable(self.state1), (self.bm1_hx1, self.bm1_cx1, self.bm1_hx2, self.bm1_cx2)))
            
            if self.args.basis_model2 == 'CONV6':
                self.state2 = self.state2.unsqueeze(0)  #### reshape(mini-batch,...) for picture input
                value2, mu2, sigma2, (self.bm2_hx1, self.bm2_cx1, self.bm2_hx2, self.bm2_cx2) = self.bm2_model(
                    (Variable(self.state2), (self.bm2_hx1, self.bm2_cx1, self.bm2_hx2, self.bm2_cx2)))
        
        
        if self.args.model == 'CONV_Choice1':
            self.state = self.state.unsqueeze(0)  #### reshape(mini-batch,...) for picture input
            value, logits, (self.hx, self.cx) = self.model(
                (Variable(self.state), (self.hx, self.cx)))
        
        
        if self.args.use_discrete_model:
            if self.env.observation_space.shape[0] == 28 and self.args.discrete_number == 4:
                (h4,j01,g0,j23,g2,a2,so4)=torch.split(self.state, [4,4,1,4,1,10,4],dim=2)
                so4b=so4.view(4,4)[-1]   # get last only
                so4bt= so4b.reshape(1,self.args.discrete_number) # 答え（定数）なのでgrad 不要か？
                so4btc= torch.clamp(so4bt, min=0.0, max=1.0) # 注意：normalizationされているので値が１になっていない。
                
                # softmax is sum of probabilities becomes 1
                probs = F.softmax(logits, dim=1)
                
                # compute loss
                loss = self.loss_func( probs, so4btc)
                self.loss_state_out.append(loss)
                
                if so4b[1] >0:
                    mu=mu1
                else:
                    mu=mu2
                
                #Clamp all elements in input into the range [ min, max ] and return a resulting tensor:
                mu = torch.clamp(mu.data, -1.0, 1.0)
                # When eval, action is directory to use mu,  vs when train prob...
                action = mu.data
                act = Variable(action)  #  set act as Variable due to mu.data, .data is requires_grad=False
                
            else:
                #
                # softmax is sum of probabilities becomes 1
                probs = F.softmax(logits, dim=1)
                
                m= torch.distributions.Categorical(probs)
                choice= m.sample()[0] # 確率分布に従って1個取り出す。最大確率のものだと、固定してしまい、振ってみる学習ができない。
                choice0= choice.cpu().numpy()  # 仮想的に choiceと云うactionを行ったと見える。basis_model1 又はbasis_model2を通じてrewardに反映される。
                
                
                if choice0 == 0:
                    mu=mu1
                else:
                    mu=mu2
                ###print ('choice', probs.data, choice0)
                
                #Clamp all elements in input into the range [ min, max ] and return a resulting tensor:
                mu = torch.clamp(mu.data, -1.0, 1.0)
                # When eval, action is directory to use mu,  vs when train prob...
                action = mu.data
                act = Variable(action)  #  set act as Variable due to mu.data, .data is requires_grad=False
                #
                log_prob =  m.log_prob(choice)  #(prob + 1e-6).log()
                self.log_probs.append(log_prob)
                
                
        else: # continous model
            # value: critic
            # mu: softsign liner out　action
            # sigma: liner out action
            # hx lstm
            # cx lstm
            
            mu = torch.clamp(mu, -1.0, 1.0)
            # SoftPlus is a smooth approximation to the ReLU function and can be used to constrain the output of a machine 
            # to always be positive.
            # 1/beta * log(1+exp(beta*x))
            sigma = F.softplus(sigma) + 1e-5
            
            # Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 
            # (also called the standard normal distribution).
            # The shape of the tensor is defined by the variable argument size
            eps = torch.randn(mu.size())
            
            
            pi = np.array([math.pi])
            pi = torch.from_numpy(pi).float()
            if self.gpu_id >= 0:
                with torch.cuda.device(self.gpu_id):
                    eps = Variable(eps).cuda()
                    pi = Variable(pi).cuda()
            else:
                eps = Variable(eps)
                pi = Variable(pi)
            
            action = (mu + sigma.sqrt() * eps).data
            act = Variable(action)
            
            # prob
            #     normal
            #      a = (-1 * (x - mu).pow(2) / (2 * sigma)).exp()
            #      b = 1 / (2 * sigma * pi.expand_as(sigma)).sqrt()
            #     return a * b
            prob = normal(act, mu, sigma, self.gpu_id, gpu=self.gpu_id >= 0)
            action = torch.clamp(action, -1.0, 1.0)
            
            # entropy 
            # Expand this tensor to the same size as other . 
            # self.expand_as(other) is equivalent to self.expand(other.size()) 
            entropy = 0.5 * ((sigma * 2 * pi.expand_as(sigma)).log() + 1)
            
            self.entropies.append(entropy)
            log_prob = (prob + 1e-6).log()
            self.log_probs.append(log_prob)
        
        
        state, reward, self.done, self.info = self.env.step(
            action.cpu().numpy()[0])
        reward = max(min(float(reward), 1.0), -1.0)
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1
        self.done = self.done or self.eps_len >= self.args.max_episode_length
        self.values.append(value)
        self.rewards.append(reward)
        return self

    def action_test(self):
        with torch.no_grad():  # without save parameters, only forward
            if self.done:
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        # use for CONV_Choice1_Net
                        self.cx = Variable(torch.zeros(
                            1, 128).cuda())
                        self.hx = Variable(torch.zeros(
                            1, 128).cuda())
                        
                        if self.args.basis_model1 == 'CONV6':
                            self.bm1_cx1 = Variable(torch.zeros(
                                1,128).cuda())
                            self.bm1_hx1 = Variable(torch.zeros(
                                1,128).cuda())
                            self.bm1_cx2 = Variable(torch.zeros(
                                1,128).cuda())
                            self.bm1_hx2 = Variable(torch.zeros(
                                1,128).cuda())
                        if self.args.basis_model2 == 'CONV6':
                            self.bm2_cx1 = Variable(torch.zeros(
                                1,128).cuda())
                            self.bm2_hx1 = Variable(torch.zeros(
                                1,128).cuda())
                            self.bm2_cx2 = Variable(torch.zeros(
                                1,128).cuda())
                            self.bm2_hx2 = Variable(torch.zeros(
                                1,128).cuda())
                else:
                    # use for CONV_Choice1_Net
                    self.cx = Variable(torch.zeros(1, 128))
                    self.hx = Variable(torch.zeros(1, 128))
                    
                    if self.args.basis_model1 == 'CONV6':
                        self.bm1_cx1 = Variable(torch.zeros(1, 128))
                        self.bm1_hx1 = Variable(torch.zeros(1, 128))
                        self.bm1_cx2 = Variable(torch.zeros(1, 128))
                        self.bm1_hx2 = Variable(torch.zeros(1, 128))
                    if self.args.basis_model2 == 'CONV6':
                        self.bm2_cx1 = Variable(torch.zeros(1, 128))
                        self.bm2_hx1 = Variable(torch.zeros(1, 128))
                        self.bm2_cx2 = Variable(torch.zeros(1, 128))
                        self.bm2_hx2 = Variable(torch.zeros(1, 128))
            else:
                # use for CONV_Choice1_Net
                self.cx = Variable(self.cx.data)
                self.hx = Variable(self.hx.data)
                if self.args.basis_model1 == 'CONV6':
                    self.bm1_cx1 = Variable(self.bm1_cx1.data)
                    self.bm1_hx1 = Variable(self.bm1_hx1.data)
                    self.bm1_cx2 = Variable(self.bm1_cx2.data)
                    self.bm1_hx2 = Variable(self.bm1_hx2.data)
                if self.args.basis_model2 == 'CONV6':
                    self.bm2_cx1 = Variable(self.bm2_cx1.data)
                    self.bm2_hx1 = Variable(self.bm2_hx1.data)
                    self.bm2_cx2 = Variable(self.bm2_cx2.data)
                    self.bm2_hx2 = Variable(self.bm2_hx2.data)
            #
            self.state1= self.state
            self.state2= self.state
            if self.args.basis_model1 == 'CONV6':
                self.state1 = self.state1.unsqueeze(0)  #### reshape(mini-batch,...) for picture input
                value1, mu1, sigma1, (self.bm1_hx1, self.bm1_cx1, self.bm1_hx2, self.bm1_cx2) = self.bm1_model(
                    (Variable(self.state1), (self.bm1_hx1, self.bm1_cx1, self.bm1_hx2, self.bm1_cx2)))
            
            if self.args.basis_model2 == 'CONV6':
                self.state2 = self.state2.unsqueeze(0)  #### reshape(mini-batch,...) for picture input
                value2, mu2, sigma2, (self.bm2_hx1, self.bm2_cx1, self.bm2_hx2, self.bm2_cx2) = self.bm2_model(
                    (Variable(self.state2), (self.bm2_hx1, self.bm2_cx1, self.bm2_hx2, self.bm2_cx2)))
            
            if self.args.model == 'CONV_Choice1':
                self.state = self.state.unsqueeze(0)  #### reshape(mini-batch,...) for picture input
                value, logits, (self.hx, self.cx) = self.model(
                    (Variable(self.state), (self.hx, self.cx)))
            
            
            if self.args.use_discrete_model:
                if self.env.observation_space.shape[0] == 28 and self.args.discrete_number == 4:
                    (h4,j01,g0,j23,g2,a2,so4)=torch.split(self.state, [4,4,1,4,1,10,4],dim=2)
                    so4b=so4.view(4,4)[-1]   # get last only
                    so4bt= so4b.reshape(1,self.args.discrete_number) # 答え（定数）なのでgrad 不要か？
                    so4btc= torch.clamp(so4bt, min=0.0, max=1.0) # 注意：normalizationされているので値が１になっていない。
                    
                    # softmax is sum of probabilities becomes 1
                    probs = F.softmax(logits, dim=1)
                    
                    # compute loss
                    loss = self.loss_func( probs, so4btc)
                    self.loss_state_out.append(loss)
                    
                    # check minus loss
                    if loss.detach().numpy() < 0.0:
                        print ( probs, so4bt)
                    
                    if 1:  # same as train condition
                        m= torch.distributions.Categorical(probs)
                        choice= m.sample().cpu().numpy()[0] # 確率分布に従って1個取り出す
                    else:
                        # get maximum probability one　確率の一番大きなものを選択する
                        #prob =  torch.max(probs)  
                        choice= torch.argmax(probs)
                        choice= choice.cpu().numpy()
                    
                    # 選択したものが当たりの場合
                    if choice == torch.argmax(so4b):
                        self.hit = True
                    else:
                        self.hit = False
                    
                    if choice == 1:  # stump
                        mu=mu2
                    else:
                        mu=mu1
                    
                    
                else:  # --- old ---
                    # choice which mu
                    # softmax is sum of probabilities becomes 1
                    probs = F.softmax(logits, dim=1).data # .data is requires_grad=False due to just forward
                    
                    if 1:  # same as train condition
                        m= torch.distributions.Categorical(probs)
                        choice= m.sample().cpu().numpy()[0] # 確率分布に従って1個取り出す
                    else:
                        # get maximum probability one　確率の一番大きなものを選択する
                        #prob =  torch.max(probs)  
                        choice= torch.argmax(probs)
                        choice= choice.cpu().numpy()
                    
                    if choice == 0:
                        mu=mu1
                    else:
                        mu=mu2
                    
                    if 0: # show choice
                        if choice == 0:
                            print ('choice mu1 at action_test()', probs)
                        else:
                            print ('choice mu2 at action_test()', probs)
                    
            else: # continous model
                # when  use basis_model1
                mu= mu1
            
            
        #Clamp all elements in input into the range [ min, max ] and return a resulting tensor:
        mu = torch.clamp(mu.data, -1.0, 1.0)
        
        # When test, action is directory to use mu,  vs when train prob...
        #
        action = mu.cpu().numpy()[0]
        
        state, self.reward, self.done, self.info = self.env.step(action)
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1
        self.done = self.done or self.eps_len >= self.args.max_episode_length
        return self

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        
        self.loss_state_out=[] # add
        
        return self
