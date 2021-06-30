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
       Add CONV_Choice1_Net
       Add observation_num for state_out

"""


from __future__ import division
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from utils import norm_col_init, weights_init, weights_init_mlp


class A3C_CONV(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(A3C_CONV, self).__init__()
        # nn.Conv1d(in_channels, out_channels, kernel_size, strid
        self.conv1 = nn.Conv1d(num_inputs, 32, 3, stride=1, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv1d(32, 32, 3, stride=1, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.conv3 = nn.Conv1d(32, 64, 2, stride=1, padding=1)
        self.lrelu3 = nn.LeakyReLU(0.1)
        self.conv4 = nn.Conv1d(64, 64, 1, stride=1)
        self.lrelu4 = nn.LeakyReLU(0.1)


        """
              nn.Conv1d(in_channels, out_channels, kernel_size, strid
        torch.nn.LSTMCell(input_size: int, hidden_size(=output_size): int, bias: bool = True)

        """
        self.lstm = nn.LSTMCell(1600, 128)
        num_outputs = action_space.shape[0]
        self.critic_linear = nn.Linear(128, 1)
        self.actor_linear = nn.Linear(128, num_outputs)
        self.actor_linear2 = nn.Linear(128, num_outputs)

        self.apply(weights_init)
        lrelu_gain = nn.init.calculate_gain('leaky_relu')
        self.conv1.weight.data.mul_(lrelu_gain)
        self.conv2.weight.data.mul_(lrelu_gain)
        self.conv3.weight.data.mul_(lrelu_gain)
        self.conv4.weight.data.mul_(lrelu_gain)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.actor_linear2.weight.data = norm_col_init(
            self.actor_linear2.weight.data, 0.01)
        self.actor_linear2.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        x, (hx, cx) = inputs

        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.conv2(x))
        x = self.lrelu3(self.conv3(x))
        x = self.lrelu4(self.conv4(x))

        x = x.view(x.size(0), -1)  # Auto Size Adjust like reshape
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), F.softsign(self.actor_linear(x)), self.actor_linear2(x), (hx, cx)



class A3C_MLP(torch.nn.Module):
    def __init__(self, num_inputs, action_space, n_frames):
        super(A3C_MLP, self).__init__()
        # Applies a linear transformation to the incoming data: y=xAT+b
        self.fc1 = nn.Linear(num_inputs, 256)
        self.lrelu1 = nn.LeakyReLU(0.1) # negative_slope 0.1
        self.fc2 = nn.Linear(256, 256)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.fc3 = nn.Linear(256, 128)
        self.lrelu3 = nn.LeakyReLU(0.1)
        self.fc4 = nn.Linear(128, 128)
        self.lrelu4 = nn.LeakyReLU(0.1)
        
        self.m1 = n_frames * 128  # n_frames : stack_frame of gym
        self.lstm = nn.LSTMCell(self.m1, 128)
        
        # critic is output only onr value
        # actior uses two linear and output number is action_space.shape[0]
        num_outputs = action_space.shape[0]
        self.critic_linear = nn.Linear(128, 1)
        self.actor_linear = nn.Linear(128, num_outputs)
        self.actor_linear2 = nn.Linear(128, num_outputs)

        self.apply(weights_init_mlp)
        lrelu = nn.init.calculate_gain('leaky_relu')
        self.fc1.weight.data.mul_(lrelu)
        self.fc2.weight.data.mul_(lrelu)
        self.fc3.weight.data.mul_(lrelu)
        self.fc4.weight.data.mul_(lrelu)

        # init weights
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.actor_linear2.weight.data = norm_col_init(
            self.actor_linear2.weight.data, 0.01)
        self.actor_linear2.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)
        
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()  # Sets the module in training mode.

    def forward(self, inputs):
        x, (hx, cx) = inputs

        x = self.lrelu1(self.fc1(x))
        x = self.lrelu2(self.fc2(x))
        x = self.lrelu3(self.fc3(x))
        x = self.lrelu4(self.fc4(x))

        x = x.view(1, self.m1)           # input [n_frames, 128]
        hx, cx = self.lstm(x, (hx, cx))  # output 128
        x = hx

        return self.critic_linear(x), F.softsign(self.actor_linear(x)), self.actor_linear2(x), (hx, cx)


#-------------------------------------------------------------------------
# CONV6    continuous model
#          CONV4_Netに
#          Lidar CONV出力にLSTM層を追加
#
class CONV6_Net(torch.nn.Module):
    def __init__(self, num_inputs, action_space, observation_num=24): # add observation_num
        super(CONV6_Net, self).__init__()
        # nn.Conv1d(in_channels, out_channels, kernel_size, strid
        # 10 lindar
        self.dim2= 4 # num_inputs, stack_frame is 4 fixed
        self.conv1 = nn.Conv1d(self.dim2, 32, 3, stride=1, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv1d(32, 32, 3, stride=1, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.conv3 = nn.Conv1d(32, 64, 2, stride=1, padding=1)   # 1,64,11= 704
        self.lrelu3 = nn.LeakyReLU(0.1)
        ###self.conv4 = nn.Conv1d(64, 64, 1, stride=1)   # # 1,64,11= 704
        self.conv4 = nn.Conv1d(64, 32, 1, stride=1)   # # 1,32,11= 352
        self.lrelu4 = nn.LeakyReLU(0.1)
        
        #--- separate network for move etc
        #   4->64->64
        #   4->64->64
        #   4->64->64
        #   2->32->32
        #
        #   256->128
        #
        self.h4fc1 = nn.Linear(7, 64)
        self.lrelu1 = nn.LeakyReLU(0.1) # negative_slope 0.1
        self.h4fc2 = nn.Linear(64, 64)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.j01fc1 = nn.Linear(6, 64)
        self.lrelu3 = nn.LeakyReLU(0.1)
        self.j01fc2 = nn.Linear(64, 64)
        self.lrelu4 = nn.LeakyReLU(0.1)
        self.j23fc1 = nn.Linear(6, 64)
        self.lrelu5 = nn.LeakyReLU(0.1)
        self.j23fc2 = nn.Linear(64, 64)
        self.lrelu6 = nn.LeakyReLU(0.1)
        self.g02fc1 = nn.Linear(2,32)
        self.lrelu7 = nn.LeakyReLU(0.1)
        self.g02fc2 = nn.Linear(32, 32)
        self.lrelu8 = nn.LeakyReLU(0.1)
        self.fc3 = nn.Linear(224, 128)  # 224= 64+64+64+32
        self.lrelu9 = nn.LeakyReLU(0.1)
        
        
        #---
        self.dim3= 128 + 128 # mov output + conv output lstm
        self.lstm1 = nn.LSTMCell(self.dim3, 128)
        
        #---
        # Add LSTM-layer to Liadr conv output 
        self.dim4=  352 ## 704
        self.lstm2 = nn.LSTMCell(self.dim4, 128)

        num_outputs = action_space.shape[0]
        self.critic_linear = nn.Linear(128, 1)
        self.actor_linear = nn.Linear(128, num_outputs)
        self.actor_linear2 = nn.Linear(128, num_outputs)
        
        self.apply(weights_init)
        lrelu_gain = nn.init.calculate_gain('leaky_relu')
        self.conv1.weight.data.mul_(lrelu_gain)
        self.conv2.weight.data.mul_(lrelu_gain)
        self.conv3.weight.data.mul_(lrelu_gain)
        self.conv4.weight.data.mul_(lrelu_gain)
        
        
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.actor_linear2.weight.data = norm_col_init(
            self.actor_linear2.weight.data, 0.01)
        self.actor_linear2.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.apply(weights_init_mlp)
        lrelu = nn.init.calculate_gain('leaky_relu')
        self.h4fc1.weight.data.mul_(lrelu)
        self.h4fc2.weight.data.mul_(lrelu)
        self.j01fc1.weight.data.mul_(lrelu)
        self.j01fc2.weight.data.mul_(lrelu)
        self.j23fc1.weight.data.mul_(lrelu)
        self.j23fc2.weight.data.mul_(lrelu)
        self.g02fc1.weight.data.mul_(lrelu)
        self.g02fc2.weight.data.mul_(lrelu)
        self.fc3.weight.data.mul_(lrelu)
        
        #---
        self.lstm1.bias_ih.data.fill_(0)
        self.lstm2.bias_ih.data.fill_(0)
        
        self.observation_num=observation_num # add
        
        self.train()
        
        
    def forward(self, inputs):
        x, (hx1, cx1, hx2, cx2) = inputs
        
        # change
        if self.observation_num == 28: # with state_out
            # hull(4) joint0/1(4) ground_contact(1) joint2/3(4) ground_contact(1) 
            (h4,j01,g0,j23,g2,a2,so4)=torch.split(x, [4,4,1,4,1,10,4],dim=2)
        else:
            # hull(4) joint0/1(4) ground_contact(1) joint2/3(4) ground_contact(1) 
            (h4,j01,g0,j23,g2,a2)=torch.split(x, [4,4,1,4,1,10],dim=2)
        
        
        x2 = self.lrelu1(self.conv1(a2))
        x2 = self.lrelu2(self.conv2(x2))
        x2 = self.lrelu3(self.conv3(x2))
        x2 = self.lrelu4(self.conv4(x2))
        x2 = x2.view(x2.size(0), -1)  # Auto Size Adjust like reshape
        
        
        hx2, cx2 = self.lstm2(x2, (hx2, cx2))
        x3= hx2
        
        #--------------------------------------------------------
        h4b=h4.view(4,4)[-1]   # get last only
        j01b=j01.view(4,4)[-1] # get last only
        j23b=j23.view(4,4)[-1] # get last only
        g02b=torch.cat([g0,g2],dim=2).view(4,2)[-1] # cat ground touch, get last only
        
        h4c=h4.view(4,4)[-2]   # get previous only
        j01c=j01.view(4,4)[-2] # get previous only
        j23c=j23.view(4,4)[-2] # get previous only
        
        # compute current - previous
        h4bcsub=torch.sub(h4b,h4c)
        j01bcsub=torch.sub(j01b,j01c)
        j23bcsub=torch.sub(j23b,j23c)
        
        
        h4bplus= torch.cat([h4b,h4bcsub[1:]],dim=0) # hull.angularVelocity/FPS, vel.x*(VIEWPORT_W/SCALE)/FPS, vel.y*(VIEWPORT_H/SCALE)/FPS
        j01bplus= torch.cat([j01b, j01bcsub[1:2], j01bcsub[3:] ],dim=0) # joints[0].speed / SPEED_HIP, joints[1].speed / SPEED_KNEE,
        j23bplus= torch.cat([j23b, j23bcsub[1:2], j23bcsub[3:] ],dim=0) # joints[2].speed / SPEED_HIP, joints[3].speed / SPEED_KNEE,
        
        
        xh4 = self.lrelu1(self.h4fc1(h4bplus))
        xh4 = self.lrelu2(self.h4fc2(xh4))
        xj01= self.lrelu3(self.j01fc1(j01bplus))
        xj01= self.lrelu4(self.j01fc2(xj01))
        xj23= self.lrelu5(self.j23fc1(j23bplus))
        xj23= self.lrelu6(self.j23fc2(xj23))
        xg02= self.lrelu7(self.g02fc1(g02b))
        xg02= self.lrelu8(self.g02fc2(xg02))
        
        xfc3 = torch.cat([xh4,xj01,xj23,xg02],dim=0)
        xfc3= self.lrelu9(self.fc3(xfc3))
        
        x1 = xfc3.view(1, 128)
        
        x4 = torch.cat([x1, x3],dim=1)
        #x = x.view(1, self.dim3)  
        hx1, cx1 = self.lstm1(x4, (hx1, cx1))
        x4 = hx1
        
        return self.critic_linear(x4), F.softsign(self.actor_linear(x4)), self.actor_linear2(x4), (hx1, cx1, hx2, cx2)

#-------------------------------------------------------------------------
# trial 1st   CONV-Choice1-Net  discrete model
#             選択用のモデル
#             その１  とりあえずLidarのみのCONV出力にLSTM層
#
class CONV_Choice1_Net(torch.nn.Module):
    def __init__(self, num_inputs, action_space, discrete_number, observation_num=24): # add observation_num
        super(CONV_Choice1_Net, self).__init__()
        #
        self.discrete_number= discrete_number
        # nn.Conv1d(in_channels, out_channels, kernel_size, strid
        # 10 lindar
        self.dim2= 4 # num_inputs, stack_frame is 4 fixed
        self.conv1 = nn.Conv1d(self.dim2, 32, 3, stride=1, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv1d(32, 32, 3, stride=1, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.conv3 = nn.Conv1d(32, 64, 2, stride=1, padding=1)   # 1,64,11= 704
        self.lrelu3 = nn.LeakyReLU(0.1)
        ###self.conv4 = nn.Conv1d(64, 64, 1, stride=1)   # # 1,64,11= 704
        self.conv4 = nn.Conv1d(64, 32, 1, stride=1)   # # 1,32,11= 352
        self.lrelu4 = nn.LeakyReLU(0.1)
        
        
        #---
        # Add LSTM-layer to Liadr conv output 
        self.dim4=  352 ## 704
        self.lstm2 = nn.LSTMCell(self.dim4, 128)
        
        num_outputs = action_space.shape[0]
        self.critic_linear = nn.Linear(128, 1)
        self.actor_linear = nn.Linear(128, self.discrete_number)
        
        self.apply(weights_init)
        lrelu_gain = nn.init.calculate_gain('leaky_relu')
        self.conv1.weight.data.mul_(lrelu_gain)
        self.conv2.weight.data.mul_(lrelu_gain)
        self.conv3.weight.data.mul_(lrelu_gain)
        self.conv4.weight.data.mul_(lrelu_gain)
        
        
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)
        
        #---
        self.lstm2.bias_ih.data.fill_(0)
        
        self.observation_num=observation_num # add
        
        self.train()
        
        
    def forward(self, inputs):
        x, (hx, cx) = inputs
        
        # change
        if self.observation_num == 28: # with state_out (GRASS=0, STUMP=1, STAIRS=2, PIT=3)
            # hull(4) joint0/1(4) ground_contact(1) joint2/3(4) ground_contact(1) 
            (h4,j01,g0,j23,g2,a2,so4)=torch.split(x, [4,4,1,4,1,10,4],dim=2)
            so4b=so4.view(4,4)[-1]   # get last only
        else:
            # hull(4) joint0/1(4) ground_contact(1) joint2/3(4) ground_contact(1) 
            (h4,j01,g0,j23,g2,a2)=torch.split(x, [4,4,1,4,1,10],dim=2)
        
        
        x2 = self.lrelu1(self.conv1(a2))
        x2 = self.lrelu2(self.conv2(x2))
        x2 = self.lrelu3(self.conv3(x2))
        x2 = self.lrelu4(self.conv4(x2))
        x2 = x2.view(x2.size(0), -1)  # Auto Size Adjust like reshape
        
        
        hx, cx = self.lstm2(x2, (hx, cx))
        x2= hx
        
        # change
        self.test_flag = False  # 固定出力にするテスト用フラグ。但し、discrete_numberが2の場合のみ。
        if self.observation_num == 28 and self.test_flag: # with state_out
             # 以下は、discrete_numberが2の場合
             if so4b[1] >0:  # 注意：値はnormalizationされているので１ではない。
                 return self.critic_linear(x2),  torch.tensor([[0.0,100.0]]), (hx, cx)  # choice mu2 
             else:
                 return self.critic_linear(x2),  torch.tensor([[100.0,0.0]]), (hx, cx)  # choice mu1
        else:
            return self.critic_linear(x2), self.actor_linear(x2), (hx, cx)   # removed F.softsign 
