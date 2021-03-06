# -*- coding: utf-8 -*-

# The following code is modified from openai/gym (https://github.com/openai/gym) under the MIT License.

#    state out　障害の状況を出力するもの。
#    stateに4要素(GRASS=0, STUMP=1, STAIRS=2, PIT=3)を追加　計24+4＝28
#    STUMP=1 切り株　有効  counter=2 の発生確率を大きくしたもの。
#    STAIRS=2, 階段　　有効
#    PIT=3　落とし穴　3,4 有効
#  

import sys
import math

import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import colorize, seeding, EzPickle



# This is simple 4-joints walker robot environment.
#
# There are two versions:
#
# - Normal, with slightly uneven terrain.
#
# - Hardcore with ladders, stumps, pitfalls.
#
# REWARD リワード　前に進むと与えられる。　遠くに行くと300ポイント加算される
#　　　　　　　　　ロボットが倒れると　100ポイント減点される
#
# Reward is given for moving forward, total 300+ points up to the far end. If the robot falls,
# it gets -100. Applying motor torque costs a small amount of points, more optimal agent
# will get better score.
#
# Heuristic 　ヒューリスティックな手法がが提供されている　デモ用
# Heuristic is provided for testing, it's also useful to get demonstrations to
# learn from. To run heuristic:
#
# python gym/envs/box2d/bipedal_walker.py
##
#
# Lidar ライダー（奥行き距離検出）は　自動運転ではないので　とりあえず　不要にしてみるか？
#
# State consists of hull angle speed, angular velocity, horizontal speed, vertical speed,
# position of joints and joints angular speed, legs contact with ground, and 10 lidar
# rangefinder measurements to help to deal with the hardcore version. There's no coordinates
# in the state vector. Lidar is less useful in normal version, but it works.
#
#
# ゲームでは　1600ステップ（32秒per 50step/sec)で300ポイント獲得する必要がある
# To solve the game you need to get 300 points in 1600 time steps.
#
# hardcore　落とし穴がある方は、　2000ステップで300ポイント
# To solve hardcore version you need 300 points in 2000 time steps.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

FPS    = 50     # 1秒間に50フレーム
SCALE  = 30.0   # ？　affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 80  # モータートルク
SPEED_HIP     = 4   #　腰下の速度
SPEED_KNEE    = 6   #　ひざ下の速度
LIDAR_RANGE   = 160/SCALE  #　ライダーの到達範囲

INITIAL_RANDOM = 5

# 胴体　
HULL_POLY =[
    (-30,+9), (+6,+9), (+34,+1),
    (+34,-8), (-30,-8)
    ]
LEG_DOWN = -8/SCALE
LEG_W, LEG_H = 8/SCALE, 34/SCALE


VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP   = 14/SCALE
TERRAIN_LENGTH = 200     # in steps
TERRAIN_HEIGHT = VIEWPORT_H/SCALE/4
TERRAIN_GRASS    = 10    # low long are grass spots, in steps
TERRAIN_STARTPAD = 20    # in steps
FRICTION = 2.5

# 以下　胴体と足の絵の描き方か？
HULL_FD = fixtureDef(
                shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in HULL_POLY ]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0020,   #Class: b2.Fixture   categoryBits: (number) The collision category bits
                maskBits=0x001,  # collide only with ground
                restitution=0.0) # 0.99 bouncy

LEG_FD = fixtureDef(
                    shape=polygonShape(box=(LEG_W/2, LEG_H/2)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)

LOWER_FD = fixtureDef(
                    shape=polygonShape(box=(0.8*LEG_W/2, LEG_H/2)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)

# 
class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        if self.env.hull==contact.fixtureA.body or self.env.hull==contact.fixtureB.body:
            self.env.game_over = True #　BODYが　したので　ゲームオーバー
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = True #　足が地面についた
    def EndContact(self, contact):
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = False #

class BipedalWalkerEdit2(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    hardcore = False
    


    def __init__(self):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None
        
        self.world = Box2D.b2World()  # 2Dワールド
        self.terrain = None  # 地形
        self.hull = None     # 胴体

        self.prev_shaping = None

        self.fd_polygon = fixtureDef(
                        shape = polygonShape(vertices=
                        [(0, 0),
                         (1, 0),
                         (1, -1),
                         (0, -1)]),
                        friction = FRICTION)

        self.fd_edge = fixtureDef(
                    shape = edgeShape(vertices=
                    [(0, 0),
                     (1, 1)]),
                    friction = FRICTION,
                    categoryBits=0x0001,
                )
        #-- add for state_out. Some of following init also call in def reset
        self.add_state_out= True
        self.dump_out =  False # True
        self.c0=0  # this is a just counter.
        self.state0 = ''
        self.state_out=[ 0, 0, 0, 0] # add three elements, (GRASS=0, STUMP=1, STAIRS=2, PIT=3)
        self.x_see_length= np.zeros(10) # add
        self.obstacle_x_end= np.zeros(4) # x point of current obstacle end
        self.sp1=3  # 先行して障害を見つける位置　Lidarの10個の位置のどれか。 7だと小さい切り株がひかからない.5も上手く行かない.
        #------------------------------------
        self.reset()
        
        
        
        # Add three element
        if self.add_state_out:
            high = np.array([np.inf] * 28) # 28
        else:
            high = np.array([np.inf] * 24) # 24
        
        # アクション空間 のの変域ー１から１
        self.action_space = spaces.Box(np.array([-1, -1, -1, -1]), np.array([1, 1, 1, 1]), dtype=np.float32)
        # 観測空間の変域　マイナス無限大から無限大
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.terrain: return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []
        self.world.DestroyBody(self.hull)
        self.hull = None
        for leg in self.legs:
            self.world.DestroyBody(leg)
        self.legs = []
        self.joints = []
        self.terrain_state = [] # add
        

    def _generate_terrain(self, hardcore): # 最初のself.reset()から呼ばれる
        # _generate_terrain　動作開始前に全体の障害物の作成するみたい
        GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)  # GRASS=0, STUMP=1, STAIRS=2, PIT=3, _STATE_=4
        
        state    = GRASS  # start GRASS
        velocity = 0.0
        y        = TERRAIN_HEIGHT
        counter  = TERRAIN_STARTPAD
        oneshot  = False # onseshot = False
        self.terrain   = []
        self.terrain_x = []
        self.terrain_y = []
        self.terrain_state = [] # add  STUMP, STAIRS, PIT, GROUND　のどれかが入る　GRASSは入らないよ！
        
        
        
        #--- for loop ---
        for i in range(TERRAIN_LENGTH):  # TERRAIN_LENGTH = 200
            x = i*TERRAIN_STEP     # i * TERRAIN_STEP=14/30
            ###print (' terrain_length loop i, x, y', i,x,y)
            self.terrain_x.append(x)

            if state==GRASS and not oneshot:
                velocity = 0.8*velocity + 0.01*np.sign(TERRAIN_HEIGHT - y)  # TERRAIN_STARTPAD = 20    # in steps
                if i > TERRAIN_STARTPAD: velocity += self.np_random.uniform(-1, 1)/SCALE   #1
                y += velocity

            elif state==PIT and oneshot:  # pit 落とし穴
                #-----------------------------------------------
                counter = self.np_random.randint(3, 5)  # 3,4  original   # TERRAIN_STEP   = 14/SCALE(=30)
                poly = [
                    (x,              y),
                    (x+TERRAIN_STEP, y),
                    (x+TERRAIN_STEP, y-4*TERRAIN_STEP),
                    (x,              y-4*TERRAIN_STEP),
                    ]
                self.fd_polygon.shape.vertices=poly
                t = self.world.CreateStaticBody(
                    fixtures = self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)
                

                self.fd_polygon.shape.vertices=[(p[0]+TERRAIN_STEP*counter,p[1]) for p in poly]
                t = self.world.CreateStaticBody(
                    fixtures = self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)
                
                # add
                self.terrain_state.append((PIT, x ,x+TERRAIN_STEP*counter))
                
                counter += 2
                original_y = y
                


            elif state==PIT and not oneshot:
                y = original_y
                if counter > 1:
                    y -= 4*TERRAIN_STEP

            elif state==STUMP and oneshot:  # 切り株　凸
                counter = self.np_random.randint(1, 3) # 1, 2
                poly = [
                    (x,                      y),
                    (x+counter*TERRAIN_STEP, y),
                    (x+counter*TERRAIN_STEP, y+counter*TERRAIN_STEP),
                    (x,                      y+counter*TERRAIN_STEP),
                    ]
                self.fd_polygon.shape.vertices=poly
                t = self.world.CreateStaticBody(
                    fixtures = self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)
                
                # add
                self.terrain_state.append((STUMP,x,x+counter*TERRAIN_STEP))

            elif state==STAIRS and oneshot:
                stair_height = +1 if self.np_random.rand() > 0.5 else -1
                stair_width = self.np_random.randint(4, 5)
                stair_steps = self.np_random.randint(3, 5)
                original_y = y
                for s in range(stair_steps):
                    poly = [
                        (x+(    s*stair_width)*TERRAIN_STEP, y+(   s*stair_height)*TERRAIN_STEP),
                        (x+((1+s)*stair_width)*TERRAIN_STEP, y+(   s*stair_height)*TERRAIN_STEP),
                        (x+((1+s)*stair_width)*TERRAIN_STEP, y+(-1+s*stair_height)*TERRAIN_STEP),
                        (x+(    s*stair_width)*TERRAIN_STEP, y+(-1+s*stair_height)*TERRAIN_STEP),
                        ]
                    self.fd_polygon.shape.vertices=poly
                    t = self.world.CreateStaticBody(
                        fixtures = self.fd_polygon)
                    t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                    self.terrain.append(t)
                
                # add
                s=stair_steps-1
                self.terrain_state.append((STAIRS,x,x+((1+s)*stair_width)*TERRAIN_STEP))
                
                counter = stair_steps*stair_width

            elif state==STAIRS and not oneshot:  # 階段
                s = stair_steps*stair_width - counter - stair_height
                n = s/stair_width
                y = original_y + (n*stair_height)*TERRAIN_STEP
            
            
            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter==0:
                counter = self.np_random.randint(TERRAIN_GRASS/2, TERRAIN_GRASS)
                if state==GRASS and hardcore:
                	#####--- GRASS=0, STUMP=1, STAIRS=2, PIT=3, _STATE_=4
                    state = self.np_random.randint(1, _STATES_) # 1,2,3
                    oneshot = True
                else:
                    state = GRASS
                    oneshot = True
        #--- end of for loop ---
        
        
        self.terrain_poly = []
        for i in range(TERRAIN_LENGTH-1):
            poly = [
                (self.terrain_x[i],   self.terrain_y[i]),
                (self.terrain_x[i+1], self.terrain_y[i+1])
                ]
            self.fd_edge.shape.vertices=poly
            t = self.world.CreateStaticBody(
                fixtures = self.fd_edge)
            color = (0.3, 1.0 if i%2==0 else 0.8, 0.3)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.4, 0.6, 0.3)
            poly += [ (poly[1][0], 0), (poly[0][0], 0) ]
            self.terrain_poly.append( (poly, color) )
        self.terrain.reverse()       # Why reverse ???
        
        
    def _generate_clouds(self):
        # Sorry for the clouds, couldn't resist
        self.cloud_poly   = []
        for i in range(TERRAIN_LENGTH//20):
            x = self.np_random.uniform(0, TERRAIN_LENGTH)*TERRAIN_STEP
            y = VIEWPORT_H/SCALE*3/4
            poly = [
                (x+15*TERRAIN_STEP*math.sin(3.14*2*a/5)+self.np_random.uniform(0,5*TERRAIN_STEP),
                 y+ 5*TERRAIN_STEP*math.cos(3.14*2*a/5)+self.np_random.uniform(0,5*TERRAIN_STEP) )
                for a in range(5) ]
            x1 = min( [p[0] for p in poly] )
            x2 = max( [p[0] for p in poly] )
            self.cloud_poly.append( (poly,x1,x2) )

    def reset(self):
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.prev_shaping = None
        self.scroll = 0.0
        self.lidar_render = 0

        # 表示するWxHを定義している
        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE

        # _generate_terrain　障害物の作成　雲の作成
        self._generate_terrain(self.hardcore)
        self._generate_clouds()

        init_x = TERRAIN_STEP*TERRAIN_STARTPAD/2
        init_y = TERRAIN_HEIGHT+2*LEG_H
        self.hull = self.world.CreateDynamicBody(
            position = (init_x, init_y),
            fixtures = HULL_FD
                )
        self.hull.color1 = (0.5,0.4,0.9)
        self.hull.color2 = (0.3,0.3,0.5)
        self.hull.ApplyForceToCenter((self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), True)

        self.legs = []
        self.joints = []
        for i in [-1,+1]:
            leg = self.world.CreateDynamicBody(
                position = (init_x, init_y - LEG_H/2 - LEG_DOWN),
                angle = (i*0.05),
                fixtures = LEG_FD
                )
            leg.color1 = (0.6-i/10., 0.3-i/10., 0.5-i/10.)
            leg.color2 = (0.4-i/10., 0.2-i/10., 0.3-i/10.)
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=leg,
                localAnchorA=(0, LEG_DOWN),
                localAnchorB=(0, LEG_H/2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed = i,
                lowerAngle = -0.8,
                upperAngle = 1.1,
                )
            self.legs.append(leg)
            self.joints.append(self.world.CreateJoint(rjd))

            lower = self.world.CreateDynamicBody(
                position = (init_x, init_y - LEG_H*3/2 - LEG_DOWN),
                angle = (i*0.05),
                fixtures = LOWER_FD
                )
            lower.color1 = (0.6-i/10., 0.3-i/10., 0.5-i/10.)
            lower.color2 = (0.4-i/10., 0.2-i/10., 0.3-i/10.)
            rjd = revoluteJointDef(
                bodyA=leg,
                bodyB=lower,
                localAnchorA=(0, -LEG_H/2),
                localAnchorB=(0, LEG_H/2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed = 1,
                lowerAngle = -1.6,
                upperAngle = -0.1,
                )
            lower.ground_contact = False
            self.legs.append(lower)
            self.joints.append(self.world.CreateJoint(rjd))

        self.drawlist = self.terrain + self.legs + [self.hull]
        
        
        #--- add
        self.c0=0  # this is a just counter.
        self.state0 = ''
        self.state_out=[ 0, 0, 0, 0] # add three elements
        self.x_see_length= np.zeros(10) # add
        self.obstacle_x_end= np.zeros(4) # x point of current obstacle end
        #---
        
        """
  // Ray-cast input data. The ray extends from p1 to p1 + maxFraction * (p2 - p1).
  struct b2RayCastInput
  {
      b2Vec2 p1, p2;
      float32 maxFraction;
  };
   The points p1 and p2 are used to define a direction for the ray, and the maxFraction specifies how far along the ray
   
  // Ray-cast output data. The ray hits at p1 + fraction * (p2 - p1), where p1 and p2
  // come from b2RayCastInput.
  struct b2RayCastOutput
  {
      b2Vec2 normal;
      float32 fraction;
  };
  If the ray does intersect the shape, b2Fixture::RayCast will return true and we can look in the output struct 
  to find the actual fraction of the intersect point, and the norma垂線? of the fixture取り付けて動かせない物 'surface' at that point: 
        """
        class LidarCallback(Box2D.b2.rayCastCallback): #　ライダー
             # -1 to filter, 0 to terminate, fraction to clip the ray for closest hit, 1 to continue
             def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:  # Class: b2.Fixture   categoryBits: (number) The collision category bits
                    return 1
                self.p2 = point
                self.fraction = fraction
                return 0
        self.lidar = [LidarCallback() for _ in range(10)]

        return self.step(np.array([0,0,0,0]))[0]

    #=======================================================================================================
    #
    #
    #
    #
    def step(self, action):
        #
        
        
        
        #self.hull.ApplyForceToCenter((0, 20), True) -- Uncomment this to receive a bit of stability help
        control_speed = False  # Should be easier as well
        if control_speed:
            self.joints[0].motorSpeed = float(SPEED_HIP  * np.clip(action[0], -1, 1))
            self.joints[1].motorSpeed = float(SPEED_KNEE * np.clip(action[1], -1, 1))
            self.joints[2].motorSpeed = float(SPEED_HIP  * np.clip(action[2], -1, 1))
            self.joints[3].motorSpeed = float(SPEED_KNEE * np.clip(action[3], -1, 1))
        else:
            self.joints[0].motorSpeed     = float(SPEED_HIP     * np.sign(action[0]))
            self.joints[0].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[0]), 0, 1))
            self.joints[1].motorSpeed     = float(SPEED_KNEE    * np.sign(action[1]))
            self.joints[1].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[1]), 0, 1))
            self.joints[2].motorSpeed     = float(SPEED_HIP     * np.sign(action[2]))
            self.joints[2].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[2]), 0, 1))
            self.joints[3].motorSpeed     = float(SPEED_KNEE    * np.sign(action[3]))
            self.joints[3].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[3]), 0, 1))

        self.world.Step(1.0/FPS, 6*30, 2*30)

        pos = self.hull.position
        vel = self.hull.linearVelocity
        
        # 最大で　math.sin(1.5*i/10.0)*LIDAR_RANGE= around 5.2　の距離まで光線が出る
        # ライダー
        
        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5*i/10.0)*LIDAR_RANGE,
                pos[1] - math.cos(1.5*i/10.0)*LIDAR_RANGE)
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)
            self.x_see_length[i]=  self.lidar[i].p2[0]  # add
        
        # add
        #GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)  # GRASS=0, STUMP=1, STAIRS=2, PIT=3, _STATE_=4
        self.state0= 'GRASS '
        for s0,x1,x2 in self.terrain_state:
            
            if pos.x >= x1 and pos.x <= x2:
                if  s0 == 1:
                    self.state0= 'STUMP'
                elif  s0 == 2:
                    self.state0 = 'STAIRS'
                elif  s0 == 3:
                    self.state0 = 'PIT'
            
            # sp1先に　障害を発見したら, 1をセットする。  Lidarの１スパンより障害物が大きいことを満たすか？？？ 微妙だね。。。
            # 
            
            if  self.x_see_length[self.sp1] >= x1 and self.x_see_length[self.sp1+1] <= x2:
                if  s0 == 1 and self.state_out[1] ==0:
                    self.state_out[1] = 1  # see STUMP
                    self.obstacle_x_end[1]= x2
                    self.state_out[0] = 0  # clear GRASS state at onnce here! 
                if s0 == 2 and self.state_out[2] ==0:
                    self.state_out[2] = 1  # see STAIRS
                    self.obstacle_x_end[2]= x2
                    self.state_out[0] = 0  # clear GRASS state at onnce here! 
                if s0 == 3 and self.state_out[3] ==0:
                    self.state_out[3] = 1  # see PIT
                    self.obstacle_x_end[3]= x2
                    self.state_out[0] = 0  # clear GRASS state at onnce here! 
                
                
                
            
            #　障害を通過したら0にリセットする。 HULLの後方が通過するまで。
            if  self.state_out[1] > 0 and (pos.x + HULL_POLY[0][0]/SCALE) > self.obstacle_x_end[1]:
                self.state_out[1] = 0  #  passed STUMP
            if  self.state_out[2] > 0 and (pos.x + HULL_POLY[0][0]/SCALE) > self.obstacle_x_end[2]:
                self.state_out[2] = 0  #  passed STAIRS
            if  self.state_out[3]  > 0 and (pos.x + HULL_POLY[0][0]/SCALE) > self.obstacle_x_end[3]:
                self.state_out[3] = 0  #  passed PIT
        
        # 障害のflagがすべて0なら GRASSをセットする
        if self.state_out[1] == 0 and self.state_out[2] == 0 and self.state_out[3] == 0:
            self.state_out[0]=1
        
        # add
        if self.dump_out:
            print('def step', self.c0, pos.x, pos.y, self.state0, self.state_out ); self.c0 += 1
            if sum(self.state_out) > 1: print ('error: self.state_out  > 1')
        
        
        state = [
            self.hull.angle,        # Normal angles up to 0.5 here, but sure more is possible.
            2.0*self.hull.angularVelocity/FPS,
            0.3*vel.x*(VIEWPORT_W/SCALE)/FPS,  # Normalized to get -1..1 range
            0.3*vel.y*(VIEWPORT_H/SCALE)/FPS,
            self.joints[0].angle,   # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
            self.joints[0].speed / SPEED_HIP,
            self.joints[1].angle + 1.0,
            self.joints[1].speed / SPEED_KNEE,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.joints[2].angle,
            self.joints[2].speed / SPEED_HIP,
            self.joints[3].angle + 1.0,
            self.joints[3].speed / SPEED_KNEE,
            1.0 if self.legs[3].ground_contact else 0.0
            ]
        state += [l.fraction for l in self.lidar]
        
        
        # Add three element
        if self.add_state_out:
            state += self.state_out
            assert len(state)==28  # 28
        else:
            assert len(state)==24  # 24

        self.scroll = pos.x - VIEWPORT_W/SCALE/5


        # REWARDの計算をするようだが・・・ shaping が関係しているのはなぜ？ pos[0]は胴体ポジション
        shaping  = 130*pos[0]/SCALE   # moving forward is a way to receive reward (normalized to get 300 on completion)
        # shaping から減点している
        shaping -= 5.0*abs(state[0])  # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping  # 前回のshapingの差として REWARDが定義されている！
        self.prev_shaping = shaping

        # モータートルクに応じて　 REWARD が減点されている
        for a in action:
            reward -= 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        done = False
        if self.game_over or pos[0] < 0:
            reward = -100
            done   = True
        if pos[0] > (TERRAIN_LENGTH-TERRAIN_GRASS)*TERRAIN_STEP:
            done   = True
        return np.array(state), reward, done, {}

    def render(self, mode='human'):  # 描く！
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        self.viewer.set_bounds(self.scroll, VIEWPORT_W/SCALE + self.scroll, 0, VIEWPORT_H/SCALE)

        self.viewer.draw_polygon( [
            (self.scroll,                  0),
            (self.scroll+VIEWPORT_W/SCALE, 0),
            (self.scroll+VIEWPORT_W/SCALE, VIEWPORT_H/SCALE),
            (self.scroll,                  VIEWPORT_H/SCALE),
            ], color=(0.9, 0.9, 1.0) )
        for poly,x1,x2 in self.cloud_poly:
            if x2 < self.scroll/2: continue
            if x1 > self.scroll/2 + VIEWPORT_W/SCALE: continue
            self.viewer.draw_polygon( [(p[0]+self.scroll/2, p[1]) for p in poly], color=(1,1,1))
        
        # 　地面の部分を塗りつぶすため
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll: continue
            if poly[0][0] > self.scroll + VIEWPORT_W/SCALE: continue
            self.viewer.draw_polygon(poly, color=color)
        
        self.lidar_render = (self.lidar_render+1) % 100  # 100 
        i = self.lidar_render
        if i < 2*len(self.lidar):
            l = self.lidar[i] if i < len(self.lidar) else self.lidar[len(self.lidar)-i-1]
            self.viewer.draw_polyline( [l.p1, l.p2], color=(1,0,0), linewidth=1 )
            
        
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                    
                else:  # BipedalWalker と　環境を描く
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)
                
        flagy1 = TERRAIN_HEIGHT
        flagy2 = flagy1 + 50/SCALE
        x = TERRAIN_STEP*3
        self.viewer.draw_polyline( [(x, flagy1), (x, flagy2)], color=(0,0,0), linewidth=2 )
        f = [(x, flagy2), (x, flagy2-10/SCALE), (x+25/SCALE, flagy2-5/SCALE)]
        self.viewer.draw_polygon(f, color=(0.9,0.2,0) )
        self.viewer.draw_polyline(f + [f[0]], color=(0,0,0), linewidth=2 )

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

class BipedalWalkerHardcoreEdit2(BipedalWalkerEdit2):
    hardcore = True
    



if __name__=="__main__":
    # Heurisic: suboptimal, have no notion of balance.
    env = BipedalWalkerHardcoreEdit2()
    env.reset() # class init 以降で　resetをもう一度よんでいる
    steps = 0
    total_reward = 0
    a = np.array([0.0, 0.0, 0.0, 0.0])
    STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1,2,3
    SPEED = 0.29  # Will fall forward on higher speed
    state = STAY_ON_ONE_LEG
    moving_leg = 0
    supporting_leg = 1 - moving_leg
    SUPPORT_KNEE_ANGLE = +0.1
    supporting_knee_angle = SUPPORT_KNEE_ANGLE
    while True:
        s, r, done, info = env.step(a)
        total_reward += r
        
        
        # show status ....
        if 0:
            if steps % 20 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                print("hull " + str(["{:+0.2f}".format(x) for x in s[0:4] ]))
                print("leg0 " + str(["{:+0.2f}".format(x) for x in s[4:9] ]))
                print("leg1 " + str(["{:+0.2f}".format(x) for x in s[9:14]]))
        steps += 1

        contact0 = s[8]
        contact1 = s[13]
        moving_s_base = 4 + 5*moving_leg
        supporting_s_base = 4 + 5*supporting_leg

        hip_targ  = [None,None]   # -0.8 .. +1.1
        knee_targ = [None,None]   # -0.6 .. +0.9
        hip_todo  = [0.0, 0.0]
        knee_todo = [0.0, 0.0]

        if state==STAY_ON_ONE_LEG:
            hip_targ[moving_leg]  = 1.1
            knee_targ[moving_leg] = -0.6
            supporting_knee_angle += 0.03
            if s[2] > SPEED: supporting_knee_angle += 0.03
            supporting_knee_angle = min( supporting_knee_angle, SUPPORT_KNEE_ANGLE )
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[supporting_s_base+0] < 0.10: # supporting leg is behind
                state = PUT_OTHER_DOWN
        if state==PUT_OTHER_DOWN:
            hip_targ[moving_leg]  = +0.1
            knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[moving_s_base+4]:
                state = PUSH_OFF
                supporting_knee_angle = min( s[moving_s_base+2], SUPPORT_KNEE_ANGLE )
        if state==PUSH_OFF:
            knee_targ[moving_leg] = supporting_knee_angle
            knee_targ[supporting_leg] = +1.0
            if s[supporting_s_base+2] > 0.88 or s[2] > 1.2*SPEED:
                state = STAY_ON_ONE_LEG
                moving_leg = 1 - moving_leg
                supporting_leg = 1 - moving_leg

        if hip_targ[0]: hip_todo[0] = 0.9*(hip_targ[0] - s[4]) - 0.25*s[5]
        if hip_targ[1]: hip_todo[1] = 0.9*(hip_targ[1] - s[9]) - 0.25*s[10]
        if knee_targ[0]: knee_todo[0] = 4.0*(knee_targ[0] - s[6])  - 0.25*s[7]
        if knee_targ[1]: knee_todo[1] = 4.0*(knee_targ[1] - s[11]) - 0.25*s[12]

        hip_todo[0] -= 0.9*(0-s[0]) - 1.5*s[1] # PID to keep head strait
        hip_todo[1] -= 0.9*(0-s[0]) - 1.5*s[1]
        knee_todo[0] -= 15.0*s[3]  # vertical speed, to damp oscillations
        knee_todo[1] -= 15.0*s[3]

        a[0] = hip_todo[0]
        a[1] = knee_todo[0]
        a[2] = hip_todo[1]
        a[3] = knee_todo[1]
        a = np.clip(0.5*a, -1.0, 1.0)

        env.render()
        if done: break
