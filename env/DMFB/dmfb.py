import copy
import math
import os
import random
import sys
import time
from datetime import MINYEAR, datetime
from enum import IntEnum
from typing_extensions import runtime

from numpy.lib.function_base import select
import gym
import numpy as np
from gym import spaces, wrappers
from gym.utils import seeding
from numpy.random import poisson
from pettingzoo.utils.env import ParallelEnv
from PIL import Image
import cv2
'''
DMFBs MARL enviroment created by R.Q. Yang
'''


# action space
class Action(IntEnum):
    STALL = 0
    RIGHT= 1
    LEFT = 2
    DOWN = 3
    UP = 4

# class block and inter method
class Block:
    def __init__(self, x_min, x_max, y_min, y_max):
        if x_min > x_max or y_min > y_max:
            raise TypeError('Module() inputs are illegal')
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def __repr__(self):
        return 'Blocks Occupies the grid from ({},{}) to ({},{})'\
            .format(self.x_min, self.y_min, self.x_max, self.y_max)

    def isPointInside(self, point):
        ''' point is in the form of (x, y) '''
        for i in point:
            if i[0] >= self.x_min and i[0] <= self.x_max and\
                    i[1] >= self.y_min and i[1] <= self.y_max:
                return True
        else:
            return False

    def isBlockOverlap(self, m):
        if self._isLinesOverlap(self.x_min, self.x_max, m.x_min, m.x_max) and\
                self._isLinesOverlap(self.y_min, self.y_max, m.y_min, m.y_max):
            return True
        else:
            return False

    def _isLinesOverlap(self, xa_1, xa_2, xb_1, xb_2):
        if xa_1 > xb_2:
            return False
        elif xb_1 > xa_2:
            return False
        else:
            return True

# droplet class


class Droplet():
    def __init__(self, coordinate_x, coordinate_y, destination_x, destination_y):
        self.x = int(coordinate_x)
        self.y = int(coordinate_y)
        self.des_x = int(destination_x)
        self.des_y = int(destination_y)

    def __repr__(self):
        return 'Droplet is at ({},{}) and its target destination is ({},{})'.format(
            self.x, self.y, self.des_x, self.des_y)

    def __eq__(self, droplet):
        if isinstance(droplet, Droplet):
            flag = (self.x == droplet.x) and (self.y == droplet.y) and (self.des_x == droplet.des_x)\
                and (self.des_y == droplet.des_y)
            return flag
        else:
            return False

    @property
    def distance(self):
        return abs(self.x-self.des_x)+abs(self.y-self.des_y)

    def shfit_x(self, step):
        self.x += step

    def shfit_y(self, step):
        self.y += step

    def move(self, action, width, length):
        # width :vertical cell number, length:horizental cell number
        if action == Action.STALL:
            pass
        elif action == Action.UP:
            self.shfit_y(1)
        elif action == Action.DOWN:
            self.shfit_y(-1)
        elif action == Action.LEFT:
            self.shfit_x(-1)
        elif action == Action.RIGHT:
            self.shfit_x(1)
        else:
            raise TypeError('action is illegal')
        if self.x > width-1:
            self.x = width-1
        elif self.x < 0:
            self.x = 0
        if self.y > length-1:
            self.y = length-1
        elif self.y < 0:
            self.y = 0


class RoutingTaskManager:
    def __init__(self, width, length, n_droplets,
                 n_blocks=0, fov=5, stall=True, b_degrade=False, per_degrade=0.1):
        self.width = width
        self.length = length
        self.n_droplets = n_droplets
        self.n_blocks = n_blocks
        self.droplets = []
        self.starts = np.zeros((self.n_droplets, 2), dtype=int)
        self.ends = np.zeros((self.n_droplets, 2), dtype=int)
        self.distances = np.zeros((self.n_droplets,), dtype=int)
        self.blocks = []
        if fov > min(width, length):
            raise RuntimeError('Fov is too large')
        self.fov = fov
        self.stall = stall
        self.global_obs = np.zeros((3, width, length), dtype=int)
        droplet_limit = int((self.width+1)*(self.length+1)/9)
        if n_droplets > droplet_limit:
            raise TypeError('Too many droplets for DMFB')
        self.m_health = np.ones((width, length))
        self.m_usage = np.zeros((width, length))
        self.b_degrade = b_degrade
        self.per_degrade = per_degrade
        self.m_degrade = self._random_health_statue()
        # variables below change every game
        self.step_count = 0
        random.seed(datetime.now())
        self.Generate_task()

    def _random_health_statue(self):
        if self.b_degrade:
            m_degrade = np.random.rand(self.width, self.length)
            m_degrade = m_degrade * 0.4 + 0.6
            selection = np.random.rand(self.width, self.length)
            per_healthy = 1. - self.per_degrade
            m_degrade[selection < per_healthy] = 1.0  # degradation factor
            return m_degrade
        else:
            return np.ones((self.width, self.length))

    def Generate_task(self):
        self.GenDroplets()
        self.distances = np.sum(np.abs(self.starts - self.ends), axis=1)
        self.GenRandomBlocks()

    # reset the enviorment
    def refresh(self, new=False):
        self.droplets.clear()
        self.blocks.clear()
        self.Generate_task()
        if new:
            self.m_health = np.ones((self.width, self.length))
            self.m_usage = np.zeros((self.width, self.length))
            self.m_degrade = self._random_health_statue()
        else:
            self.updateHealth()

    def restartforall(self):
        self.droplets.clear()
        for i in range(0, self.n_droplets):
            self.droplets.append(
                Droplet(self.starts[i][0], self.starts[i][1], self.ends[i][0], self.ends[i][1]))
        self.distances = np.sum(np.abs(self.starts - self.ends), axis=1)

    def GenDroplets(self):
        Start_End = self._Generate_Start_End()
        self.starts = Start_End[0:self.n_droplets]
        self.ends = Start_End[self.n_droplets:]
        for i in range(0, self.n_droplets):
            self.droplets.append(
                Droplet(self.starts[i][0], self.starts[i][1], self.ends[i][0], self.ends[i][1]))

    def compute_norm_squared_EDM(self, x):
        x = x.T
        m, n = x.shape
        G = np.dot(x.T, x)
        H = np.tile(np.diag(G), (n, 1))
        return H+H.T-2*G

    def _Generate_Start_End(self):
        def randomXY(w, l, n):
            y = np.random.randint(0, l, size=(n*2, 1))
            x = np.random.randint(0, w, size=(n*2, 1))
            Start_End = np.hstack((x, y))
            return Start_End
        Start_End = randomXY(self.width, self.length, self.n_droplets)
        dis = self.compute_norm_squared_EDM(Start_End)
        strided = np.lib.stride_tricks.as_strided
        s0, s1 = dis.strides
        m = dis.shape[0]
        out = strided(dis.ravel()[1:], shape=(m-1, m),
                      strides=(s0+s1, s1)).reshape(m, -1)
        while out.min() <= 2:
            Start_End = randomXY(self.width, self.length, self.n_droplets)
            dis = self.compute_norm_squared_EDM(Start_End)
            s0, s1 = dis.strides
            out = strided(dis.ravel()[1:], shape=(
                m-1, m), strides=(s0+s1, s1)).reshape(m, -1)
        return Start_End

    def GenRandomBlocks(self):
        # Generate random blocks up to n_blocks
        if self.width < 5 or self.length < 5:
            return []
        if self.n_blocks * 4 / (self.width * self.length) > 0.2:
            print('Too many required modules in the environment.')
            return []
        self.blocks = []

        def isblocksoverlap(m, blocks):
            for mdl in blocks:
                if mdl.isBlockOverlap(m):
                    return True
            return False

        for i in range(self.n_blocks):
            y = np.random.randint(0, self.length-3)
            x = np.random.randint(0, self.width-3)
            m = Block(x, x+1, y, y+1)
            while m.isPointInside(np.vstack((self.starts, self.ends))) or isblocksoverlap(m, self.blocks):
                y = random.randrange(0, self.length - 3)
                x = random.randrange(0, self.width - 3)
                m = Block(x, x+1, y, y+1)
            self.blocks.append(m)

    def moveDroplets(self, actions):
        def comflic_static(n_droplets, all_cur_position):
            static_conflic = [0] * n_droplets
            for i in range(n_droplets - 1):
                for j in range(i+1, n_droplets):
                    if np.linalg.norm(all_cur_position[i]-all_cur_position[j]) < 2:
                        static_conflic[i] += 1
                        static_conflic[j] += 1
            return static_conflic

        def comflic_dynamic(n_droplets, all_cur_position, all_past_pisition):
            dynamic_conflict = [0] * n_droplets
            for i in range(n_droplets):
                for j in range(n_droplets):
                    if i != j:
                        if np.linalg.norm(all_past_pisition[i] - all_cur_position[j]) < 2:
                            dynamic_conflict[i] += 1
                            dynamic_conflict[j] += 1
            return dynamic_conflict
        if len(actions) != self.n_droplets:
            raise RuntimeError("The number of actions is not the same"
                               " as n_droplets")
        rewards = []
        pasts = []
        curs = []
        dones = self.getTaskStatus()
        for i in range(self.n_droplets):
            reward, past, cur = self.moveOneDroplet(i, actions[i])
            rewards.append(reward)
            pasts.append(past)
            curs.append(cur)
        sta = np.array(comflic_static(self.n_droplets, np.array(curs)))
        dy = np.array(comflic_dynamic(self.n_droplets, curs, pasts))
        # the nunber of obey the constraints
        constraints = np.sum(sta)+np.sum(dy)
        rewards = np.array(rewards)-2*sta-2*dy
        if self.stall:
            for i in range(self.n_droplets):
                if dones[i]:
                    rewards[i] = 0
        if np.all(self.getTaskStatus()) == True:
            rewards = [i+10 for i in rewards]
            if constraints==0:
                rewards = [i+10 for i in rewards]
        rewards = list(rewards)

        return rewards, constraints

    def _isTouchingBlocks(self, point):
        for m in self.blocks:
            if point[0] >= m.x_min and\
                    point[0] <= m.x_max and\
                    point[1] >= m.y_min and\
                    point[1] <= m.y_max:
                return True
        return False

    def _isinvalidaction(self):
        position = np.zeros((self.n_droplets, 2))
        for i, d in enumerate(self.droplets):
            position[i][0], position[i][1] = d.x, d.y
        dis = self.compute_norm_squared_EDM(position)
        m = dis.shape[0]
        strided = np.lib.stride_tricks.as_strided
        s0, s1 = dis.strides
        out = strided(dis.ravel()[1:], shape=(m-1, m),
                      strides=(s0+s1, s1)).reshape(m, -1)
        if out.min() == 0:
            return True
        else:
            return False

    def moveOneDroplet(self, droplet_index, action):
        if droplet_index >= self.n_droplets:
            raise RuntimeError(
                "The droplet index {} is out of bound".format(droplet_index))
        x = self.droplets[droplet_index].x
        y = self.droplets[droplet_index].y
        if self.stall and self.distances[droplet_index] == 0:
            reward = 0.0
        else:
            prob = self.getMoveProb(self.droplets[droplet_index])
            if random.random() <= prob:
                self.droplets[droplet_index].move(
                    action, self.width, self.length)
                if self._isTouchingBlocks([self.droplets[droplet_index].x, self.droplets[droplet_index].y]):
                    self.droplets[droplet_index].x = x
                    self.droplets[droplet_index].y = y
                if self._isinvalidaction():
                    self.droplets[droplet_index].x = x
                    self.droplets[droplet_index].y = y
            new_dist = self.droplets[droplet_index].distance
            if new_dist == self.distances[droplet_index] and self.distances[droplet_index] == 0:
                reward = -0.1
            # stall in past pisition
            elif new_dist == self.distances[droplet_index] and action == 0:
                reward = -0.25
            # closer to the destination
            elif new_dist < self.distances[droplet_index]:
                reward = -0.1
            else:
                reward = -0.4  # penalty for taking one more step
            self.distances[droplet_index] = new_dist
        past = np.array([x, y])
        cur = np.array([self.droplets[droplet_index].x,
                       self.droplets[droplet_index].y])
        return reward, past, cur

    def getMoveProb(self, droplet):
        prob = self.m_health[droplet.x][droplet.y]
        return prob

    def getTaskStatus(self):
        return [i == 0 for i in self.distances]

    def getglobalobs(self):
        '''
        format of gloabal observed
        3-l-w
        Droplets               in layer 0  [id 0 0] 
        Goal                   in layer 1  [x id 0]
        Obstacles              in layer 2  [0  0 1]
        '''
        def add_blocks_In_gloabal_Obs(g_obs):
            for m in self.blocks:
                for x in range(m.x_min, m.x_max + 1):
                    for y in range(m.y_min, m.y_max + 1):
                        g_obs[2][x][y] = 1
            return g_obs

        def add_droplets_in_gloabal_Obs(g_obs):
            for i in range(self.n_droplets):
                g_obs[0][self.droplets[i].x][self.droplets[i].y] = i+1
                g_obs[1][self.droplets[i].des_x][self.droplets[i].des_y] = i+1
            return g_obs
        global_obs = np.zeros((3, self.width, self.length), dtype=int) # ?
        global_obs = add_blocks_In_gloabal_Obs(global_obs)
        global_obs = add_droplets_in_gloabal_Obs(global_obs)
        self.global_obs = global_obs
        return global_obs

    # partitial observed sertting for droplet-index
    def getOneObs(self, agent_i):
        '''
        format of gloabal observed
        3-l-w
        Droplets               in layer 0  [id 0 0 0]
        other's Goal           in layer 2  [x x id 0]
        Obstacles              in layer 3  [x 0  0 1]
        '''
        fov = self.fov  # 正方形fov边长
        hf=fov//2
        obs_i = np.zeros((3, fov, fov), dtype=np.int8)
        center_x, center_y = self.droplets[agent_i].x, self.droplets[agent_i].y
        tar_x, tar_y = self.droplets[agent_i].des_x, self.droplets[agent_i].des_y
        origin = (center_x-fov//2, center_y-fov//2)
        # get droplet layer 0
        for idx, d in enumerate(self.droplets):
            x, y = d.x-origin[0], d.y-origin[1]
            if (0 <= x < fov) and (0 <= y < fov):
                obs_i[0][x][y] = idx+1       
        ###333333
        # get other's Goal layer 1
        for idx, d in enumerate(self.droplets):
            if idx != agent_i and (abs(d.x-center_x)<fov/2 and abs(d.y-center_y)<fov/2):
                x = np.clip(d.des_x-origin[0], 0, fov-1)
                y = np.clip(d.des_y-origin[1], 0, fov-1)
                obs_i[1][x][y] = idx+1
        # get blocks layer 2 (
        for block in self.blocks:
            for i in range(block.x_min, block.x_max+1):
                for j in range(block.y_min, block.y_max+1):
                    if (0 <= i < fov) and (0 <= j < fov):
                        obs_i[2][i][j] = 1
        # add boundary
        leftbound = hf-center_x
        rightbound = hf-(self.width-1-center_x)
        if leftbound > 0:
            obs_i[2, 0:leftbound, :] = 1
        elif rightbound > 0:
            obs_i[2, -rightbound:, :] = 1
        upbound = hf-center_y
        downbound = hf-(self.length-1-center_y)
        if upbound > 0:
            obs_i[2, :, 0:upbound] = 1
        elif downbound > 0:
            obs_i[2, :, -downbound:] = 1

        # direct victor (goal inside fov: exact value; goal outside fov: zoom to 10*10
        drx=tar_x-center_x
        dry=tar_y-center_y
        if abs(drx) > hf:
            if drx>0:
                drx= round((drx-hf)/((self.width-hf)/(10-hf)))+hf
            else:
                drx= round((drx+hf)/((self.width-hf)/(10-hf)))-hf
        if abs(dry) > hf:
            if dry>0:
                dry= round((dry-hf)/((self.length-hf)/(10-hf)))+hf
            else:
                dry= round((dry+hf)/((self.length-hf)/(10-hf)))-hf
        dirct =np.array([drx, dry], dtype=np.int8)
        # dir = np.array([(tar_x-center_x)/self.width, (tar_y-center_y)/self.length]) #之前x在后y在前是这么定义的
        # dir = np.array([(tar_y - center_y) / self.length, (tar_x - center_x) / self.width])
        return obs_i,dirct

    def addUsage(self):
        done = self.getTaskStatus()
        for i in range(self.n_droplets):
            if not done[i]:
                self.m_usage[self.droplets[i].x][self.droplets[i].y] += 1

    def updateHealth(self):
        # if not self.b_degrade:
        #     return
        # self.m_health = np.exp((self.m_usage/50)*np.log(self.m_degrade))
        index = self.m_usage > 50.0  # degrade here
        self.m_health[index] = self.m_health[index] * self.m_degrade[index]
        self.m_usage[index] = 0


class DMFBenv(ParallelEnv):
    """ A DMFB biochip environment
    [0,0]
        +---l---+-> x
        w       |
        +-------+
        |     [1,2]
        V
        y
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    # 环境初始化

    def __init__(self, width, length, n_agents, n_blocks=0, fov=5, stall=True, b_degrade=False, per_degrade=0.1, show=False, savemp4=False):
        super(DMFBenv, self).__init__()
        assert width >= 5 and length >= 5
        assert n_agents > 0
        self.mode='human' if show else None
        self.actions = Action
        self.agents = ["player_{}".format(i) for i in range(n_agents)]
        # self.agents=list(range(n_agents))
        self.possible_agents = self.agents[:]
        self.action_spaces = {name: spaces.Discrete(len(self.actions))
                              for name in self.agents}
        self.observation_spaces = {name: spaces.Box(
            low=0, high=n_agents, shape=(3, width, length), dtype='uint8')
            for name in self.agents}
        self.rewards = {i: 0. for i in self.agents}
        self.dones = {i: False for i in self.agents}
        # Other data members
        self.width = width
        self.length = length
        self.routing_manager = RoutingTaskManager(
            width, length, n_agents, n_blocks, fov, stall, b_degrade, per_degrade)
        self.max_step = (width + length)*2
        # variables below change every game
        self.step_count = 0
        self.constraints = 0
        # used for render
        self.agent_redender = [None]*len(self.agents)
        self.agenttrans = [None]*len(self.agents)
        self.u_size = 40  # size for cell pixels
        # self.env_width = self.u_size * \
        #     (self.width)    # scenario width (pixels)
        # self.env_length = self.u_size * (self.length)  # height
        self.viewer = None
        self.color_table = [
            [0.98039216, 0.92156863, 0.84313725],
            [0., 1., 1.],
            [0.49803922, 1., 0.83137255],
            [0.39215686, 0.58431373, 0.92941176],
            [0.33333333, 0.41960784, 0.18431373],
            [0.96078431, 0.96078431, 0.8627451],
            [1., 0.89411765, 0.76862745],
            [0., 0., 1.],
            [0.54117647, 0.16862745, 0.88627451],
            [0.64705882, 0.16470588, 0.16470588],
            [0.87058824, 0.72156863, 0.52941176],
            [0.8627451, 0.07843137, 0.23529412],
            [0., 0., 0.54509804],
            [0., 0.54509804, 0.54509804],
            [0., 0.39215686, 0.],
            [0.54509804, 0., 0.54509804],
            [1., 0.54901961, 0.],
            [0.37254902, 0.61960784, 0.62745098],
            [0.49803922, 1., 0.],
            [1., 0.49803922, 0.31372549],
            [0.54509804, 0., 0.]
        ]
        self.save = False
        self.screen_length = self.u_size * length
        self.screen_width = self.u_size * width
        self.video=None
        if savemp4:
            self.save=True
            self.mode = 'human'
            file_path = 'video/{}by{}-{}d{}b'.format(
                width, length, n_agents, n_blocks) + str(
                int(time.time())) + ".avi"  # 导出路径

            fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')  # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）
            fps=12
            self.video = cv2.VideoWriter(file_path, fourcc, fps, (self.screen_width, self.screen_length))



    def step(self, actions, record=True):
        self.step_count += 1
        success = 0
        if isinstance(actions, dict):
            acts = [actions[agent] for agent in self.agents]
        elif isinstance(actions, list):
            acts = actions
        else:
            raise TypeError('wrong actions')
        rewards, constraints = self.routing_manager.moveDroplets(acts)
        if record:
            self.routing_manager.addUsage()
        self.constraints += constraints
        for key, r in zip(self.agents, rewards):
            self.rewards[key] = r
        self.routing_manager.getglobalobs()  # update the state
        obs = self.getObs()  # patitial observed consist of the Obs
        if self.step_count < self.max_step:
            status = self.routing_manager.getTaskStatus()
            if np.all(status) and self.constraints == 0:
                success = 1
            for key, s in zip(self.agents, status):
                self.dones[key] = s
        else:
            for key in self.agents:
                self.dones[key] = True
        info = {'constraints': constraints, 'success': success}
        return obs, self.rewards, self.dones, info

    def reset(self, new=False):
        self.rewards = {i: 0 for i in self.agents}
        self.dones = {i: False for i in self.agents}
        self.step_count = 0
        self.constraints = 0
        self.routing_manager.refresh(new=new) # 会更新degrade, 是new就是新生成的
        obs = self.getObs()
        self.render()
        return obs

    def restart(self, index=None):
        self.routing_manager.restartforall()
        self.rewards = {i: 0.0 for i in self.agents}
        self.dones = {i: False for i in self.agents}
        self.step_count = 0
        self.constraints = 0
        return self.getObs()

    def seed(self, seed=None):
        pass

    def close(self):
        if self.viewer:
            self.render(close=True)

    def getOneObs(self, agent):
        if type(agent) == str:
            index = int(agent[-1])
        else:
            index = agent
        pixel,drc=self.routing_manager.getOneObs(index)
        return np.append(pixel,drc)

    def getObs(self):  # partitial observertion for all droplets
        observations = {}
        for i, agent in enumerate(self.agents):
            observations[agent] = self.getOneObs(i)
        return [observations[agent].reshape(-1) for agent in self.agents]


# 2021 531添加
# used for qmix


    def get_env_info(self):
        pixel, vector= self.routing_manager.getOneObs(0)
        env_info = {"n_actions": len(self.actions),
                    "n_agents": len(self.agents),
                    # "state_shape": self.routing_manager.getglobalobs().flatten().shape[-1],
                    "obs_shape": pixel.shape+(vector.size,pixel.size+vector.size), #(channel,fov,fov,vector lenth, whole size) whole size:obs_shape[-1]
                    "episode_limit": self.max_step}
        return env_info

    def render(self, close=False):
        if self.mode is None:
            return
        def is_goal(position):
            for i in range(len(self.agents)):
                if np.array_equal(position, self.routing_manager.ends[i]):
                    return i
            return False

        def is_out_edge(position):
            if 2 > position[0] or position[0] > self.length+1 or 2 > position[1] or position[1] > self.width+1:
                return True
            return False

        def drawcell(u_size):
            cell = np.ones((u_size, u_size, 3), dtype='uint8') * 255
            cell[:, [0, -2, -1], :] = 0
            cell[[0, -2, -1], :, :] = 0
            return cell


        import pygame

        if close:
            if self.viewer is not None:
                pygame.display.set_mode((800, 800))
                pygame.display.flip()
                # pygame.quit()
                self.viewer = None
            if self.video is not None:
                self.video.release()
            return None

        u_size = self.u_size  # cell 尺寸
        m = 2  # cell 间隔
        img = np.zeros([self.width, self.length, 3])
        screen_length = self.screen_length
        screen_width = self.screen_width
        if self.viewer is None:
            # pygame.init()
            self.viewer = pygame.display.set_mode((screen_width, screen_length)) #rendering.Viewer(self.env_length, self.env_width)
            # 背景
            background = pygame.Surface((screen_width, screen_length))
            background.fill('grey')
            cellarr = drawcell(u_size - 2 * m)
            cell = pygame.image.frombuffer(cellarr.flatten(), (u_size - 2 * m, u_size - 2 * m), 'RGB').convert()
            for x in range(self.width):
                for y in range(self.length):
                    background.blit(cell, (x * u_size + m, y * u_size + m))  # 普通格子

            self.background = background

        self.viewer.blit(self.background, (0, 0))

        # goal
        for i in range(len(self.agents)):
            goalfile = '../fig/goal{}.png'.format(i)
            goal = pygame.image.load(goalfile).convert_alpha()
            goal = pygame.transform.scale(goal, (u_size - m, u_size - m))
            self.viewer.blit(goal, (self.routing_manager.ends[i][0] * u_size, self.routing_manager.ends[i][1] * u_size))

            # # 保存为背景
            # self.background = background



        for i in range(len(self.agents)):
            self.agent_redender[i] = pygame.image.frombuffer(Image.open('../fig/droplet{}.png'.format(i)).resize((u_size,u_size)).tobytes(),(u_size,u_size),'RGBA').convert_alpha()   # 液滴圆圈)
            [x, y] = self.routing_manager.droplets[i].x, self.routing_manager.droplets[i].y
            self.viewer.blit(self.agent_redender[i], (x * u_size, y * u_size))
            img[x][y] = np.multiply(self.color_table[i], 255)

        if self.save:
            imagestring = pygame.image.tostring(self.viewer.subsurface(0, 0,screen_width, screen_length), "RGB")
            pilImage = Image.frombytes("RGB", (screen_width, screen_length), imagestring)
            imag = cv2.cvtColor(np.asarray(pilImage), cv2.COLOR_RGB2BGR)
            self.video.write(imag)

        return pygame.display.flip() # if mode == 'human' else img


class DMFBenv_v0_1(DMFBenv):
    def __init__(self, w, l, n, **kwargs):
        super().__init__(w, l, n, **kwargs)

    def getOneObs(self, agent_i):
        '''
        format of gloabal observed
        4-l-w
        Droplets               in layer 0  [id 0 0 0]
        current droplet's goal in layer 1  [x id x 0]
        other's Goal           in layer 2  [x x id 0]
        Obstacles              in layer 3  [x 0  0 1]
        '''
        fov = self.routing_manager.fov  # 正方形fov边长
        obs_i = np.zeros((4, fov, fov))
        droplets = self.routing_manager.droplets
        center_x, center_y = droplets[agent_i].x, droplets[agent_i].y
        tar_x, tar_y = droplets[agent_i].des_x, droplets[agent_i].des_y
        origin = (center_x-fov//2, center_y-fov//2)
        seeing=[]
        # get droplet layer 0
        for idx, d in enumerate(droplets):
            x, y = d.x-origin[0], d.y-origin[1]
            if (0 <= x < fov) and (0 <= y < fov):
                obs_i[0][x][y] = idx+1
                # modified
                if idx != agent_i:
                    seeing.append((idx, x, y, d, d.distance))

        # get current droplet's goal layer 1
        # # 投影过来 （50 for 4 drop)
        if len(self.agents) < 10:
            x = np.clip(droplets[agent_i].des_x - origin[0], 0, fov-1)
            y = np.clip(droplets[agent_i].des_y - origin[1], 0, fov-1)
            obs_i[1][x][y] = agent_i+1
        ######
        # 不投影 (10 drop)
        else:
            x = droplets[agent_i].des_x - origin[0]
            y = droplets[agent_i].des_y - origin[1]
            if (0 <= x < fov) and (0 <= y < fov):
                obs_i[1][x][y] = agent_i+1
        ###333333
        # get other's Goal layer 2  #modified
        seeing.sort(key=lambda x: x[-1])
        import math
        for idx, x, y, d, _ in seeing:
            dx = d.des_x-d.x
            dy = d.des_y-d.y
            boundx = fov - 1 - x if dx >= 0 else -x
            boundy = fov - 1 - y if dy >= 0 else -y
            if abs(dx)<=abs(boundx) and abs(dy)<=abs(boundy):
                clipdx=dx
                clipdy=dy
            elif dx==0:
                clipdx=0
                clipdy=boundy
            elif dy==0:
                clipdx=boundx
                clipdy=0
            else:
                if dx>=0:
                    clipdx = min(boundx, math.ceil(dx / dy * boundy))
                else:
                    clipdx = max(boundx, math.floor(dx / dy * boundy))
                if dy>=0:
                    clipdy = min(boundy, math.ceil(dy * boundx / dx))
                else:
                    clipdy = max(boundy, math.floor(dy * boundx / dx))
            i, j = x + clipdx, y + clipdy
            if obs_i[2][i][j] == 0:  # 如果没被占
                obs_i[2][i][j] = idx+1
            else:
                if i == x and j == y:
                    continue
                if i+1 < fov and obs_i[2][i + 1][j] == 0:
                    obs_i[2][i + 1][j] = idx+1
                    continue
                if i-1 >=0 and obs_i[2][i - 1][j] == 0:  # 如果没被占
                    obs_i[2][i - 1][j] = idx+1
                    continue
                if j+1 < fov and obs_i[2][i][j + 1] == 0:  # 如果没被占
                    obs_i[2][i][j + 1] = idx+1
                    continue
                if j-1 >=0 and obs_i[2][i][j - 1] == 0:  # 如果没被占
                    obs_i[2][i][j - 1] = idx+1
                    continue



        # get blocks layer 3
        for block in self.routing_manager.blocks:
            for i in range(block.x_min, block.x_max+1):
                for j in range(block.y_min, block.y_max+1):
                    if (0 <= i < fov) and (0 <= j < fov):
                        obs_i[3][i][j] = 1
        # add boundary
        leftbound = fov//2-center_x
        rightbound = fov//2-(self.width-1-center_x)
        if leftbound > 0:
            obs_i[3, 0:leftbound, :] = 1
        elif rightbound > 0:
            obs_i[3, -rightbound:, :] = 1
        upbound = fov//2-center_y
        downbound = fov//2-(self.length-1-center_y)
        if upbound > 0:
            obs_i[3, :, 0:upbound] = 1
        elif downbound > 0:
            obs_i[3, :, -downbound:] = 1
        # dir = np.array([(tar_x-center_x)/self.width, (tar_y-center_y)/self.length]) #之前x在后y在前是这么定义的
        dir = np.array([(tar_y - center_y) / self.length, (tar_x - center_x) / self.width])
        obs_i = np.append(obs_i, dir)
        return obs_i
