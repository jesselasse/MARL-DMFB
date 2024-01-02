import random
import numpy as np
from enum import IntEnum
import time
from datetime import MINYEAR, datetime


# action space
class Action(IntEnum):
    STALL = 0
    RIGHT = 1  # or right for T2
    LEFT = 2  # or left for T2
    DOWN = 3  # or forward for T2
    UP = 4  # or reverse for T2


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
        return 'Blocks Occupies the grid from ({},{}) to ({},{})' \
            .format(self.x_min, self.y_min, self.x_max, self.y_max)

    def isPointInside(self, points):
        ''' point is in the form of (x, y); points: ((x,y),...) '''
        for i in points:
            if i[0] >= self.x_min and i[0] <= self.x_max and \
                    i[1] >= self.y_min and i[1] <= self.y_max:
                return True
        else:
            return False

    def isBlockOverlap(self, m):
        if self._isLinesOverlap(self.x_min, self.x_max, m.x_min, m.x_max) and \
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

class Droplet:

    def __init__(self, coordinate, opid):
        self.pos: list = coordinate
        self.start_x, self.start_y = coordinate
        self.opid = opid
        self.hidden_state = None
        self.last_action = None
        self.difficulty=0

    def get_position(self):
        return (self.pos[0], self.pos[1])

    def set_position(self, x, y):
        self.pos[0] = x
        self.pos[1] = y

    def refresh(self):
        self.pos = [self.start_x, self.start_y]

    def shfit_x(self, step):
        self.pos[0] += step
        # if self.pos[0] > self.max_x:
        #     self.pos[0] = self.max_x
        # elif self.pos[0] < 0:
        #     self.pos[0] = 0


    def shfit_y(self, step):
        self.pos[1] += step
        # if self.pos[1] > self.max_y:
        #     self.pos[1] = self.max_y
        # elif self.pos[1] < 0:
        #     self.pos[1] = 0

    def move(self, action):
        ''' try to move the droplet'''
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

    def fail2movestep(self):
        pass



    def getdir(self, action):
        if action == Action.STALL:
            return (0,0)
        elif action == Action.UP:
            return (0,1)
        elif action == Action.DOWN:
            return (0,-1)
        elif action == Action.LEFT:
            return (-1,0)
        elif action == Action.RIGHT:
            return (1,0)
        else:
            raise TypeError('action is illegal')


class Droplet_T1(Droplet):
    type = 0

    def __init__(self, start, destination=(3,3), opid=0, partner=None):
        # start, destination是列表, destination 可以是元组： out or non-config
        super().__init__(start, opid)
        self.des = destination
        self.difficulty = 0.1*self.distance
        self.partner = partner

    def __repr__(self):
        return 'T1 Droplet at ({},{}) and its target destination is ({},{})'.format(
            self.pos[0], self.pos[1], self.des[0], self.des[1])

    def __eq__(self, droplet):
        if isinstance(droplet, Droplet_T1):
            flag = (self.pos[0] == droplet.pos[0]) and (self.pos[1] == droplet.pos[1]) and (
                        self.des[0] == droplet.des[1]) \
                   and (self.des[0] == droplet.des[1])
            return flag
        else:
            return False

    @property
    def distance(self):
        # Manhattan's distance
        return abs(self.pos[0] - self.des[0]) + abs(self.pos[1] - self.des[1])

    @property
    def finished(self):
        return not self.distance

    def try2move(self, action):
        return np.array(self.pos)+np.array(self.getdir(action))

    def move_reward(self, action): # ensured to move successfully
        o_distance = self.distance
        super().move(action)
        new_dist = self.distance
        # stall in final position
        if new_dist == o_distance and o_distance == 0:
            return -0.1
        # stall in past pisition
        elif new_dist == o_distance and action == 0:
            return -0.25
        # closer to the destination
        elif new_dist < o_distance:
            return -0.1
        # penalty for taking one more step
        else:
            return -0.4

    def direct_vector(self,width,length,hf):
        tar=np.array(self.des)
        pos=np.array(self.pos)
        drx, dry = tar - pos
        # direct victor for T1 (goal inside fov: exact value; goal outside fov: zoom to 10*10
        if abs(drx) > hf:
            if drx > 0:
                drx = round((drx - hf) / ((width - hf) / (10 - hf))) + hf
            else:
                drx = round((drx + hf) / ((width - hf) / (10 - hf))) - hf
        if abs(dry) > hf:
            if dry > 0:
                dry = round((dry - hf) / ((length - hf) / (10 - hf))) + hf
            else:
                dry = round((dry + hf) / ((length - hf) / (10 - hf))) - hf
        dirct = np.array([drx, dry])
        return dirct



class Droplet_T2(Droplet):
    type = 1
    d = ['+x', '-x', '-y', '+y']
    fullmix = 1

    def __init__(self, coordinate, opid=0, mix_update=(0.0029, 0.0058, 0.001, -0.005)):
        # coordinat_x, coordinate_y: int
        # mix_update: [0.0029,0.0058, 0.001, -0.005]
        super().__init__(coordinate, opid)
        self.mix_percent = 0.0
        self.last_actions = [1, 0] #默认朝+x
        self.direction = {'left': {1: 4, 2: 3, 3: 1, 4: 2}, 'right': {1: 3, 2: 4, 3: 2, 4: 1},
                          'reverse': {1: 2, 2: 1, 3: 4, 4: 3}}
        self.mix_update = mix_update
        # self.half_reward = [True, True, True, True, True, True, True, True, True]
        # self.reward_thr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __eq__(self, droplet):
        if isinstance(droplet, Droplet_T2):
            flag = (self.pos[0] == droplet.pos[0]) and (self.pos[1] == droplet.pos[1]) and (
                        self.mix_percent == droplet.mix_percent)
            return flag
        else:
            return False

    def __repr__(self):
        return 'T2 Droplet at ({},{}) and head to {}'.format(self.pos[0], self.pos[1], self.d[self.headto-1])

    @property
    def finished(self):
        return self.mix_percent >= self.fullmix

    def try2move(self, direction):
        if direction == 0:
            action = 0
        elif direction == 1:  # forward  p 3
            action = self.last_actions[0]
        elif direction == 2:  # backward  p 4
            action = self.direction['reverse'][self.last_actions[0]]
        elif direction == 3:  # p 1
            action = self.direction['right'][self.last_actions[0]]
        elif direction == 4:  # p 2
            action = self.direction['left'][self.last_actions[0]]
        return np.array(self.pos)+np.array(self.getdir(action))

    def move_reward(self, direction):
        if direction == 0:
            self.last_actions[1] = 0 #防止出现连续两步向前的计算
            return -0.3
        if direction == 1:  # forward
            action = self.last_actions[0]
            if self.last_actions[0] == self.last_actions[1]:# 连续两步向前
                addpercent = 1
            else: # 向前1步
                addpercent = 0
            reward = -0.1
        elif direction == 2:  # backward
            action = self.direction['reverse'][self.last_actions[0]]
            addpercent = 3
            reward = -0.5
        elif direction == 3:
            action = self.direction['right'][self.last_actions[0]]
            addpercent = 2
            reward = -0.2
        elif direction == 4:
            action = self.direction['left'][self.last_actions[0]]
            addpercent = 2
            reward = -0.2
        super().move(action)
        self.last_actions[1] = self.last_actions[0]
        self.last_actions[0] = action
        self.mix_percent = min(1.0, self.mix_percent + self.mix_update[addpercent])
        self.mix_percent= max(0,self.mix_percent)
        # for i in range(len(self.half_reward)):
        #     if self.half_reward[i] and self.mix_percent > self.reward_thr[i]:
        #         reward += 1
        #         self.half_reward[i] = False
        #         return reward
        return reward
        # if action == 0:
        #     self.last_actions[1] = 0
        #     return 0.3
        # if action == self.last_actions[0]:
        #     reward = -0.1
        #     if self.last_actions[0] == self.last_actions[1]:# 连续两步向前
        #         addpercent = 1
        #     else:
        #         addpercent = 0
        # elif action == self.direction['reverse'][self.last_actions[0]]:
        #     reward = -0.4
        # else:
        #     reward = -0.25
        # super().move(action)
        # self.last_actions[1] = self.last_actions[0]
        # self.last_actions[0] = action
        # self.mix_percent = min(1.0, self.mix_percent + self.mix_update[addpercent])
        # for i in range(len(self.half_reward)):
        #     if self.half_reward[i] and self.mix_percent > self.reward_thr[i]:
        #         reward += 1
        #         self.half_reward[i] = False
        #         return reward
        # return reward

    def fail2movestep(self):
        self.last_actions[1] = 0

    @property
    def headto(self):
        return self.last_actions[0]

    def direct_vector(self, *args):
        return super().getdir(self.last_actions[0])


class Droplet_store(Droplet):
    type = 2

    def __init__(self, start, opid=0):
        # start, destination是列表, destination 可以是元组： out or non-config
        super().__init__(start, opid)
        self.ref = 0
        self.duration = None
        self.time = 0

    def move_reward(self, action):
        super().move(action)
        self.time += 1
        # stall in final position
        return 0

    def direct_vector(self, *args):
        return (0,0)

    @property
    def finished(self):
        if self.duration:
            return self.time >= self.duration
        return False



class Droplets(list):
    def __init__(self, *args):
        super().__init__(*args)

    def append(self, drop) -> None:
        super().append(drop)

    def add(self, task, *args, opid=0, difficulty=0):
        Drop = [Droplet_T1, Droplet_T2, Droplet_store]
        d = Drop[task](*args, opid=opid)
        d.difficulty+=difficulty
        super().append(d)
        return d

    def addcp(self, pos1, pos2, opid=0):
        d1 = Droplet_T1(pos1, pos2, opid=opid)
        d2 = Droplet_T1(pos2, pos1, opid=opid)
        d1.partner = d2
        d2.partner = d1
        super().extend([d1, d2])


class Chip:
    def __init__(self, w, l, n_block=0, b_degrade=False, per_degrade=0.1):
        self.size = (w, l)
        self.width = w
        self.length = l
        self.blocks = []
        self.ports = []
        self.m_health = np.ones((w, l))
        self.m_usage = np.zeros((w, l))
        self.b_degrade = b_degrade
        self.chip_changes = b_degrade
        self.m_degrade = self._random_health_statue()
        self.per_degrade = per_degrade
        self.generate_random_chip(n_block=n_block)

    def generate_random_chip(self, points=[], n_dis=8, n_block=0):
        self.ports = []
        self.blocks = []
        self.create_dispense(n_dis)
        self.create_block(n_block, points)

    def create_block(self, n_block, points=[]):
        # Generate random blocks up to n_blocks
        if self.width < 5 or self.length < 5:
            return []
        if n_block * 4 / (self.width * self.length) > 0.2:
            print('Too many required modules in the environment.')
            return []
        self.blocks = []

        def isblocksoverlap(m, blocks):
            for mdl in blocks:
                if mdl.isBlockOverlap(m):
                    return True
            return False

        for i in range(n_block):
            size = 1
            y = np.random.randint(0, self.length - 3)
            x = np.random.randint(0, self.width - 3)
            m = Block(x, x + size, y, y + size)
            while m.isPointInside(points) or isblocksoverlap(m, self.blocks):
                y = random.randrange(0, self.length - 3)
                x = random.randrange(0, self.width - 3)
                m = Block(x, x + 1, y, y + 1)
            self.blocks.append(m)

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

    def create_dispense(self, n):
        length = self.length
        width = self.width
        boundary = [(0, i) for i in range(length)] + [(i, 0) for i in range(1, width)] + [(width - 1, i) for i in
                                                                                          range(length)] + [
                       (i, length - 1) for i in range(1, width - 1)]
        chosen = []
        while len(chosen) < n:
            pos = np.random.choice(len(boundary))
            if all(abs(boundary[pos][0] - c[0]) + abs(boundary[pos][1] - c[1]) > 2 or (
                    abs(boundary[pos][0] - c[0]) != abs(boundary[pos][1] - c[1]) and abs(boundary[pos][0] - c[0]) + abs(
                boundary[pos][1] - c[1]) == 2) for c in chosen):
                chosen.append(boundary.pop(pos))
        self.ports = chosen

    def addUsage(self, pos):
        if not pos:
            return
        if type(pos[0]) == int:
            self.m_usage[pos] += 1
        else:
            for p in pos:
                self.m_usage[tuple(p)] += 1

    def updateHealth(self):
        index = self.m_usage > 50.0  # degrade here
        self.m_health[index] = self.m_health[index] * self.m_degrade[index]
        self.m_usage[index] = 0

    def _isinsideBlocks(self, points):
        for m in self.blocks:
            if m.isPointInside(points):
                return True
        return False


def randomXY(w, l, n):
    y = np.random.randint(0, l, size=(n, 1))
    x = np.random.randint(0, w, size=(n, 1))
    points = np.hstack((x, y))
    return points


def compute_norm_squared_EDM(x):
    x = x.T
    m, n = x.shape
    G = np.dot(x.T, x)
    H = np.tile(np.diag(G), (n, 1))
    return H + H.T - 2 * G

def euclidean_distance_matrix(points):
    # 计算点与点之间的差值
    point_diff = points[:, np.newaxis] - points

    # 计算差值的平方并按行求和
    distance_matrix = np.sqrt(np.sum(point_diff ** 2, axis=2))

    return distance_matrix