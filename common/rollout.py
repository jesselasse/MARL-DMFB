import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time
import collections


Experience=collections.namedtuple('Experience', field_names=['o','u','r','avail_u', 'terminated','padded'])

class Evaluator:
    # 评估使用的类
    def __init__(self, env, agents):
        self.agents = agents
        self.env = env
        # self.n_agents=agents.n_agents
        self.n_actions=agents.n_actions

    def one_step(self, obs, hidden_states, epsilon=0):
        avail_actions=self.env.routing_manager.get_avail_action(self.n_actions)
        actions, hidden_state = self.agents.choose_action4all(obs, avail_actions, epsilon, hidden_states)
        # for agent_id in range(self.env.n_agents):
        #     d = self.env.routing_manager.droplets[agent_id]
        #     # action, hidden_state = self.agents.choose_action(
        #     #     obs[agent_id], avail_actions[agent_id], epsilon, d.last_action, d.hidden_state)
        #     action, hidden_state = np.random.choice(avail_actions[agent_id]), d.hidden_state # for test
        #     # generate onehot vector of th action
        #     action_onehot = np.zeros(self.n_actions)
        #     action_onehot[action] = 1
        #     actions.append(int(action))
        #     actions_onehot.append(action_onehot)
        #     d.hidden_state=hidden_state
        #     d.last_action = action_onehot
        # taction, _ = self.agents.choose_action(obs[0].reshape((1, -1)), avail_actions[0], epsilon,
        #                                       np.array([0, 1, 0, 0, 0]), hidden_states[0].reshape((1, 1, -1)))
        new_obs, r, terminated, info = self.env.step(actions)
        r = np.sum(r) / len(r)
        experience = Experience(new_obs, np.reshape(actions, [self.env.n_agents, 1]), [r], avail_actions, [terminated], [0.])
        self.env.render()
        return experience, info, new_obs, hidden_state

    def _generate_episode(self, drop_num=None): #evaluate
        # if self.args.replay_dir != '' and episode_num == 0:  # prepare for save replay of evaluation
        #     self.env.close()
        episode_limit= self.env.routing_manager.max_step
        obs=self.env.reset(drop_num) # 会更新degrade
        terminated = False
        step = 0
        success = 0
        reward = 0  # cumulative rewards
        constraints = 0
        hidden_states = None
        while not terminated and step < episode_limit:
            experience, info, obs, hidden_states= self.one_step(obs,hidden_states)
            reward += experience.r[0]
            constraints += info['constraints']
            success += info['success']
            terminated = experience.terminated[0]
            step += 1
        if not success:
            step = episode_limit

        # if episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
        #     # self.env.save_replay()
        #     self.env.close()

        # 之前测试task2用的
        # if self.env.routing_manager.task ==1:
        #     mp=[]
        #     for d in self.env.routing_manager.droplets:
        #         mp.append(d.mix_percent)
        #     step=np.array((mp)).mean()

        return reward, step, constraints, success

    def evaluate(self, task_num, drop_num):
        # 运行task_num个episode, 输出指标的平均值
        episode_rewards = 0
        episode_steps = 0
        episode_constraints = 0
        total_success = 0
        # 2022.6.1 jc修改平均步长计算，不成功按最大长度算
        for epoch in range(task_num):
            # for epoch in range(2)
            # 2021.6.7 添加每个epoch的总steps
            episode_reward, total_step, total_constraints, success = self._generate_episode(drop_num)
            episode_rewards += episode_reward
            episode_steps += total_step  # 计算所有的步长
            episode_constraints += total_constraints
            total_success += success
        self.env.close()
        return episode_rewards / task_num, episode_steps / task_num, episode_constraints / task_num, total_success / task_num



class RolloutWorker(Evaluator):
    # 训练时用的类，会将每一步信息保存下来
    def __init__(self, env, agents, args):
        super().__init__(env, agents)
        self.n_actions = args.n_actions
        self.epsilon_anneal_scale = args.train.epsilon_anneal_scale
        self.epsilon = args.train.epsilon
        self.min_epsilon = args.train.min_epsilon
        self.anneal_epsilon = (args.train.epsilon - args.train.min_epsilon) / args.train.anneal_steps
        print('Init RolloutWorker')

    def generate_episode(self, drop_num=None):
        episode_limit = self.env.routing_manager.max_step
        episode = collections.defaultdict(list)
        obs=self.env.reset(drop_num) # 生成新的任务，会更新degrade
        terminated = False
        step = 0
        success = 0
        reward = 0  # cumulative rewards
        constraints = 0

        # epsilon
        epsilon = self.epsilon
        if self.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        episode['o'].append(obs)
        hidden_states=None

        while not terminated and step < episode_limit:
            experience, info, obs, hidden_states = self.one_step(obs, hidden_states, epsilon)
            for i in range(experience.__len__()):
                episode[experience._fields[i]].append(experience[i])
            reward += experience.r[0]
            constraints += info['constraints']
            success += info['success']
            terminated = experience.terminated[0]
            if self.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
            step += 1

        # if step < self.episode_limit，padding
        if episode:
            episode['terminated'][-1]=[1]


        # add episode dim
        for key in episode.keys():
            episode[key] = torch.tensor(np.array([episode[key]]))
        self.epsilon = epsilon
        # jc加的,不成功按最大步长算
        if not success:
            step = episode_limit
        return reward, step, constraints, success, episode
