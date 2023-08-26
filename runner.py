import matplotlib.pyplot as plt
from common.replay_buffer import ReplayBuffer
from agent.agent import Agents
from common.rollout import RolloutWorker
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
plt.switch_backend('agg')


class Runner:
    def __init__(self, env, args):
        self.env = env
        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        self.buffer = ReplayBuffer(args)
        self.args = args
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_constraints = []
        self.success_rate = []
        # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + args.alg + '/' + '{}by{}-{}d{}b'.format(
            self.args.chip_size, self.args.chip_size, self.args.drop_num, self.args.block_num)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        # 进行n_step次训练，其中每evaluation cycle个step进行一次评估；每运行n_episode把数据保存下来训练网络，更新 train_steps次参数

        time_steps, train_steps, evaluate_steps = 0, 0, -1
        # n_step 每一次完整实验的总steps.\
        zqq_count = 0
        while time_steps < self.args.n_steps:
            if np.mod(zqq_count, 50) == 0:
                print('Run {}, time_steps {}'.format(num, time_steps))
            zqq_count += 1
            if time_steps // self.args.evaluate_cycle > evaluate_steps:
                average_episode_reward, average_episode_steps, average_episode_constraints, success_rate = self.evaluate(self.args.evaluate_epoch)
                self.episode_rewards.append(average_episode_reward)
                self.episode_steps.append(average_episode_steps)
                self.episode_constraints.append(average_episode_constraints)
                self.success_rate.append(success_rate)
                self.plt(num)
                self.train_data_save(num)
                self.agents.policy.save_model(num, train_steps) # jc改，每次记录数据和保存模型对应起来吧
                evaluate_steps += 1
            episodes = []
            # 收集self.args.n_episodes个episodes
            # n_eisode 'the number of episodes before once training'
            for episode_idx in range(self.args.n_episodes):
                _, steps, _, _, episode = self.rolloutWorker.generate_episode(
                    episode_idx)
                episodes.append(episode)
                time_steps += steps
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate(
                        (episode_batch[key], episode[key]), axis=0)
            self.buffer.store_episode(episode_batch)
            for train_step in range(self.args.train_steps):
                mini_batch = self.buffer.sample(
                    min(self.buffer.current_size, self.args.batch_size))
                self.agents.train(mini_batch, train_steps)
                train_steps += 1
        average_episode_reward, average_episode_steps, average_episode_constraints, success_rate = self.evaluate(self.args.evaluate_epoch)
        self.episode_rewards.append(average_episode_reward)
        self.episode_steps.append(average_episode_steps)
        self.episode_constraints.append(average_episode_constraints)
        self.success_rate.append(success_rate)
        self.plt(num)
        self.train_data_save(num)
        self.agents.policy.save_final_model(num)

    def evaluate(self, evaluate_epoch):
        # 运行evaluate epoch个episode, 输出指标的平均值
        episode_rewards = 0
        episode_steps = 0
        episode_constraints = 0
        total_success = 0
        # 2022.6.1 jc修改平均步长计算，不成功按最大长度算
        for epoch in range(evaluate_epoch):
            # for epoch in range(2)
            # 2021.6.7 添加每个epoch的总steps
            episode_reward, total_step, total_constraints, success, _ = self.rolloutWorker.generate_episode(epoch,
                                                                                                            evaluate=True)
            episode_rewards += episode_reward
            episode_steps += total_step  # 计算所有的步长
            episode_constraints += total_constraints
            total_success += success
        return episode_rewards / evaluate_epoch, episode_steps / evaluate_epoch, episode_constraints / evaluate_epoch, total_success / evaluate_epoch

    def plt(self, num):
        import matplotlib.pylab as pylab
        params = {'legend.fontsize': 'x-large',
                  'figure.figsize': (10, 10),
                  'axes.labelsize': 20,
                  'axes.titlesize': 20,
                  'xtick.labelsize': 20,
                  'ytick.labelsize': 20,
                  'figure.dpi': 400,
                  'savefig.dpi': 400}
        pylab.rcParams.update(params)
        y_name = ['Rewards', '$T_{latest}$', 'Constrains', 'success\_rate']
        x_name = 'step*' + str(self.args.evaluate_cycle)
        data = [self.episode_rewards, self.episode_steps,
                self.episode_constraints, self.success_rate]
        for i in range(4):
            plt.subplot(4, 1, i + 1)
            plt.plot(data[i], linewidth=2)
            plt.xlabel(x_name)
            plt.ylabel(y_name[i])
        plt.tight_layout()
        plt.savefig(self.save_path + '/plt_{}.png'.format(num),
                    format='png', dpi=400)

    def train_data_save(self, num):
        prefix = '/{}'.format(self.args.alg) + '_env({},{},{},{},{},{})'.format(
            self.args.chip_size, self.args.chip_size, self.args.drop_num, self.args.block_num, self.args.fov, self.args.stall)
        np.save(self.save_path + prefix +
                'Rewards_{}'.format(num), self.episode_rewards)
        np.save(self.save_path + prefix +
                'steps_{}'.format(num), self.episode_steps)
        np.save(self.save_path + prefix +
                'constraints_{}'.format(num), self.episode_constraints)
        np.save(self.save_path + prefix +
                'success_rate_{}'.format(num), self.success_rate)
