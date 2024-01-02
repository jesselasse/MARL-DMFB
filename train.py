import matplotlib.pyplot as plt
from common.replay_buffer import ReplayBuffer
from agent.agent import Agents
from common.rollout import Evaluator, RolloutWorker
import numpy as np
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
plt.switch_backend('agg')


class Trainer:
    def __init__(self, env, args, evaluate=False):
        self.env = env
        if not evaluate:
            self.agents = Agents(args, task=args.task)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)
        self.buffer = ReplayBuffer(args, args.episode_limit)
        self.args = args
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_constraints = []
        self.success_rate = []
        self.time_cost = []
        # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + args.alg + '/fov{}/task{}'.format(args.netdata[args.task].fov, args.task)
        if args.max_n_drop:
            self.save_path += '/CL'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, online_evaluate=False):
        # 进行n_step次训练，其中每evaluation cycle个step进行一次评估；每运行n_episode把数据保存下来训练网络，更新 train_time次参数

        time_steps, trained_times, evaluate_steps = 0, 0, -1
        # n_step 每一次完整实验的总steps.\
        # zqq_count = 0
        start_time = time.time()
        drop_num = self.args.drop_num
        up_drop = None
        if self.args.max_n_drop:
            up_drop = True
            up_drop_steps = [1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
            self.up_drop_sizes = [5, 10, 10, 10, 15, 16, 17, 18, 19, 20, 21]
            # up_drop_steps = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
            self.args.n_steps = sum(up_drop_steps[i]
                                    for i in range(self.args.max_n_drop))*100000
            print('total step:', self.args.n_steps)
            # up_drop = self.args.n_steps//self.args.max_n_drop
            drop_num = 1
            pre_step=0

        while time_steps < self.args.n_steps:
            # if np.mod(zqq_count, 50) == 0:
            #     print('Run {}, time_steps {}'.format(self.args.ith_run, time_steps))
            # zqq_count += 1
            if time_steps // self.args.evaluate_cycle > evaluate_steps:
                evaluate_steps += 1
                temp=time.time()
                self.time_cost.append(temp-start_time)
                print('Run {}, time_steps {}, evaluate {}'.format(self.args.ith_run, time_steps, evaluate_steps), temp-start_time)
                self.agents.policy.save_model(self.args.ith_run, evaluate_steps)  # jc改，每次记录数据和保存模型对应起来吧
                if online_evaluate:
                    average_episode_reward, average_episode_steps, average_episode_constraints, success_rate = self.rolloutWorker.evaluate(
                        self.args.evaluate_task, drop_num)
                    self.episode_rewards.append(average_episode_reward)
                    self.episode_steps.append(average_episode_steps)
                    self.episode_constraints.append(average_episode_constraints)
                    self.success_rate.append(success_rate)
                    self.plt()
                    self.train_data_save()
                    print('success rate:', success_rate, ' evaluate cost:', time.time()-temp)
            episodes = []
            # 收集self.args.n_episodes个episodes
            self.env.routing_manager.set4traintask2()
            self.args.episode_limit = self.env.routing_manager.max_step
            # changing drop_num here
            if up_drop:
                if time_steps >= up_drop_steps[drop_num-1] * 100000+pre_step:
                    pre_step=time_steps
                    drop_num += 1
                    print('time_steps:', time_steps, '//drop_num:', drop_num)
                    # # 如果drop_num 除以5余0，尺寸加5 即，5，10各加5
                    # a,b=divmod(drop_num,5)
                    # self.rolloutWorker.epsilon=0.3
                    # if b == 0:
                    #     w=10+a*5
                    #     print('chip_size updated:', w)
                    #     self.env.reset_chip(w, w)
                    #     args.episode_limit=self.env.routing_manager.max_step
                    #     self.buffer.episode_limit = args.episode_limit
                    if self.env.routing_manager.width != self.up_drop_sizes[drop_num-1]:
                        w=self.up_drop_sizes[drop_num-1]
                        print('chip_size updated:', w)
                        self.env.reset_chip(w, w)
                        self.args.episode_limit=self.env.routing_manager.max_step
                        self.buffer.episode_limit = self.args.episode_limit
            # n_episode 'the number of episodes before once training'
            for episode_idx in range(self.args.train.n_episodes):
                _, steps, _, _, episode = self.rolloutWorker.generate_episode(drop_num=drop_num)
                episodes.append(episode)
                time_steps += steps

            self.buffer.store_episode(episodes)
            for _ in range(self.args.train.train_time):
                mini_batchs = self.buffer.sample(self.args.train.batch_size)
                for mini_batch in mini_batchs:
                    self.agents.train(mini_batch, trained_times)
                trained_times += 1

            self.env.routing_manager.set4traintask2(revers=True)

        self.agents.policy.save_model(self.args.ith_run)
        self.time_cost.append(time.time()-start_time)
        print('Run {}, time_steps {}, evaluate {}'.format(self.args.ith_run, time_steps, evaluate_steps+1),
              temp - start_time)
        if online_evaluate:
            average_episode_reward, average_episode_steps, average_episode_constraints, success_rate = self.rolloutWorker.evaluate(self.args.evaluate_task, drop_num, self.args.episode_limit)
            self.episode_rewards.append(average_episode_reward)
            self.episode_steps.append(average_episode_steps)
            self.episode_constraints.append(average_episode_constraints)
            self.success_rate.append(success_rate)
            self.plt()
            self.train_data_save()
        else:
            self.evaluate_total()


    def evaluate_total(self, show=False):
        args = self.args
        args.load_model=True
        if self.args.max_n_drop:
            record = [{}]*4
            dropnums = range(3, self.args.max_n_drop + 1)
            for drop_num in dropnums:
                for r in record:
                    r[drop_num] = []
            for load_name in range(args.n_steps // args.evaluate_cycle + 1):
                if load_name == args.n_steps // args.evaluate_cycle:
                    args.netdata[args.task].load_model_name = '{}_'.format(args.ith_run)
                else:
                    args.netdata[args.task].load_model_name = '{}_{}_'.format(args.ith_run, load_name)
                print(args.netdata[args.task].load_model_name)
                evaluator = Evaluator(self.env, Agents(args, task=args.task))
                for drop_num in dropnums:
                    size = self.up_drop_sizes[drop_num - 1]
                    # print('//drop_num:', drop_num,'chip_size updated:', w)
                    self.env.reset_chip(size, size)
                    onerecord = evaluator.evaluate(
                        args.evaluate_task, drop_num)
                    # print(drop_num, onerecord)
                    for j in range(4):
                        record[j][drop_num].append(onerecord[j])
                    print('drop_num:', drop_num, 'step:', onerecord[1], 'constraints:', onerecord[2], 'success:',
                          onerecord[3])
            return record
        for load_name in range(args.n_steps // args.evaluate_cycle+1):
            if load_name == args.n_steps // args.evaluate_cycle:
                args.netdata[args.task].load_model_name = '{}_'.format(args.ith_run)
            else:
                args.netdata[args.task].load_model_name = '{}_{}_'.format(args.ith_run, load_name)
            print(args.load_model_name)
            evaluator = Evaluator(self.env, Agents(args, task=args.task))
            average_episode_reward, average_episode_steps, average_episode_constraints, success_rate = evaluator.evaluate(
                args.evaluate_task, self.args.drop_num)
            self.episode_rewards.append(average_episode_reward)
            self.episode_steps.append(average_episode_steps)
            self.episode_constraints.append(average_episode_constraints)
            self.success_rate.append(success_rate)
        self.plt()
        self.train_data_save()
        return

    def plt(self):
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
        y_name = ['Rewards', '$T_{latest}$', 'Constrains', 'success\_rate', 'run time']
        x_name = 'evaluate times, eq {} steps'.format(self.args.evaluate_cycle)
        data = [self.episode_rewards, self.episode_steps,
                self.episode_constraints, self.success_rate, self.time_cost]
        for i in range(5):
            plt.subplot(5, 1, i + 1)
            plt.plot(data[i], linewidth=2)
            plt.xlabel(x_name)
            plt.ylabel(y_name[i])
        plt.tight_layout()
        plt.savefig(self.save_path + '/plt_{}.png'.format(self.args.ith_run),
                    format='png', dpi=400)
        plt.close()

    def train_data_save(self):
        num = self.args.ith_run
        if self.args.max_n_drop:
            prefix='/max_n_drop_{}'.format(self.args.max_n_drop)
        else:
            prefix ='/{}'.format(self.args.alg) +'_env({}by{}-{}d{}b-f{})'.format(
            self.args.width, self.args.length, self.args.drop_num, self.args.block_num, self.args.fov)
        np.save(self.save_path + prefix +
                'Rewards_{}'.format(num), self.episode_rewards)
        np.save(self.save_path + prefix +
                'steps_{}'.format(num), self.episode_steps)
        np.save(self.save_path + prefix +
                'constraints_{}'.format(num), self.episode_constraints)
        np.save(self.save_path + prefix +
                'success_rate_{}'.format(num), self.success_rate)
        np.save(self.save_path + prefix +
                'runtime_{}'.format(num), self.time_cost)


if __name__ == '__main__':
    from common.arguments import get_train_args
    args, ENV, Chip, Manager = get_train_args()
    nargs=args.netdata
    # ----一次运行FF
    manager = Manager(Chip(args.width, args.length), task=args.task, fov=[nargs[args.task].fov,nargs[args.task].fov], oc=args.oc, stall=args.stall)
    env = ENV(manager)
    print(args)
    args.__dict__.update(env.get_env_info())
    nargs = args.netdata[args.task]
    # args.obs_shape = args.obs_shape[args.task]
    runner = Trainer(env, args)
    runner.run(online_evaluate=args.online_eval)