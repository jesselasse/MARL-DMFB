from common.arguments import get_train_args
import numpy as np
from agent.agent import Agents
from common.rollout import Evaluator
from train import Trainer


def print_evaluate_for_drop_num():
    args, ENV, Chip, Manager = get_train_args()

    def plts(data, dropnums):
        save_path = args.result_dir + '/' + args.alg + '/fov{}/task{}'.format(args.fov, args.task) + '/CL'
        import matplotlib.pyplot as plt
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
        x_name = 'evaluate times,, success_rate'
        for i in range(len(dropnums)):
            plt.subplot(len(dropnums), 1, i + 1)
            plt.plot(data[dropnums[i]], linewidth=2)
            plt.xlabel(x_name)
            plt.ylabel(dropnums[i])
        plt.tight_layout()
        plt.savefig(save_path + '/pltddrop_{}.png'.format(args.ith_run),
                    format='png', dpi=400)
        plt.close()

    # ----一次运行FF
    assayManager = Manager(Chip(args.width, args.length), task=args.task, fov=[args.fov,args.fov], stall=args.stall)
    env = ENV(assayManager)
    print(args)
    args.__dict__.update(env.get_env_info())
    args.obs_shape = args.obs_shape[args.task]
    episode_rewards, episode_steps, episode_constraints, success_rate = {}, {}, {}, {}
    record = [episode_rewards, episode_steps, episode_constraints, success_rate]
    dropnums= [3, 5, 7, 9, 10]
    for drop_num in dropnums:
        for r in record:
            r[drop_num] = []


    for load_name in range(args.n_steps // args.evaluate_cycle):
        args.load_model_name = '{}_{}_'.format(args.ith_run, load_name)
        print(args.load_model_name)
        env.reset_chip(10, 10)
        print('chip_size updated:', 10)
        for drop_num in [3,5,7,9,10]:
            a, b = divmod(drop_num, 5)
            if b == 0:
                w = 10 + a * 5
                print('//drop_num:', drop_num,'chip_size updated:', w)
                env.reset_chip(w, w)
            evaluator = Evaluator(env, Agents(args, task=args.task), env.routing_manager.max_step)
            onerecord= evaluator.evaluate(
                args.evaluate_task, drop_num)
            print(drop_num, onerecord)
            for i in range(4):
                record[i][drop_num].append(onerecord[i])



        plts(record[-1], dropnums)



def print_evaluate_total():
    args, ENV, Chip, Manager = get_train_args()
    # ----一次运行FF
    assayManager = Manager(Chip(args.width, args.length), task=args.task, fov=[args.fov,args.fov], stall=args.stall)
    env = ENV(assayManager)
    print(args)
    args.__dict__.update(env.get_env_info())
    args.obs_shape = args.obs_shape[args.task]
    runner = Trainer(env, args)
    runner.evaluate_total()
    rewards=runner.episode_rewards
    steps=runner.episode_steps
    constraints=runner.episode_constraints
    success_rate=runner.success_rate
    runtime=runner.time_cost
    print('The rewards are:  {}'.format(rewards))
    print('The steps is: {}'.format(steps))
    print('The successful rate are: {}'.format(success_rate))
    print('The runtime are: {}'.format(runtime))
    print('The constraints are: {}'.format(constraints))

def just_print():
    args, ENV, Chip, Manager = get_train_args()
    path = args.result_dir + '/' + args.alg + '/fov{}/{}by{}-{}d{}b/{}_env({},{},{},{},{},{})'.format(args.fov,
                                                                                                      args.width,
                                                                                                      args.length,
                                                                                                      args.drop_num,
                                                                                                      args.block_num,
                                                                                                      args.alg,
                                                                                                      args.width,
                                                                                                      args.length,
                                                                                                      args.drop_num,
                                                                                                      args.block_num,
                                                                                                      args.fov,
                                                                                                      args.stall)
    rewards=np.load(path+'Rewards_{}.npy'.format(args.ith_run))
    steps=np.load(path+'steps_{}.npy'.format(args.ith_run))
    constraints=np.load(path+'constraints_{}.npy'.format(args.ith_run))
    success_rate=np.load(path+'success_rate_{}.npy'.format(args.ith_run))
    runtime=np.load(path+'runtime_{}.npy'.format(args.ith_run))
    print('The rewards are:  {}'.format(rewards))
    print('The steps is: {}'.format(steps))
    print('The successful rate are: {}'.format(success_rate))
    print('The runtime are: {}'.format(runtime))
    print('The constraints are: {}'.format(constraints))

if __name__ == '__main__':
    print_evaluate_for_drop_num()
