from common.arguments import common_args, process_evaluate_args, get_train_args
import numpy as np
from agent.agent import Agents
from common.rollout import Evaluator
from train import Trainer


def print_evaluate_for_drop_num():
    def extended_parser():
        parser = common_args()
        parser.add_argument('--ints', metavar='N', type=int, nargs='*', help='挑选哪一次或几次训练需要展示，默认就是文件夹里所有的模型')
        args = parser.parse_args()
        print(args.ints)
        return process_evaluate_args(args)

    args, ENV, Chip, Manager = extended_parser()
    nargs = args.netdata
    save_path = args.result_dir + '/' + args.alg + '/fov{}/task{}'.format(nargs[args.task].fov, args.task) + '/CL'

    def plts(data, dropnums):
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
    assayManager = Manager(Chip(args.width, args.length), task=args.task, fov=[nargs[args.task].fov,nargs[args.task].fov],  oc=args.oc, stall=args.stall)
    env = ENV(assayManager)
    print(args)
    args.__dict__.update(env.get_env_info())
    episode_rewards, episode_steps, episode_constraints, success_rate = {}, {}, {}, {}
    record = [episode_rewards, episode_steps, episode_constraints, success_rate]
    if args.task ==0:
        dropnums= [3, 4, 5, 6, 8, 9, 10]
        chip_sizes = [10, 10, 15, 16, 18, 19, 20]
    if args.task==1:
        dropnums = [3, 4, 5]
        chip_sizes= [10, 10, 15]
    for drop_num in dropnums:
        for r in record:
            r[drop_num] = []


    for load_name in range(args.n_steps // args.evaluate_cycle+1):
        if load_name == args.n_steps // args.evaluate_cycle:
            nargs[args.task].load_model_name = '{}_'.format(args.ith_run)
        else:
            nargs[args.task].load_model_name = '{}_{}_'.format(args.ith_run, load_name)
        print(nargs[args.task].load_model_name)
        # print('chip_size updated:', 10)
        evaluator = Evaluator(env, Agents(args, task=args.task))
        args.episode_limit = env.routing_manager.max_step
        for drop_num,size in zip(dropnums, chip_sizes):
                # print('//drop_num:', drop_num,'chip_size updated:', w)
            env.reset_chip(size, size)
            args.episode_limit = env.routing_manager.max_step
            onerecord= evaluator.evaluate(
                args.evaluate_task, drop_num, args.episode_limit)
            # print(drop_num, onerecord)
            for i in range(4):
                record[i][drop_num].append(onerecord[i])
            print('drop_num:', drop_num, 'step:', onerecord[1], 'constraints:', onerecord[2], 'success:', onerecord[3])



        plts(record[-1], dropnums)
        np.save(save_path + '/{}r,step,constraint,success_{}'.format(dropnums, args.ith_run), record)



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
