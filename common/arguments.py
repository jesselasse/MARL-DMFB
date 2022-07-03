import argparse

"""
Here are the param for the training

"""


def get_common_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--seed', type=int, default=12, help='random seed')
    parser.add_argument('--ith_run', type=int, default=0, help='save for ith running. 第几次运行，主要体现在保存文件的后缀名上')
    parser.add_argument('--replay_dir', type=str, default='',
                        help='absolute path to save the replay')
    parser.add_argument('--alg', type=str, default='vdn',
                        help='the algorithm to train the agent')
    parser.add_argument('--n_steps', type=int,
                        default=3000000, help='total time steps')
    parser.add_argument('--n_episodes', type=int, default=2,
                        help='the number of episodes before once training')
    parser.add_argument('--last_action', default=True, action='store_false',
                        help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', default=True, action='store_false',
                        help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float,
                        default=0.99, help='discount factor')
    parser.add_argument('--cuda', default=True, action='store_false',
                        help='whether to use the GPU')
    parser.add_argument('--optimizer', type=str,
                        default="ADAM", help='optimizer')
    parser.add_argument('--evaluate_cycle', type=int,
                        default=100000, help='how often to evaluate the model')
    parser.add_argument('--evaluate_epoch', type=int, default=100,
                        help='number of the epoch to evaluate the agent')
    parser.add_argument('--model_dir', type=str,
                        default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str,
                        default='./TrainResult', help='result directory of the policy')
    parser.add_argument('--load_model', default=False, action='store_true',
                        help='whether to load the pretrained model')
    parser.add_argument('--load_model_name', type=str, default='', help=' we can choose the file name, 例：1_7500 '
                             '会找到指定1_7500_rnn_net_params.pkl文件')
    parser.add_argument('--evaluate', default=False, action='store_true',
                        help='whether to evaluate the model')
    parser.add_argument('--show', default=False, action='store_true',
                        help='show the droplet env')
    parser.add_argument('--stall', default=True, action='store_false',
                        help='whether the droplet can move or not after reach the target')
    parser.add_argument('--chip_size', type=int, default=20, help='chip_size')
    parser.add_argument('--drop_num', type=int, default=10,
                        help='the number of droplet')
    parser.add_argument('--block_num', type=int, default=0,
                        help='the number of block')
    parser.add_argument('--net', type=str, default='crnn',
                        help='the architecture of policy')
    parser.add_argument('--fov', type=int, default=9, help='the fov value')
    args = parser.parse_args()
    return args


# arguments of vnd、 qmix
def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 128
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = True
    args.hyper_hidden_dim = 32
    args.lr = 5e-4

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 50000 # 10d 50000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the train steps in one epoch
    args.train_steps = 1

    # experience replay
    args.batch_size = 256
    args.buffer_size = int(1e4)

    # # how often (how many train steps) to save the model
    # args.save_cycle = 2500

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 9

    return args
