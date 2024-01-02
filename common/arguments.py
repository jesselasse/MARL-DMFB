import argparse
from argparse import Namespace
import yaml
from common.config import config

"""
Here are the param for the training

"""


def common_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('name', default='dmfb', choices=['dmfb', 'meda'], help='dmfb or meda')
    parser.add_argument('--seed', type=int, default=12, help='random seed')
    parser.add_argument('--alg', type=str, default='vdn',
                        help='the algorithm to train the agent')
    # parser.add_argument('--n_episodes', type=int, default=2,
    #                     help='the number of episodes before once training')
    parser.add_argument('--cuda', default=True, action='store_false',
                        help='whether to use the GPU')
    parser.add_argument('--evaluate_task', type=int, default=100,
                        help='evaluate the model: the average performance over #evaluate_task random generated routing tasks')
    parser.add_argument('--model_dir', type=str,
                        default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str,
                        default='./TrainResult', help='result directory of the policy')
    parser.add_argument('--load_model', default=False, action='store_true',
                        help='whether to load the pretrained model')
    parser.add_argument('--load_model_name', type=str, default='', help=' we can choose the file name, 例：1_7500 '
                             '会找到指定1_7500_rnn_net_params.pkl文件')
    # parser.add_argument('--evaluate','-e', default=False, action='store_true',
    #                     help='whether to evaluate the model')

    parser.add_argument('--stall', default=True, action='store_false',
                        help='whether the droplet can move or not after reach the target')
    parser.add_argument('--drop_num', '-d', type=int, default=4,
                        help='the number of droplet')
    parser.add_argument('--oc', type=int, default=None,
                        help='which obs definition do we use')
    parser.add_argument('--max_n_drop', '-m', type=int, default=None,
                        help='the maximum number of droplet while training if we use curriculum learning')
    parser.add_argument('--block_num', type=int, default=0,
                        help='the number of block')
    parser.add_argument('--net', type=str, default=None, choices=['crnn', 'FC4','CRatt'],
                        help='the architecture of policy')
    parser.add_argument('--width', '-w', '--chip_size', help='Width of the biochip', type=int, default=None)
    parser.add_argument('--length', '-l', help='Length of the biochip', type=int, default=None)
    parser.add_argument('--version', '-v', help='version: None or 0.1', type=str, default=None)
    parser.add_argument('--task', default=0, choices=[-1, 0, 1], type=int,
                        help='which task to train. choices: 0--trainsporting'
                             ' droplets; 1--mixing; all--containing all kind of droplets (transport, mix, store)')
    return parser

def set_default(args):
    if args.name=='dmfb':
        if args.width is None:
            args.width=10
            args.length=10
        elif args.length is None:
            args.length=args.width
    elif args.name == 'meda':
        if args.version is None:
            args.version = '0.2'
        if args.width is None:
            if args.drop_num == 10:
                args.width = 80
                args.length = 80
            else:
                args.width = 30
                args.length = 60
        elif args.length is None:
            args.length = args.width
    obs_set={'crnn':5, 'FC4':0, 'CRatt':1}
    # 允許訓練時在外面指定？
    # if args.net is not None:
    #     args.netdata[args.task].net = args.net
    if args.oc is None:
        args.oc=5
    # if args.oc is not None:
    #     args.netdata[args.task].oc = args.oc
    if args.load_model_name:
        args.load_model=True
        args.netdata[args.task].load_model_name=args.load_model_name
    return args

def train_args(parser):
    parser.add_argument('--n_steps', type=int,
                        default=30, help='total time steps for training *100000')
    parser.add_argument('--ith_run', '-i', type=int, default=0, help='save for ith running. 第几次运行，主要体现在保存文件的后缀名上')
    parser.add_argument('--replay_dir', type=str, default='',
                        help='absolute path to save the replay')
    parser.add_argument('--evaluate_cycle', type=int,
                        default=100000, help='how often to evaluate the model; we evaluate the model each \'evaluate_cycle\' time steps\' training')
    parser.add_argument('--online_eval', default=True, action='store_false',
                        help='evaluate until training finish. default: evaluate while training')

    return parser

def process_train_args(args, pri=True):
    args = set_default(args)
    ENV, Chip, Manager=config(args.name, args.version)
    if args.task == 0:
        filename ='TrainParas/Task1.yaml'
    elif args.task == 1:
        filename = 'TrainParas/Task2.yaml'
    else:
        filename = 'TrainParas/4d.yaml'
    with open(filename) as f:
        netdata, traindata = yaml.safe_load_all(f.read())
    args.netdata, args.train={},{}
    args.netdata[args.task] = Namespace(**netdata)
    args.train=Namespace(**traindata)
    args.train.gamma=args.netdata[args.task].gamma
    # args.max_n_agents=args.drop_num+1
    args.n_steps=args.n_steps*100000
    if pri:
        # print('drop number:', args.drop_num)
        print('chip size:', args.width, '*', args.length)
        print('FOV size:', args.netdata[args.task].fov)
        print('observation id:', args.oc)
    return args, ENV, Chip, Manager


def get_train_args(recieve=None, pri=True):
    parser = common_args()
    parser = train_args(parser)
    args = parser.parse_args(recieve)
    return process_train_args(args, pri)

def evaluate_args(parser):
    parser.add_argument('--show', default=False, action='store_true',
                        help='show the droplet env')
    parser.add_argument('--show_save', default=False, action='store_true',
                        help='show the droplet env and save video')
    parser.add_argument('--b-degrade', default=True)
    parser.add_argument('--per-degrade', help='Percentage of degrade', type=float, default=0)
    parser.add_argument('--evaluate_epoch', help='used for degree evaluation: evaluate #evaluate_epoch*#task_size episodes, performance is calculated every #task_size episodes and generate #evaluate_epoch results', type=float, default=20)
    parser.set_defaults(load_model=True)
    return parser

def process_evaluate_args(args):
    args = set_default(args)
    ENV, Chip, Manager= config(args.name, args.version)
    if args.task == 0:
        filename ='TrainParas/Task1.yaml'
    else:
        # filename = 'TrainParas/Task1.yaml'
        filename = 'TrainParas/Task2.yaml'
    with open(filename) as f:
        netdata, data = yaml.safe_load_all(f.read())
    args.netdata = {}
    args.netdata[args.task] = Namespace(**netdata)
    args.netdata[args.task].load_model_name = args.load_model_name


    return args, ENV, Chip, Manager

def get_evaluate_args():
    parser = common_args()
    parser = evaluate_args(parser)
    args = parser.parse_args()
    return process_evaluate_args(args)

def process_assay_args(args):
    args = set_default(args)
    ENV, Chip, Manager = config(args.name, args.version, train=False)
    args.netdata = []
    for filename in ['TrainParas/Task1.yaml', 'TrainParas/Task2.yaml']:
        with open(filename) as f:
            nd, _ = yaml.safe_load_all(f.read())
            args.netdata.append(Namespace(**nd))
    if not hasattr(args.netdata[0], 'load_model_name'):
        args.netdata[0].load_model_name=args.load_model_name
        args.netdata[1].load_model_name=args.load_model_name2
    return args, ENV, Chip, Manager


if __name__ == '__main__':
    args=get_train_args()
    print(args)