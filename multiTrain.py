from common.arguments import get_train_args
from train import Trainer
from printcompare import processdata, pltsame
import sys


# args, ENV, Chip, Manager  = get_train_args(recieve=sys.argv[1:]+['dmfb', '--task=0', '-m=10', '--net=crnn', '--oc=3'], pri=False)
args, ENV, Chip, Manager = get_train_args()
args.model_dir='./model/oc{}'.format(args.oc)
args.result_dir='./TrainResult/oc{}'.format(args.oc)
all_data =[]
for i in range(6,10):
    args.ith_run = i
    nargs=args.netdata
    manager = Manager(Chip(args.width, args.length), task=args.task, fov=[nargs[args.task].fov,nargs[args.task].fov], oc=args.oc,
                      stall=args.stall)
    env = ENV(manager)
    print(args)
    args.__dict__.update(env.get_env_info())
    # args.obs_shape = args.obs_shape[args.task]
    runner = Trainer(env, args)
    runner.run()# 不在线评估
    record = runner.evaluate_total()
    all_data.append(record)

dropnums = range(3, args.max_n_drop + 1)
total_data = processdata(all_data, dropnums, f'task{args.task+1}')
for d in dropnums:
    fig = pltsame(total_data[d], d)
    fig.savefig(runner.save_path + f'/pltddrop_{d}.png',
                bbox_inches='tight')
# 多个fov多个液滴数目测试
# for f in [7,5,9]:
#     args.fov=f
#     for d in [3,4]:
#         args.drop_num=d
#         args.ith_run=5
#         # args.fov = fov
#         args.load_model = False
#         print('drop number:', args.drop_num)
#         print('chip size:', args.width, '*', args.length)
#         print('FOV size:', args.fov)
#         # ----一次运行FF
#         env = ENV(args.width, args.length, args.drop_num,
#                   n_blocks=args.block_num, fov=args.fov, stall=args.stall)
#         args.__dict__.update(env.get_env_info())
#         runner = Trainer(env, args)
#         runner.run()  # 不在线评估


#
# args, ENV = get_train_args(recieve=['meda', '-d=10', '--n_steps=80', '-v=0.1'])
# for i in [0, 1, 4, 5, 6]:
#     args.ith_run = i
#     # ----一次运行FF
#     env = ENV(args.width, args.length, args.drop_num,
#               n_blocks=args.block_num, fov=args.fov, stall=args.stall)
#     args.__dict__.update(env.get_env_info())
#     runner = Trainer(env, args)
#     runner.run(online_evaluate=True)  # 在线评估