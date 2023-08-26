from common.arguments import get_train_args
from train import Trainer
import sys


args, ENV = get_train_args(recieve=sys.argv[1:]+['dmfb', '--n_steps=20'], pri=False)

for f in [7,5,9]:
    args.fov=f
    for d in [3,4]:
        args.drop_num=d
        args.ith_run=5
        # args.fov = fov
        args.load_model = False
        print('drop number:', args.drop_num)
        print('chip size:', args.width, '*', args.length)
        print('FOV size:', args.fov)
        # ----一次运行FF
        env = ENV(args.width, args.length, args.drop_num,
                  n_blocks=args.block_num, fov=args.fov, stall=args.stall)
        args.__dict__.update(env.get_env_info())
        runner = Trainer(env, args)
        runner.run()  # 不在线评估


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