from common.arguments import get_train_args
import numpy as np
from train import Trainer

if __name__ == '__main__':
    args, ENV = get_train_args()
    # path = args.result_dir + '/' + args.alg + '/fov{}/{}by{}-{}d{}b/{}_env({},{},{},{},{},{})'.format(args.fov,
    #                                         args.width,args.length, args.drop_num, args.block_num,
    #     args.alg, args.width, args.length, args.drop_num, args.block_num, args.fov, args.stall)

    if args.load_model:
        env = ENV(args.width, args.length, args.drop_num,
                      args.block_num, fov=args.fov, stall=args.stall)
        args.__dict__.update(env.get_env_info())
        runner = Trainer(env, args, evaluate=True)
        runner.evaluate_total()
        rewards=runner.episode_rewards
        steps=runner.episode_steps
        constraints=runner.episode_constraints
        success_rate=runner.success_rate
        runtime=runner.time_cost
    else:
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