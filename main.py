from runner import Runner
from common.arguments import get_common_args, get_mixer_args
from dmfb import*

if __name__ == '__main__':
    Psuccess=[]
    Tavg=[]
    args = get_common_args()
    args = get_mixer_args(args)
    # ----一次运行FF
    env = DMFBenv(args.chip_size, args.chip_size, args.drop_num,
                  args.block_num, fov=args.fov, stall=args.stall)
    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"] # 就是max_step 最大步长约束
    runner = Runner(env, args)
    if not args.evaluate:
        runner.run(args.ith_run)
    else:
        average_episode_rewards, average_episode_steps, average_episode_constraints, success_rate = runner.evaluate()
        Psuccess.append(success_rate)
        Tavg.append(average_episode_steps)
        print('The averege total_rewards of {} is  {}'.format(
            args.alg, average_episode_rewards))
        print('The each epoch total_steps is: {}'.format(
            average_episode_steps))
        print('The successful rate is: {}'.format(success_rate))
    env.close()
    # ----
    if args.evaluate:
      np.save('Psuccess_{}_{}'.format(args.chip_size,args.drop_num),Psuccess)
      np.save('Tavg_{}_{}'.format(args.chip_size,args.drop_num),Tavg)
