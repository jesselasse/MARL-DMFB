from train import Trainer
from common.arguments import get_common_args, get_mixer_args
from dmfb import DMFBenv
import numpy as np

if __name__ == '__main__':
    Psuccess=[]
    Tavg=[]
    args = get_common_args()
    args = get_mixer_args(args)
    # ----一次运行FF
    env = DMFBenv(args.chip_size, args.chip_size, args.drop_num,
                  args.block_num, fov=args.fov, stall=args.stall, savemp4=args.show_save)
    runner = Trainer(env, args)
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
