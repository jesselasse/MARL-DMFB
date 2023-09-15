from agent.agent import Agents
from common.rollout import Evaluator
from common.arguments import get_evaluate_args
import time


if __name__ == '__main__':
    Psuccess=[]
    Tavg=[]
    args, ENV, Chip, Manager = get_evaluate_args()
    for i in range(1):
        # ----一次运行FF
        start=time.time()
        assayManager = Manager(Chip(args.width, args.length), task=args.task, fov=[args.fov, args.fov], stall=args.stall)
        env = ENV(assayManager, show=args.show, savemp4=args.show_save)
        args.__dict__.update(env.get_env_info())
        args.obs_shape = args.obs_shape[args.task]
        evaluator = Evaluator(env, Agents(args, task=args.task), args.episode_limit)
        average_episode_rewards, average_episode_steps, _, success_rate = evaluator.evaluate(args.evaluate_task, args.drop_num)
        Psuccess.append(success_rate)
        Tavg.append(average_episode_steps)
        print('time:',time.time()-start)
        print('The average total_rewards of {} is  {}'.format(
            args.alg, average_episode_rewards))
        print('The average total_steps is: {}'.format(
            average_episode_steps))
        print('The successful rate is: {}'.format(success_rate))
    # ----
    # if args.evaluate:
    # np.save('Psuccess_{}_{}_{}'.format(args.width, args.drop_num, args.fov), Psuccess)
    # np.save('Tavg_{}_{}_{}'.format(args.width, args.drop_num, args.fov), Tavg)

